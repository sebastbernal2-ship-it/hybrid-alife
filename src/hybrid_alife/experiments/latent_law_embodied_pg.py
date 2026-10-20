"""Policy-gradient training for the embodied latent-law study.

REINFORCE over a Gaussian movement mean, mirroring the point-organism trainer in
``latent_law_continuous.py``. The rest of the embodied controller stays
deterministic: gates and messages come from the same genome, but only the two
movement means receive exploration noise and a score-function gradient.

The training reward is the same alignment signal the gate measures:
``dot(movement, unit_law)``. Free advection is excluded because it is identical
for every organism.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_alife.agents.controller import forward as controller_forward
from hybrid_alife.agents.embodied import apply_embodied_actions, observe_embodied
from hybrid_alife.experiments.latent_law_embodied import (
    _impose_flow,
    embodied_law_config,
)
from hybrid_alife.experiments.runner import initialize_sim
from hybrid_alife.world.env import positions_to_grid


def single_agent_config(config):
    """Base config with one agent slot, so the genome has batch size one."""
    return replace(
        config,
        embodied=replace(
            config.embodied, population_size=1, initial_alive_fraction=1.0
        ),
    )


def init_trainable_genomes(config) -> dict[str, jax.Array]:
    """Seed one genome from a single-agent population."""
    state = initialize_sim(single_agent_config(config))
    return {name: jnp.asarray(value) for name, value in state.embodied.genomes.items()}


def _unit_flow_at_positions(state) -> jax.Array:
    positions = state.embodied.positions
    grid = positions_to_grid(
        positions, state.world.flow.shape[0], state.world.flow.shape[1]
    )
    flow = state.world.flow[grid[:, 0], grid[:, 1]]
    return flow / jnp.maximum(jnp.linalg.norm(flow, axis=-1, keepdims=True), 1e-8)


def collect_episode(
    genomes: dict[str, jax.Array],
    config,
    *,
    seed: int,
    direction: tuple[float, float],
    steps: int,
    sigma: float,
    key: jax.Array,
    magnitude: float = 0.2,
    warmup_steps: int = 0,
) -> dict[str, jax.Array]:
    """Roll out one episode under the current policy with exploration noise."""
    state = initialize_sim(replace(single_agent_config(config), seed=seed))
    state.embodied.genomes = {name: jnp.asarray(v) for name, v in genomes.items()}
    state = _impose_flow(state, direction, magnitude)
    hidden = state.embodied.hidden
    temperature = max(float(config.embodied.action_temperature), 1e-6)

    observations: list[jax.Array] = []
    actions: list[jax.Array] = []
    hiddens: list[jax.Array] = []
    rewards: list[jax.Array] = []
    for step_index in range(warmup_steps + steps):
        state.embodied.hidden = hidden
        obs = observe_embodied(
            state.embodied, state.world, config.embodied, config.world, key
        )
        raw, new_hidden = controller_forward(
            state.embodied.genomes, obs, hidden, config.embodied.use_memory
        )
        mean = jnp.tanh(raw[:, 0:2] / temperature)
        key, k_noise, k_act = jax.random.split(key, 3)
        move = mean + sigma * jax.random.normal(k_noise, mean.shape)
        gates = jax.nn.sigmoid(raw[:, 2:8] / temperature)
        action = jnp.concatenate([move, gates, raw[:, 8:]], axis=-1)

        unit_flow = _unit_flow_at_positions(state)
        reward = jnp.sum(move * unit_flow, axis=-1)
        state.embodied, state.world, _ = apply_embodied_actions(
            state.embodied, action, state.world, config.embodied, config.world, k_act
        )
        if step_index >= warmup_steps:
            observations.append(obs[0])
            actions.append(move[0])
            hiddens.append(hidden[0])
            rewards.append(reward[0])
        hidden = new_hidden
        state = _impose_flow(state, direction, magnitude)

    return {
        "observations": jnp.stack(observations),
        "actions": jnp.stack(actions),
        "hiddens": jnp.stack(hiddens),
        "rewards": jnp.stack(rewards),
    }


def _returns_to_go(rewards: jax.Array) -> jax.Array:
    def step(carry, reward):
        running = reward + carry
        return running, running

    _, out = jax.lax.scan(step, 0.0, rewards[::-1])
    return out[::-1]


def _pg_loss(
    genomes: dict[str, jax.Array],
    observations: jax.Array,
    hiddens: jax.Array,
    actions: jax.Array,
    advantages: jax.Array,
    config,
    sigma: float,
) -> jax.Array:
    temperature = max(float(config.embodied.action_temperature), 1e-6)

    def step(obs, hidden, action, advantage):
        raw, _ = controller_forward(
            genomes, obs[None], hidden[None], config.embodied.use_memory
        )
        mean = jnp.tanh(raw[:, 0:2] / temperature)[0]
        residual = (action - mean) / sigma
        log_prob = -0.5 * jnp.sum(residual**2)
        return -log_prob * advantage

    return jnp.mean(jax.vmap(step)(observations, hiddens, actions, advantages))


def train_policy_gradient(
    config,
    *,
    directions: tuple[tuple[float, float], ...],
    updates: int = 50,
    episodes_per_update: int = 8,
    steps: int = 8,
    learning_rate: float = 0.05,
    sigma: float = 0.4,
    magnitude: float = 0.2,
    warmup_steps: int = 1,
    random_directions: bool = False,
    key: jax.Array | None = None,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """REINFORCE on the embodied movement mean. Returns trained and initial genomes."""
    if not directions:
        raise ValueError("directions must not be empty")
    if updates < 1 or episodes_per_update < 2:
        raise ValueError("updates must be positive and episodes_per_update at least two")

    genomes = init_trainable_genomes(config)
    initial = {name: jnp.array(value) for name, value in genomes.items()}
    rng = key if key is not None else jax.random.PRNGKey(config.seed + 77_000)
    loss_and_grad = jax.jit(jax.value_and_grad(_pg_loss), static_argnums=(5,))

    for update in range(updates):
        rng, *episode_keys = jax.random.split(rng, episodes_per_update + 1)
        batches = []
        for index in range(episodes_per_update):
            if random_directions:
                angle = 2.0 * jnp.pi * jax.random.uniform(
                    jax.random.fold_in(episode_keys[index], 1)
                )
                direction = (float(jnp.cos(angle)), float(jnp.sin(angle)))
            else:
                direction = directions[
                    (update * episodes_per_update + index) % len(directions)
                ]
            batches.append(
                collect_episode(
                    genomes,
                    config,
                    seed=config.seed + update * episodes_per_update + index,
                    direction=direction,
                    steps=steps,
                sigma=sigma,
                key=episode_keys[index],
                magnitude=magnitude,
                    warmup_steps=warmup_steps,
                )
            )
        observations = np.stack([np.asarray(b["observations"]) for b in batches])
        hiddens = np.stack([np.asarray(b["hiddens"]) for b in batches])
        actions = np.stack([np.asarray(b["actions"]) for b in batches])
        returns = np.stack(
            [np.asarray(_returns_to_go(jnp.asarray(b["rewards"]))) for b in batches]
        )
        # Baseline across episodes at each timestep, then normalize globally.
        # A per-episode baseline would erase the cross-episode signal that tells
        # a flow-following policy apart from a fixed action.
        advantages = returns - returns.mean(axis=0, keepdims=True)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        _, gradients = loss_and_grad(
            genomes,
            jnp.asarray(observations).reshape((-1,) + observations.shape[2:]),
            jnp.asarray(hiddens).reshape((-1,) + hiddens.shape[2:]),
            jnp.asarray(actions).reshape((-1,) + actions.shape[2:]),
            jnp.asarray(advantages).reshape(-1),
            config,
            sigma,
        )
        genomes = jax.tree_util.tree_map(
            lambda value, gradient: value
            - learning_rate * jnp.clip(gradient, -1.0, 1.0),
            genomes,
            gradients,
        )
    return genomes, initial


__all__ = [
    "collect_episode",
    "embodied_law_config",
    "init_trainable_genomes",
    "single_agent_config",
    "train_policy_gradient",
]
