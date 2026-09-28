"""A small, reproducible paired-population PPO-POET experiment.

This module is intentionally separate from the legacy simulator runner.  It
provides a fixed-shape navigation world, neural actor-value policies, PPO
updates, environment mutation, frozen-policy transfer evaluation, and JSON
artifacts suitable for a preregistered campaign.
"""

from __future__ import annotations

import json
import pickle
import platform
import subprocess
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from hybrid_alife.agents.avida_vm import initialize_avida_population, step_avida_population
from hybrid_alife.types import (
    AvidaConfig,
    AvidaPopulationState,
    SimState,
    WorldConfig,
    WorldState,
)
from hybrid_alife.world.env import (
    _compute_enrichment_proxy,
    _compute_lift_proxy,
    _compute_shear,
    _compute_shear_gradient,
    initialize_world,
    step_world,
)

OBS_DIM = 8


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-3
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epochs: int = 3


@dataclass(frozen=True)
class POETConfig:
    width: int = 12
    height: int = 12
    terrain_channels: int = 1
    population_size: int = 4
    policy_hidden: int = 16
    action_count: int = 5
    rollout_steps: int = 16
    ppo: PPOConfig = field(default_factory=PPOConfig)
    environment_param_low: tuple[float, ...] = (0.1, 0.1, 0.1, 0.0, -0.2, -0.2)
    environment_param_high: tuple[float, ...] = (0.9, 0.9, 1.0, 1.0, 0.2, 0.2)
    environment_mutation_std: float = 0.08
    terrain_mutation_std: float = 0.12
    terrain_mutation_prob: float = 0.12
    policy_mutation_std: float = 0.01
    policy_mutation_prob: float = 0.05
    transfer_steps: int = 12
    avida_enabled: bool = False
    avida_lifecycle: str = "reset"
    track: str = "isolated"
    minimal_criterion: float = -10.0
    heldout_environments: int = 4
    heldout_seed_offset: int = 1_000_003

    def __post_init__(self) -> None:
        if self.heldout_environments < 0:
            raise ValueError("heldout_environments must be non-negative")
        if self.avida_lifecycle not in {"reset", "persistent"}:
            raise ValueError("avida_lifecycle must be reset or persistent")
        if self.avida_lifecycle == "persistent" and not self.avida_enabled:
            raise ValueError("persistent Avida requires avida_enabled=True")


class PolicyParams(NamedTuple):
    w1: jax.Array
    b1: jax.Array
    w_policy: jax.Array
    b_policy: jax.Array
    w_value: jax.Array
    b_value: jax.Array


class EnvironmentGenome(NamedTuple):
    params: jax.Array
    terrain: jax.Array

    def dynamics_signature(self, cfg: POETConfig) -> tuple[float, float, float]:
        """Return values that make scalar and terrain changes observable."""
        return (
            float(self.params[2]),
            float(self.params[3]),
            float(jnp.mean(self.terrain)),
        )


@dataclass(frozen=True)
class EnvironmentArchiveEntry:
    environment_id: str
    environment: EnvironmentGenome
    parent_id: str | None
    birth_generation: int
    admission_score: float | None


@dataclass
class POETState:
    environments: tuple[EnvironmentGenome, ...]
    policies: tuple[PolicyParams, ...]
    generation: int
    policy_updates: int
    policy_mutations: int
    environment_mutations: int
    transfer_history: list[np.ndarray]
    metrics: list[dict[str, float]]
    initial_policy: np.ndarray
    final_policy: np.ndarray
    initial_environment_terrain: np.ndarray
    final_environment_terrain: np.ndarray
    track: str
    avida_enabled: bool
    avida_lifecycle: str
    environment_ids: tuple[str, ...]
    environment_archive: list[EnvironmentArchiveEntry]
    archive_transfer_history: list[np.ndarray]
    heldout_environments: tuple[EnvironmentGenome, ...]
    heldout_transfer_history: list[np.ndarray]
    environment_candidates: int
    environment_admissions: int
    environment_rejections: int
    replication_metadata: dict[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "policy_updates": self.policy_updates,
            "policy_mutations": self.policy_mutations,
            "environment_mutations": self.environment_mutations,
            "track": self.track,
            "avida_enabled": self.avida_enabled,
            "avida_lifecycle": self.avida_lifecycle,
            "metrics": self.metrics,
            "transfer_history": [matrix.round(8).tolist() for matrix in self.transfer_history],
            "archive_size": len(self.environment_archive),
            "archive_admissions": self.environment_admissions,
            "archive_rejections": self.environment_rejections,
            "environment_candidates": self.environment_candidates,
            "archive_lineage": [
                {
                    "environment_id": entry.environment_id,
                    "parent_id": entry.parent_id,
                    "birth_generation": entry.birth_generation,
                    "admission_score": entry.admission_score,
                }
                for entry in self.environment_archive
            ],
            "archive_transfer_history": [
                matrix.round(8).tolist() for matrix in self.archive_transfer_history
            ],
            "heldout_transfer_history": [
                matrix.round(8).tolist() for matrix in self.heldout_transfer_history
            ],
            "heldout_environment_count": len(self.heldout_environments),
            "replication_metadata": self.replication_metadata,
        }


@dataclass
class _Batch:
    observations: jax.Array
    actions: jax.Array
    old_log_probs: jax.Array
    advantages: jax.Array
    returns: jax.Array
    rewards: jax.Array


def _source_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _register_pytree_dataclass(cls: type) -> None:
    field_names = tuple(item.name for item in fields(cls))
    jax.tree_util.register_pytree_node(
        cls,
        lambda value: (tuple(getattr(value, name) for name in field_names), None),
        lambda _aux, children: cls(**dict(zip(field_names, children, strict=True))),
    )


_register_pytree_dataclass(WorldState)
_register_pytree_dataclass(AvidaPopulationState)
_register_pytree_dataclass(_Batch)


def _tree_copy(params: PolicyParams) -> PolicyParams:
    return jax.tree_util.tree_map(lambda x: jnp.array(x), params)


def _init_policy(cfg: POETConfig, key: jax.Array) -> PolicyParams:
    k1, k2, k3 = jax.random.split(key, 3)
    hidden = cfg.policy_hidden
    return PolicyParams(
        w1=0.15 * jax.random.normal(k1, (OBS_DIM, hidden)),
        b1=jnp.zeros((hidden,), dtype=jnp.float32),
        w_policy=0.15 * jax.random.normal(k2, (hidden, cfg.action_count)),
        b_policy=jnp.zeros((cfg.action_count,), dtype=jnp.float32),
        w_value=0.15 * jax.random.normal(k3, (hidden,)),
        b_value=jnp.zeros((), dtype=jnp.float32),
    )


def policy_forward(params: PolicyParams, observations: jax.Array) -> tuple[jax.Array, jax.Array]:
    hidden = jnp.tanh(observations @ params.w1 + params.b1)
    logits = hidden @ params.w_policy + params.b_policy
    values = hidden @ params.w_value + params.b_value
    return logits, values


def _poet_world_config(cfg: POETConfig) -> WorldConfig:
    return WorldConfig(
        width=cfg.width,
        height=cfg.height,
        toroidal=True,
        resource_channels=1,
        hazard_channels=1,
        max_agents=cfg.population_size,
        max_digital_organisms=8,
        flow_noise_std=0.0,
        concentration_decay=0.99,
        concentration_diffusion=0.02,
        metabolite_channels=2,
        metabolite_decay=0.95,
        metabolite_diffusion=0.05,
        flow_advection_strength=0.25,
    )


def _poet_avida_config() -> AvidaConfig:
    return AvidaConfig(
        enabled=True,
        population_size=8,
        genome_length=12,
        max_genome_length=24,
        registers=4,
        memory_size=8,
        cycles_per_update=3,
        point_mutation_prob=0.002,
        insertion_prob=0.0,
        deletion_prob=0.0,
        max_cycles_per_update=4,
        metabolite_uptake=0.1,
        metabolite_deposit=0.02,
    )


def _step_world_jitted(
    world: WorldState, rng: jax.Array, cfg: WorldConfig
) -> tuple[WorldState, jax.Array]:
    state = SimState(
        generation=0,
        step=0,
        rng=rng,
        world=world,
        embodied=None,
        avida=None,
    )
    state = step_world(state, cfg)
    return state.world, state.rng


_step_world_fast = jax.jit(_step_world_jitted, static_argnums=2)


def _step_avida_jitted(
    population: AvidaPopulationState,
    world: WorldState,
    cfg: AvidaConfig,
    lineage_counter: int,
    key: jax.Array,
    positions: jax.Array,
) -> tuple[AvidaPopulationState, WorldState, int]:
    population, lineage_counter = step_avida_population(
        population, world, cfg, lineage_counter, key, positions
    )
    return population, world, lineage_counter


_step_avida_fast = jax.jit(_step_avida_jitted, static_argnums=2)


def initialize_environment_world(
    env: EnvironmentGenome, cfg: POETConfig, key: jax.Array
) -> tuple[WorldState, WorldConfig]:
    """Inject an environment genome into the existing microfluidic world model."""
    world_cfg = _poet_world_config(cfg)
    world = initialize_world(world_cfg, key)
    terrain = world.terrain.at[..., 0].set(env.terrain)
    terrain = terrain.at[..., 1].set(0.5 * terrain[..., 1] + 0.5 * env.terrain)
    resources = world.resources.at[..., 0].set(
        jnp.clip(world.resources[..., 0] * env.params[2] + 0.35 * env.terrain, 0.0, 1.0)
    )
    hazards = world.hazards.at[..., 0].set(
        jnp.clip(world.hazards[..., 0] * env.params[3] + 0.08 * (1.0 - env.terrain), 0.0, 1.0)
    )
    flow = world.flow + env.params[4:6]
    shear = _compute_shear(flow)
    shear_grad = _compute_shear_gradient(shear)
    lift = _compute_lift_proxy(flow, shear, world.curvature, world_cfg)
    enrichment = _compute_enrichment_proxy(resources, hazards, lift, world.occupancy, world_cfg)
    # Seed metabolites from the injected resource landscape so the Avida branch
    # can consume and alter a non-empty shared field during PPO rollouts.
    metabolite_seed = jnp.clip(0.05 * resources[..., 0] + 0.05 * env.terrain, 0.0, 1.0)
    metabolites = jnp.broadcast_to(metabolite_seed[..., None], world.metabolites.shape)
    return (
        WorldState(
            terrain=terrain,
            resources=resources,
            hazards=hazards,
            flow=flow,
            curvature=world.curvature,
            shear=shear,
            shear_grad=shear_grad,
            enrichment=enrichment,
            lift=lift,
            concentration=world.concentration,
            metabolites=metabolites,
            occupancy=world.occupancy,
            time=world.time,
        ),
        world_cfg,
    )


def initialize_environment(cfg: POETConfig, key: jax.Array) -> EnvironmentGenome:
    k_params, k_terrain = jax.random.split(key)
    low = jnp.asarray(cfg.environment_param_low, dtype=jnp.float32)
    high = jnp.asarray(cfg.environment_param_high, dtype=jnp.float32)
    params = jax.random.uniform(k_params, low.shape, minval=low, maxval=high)
    terrain = jax.random.uniform(
        k_terrain, (cfg.height, cfg.width), minval=0.0, maxval=1.0, dtype=jnp.float32
    )
    return EnvironmentGenome(params=params, terrain=terrain)


def normalize_replication_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Return the declared metadata required to interpret a campaign cell."""
    metadata = {} if metadata is None else dict(metadata)
    normalized = {
        "replication_id": str(metadata.get("replication_id", "replication-unknown")),
        "seed_offset": int(metadata.get("seed_offset", 0)),
        "operator": str(metadata.get("operator", "unspecified")),
        "machine_label": str(metadata.get("machine_label", platform.node() or "unknown")),
        "cache_cleared": bool(metadata.get("cache_cleared", False)),
    }
    if not normalized["replication_id"]:
        raise ValueError("replication_id must not be empty")
    return normalized


def minimal_criterion_admits(raw_reward: float, cfg: POETConfig) -> bool:
    """Apply the preregistered raw-reward lower bound to a candidate world."""
    return float(raw_reward) >= cfg.minimal_criterion


def initialize_heldout_environments(cfg: POETConfig, seed: int) -> tuple[EnvironmentGenome, ...]:
    """Create deterministic targets from a seed stream excluded from training."""
    key = jax.random.PRNGKey(seed + cfg.heldout_seed_offset)
    keys = jax.random.split(key, cfg.heldout_environments)
    return tuple(initialize_environment(cfg, item) for item in keys)


def initialize_poet_state(
    cfg: POETConfig,
    seed: int,
    *,
    replication_metadata: dict[str, Any] | None = None,
) -> POETState:
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 2 * cfg.population_size)
    environments = tuple(
        initialize_environment(cfg, keys[i]) for i in range(cfg.population_size)
    )
    policies = tuple(
        _init_policy(cfg, keys[cfg.population_size + i]) for i in range(cfg.population_size)
    )
    initial_policy = np.asarray(policies[0].w_policy)
    initial_terrain = np.asarray(jnp.stack([env.terrain for env in environments]))
    archive = [
        EnvironmentArchiveEntry(
            environment_id=f"env-{index:06d}",
            environment=environment,
            parent_id=None,
            birth_generation=0,
            admission_score=None,
        )
        for index, environment in enumerate(environments)
    ]
    return POETState(
        environments=environments,
        policies=policies,
        generation=0,
        policy_updates=0,
        policy_mutations=0,
        environment_mutations=0,
        transfer_history=[],
        metrics=[],
        initial_policy=initial_policy,
        final_policy=initial_policy.copy(),
        initial_environment_terrain=initial_terrain,
        final_environment_terrain=initial_terrain.copy(),
        track=cfg.track,
        avida_enabled=cfg.avida_enabled,
        avida_lifecycle=cfg.avida_lifecycle,
        environment_ids=tuple(entry.environment_id for entry in archive),
        environment_archive=archive,
        archive_transfer_history=[],
        heldout_environments=initialize_heldout_environments(cfg, seed),
        heldout_transfer_history=[],
        environment_candidates=0,
        environment_admissions=0,
        environment_rejections=0,
        replication_metadata=normalize_replication_metadata(replication_metadata),
    )


def mutate_environment(
    parent: EnvironmentGenome, cfg: POETConfig, key: jax.Array
) -> EnvironmentGenome:
    """Mutate bounded scalar parameters and a fixed-shape terrain map."""
    k_params, k_mask, k_noise = jax.random.split(key, 3)
    low = jnp.asarray(cfg.environment_param_low, dtype=jnp.float32)
    high = jnp.asarray(cfg.environment_param_high, dtype=jnp.float32)
    params = parent.params + cfg.environment_mutation_std * jax.random.normal(
        k_params, parent.params.shape
    )
    params = jnp.clip(params, low, high)
    mask = jax.random.bernoulli(k_mask, cfg.terrain_mutation_prob, parent.terrain.shape)
    mask = mask.at[0, 0].set(True)
    terrain_noise = cfg.terrain_mutation_std * jax.random.normal(k_noise, parent.terrain.shape)
    terrain = jnp.clip(parent.terrain + mask * terrain_noise, 0.0, 1.0)
    return EnvironmentGenome(params=params, terrain=terrain)


def mutate_policy(parent: PolicyParams, cfg: POETConfig, key: jax.Array) -> PolicyParams:
    keys = jax.random.split(key, 2 * len(parent))
    leaves = []
    for i, leaf in enumerate(parent):
        mask = jax.random.bernoulli(keys[i], cfg.policy_mutation_prob, leaf.shape)
        noise = cfg.policy_mutation_std * jax.random.normal(keys[i + len(parent)], leaf.shape)
        leaves.append(leaf + mask * noise)
    return PolicyParams(*leaves)


def _grid_value(terrain: jax.Array, position: jax.Array) -> jax.Array:
    x = jnp.clip(
        jnp.floor(position[0] * terrain.shape[1]).astype(jnp.int32), 0, terrain.shape[1] - 1
    )
    y = jnp.clip(
        jnp.floor(position[1] * terrain.shape[0]).astype(jnp.int32), 0, terrain.shape[0] - 1
    )
    return terrain[y, x]


def _observation(
    env: EnvironmentGenome, position: jax.Array, step: int, cfg: POETConfig
) -> jax.Array:
    terrain = _grid_value(env.terrain, position)
    hazard = env.params[3] * (1.0 - terrain)
    return jnp.asarray(
        [
            position[0],
            position[1],
            env.params[0],
            env.params[1],
            terrain,
            hazard,
            1.0 - step / max(cfg.rollout_steps, 1),
            jnp.mean(env.terrain),
        ],
        dtype=jnp.float32,
    )


def step_environment(
    env: EnvironmentGenome, position: jax.Array, action: int, cfg: POETConfig
) -> tuple[jax.Array, jax.Array, bool]:
    moves = jnp.asarray(
        [[0.0, 0.08], [0.0, -0.08], [-0.08, 0.0], [0.08, 0.0], [0.0, 0.0]],
        dtype=jnp.float32,
    )
    move = moves[jnp.asarray(action, dtype=jnp.int32)]
    next_position = jnp.clip(position + move + env.params[4:6], 0.0, 0.999)
    terrain = _grid_value(env.terrain, next_position)
    goal = env.params[:2]
    distance = jnp.linalg.norm(next_position - goal)
    reward = 0.35 * terrain - env.params[3] * (1.0 - terrain) - distance
    done = bool(distance < 0.12)
    reward = reward + jnp.where(done, 1.0, 0.0)
    return next_position, reward.astype(jnp.float32), done


def _categorical_action(logits: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
    log_probs = jax.nn.log_softmax(logits)
    action = jax.random.categorical(key, logits)
    return action, log_probs[action]


def _gae(
    rewards: list[jax.Array],
    values: list[jax.Array],
    last_value: jax.Array,
    cfg: POETConfig,
) -> jax.Array:
    advantages: list[jax.Array] = []
    gae = jnp.asarray(0.0, dtype=jnp.float32)
    next_value = last_value
    for reward, value in zip(reversed(rewards), reversed(values), strict=True):
        delta = reward + cfg.ppo.gamma * next_value - value
        gae = delta + cfg.ppo.gamma * cfg.ppo.gae_lambda * gae
        advantages.append(gae)
        next_value = value
    return jnp.stack(advantages[::-1])


def _grid_vector(field: jax.Array, position: jax.Array) -> jax.Array:
    x = jnp.clip(
        jnp.floor(position[0] * field.shape[1]).astype(jnp.int32), 0, field.shape[1] - 1
    )
    y = jnp.clip(
        jnp.floor(position[1] * field.shape[0]).astype(jnp.int32), 0, field.shape[0] - 1
    )
    return field[y, x]


def _world_observation(
    env: EnvironmentGenome,
    world: WorldState,
    position: jax.Array,
    step: int,
    cfg: POETConfig,
    horizon: int | None = None,
) -> jax.Array:
    horizon = cfg.rollout_steps if horizon is None else horizon
    terrain = _grid_value(world.terrain[..., 0], position)
    resource = _grid_vector(world.resources[..., 0], position)
    hazard = _grid_vector(world.hazards[..., 0], position)
    return jnp.asarray(
        [
            position[0],
            position[1],
            env.params[0],
            env.params[1],
            terrain,
            resource - hazard,
            1.0 - step / max(horizon, 1),
            jnp.mean(world.metabolites) + jnp.mean(world.concentration),
        ],
        dtype=jnp.float32,
    )


def _rollout_step(
    state: SimState,
    env: EnvironmentGenome,
    position: jax.Array,
    action: int,
    cfg: POETConfig,
    world_cfg: WorldConfig,
    lineage_counter: int,
) -> tuple[SimState, jax.Array, jax.Array, bool, int]:
    state.world, state.rng = _step_world_fast(state.world, state.rng, world_cfg)
    move = jnp.asarray(
        [[0.0, 0.08], [0.0, -0.08], [-0.08, 0.0], [0.08, 0.0], [0.0, 0.0]],
        dtype=jnp.float32,
    )[jnp.asarray(action, dtype=jnp.int32)]
    flow = _grid_vector(state.world.flow, position)
    next_position = jnp.clip(position + move + 0.02 * flow, 0.0, 0.999)
    if state.avida is not None:
        state.rng, avida_key = jax.random.split(state.rng)
        avida_positions = jnp.broadcast_to(next_position, (state.avida.alive.shape[0], 2))
        state.avida, state.world, lineage_counter = _step_avida_fast(
            state.avida,
            state.world,
            _poet_avida_config(),
            lineage_counter,
            avida_key,
            avida_positions,
        )
    terrain = _grid_vector(state.world.terrain[..., 0], next_position)
    resource = _grid_vector(state.world.resources[..., 0], next_position)
    hazard = _grid_vector(state.world.hazards[..., 0], next_position)
    distance = jnp.linalg.norm(next_position - env.params[:2])
    reward = (
        0.35 * resource
        + 0.25 * terrain
        + 0.1 * jnp.mean(state.world.concentration)
        + 0.1 * jnp.mean(state.world.metabolites)
        - hazard
        - distance
    )
    done = bool(distance < 0.12)
    reward = reward + jnp.where(done, 1.0, 0.0)
    return state, next_position, reward.astype(jnp.float32), done, lineage_counter


def _collect_rollout(
    params: PolicyParams,
    env: EnvironmentGenome,
    cfg: POETConfig,
    key: jax.Array,
    *,
    stochastic: bool,
    horizon: int | None,
    initial_avida: AvidaPopulationState | None = None,
) -> tuple[_Batch, AvidaPopulationState | None]:
    horizon = cfg.rollout_steps if horizon is None else horizon
    if horizon < 1:
        raise ValueError("rollout horizon must be positive")
    key, world_key, population_key = jax.random.split(key, 3)
    world, world_cfg = initialize_environment_world(env, cfg, world_key)
    avida = initial_avida
    if avida is None and cfg.avida_enabled:
        avida = initialize_avida_population(_poet_avida_config(), population_key)
    state = SimState(
        generation=0,
        step=0,
        rng=key,
        world=world,
        embodied=None,
        avida=avida,
    )
    lineage_counter = 8
    positions = jnp.asarray([0.5, 0.5], dtype=jnp.float32)
    observations: list[jax.Array] = []
    actions: list[jax.Array] = []
    old_log_probs: list[jax.Array] = []
    rewards: list[jax.Array] = []
    values: list[jax.Array] = []
    for step in range(horizon):
        key, action_key = jax.random.split(key)
        observation = _world_observation(env, state.world, positions, step, cfg, horizon)
        logits, value = policy_forward(params, observation)
        if stochastic:
            action, log_prob = _categorical_action(logits, action_key)
        else:
            action = jnp.argmax(logits)
            log_prob = jax.nn.log_softmax(logits)[action]
        state, positions, reward, done, lineage_counter = _rollout_step(
            state, env, positions, int(action), cfg, world_cfg, lineage_counter
        )
        observations.append(observation)
        actions.append(action)
        old_log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)
        if done:
            positions = jnp.asarray([0.5, 0.5], dtype=jnp.float32)
    last_value = policy_forward(
        params, _world_observation(env, state.world, positions, horizon, cfg, horizon)
    )[1]
    advantages = _gae(rewards, values, last_value, cfg)
    returns = advantages + jnp.stack(values)
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    return _Batch(
        observations=jnp.stack(observations),
        actions=jnp.stack(actions),
        old_log_probs=jnp.stack(old_log_probs),
        advantages=advantages,
        returns=returns,
        rewards=jnp.stack(rewards),
    ), state.avida


def collect_rollout(
    params: PolicyParams,
    env: EnvironmentGenome,
    cfg: POETConfig,
    key: jax.Array,
    *,
    stochastic: bool = True,
    horizon: int | None = None,
) -> _Batch:
    batch, _ = _collect_rollout(
        params,
        env,
        cfg,
        key,
        stochastic=stochastic,
        horizon=horizon,
    )
    return batch


def ppo_update(
    params: PolicyParams, batch: _Batch, cfg: PPOConfig
) -> tuple[PolicyParams, float]:
    """Apply PPO clipped actor-value updates and return updated parameters."""
    optimizer = optax.adam(cfg.learning_rate)
    opt_state = optimizer.init(params)

    def loss_fn(current: PolicyParams) -> jax.Array:
        logits, values = policy_forward(current, batch.observations)
        log_probs = jax.nn.log_softmax(logits)
        selected = jnp.take_along_axis(log_probs, batch.actions[:, None], axis=1)[:, 0]
        ratio = jnp.exp(selected - batch.old_log_probs)
        clipped = jnp.clip(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
        actor_loss = -jnp.mean(jnp.minimum(ratio * batch.advantages, clipped * batch.advantages))
        value_loss = jnp.mean((values - batch.returns) ** 2)
        probabilities = jax.nn.softmax(logits)
        entropy = -jnp.mean(jnp.sum(probabilities * log_probs, axis=-1))
        return actor_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

    final_loss = jnp.asarray(0.0)
    for _ in range(cfg.epochs):
        final_loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
    return params, float(final_loss)


def evaluate_policy(
    params: PolicyParams, env: EnvironmentGenome, cfg: POETConfig, seed: int
) -> float:
    batch = collect_rollout(
        params,
        env,
        cfg,
        jax.random.PRNGKey(seed),
        stochastic=False,
        horizon=cfg.transfer_steps,
    )
    return float(jnp.sum(batch.rewards))


def evaluate_transfer_matrix(
    policies: tuple[PolicyParams, ...] | list[PolicyParams],
    environments: tuple[EnvironmentGenome, ...] | list[EnvironmentGenome],
    cfg: POETConfig,
    seed: int,
) -> np.ndarray:
    """Evaluate every frozen policy on every environment without updates."""
    matrix = np.zeros((len(policies), len(environments)), dtype=np.float32)
    for policy_index, policy in enumerate(policies):
        for env_index, env in enumerate(environments):
            matrix[policy_index, env_index] = evaluate_policy(
                policy, env, cfg, seed + policy_index * 1009 + env_index
            )
    return matrix


def evaluate_archive_transfer(
    policies: tuple[PolicyParams, ...] | list[PolicyParams],
    archive: list[EnvironmentArchiveEntry],
    cfg: POETConfig,
    seed: int,
) -> np.ndarray:
    """Evaluate frozen policies on every admitted environment stepping stone."""
    return evaluate_transfer_matrix(
        policies, [entry.environment for entry in archive], cfg, seed
    )


def _avida_comparison_score(matrix: np.ndarray) -> float:
    """Summarize positive transfer over each policy's own-environment score."""
    centered = matrix - np.mean(matrix, axis=1, keepdims=True)
    return float(np.mean(np.maximum(centered, 0.0)))


def _avida_comparator_steps(generations: int) -> int:
    return max(1, min(generations, 4))


def _run_avida_comparator(
    env: EnvironmentGenome, cfg: POETConfig, seed: int, generations: int
) -> float:
    """Run the existing Avida VM against the same injected world genome."""
    key = jax.random.PRNGKey(seed + 7001)
    key, world_key, pop_key, pos_key = jax.random.split(key, 4)
    world, world_cfg = initialize_environment_world(env, cfg, world_key)
    avida_cfg = _poet_avida_config()
    population = initialize_avida_population(avida_cfg, pop_key)
    positions = jax.random.uniform(pos_key, (avida_cfg.population_size, 2))
    lineage_start = avida_cfg.population_size
    state = SimState(
        generation=0,
        step=0,
        rng=key,
        world=world,
        embodied=None,
        avida=population,
    )
    for _ in range(_avida_comparator_steps(generations)):
        state.world, state.rng = _step_world_fast(state.world, state.rng, world_cfg)
        state.rng, step_key = jax.random.split(state.rng)
        state.avida, state.world, lineage_start = _step_avida_fast(
            state.avida, state.world, avida_cfg, lineage_start, step_key, positions
        )
    return float(jnp.mean(state.avida.merit))


def run_poet(
    cfg: POETConfig,
    seed: int,
    generations: int | None = None,
    out_dir: str | Path | None = None,
    *,
    replication_metadata: dict[str, Any] | None = None,
) -> POETState:
    """Run paired populations with admitted archive worlds and held-out targets."""
    metadata = normalize_replication_metadata(replication_metadata)
    effective_seed = seed + metadata["seed_offset"]
    state = initialize_poet_state(
        cfg, effective_seed, replication_metadata=metadata
    )
    generations = cfg.rollout_steps if generations is None else generations
    key = jax.random.PRNGKey(effective_seed + 17)
    persistent_avida: list[AvidaPopulationState | None] = [None] * len(state.environments)
    if cfg.avida_enabled and cfg.avida_lifecycle == "persistent":
        for index in range(len(persistent_avida)):
            key, population_key = jax.random.split(key)
            persistent_avida[index] = initialize_avida_population(
                _poet_avida_config(), population_key
            )
    for generation in range(generations):
        transfer = evaluate_transfer_matrix(
            state.policies, state.environments, cfg, effective_seed + generation
        )
        state.transfer_history.append(transfer)
        policy_scores = transfer.mean(axis=1)
        best_policy_index = int(np.argmax(policy_scores))
        trained_policies = list(state.policies)
        for index, env in enumerate(state.environments):
            parent_index = int(np.argmax(transfer[:, index]))
            key, rollout_key = jax.random.split(key)
            if cfg.avida_enabled and cfg.avida_lifecycle == "persistent":
                batch, persistent_avida[index] = _collect_rollout(
                    trained_policies[parent_index],
                    env,
                    cfg,
                    rollout_key,
                    stochastic=True,
                    horizon=None,
                    initial_avida=persistent_avida[index],
                )
            else:
                batch = collect_rollout(
                    trained_policies[parent_index], env, cfg, rollout_key, stochastic=True
                )
            trained_policies[parent_index], loss = ppo_update(
                trained_policies[parent_index], batch, cfg.ppo
            )
            state.policy_updates += 1
            state.metrics.append(
                {
                    "generation": float(generation),
                    "policy_index": float(parent_index),
                    "environment_index": float(index),
                    "transfer_score": float(transfer[parent_index, index]),
                    "ppo_loss": float(loss),
                }
            )
        worst_policy_index = int(np.argmin(policy_scores))
        key, policy_mutation_key = jax.random.split(key)
        trained_policies[worst_policy_index] = mutate_policy(
            trained_policies[best_policy_index], cfg, policy_mutation_key
        )
        state.policies = tuple(trained_policies)
        state.policy_mutations += 1

        # Candidate worlds enter the archive only after the raw-reward criterion.
        if cfg.track != "static":
            worst_environment_index = int(np.argmin(transfer[best_policy_index]))
            key, mutation_key = jax.random.split(key)
            parent_id = state.environment_ids[best_policy_index]
            child = mutate_environment(state.environments[best_policy_index], cfg, mutation_key)
            candidate_score = evaluate_policy(
                state.policies[best_policy_index], child, cfg, effective_seed + 500_000 + generation
            )
            state.environment_candidates += 1
            if minimal_criterion_admits(candidate_score, cfg):
                child_id = f"env-{len(state.environment_archive):06d}"
                state.environment_archive.append(
                    EnvironmentArchiveEntry(
                        environment_id=child_id,
                        environment=child,
                        parent_id=parent_id,
                        birth_generation=generation + 1,
                        admission_score=candidate_score,
                    )
                )
                environments = list(state.environments)
                environments[worst_environment_index] = child
                state.environments = tuple(environments)
                ids = list(state.environment_ids)
                ids[worst_environment_index] = child_id
                state.environment_ids = tuple(ids)
                state.environment_mutations += 1
                state.environment_admissions += 1
            else:
                state.environment_rejections += 1
        state.generation = generation + 1
        state.final_policy = np.asarray(state.policies[0].w_policy)
        state.final_environment_terrain = np.asarray(
            jnp.stack([environment.terrain for environment in state.environments])
        )
    final_transfer = evaluate_transfer_matrix(
        state.policies, state.environments, cfg, effective_seed + generations
    )
    state.transfer_history.append(final_transfer)
    state.archive_transfer_history.append(
        evaluate_archive_transfer(
            state.policies, state.environment_archive, cfg, effective_seed + 700_000
        )
    )
    if state.heldout_environments:
        state.heldout_transfer_history.append(
            evaluate_transfer_matrix(
                state.policies,
                state.heldout_environments,
                cfg,
                effective_seed + cfg.heldout_seed_offset,
            )
        )
    if out_dir is not None:
        write_poet_artifacts(state, cfg, seed, Path(out_dir), generations)
    return state


def save_poet_checkpoint(
    path: str | Path, state: POETState, cfg: POETConfig, seed: int
) -> Path:
    """Persist paired populations and transfer history for local restoration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(
            {
                "version": 1,
                "seed": seed,
                "source_commit": _source_commit(),
                "config": asdict(cfg),
                "state": state,
            },
            handle,
        )
    return path


def load_poet_checkpoint(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def write_poet_artifacts(
    state: POETState, cfg: POETConfig, seed: int, out_dir: Path, generations: int
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    metadata = {
        "source_commit": _source_commit(),
        "seed": seed,
        "effective_seed": seed + state.replication_metadata["seed_offset"],
        "generations": generations,
        "config": asdict(cfg),
        "replication_metadata": state.replication_metadata,
    }
    (out_dir / "config.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in state.metrics:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    final_matrix = state.transfer_history[-1]
    (out_dir / "transfer_matrix.json").write_text(
        json.dumps(
            {
                **metadata,
                "matrix": final_matrix.tolist(),
                "frozen_policy": True,
                "mode": "raw_reward_transfer",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    archive = [
        {
            "environment_id": entry.environment_id,
            "parent_id": entry.parent_id,
            "birth_generation": entry.birth_generation,
            "admission_score": entry.admission_score,
            "params": np.asarray(entry.environment.params).tolist(),
            "terrain": np.asarray(entry.environment.terrain).tolist(),
            "terrain_mean": float(jnp.mean(entry.environment.terrain)),
            "terrain_std": float(jnp.std(entry.environment.terrain)),
        }
        for entry in state.environment_archive
    ]
    (out_dir / "environment_archive.json").write_text(
        json.dumps(
            {
                **metadata,
                "minimal_criterion": cfg.minimal_criterion,
                "environments": archive,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    archive_matrix = state.archive_transfer_history[-1]
    (out_dir / "archive_transfer.json").write_text(
        json.dumps(
            {
                **metadata,
                "mode": "stepping_stone_archive_transfer",
                "frozen_policy": True,
                "matrix": archive_matrix.tolist(),
                "history": [matrix.tolist() for matrix in state.archive_transfer_history],
                "environment_ids": [entry.environment_id for entry in state.environment_archive],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    heldout_matrix = (
        state.heldout_transfer_history[-1]
        if state.heldout_transfer_history
        else np.zeros((len(state.policies), 0), dtype=np.float32)
    )
    (out_dir / "heldout_transfer.json").write_text(
        json.dumps(
            {
                **metadata,
                "mode": "heldout_transfer",
                "frozen_policy": True,
                "excluded_from_training": True,
                "matrix": heldout_matrix.tolist(),
                "history": [matrix.tolist() for matrix in state.heldout_transfer_history],
                "environment_count": len(state.heldout_environments),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_poet_checkpoint(out_dir / "poet_checkpoint.pkl", state, cfg, seed)
    comparison_score = _avida_comparison_score(final_matrix) if cfg.avida_enabled else None
    avida_comparator_steps = (
        _avida_comparator_steps(generations) if cfg.avida_enabled else None
    )
    avida_mean_merit = (
        float(
            np.mean(
                [
                    _run_avida_comparator(environment, cfg, seed + index, generations)
                    for index, environment in enumerate(state.environments)
                ]
            )
        )
        if cfg.avida_enabled
        else None
    )
    summary = state.to_jsonable()
    summary.update(
        {
            "source_commit": metadata["source_commit"],
            "config": metadata["config"],
            "seed": seed,
            "effective_seed": metadata["effective_seed"],
            "generations": generations,
            "replication_metadata": state.replication_metadata,
            "heldout_transfer_mean": (
                float(np.mean(heldout_matrix)) if heldout_matrix.size else None
            ),
            "archive_transfer_mean": (
                float(np.mean(archive_matrix)) if archive_matrix.size else None
            ),
            "comparison_score": comparison_score,
            "avida_comparator_steps": avida_comparator_steps,
            "avida_mean_merit": avida_mean_merit,
            "final_transfer_mean": float(np.mean(final_matrix)),
            "final_transfer_std": float(np.std(final_matrix)),
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_progress_figure(state, out_dir / "figures" / "poet_progress.png")


def _write_progress_figure(state: POETState, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[int, list[float]] = {}
    for row in state.metrics:
        grouped.setdefault(int(row["generation"]), []).append(row["transfer_score"])
    x = sorted(grouped)
    y = [float(np.mean(grouped[g])) for g in x]
    figure, axis = plt.subplots(figsize=(5, 3))
    axis.plot(x, y, marker="o")
    axis.set(xlabel="generation", ylabel="frozen transfer score", title="Paired PPO-POET")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def load_poet_config(path: str | Path, *, track: str | None = None) -> POETConfig:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    ppo = PPOConfig(**raw.pop("ppo", {}))
    if track is not None:
        raw["track"] = track
        raw["avida_enabled"] = track in {"avida_enabled", "avida_persistent"}
        raw["avida_lifecycle"] = "persistent" if track == "avida_persistent" else "reset"
    raw["ppo"] = ppo
    return POETConfig(**raw)
