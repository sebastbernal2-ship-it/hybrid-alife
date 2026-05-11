"""Embodied recurrent neural agent skeleton."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hybrid_alife.types import EmbodiedConfig, EmbodiedPopulationState, PRNGKey, WorldState


def initialize_embodied_population(cfg: EmbodiedConfig, key: PRNGKey) -> EmbodiedPopulationState:
    """Initialize positions, energies, recurrent state, messages, and flat genome params."""
    k_pos, k_genome = jax.random.split(key)
    n = cfg.population_size
    positions = jax.random.uniform(k_pos, (n, 2), minval=0.0, maxval=1.0)
    energy = jnp.full((n,), cfg.initial_energy, dtype=jnp.float32)
    age = jnp.zeros((n,), dtype=jnp.int32)
    alive = jnp.ones((n,), dtype=bool)
    hidden = jnp.zeros((n, cfg.hidden_size), dtype=jnp.float32)
    messages = jnp.zeros((n, cfg.message_size), dtype=jnp.float32)
    genomes = initialize_embodied_genomes(cfg, k_genome)
    lineage_id = jnp.arange(n, dtype=jnp.int32)
    parent_id = -jnp.ones((n,), dtype=jnp.int32)
    return EmbodiedPopulationState(
        positions=positions,
        energy=energy,
        age=age,
        alive=alive,
        hidden=hidden,
        messages=messages,
        genomes=genomes,
        lineage_id=lineage_id,
        parent_id=parent_id,
    )


def initialize_embodied_genomes(cfg: EmbodiedConfig, key: PRNGKey) -> dict[str, jax.Array]:
    """Flat MLP-GRU-ish placeholder genome.

    V1 uses direct arrays in a dict. Later, switch to Flax modules and pytrees.
    """
    obs_dim = embodied_observation_dim(cfg)
    action_dim = embodied_action_dim(cfg)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return {
        "w_obs": 0.1 * jax.random.normal(k1, (cfg.population_size, obs_dim, cfg.hidden_size)),
        "w_h": 0.1 * jax.random.normal(k2, (cfg.population_size, cfg.hidden_size, cfg.hidden_size)),
        "b_h": jnp.zeros((cfg.population_size, cfg.hidden_size)),
        "w_act": 0.1 * jax.random.normal(k3, (cfg.population_size, cfg.hidden_size, action_dim)),
        "b_act": 0.01 * jax.random.normal(k4, (cfg.population_size, action_dim)),
    }


def embodied_observation_dim(cfg: EmbodiedConfig) -> int:
    patch_side = 2 * cfg.obs_radius + 1
    world_channels_v1 = 4 + 3 + 2 + 2 + 1 + 2 + 1 + 2 + 2
    proprioception = 6 + cfg.message_size
    return patch_side * patch_side * world_channels_v1 + proprioception


def embodied_action_dim(cfg: EmbodiedConfig) -> int:
    continuous_actions = 8
    return continuous_actions + cfg.message_size


def observe_embodied(pop: EmbodiedPopulationState, world: WorldState, cfg: EmbodiedConfig) -> jax.Array:
    """Return placeholder observation tensor [N, obs_dim].

    V1 TODO: gather local patches around each continuous position using nearest grid cell.
    """
    del world
    n = cfg.population_size
    obs = jnp.zeros((n, embodied_observation_dim(cfg)), dtype=jnp.float32)
    proprio = jnp.concatenate(
        [
            pop.positions,
            pop.energy[:, None],
            pop.age[:, None].astype(jnp.float32),
            pop.alive[:, None].astype(jnp.float32),
            jnp.linalg.norm(pop.hidden, axis=-1, keepdims=True),
            pop.messages,
        ],
        axis=-1,
    )
    return obs.at[:, : proprio.shape[1]].set(proprio)


def act_embodied(
    pop: EmbodiedPopulationState, obs: jax.Array, cfg: EmbodiedConfig
) -> tuple[jax.Array, EmbodiedPopulationState]:
    """Compute actions and next recurrent hidden state."""
    del cfg
    h_pre = jnp.einsum("no,noh->nh", obs, pop.genomes["w_obs"])
    h_rec = jnp.einsum("nh,nhk->nk", pop.hidden, pop.genomes["w_h"])
    hidden = jnp.tanh(h_pre + h_rec + pop.genomes["b_h"])
    actions = jnp.einsum("nh,nha->na", hidden, pop.genomes["w_act"]) + pop.genomes["b_act"]
    pop.hidden = hidden
    return actions, pop


def mutate_embodied_genomes(
    genomes: dict[str, jax.Array], cfg: EmbodiedConfig, key: PRNGKey
) -> dict[str, jax.Array]:
    """Apply elementwise Gaussian perturbations under Bernoulli mask."""
    keys = jax.random.split(key, len(genomes))
    mutated: dict[str, jax.Array] = {}
    for subkey, (name, value) in zip(keys, genomes.items(), strict=True):
        k_mask, k_noise = jax.random.split(subkey)
        mask = jax.random.bernoulli(k_mask, cfg.genome_mutation_prob, value.shape)
        noise = cfg.genome_mutation_std * jax.random.normal(k_noise, value.shape)
        mutated[name] = value + mask * noise
    return mutated

