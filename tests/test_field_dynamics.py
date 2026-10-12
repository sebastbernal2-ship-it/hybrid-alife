from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_alife.agents.embodied import (
    apply_embodied_actions,
    embodied_action_dim,
    initialize_embodied_population,
)
from hybrid_alife.experiments.runner import initialize_sim, load_config
from hybrid_alife.types import EmbodiedConfig
from hybrid_alife.world.env import initialize_world, step_world


def _replace_embodied(cfg, **kwargs) -> EmbodiedConfig:
    return dataclasses.replace(cfg.embodied, **kwargs)


def test_resource_advection_is_disabled_by_default_and_causally_uses_flow():
    cfg = load_config("configs/base.yaml")
    no_advection_cfg = dataclasses.replace(
        cfg, world=dataclasses.replace(cfg.world, resource_regen=0.0)
    )
    positive_cfg = dataclasses.replace(
        cfg,
        world=dataclasses.replace(
            cfg.world, resource_regen=0.0, resource_advection_strength=0.5
        ),
    )
    baseline = initialize_sim(no_advection_cfg)
    coupled = initialize_sim(positive_cfg)
    pattern = jnp.zeros_like(baseline.world.resources).at[0, 0, :].set(1.0)
    flow = jnp.ones_like(baseline.world.flow) * jnp.array([0.2, 0.0])
    baseline.world.resources = pattern
    baseline.world.flow = flow
    coupled.world.resources = pattern
    coupled.world.flow = flow
    baseline = step_world(baseline, no_advection_cfg.world)
    coupled = step_world(coupled, positive_cfg.world)
    np.testing.assert_array_equal(
        baseline.world.resources,
        pattern,
    )
    assert bool(jnp.any(coupled.world.resources != pattern))


def test_flow_alignment_reward_is_optional_and_causal():
    cfg = load_config("configs/base.yaml")
    reward_cfg = _replace_embodied(cfg, flow_alignment_reward=1.0)
    key = jax.random.PRNGKey(71)
    world = initialize_world(cfg.world, key)
    world.flow = jnp.ones_like(world.flow) * jnp.array([0.2, 0.0])
    aligned = initialize_embodied_population(reward_cfg, cfg.world, key)
    opposed = initialize_embodied_population(reward_cfg, cfg.world, key)
    action_dim = embodied_action_dim(reward_cfg)
    aligned_actions = jnp.zeros((reward_cfg.population_size, action_dim)).at[:, 0].set(1.0)
    opposed_actions = jnp.zeros((reward_cfg.population_size, action_dim)).at[:, 0].set(-1.0)
    aligned, _, _ = apply_embodied_actions(
        aligned, aligned_actions, world, reward_cfg, cfg.world, key
    )
    opposed, _, _ = apply_embodied_actions(
        opposed, opposed_actions, world, reward_cfg, cfg.world, key
    )
    assert bool(jnp.mean(aligned.energy) > jnp.mean(opposed.energy))


def test_flow_advection_only_changes_movement_when_enabled():
    cfg = load_config("configs/base.yaml")
    key = jax.random.PRNGKey(70)
    zero_cfg = cfg.embodied
    assert zero_cfg.flow_movement_scale == 0.0
    positive_cfg = _replace_embodied(cfg, flow_movement_scale=0.5)
    pop_zero = initialize_embodied_population(zero_cfg, cfg.world, key)
    pop_flow_zero = initialize_embodied_population(zero_cfg, cfg.world, key)
    pop_flow_positive = initialize_embodied_population(positive_cfg, cfg.world, key)
    world_zero = initialize_world(cfg.world, key)
    world_flow = initialize_world(cfg.world, key)
    world_flow.flow = jnp.ones_like(world_flow.flow) * jnp.array([0.2, -0.1])
    act = jnp.zeros(
        (pop_zero.alive.shape[0], embodied_action_dim(zero_cfg)), dtype=jnp.float32
    )

    baseline, _, _ = apply_embodied_actions(
        pop_zero, act, world_zero, zero_cfg, cfg.world, key
    )
    no_coupling, _, _ = apply_embodied_actions(
        pop_flow_zero, act, world_flow, zero_cfg, cfg.world, key
    )
    coupled, _, _ = apply_embodied_actions(
        pop_flow_positive, act, world_flow, positive_cfg, cfg.world, key
    )

    np.testing.assert_array_equal(no_coupling.positions, baseline.positions)
    assert bool(jnp.any(coupled.positions != baseline.positions))
