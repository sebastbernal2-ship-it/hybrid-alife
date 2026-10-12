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
from hybrid_alife.experiments.runner import load_config
from hybrid_alife.types import EmbodiedConfig
from hybrid_alife.world.env import initialize_world


def _replace_embodied(cfg, **kwargs) -> EmbodiedConfig:
    return dataclasses.replace(cfg.embodied, **kwargs)


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
