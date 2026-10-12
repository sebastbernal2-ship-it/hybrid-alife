from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp

from hybrid_alife.agents.embodied import (
    apply_embodied_actions,
    build_world_field,
    initialize_embodied_population,
    observe_embodied,
)
from hybrid_alife.experiments.runner import load_config
from hybrid_alife.world.env import initialize_world
from hybrid_alife.world.sensing import sample_sixth_sense


def test_named_field_groups_are_hidden_without_changing_other_patch_channels() -> None:
    cfg = load_config("configs/base.yaml")
    key = jax.random.PRNGKey(0)
    world = initialize_world(cfg.world, key)

    full = build_world_field(world, cfg.world)
    hidden = build_world_field(world, cfg.world, ("flow", "concentration"))
    terrain_end = world.terrain.shape[-1]
    resource_end = terrain_end + world.resources.shape[-1]
    hazard_end = resource_end + world.hazards.shape[-1]
    flow_end = hazard_end + world.flow.shape[-1]
    curvature_end = flow_end + world.curvature.shape[-1]
    shear_end = curvature_end + world.shear.shape[-1]
    shear_grad_end = shear_end + world.shear_grad.shape[-1]
    enrichment_end = shear_grad_end + world.enrichment.shape[-1]
    lift_end = enrichment_end + world.lift.shape[-1]
    concentration_end = lift_end + world.concentration.shape[-1]

    assert jnp.array_equal(full[..., :hazard_end], hidden[..., :hazard_end])
    assert bool(jnp.all(hidden[..., hazard_end:flow_end] == 0.0))
    assert jnp.array_equal(full[..., flow_end:lift_end], hidden[..., flow_end:lift_end])
    assert bool(jnp.all(hidden[..., lift_end:concentration_end] == 0.0))


def test_decorrelated_flow_keeps_values_but_moves_them_to_other_cells() -> None:
    cfg = load_config("configs/base.yaml")
    key = jax.random.PRNGKey(4)
    world = initialize_world(cfg.world, key)
    direct = build_world_field(world, cfg.world)
    shifted = build_world_field(world, cfg.world, shuffle_flow_observation=True)
    start = world.terrain.shape[-1] + world.resources.shape[-1] + world.hazards.shape[-1]
    end = start + world.flow.shape[-1]
    expected = jnp.roll(
        world.flow,
        shift=(cfg.world.height // 2, cfg.world.width // 2),
        axis=(0, 1),
    )
    assert jnp.array_equal(shifted[..., start:end], expected)
    assert jnp.array_equal(
        jnp.sort(direct[..., start:end].reshape(-1)),
        jnp.sort(shifted[..., start:end].reshape(-1)),
    )
    assert not jnp.array_equal(direct[..., start:end], shifted[..., start:end])


def test_hidden_groups_mask_sixth_sense_but_keep_proprioception() -> None:
    cfg = load_config("configs/base.yaml")
    world_cfg = replace(cfg.world, sensory_noise_std=0.0)
    full_cfg = replace(cfg.embodied, hidden_field_groups=())
    hidden_cfg = replace(cfg.embodied, hidden_field_groups=("flow", "concentration"))
    key = jax.random.PRNGKey(1)
    world = initialize_world(world_cfg, key)
    pop = initialize_embodied_population(full_cfg, world_cfg, key)

    full_sense = sample_sixth_sense(
        world, pop.positions, pop.alive, pop.messages, world_cfg, full_cfg, key
    )
    hidden_sense = sample_sixth_sense(
        world, pop.positions, pop.alive, pop.messages, world_cfg, hidden_cfg, key
    )
    assert bool(jnp.all(hidden_sense[:, 1:3] == 0.0))
    assert bool(jnp.all(hidden_sense[:, 4:6] == 0.0))
    assert bool(jnp.all(hidden_sense[:, 6:8] == 0.0))
    assert bool(jnp.any(full_sense[:, 1:3] != 0.0))

    full_obs = observe_embodied(pop, world, full_cfg, world_cfg, key)
    hidden_obs = observe_embodied(pop, world, hidden_cfg, world_cfg, key)
    proprio_start = full_obs.shape[-1] - (6 + full_cfg.message_size)
    assert jnp.array_equal(full_obs[:, proprio_start:], hidden_obs[:, proprio_start:])


def test_unknown_field_group_fails_closed() -> None:
    cfg = load_config("configs/base.yaml")
    bad_cfg = replace(cfg.embodied, hidden_field_groups=("gravity",))
    pop = initialize_embodied_population(cfg.embodied, cfg.world, jax.random.PRNGKey(2))
    world = initialize_world(cfg.world, jax.random.PRNGKey(2))
    try:
        observe_embodied(pop, world, bad_cfg, cfg.world, jax.random.PRNGKey(2))
    except ValueError as error:
        assert "unknown hidden field groups" in str(error)
    else:
        raise AssertionError("unknown hidden field groups must fail closed")


def test_defaults_preserve_existing_observation_shape() -> None:
    cfg = load_config("configs/base.yaml")
    key = jax.random.PRNGKey(3)
    world = initialize_world(cfg.world, key)
    pop = initialize_embodied_population(cfg.embodied, cfg.world, key)
    obs = observe_embodied(pop, world, cfg.embodied, cfg.world, key)
    assert obs.shape[-1] > 0


def test_active_probe_reveals_hidden_local_flow_without_changing_defaults() -> None:
    cfg = load_config("configs/base.yaml")
    probe_cfg = replace(
        cfg.embodied,
        probe_enabled=True,
        hidden_field_groups=("flow", "shear", "shear_grad", "lift"),
    )
    key = jax.random.PRNGKey(8)
    world = initialize_world(cfg.world, key)
    pop = initialize_embodied_population(probe_cfg, cfg.world, key)
    grid = jnp.floor(
        pop.positions * jnp.asarray([cfg.world.height, cfg.world.width])
    ).astype(jnp.int32)
    local_flow = world.flow[grid[:, 0], grid[:, 1]]
    actions = jnp.zeros((probe_cfg.population_size, 9 + probe_cfg.message_size))
    actions = actions.at[:, 8].set(1.0)
    updated, _, _ = apply_embodied_actions(
        pop, actions, world, probe_cfg, cfg.world, key
    )
    assert updated.probe_result is not None
    assert jnp.allclose(updated.probe_result, local_flow)
    observation = observe_embodied(updated, world, probe_cfg, cfg.world, key)
    probe_start = observation.shape[-1] - (2 + 6 + probe_cfg.message_size)
    assert jnp.allclose(observation[:, probe_start : probe_start + 2], local_flow)


def test_probe_disabled_keeps_action_shape() -> None:
    cfg = load_config("configs/base.yaml")
    assert cfg.embodied.probe_enabled is False
    pop = initialize_embodied_population(cfg.embodied, cfg.world, jax.random.PRNGKey(9))
    assert pop.last_action.shape[-1] == 8 + cfg.embodied.message_size
