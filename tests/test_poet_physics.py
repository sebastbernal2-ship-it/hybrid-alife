import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_alife.physics import lbm
from hybrid_alife.poet import (
    PHYSICS_GENE_NAMES,
    POETConfig,
    SimState,
    _cached_physics_flow,
    _evaluate_policy_reference,
    _init_policy,
    _physics_mask,
    _poet_world_config,
    _rollout_step,
    _step_world_jitted,
    _step_world_physics,
    evaluate_policy,
    initialize_environment,
    initialize_environment_world,
    mutate_environment,
    physics_base_flow,
    run_poet,
    validate_physics_genes,
)


def cfg8(**kw):
    return POETConfig(
        **{
            "width": 8,
            "height": 8,
            "population_size": 1,
            "policy_hidden": 4,
            "action_count": 5,
            "rollout_steps": 2,
            "transfer_steps": 2,
            "avida_enabled": False,
            **kw,
        }
    )


def genes(cfg, **changes):
    p = np.mean([cfg.physics_param_low, cfg.physics_param_high], axis=0).astype(np.float32)
    for name, value in changes.items():
        p[PHYSICS_GENE_NAMES.index(name)] = value
    return jnp.asarray(p)


def env_for(cfg, physics=None):
    base = initialize_environment(
        POETConfig(**{**cfg.__dict__, "physics_genes": False}), jax.random.PRNGKey(7)
    )
    return base._replace(physics=genes(cfg) if physics is None else physics)


def test_opt_in_preserves_legacy_genome_and_world_initialization():
    legacy, physical = cfg8(), cfg8(physics_genes=True)
    key = jax.random.PRNGKey(4)
    old, new = initialize_environment(legacy, key), initialize_environment(physical, key)
    assert old.physics is None and new.physics is not None
    np.testing.assert_array_equal(old.params, new.params)
    np.testing.assert_array_equal(old.terrain, new.terrain)
    np.testing.assert_array_equal(
        mutate_environment(old, legacy, key).params, mutate_environment(new, physical, key).params
    )
    np.testing.assert_array_equal(
        mutate_environment(old, legacy, key).terrain, mutate_environment(new, physical, key).terrain
    )


def test_physics_config_validation_and_bounded_mutations():
    cfg = cfg8(physics_genes=True, physics_mutation_scale=100.0)
    with pytest.raises(ValueError):
        cfg8(physics_genes=True, physics_param_low=(0.1,))
    low = list(cfg.physics_param_low)
    low[0] = 0.04
    with pytest.raises(ValueError):
        cfg8(physics_genes=True, physics_param_low=tuple(low))
    low = list(cfg.physics_param_low)
    high = list(cfg.physics_param_high)
    low[0], high[0] = 0.2, 0.1
    with pytest.raises(ValueError):
        cfg8(physics_genes=True, physics_param_low=tuple(low), physics_param_high=tuple(high))
    with pytest.raises(ValueError):
        cfg8(physics_genes=True, physics_lbm_steps=1)
    env = initialize_environment(cfg, jax.random.PRNGKey(0))
    for i in range(12):
        env = mutate_environment(env, cfg, jax.random.PRNGKey(i + 1))
        validate_physics_genes(env.physics, cfg)
        p = np.asarray(env.physics)
        assert np.all(p >= np.asarray(cfg.physics_param_low, np.float32))
        assert np.all(p <= np.asarray(cfg.physics_param_high, np.float32))
        assert np.all(np.any(~_physics_mask(p, cfg), axis=0))
    for bad in (jnp.zeros((13,)), genes(cfg).at[0].set(jnp.nan), genes(cfg).at[0].set(0.31)):
        with pytest.raises(ValueError):
            validate_physics_genes(bad, cfg)
    with pytest.raises(ValueError):
        validate_physics_genes(jnp.zeros((14, 1)), cfg)
    # On a 4x4 channel the legal maximum obstacle can occlude a whole column.
    small = cfg8(width=4, height=4, physics_genes=True)
    blocked = genes(small, obstacle_radius=0.25, obstacle_x=0.5, obstacle_y=0.5)
    with pytest.raises(ValueError, match="blocks"):
        validate_physics_genes(blocked, small)
    with pytest.raises(ValueError, match="agree"):
        mutate_environment(env, replace(cfg, physics_genes=False), jax.random.PRNGKey(2))
    with pytest.raises(ValueError, match="agree"):
        mutate_environment(env._replace(physics=None), cfg, jax.random.PRNGKey(2))
    # Exact float32 endpoints remain valid.
    edge = genes(cfg).at[0].set(np.float32(cfg.physics_param_high[0]))
    validate_physics_genes(edge, cfg)


def test_invalid_geometry_mutation_keeps_parent_obstacle_genes(monkeypatch):
    import hybrid_alife.poet as poet

    cfg = cfg8(physics_genes=True, physics_mutation_scale=1000)
    parent = env_for(cfg, genes(cfg, obstacle_radius=0.05, obstacle_x=0.5, obstacle_y=0.5))
    monkeypatch.setattr(poet, "_valid_physics_geometry", lambda *_: False)
    child = mutate_environment(parent, cfg, jax.random.PRNGKey(3))
    np.testing.assert_array_equal(child.physics[1:4], parent.physics[1:4])


def test_flow_cache_physics_and_gene_effects():
    cfg = cfg8(physics_genes=True)
    env = env_for(cfg)
    _cached_physics_flow.cache_clear()
    f = physics_base_flow(env, cfg)
    assert not f.flags.writeable and np.isfinite(f).all()
    physics_base_flow(env, cfg)
    assert _cached_physics_flow.cache_info().hits == 1
    changed = env._replace(physics=env.physics.at[0].set(0.06))
    physics_base_flow(changed, cfg)
    assert _cached_physics_flow.cache_info().misses == 2
    # Non-geometry changes do not alter the steady-flow cache key.
    changed = env._replace(physics=env.physics.at[4].set(0.9))
    physics_base_flow(changed, cfg)
    assert _cached_physics_flow.cache_info().hits == 2
    extreme = genes(
        cfg,
        obstacle_radius=0,
        curvature_amplitude=1,
        curvature_wavenumber=4,
        dean_strength=0.5,
        flow_noise_std=0.05,
        viscosity=0.05,
    )
    ef = env._replace(physics=extreme)
    world, _ = initialize_environment_world(ef, cfg, jax.random.PRNGKey(9))
    mask = _physics_mask(extreme, cfg)
    assert np.all(np.asarray(world.flow)[mask] == 0)
    assert np.max(np.abs(np.asarray(world.flow))) * 0.05 < lbm.MAX_LATTICE_SPEED
    lower = env._replace(physics=env.physics.at[0].set(0.05))
    higher = env._replace(physics=env.physics.at[0].set(0.3))
    assert np.max(np.abs(physics_base_flow(lower, cfg))) > np.max(
        np.abs(physics_base_flow(higher, cfg))
    )
    # All genes alter their intended initialization or one-step world/rollout output.
    baseline, wc = initialize_environment_world(env, cfg, jax.random.PRNGKey(11))
    for gene, idx, field in [
        ("curvature_amplitude", 4, "curvature"),
        ("dean_strength", 6, "flow"),
        ("flow_noise_std", 7, "flow"),
        ("resource_abundance", 8, "resources"),
    ]:
        alt = env._replace(physics=env.physics.at[idx].set(cfg.physics_param_high[idx]))
        got, _ = initialize_environment_world(alt, cfg, jax.random.PRNGKey(11))
        assert not np.array_equal(getattr(baseline, field), getattr(got, field)), gene
    # The remaining genes act during dynamics or gathering; hold the action to gather.
    state = SimState(
        generation=0, step=0, rng=jax.random.PRNGKey(3), world=baseline, embodied=None, avida=None
    )
    pos = jnp.array([0.5, 0.5])
    for gene, idx in [("thermo_efficiency", 11), ("hazard_intensity", 9)]:
        alt = env._replace(physics=env.physics.at[idx].set(cfg.physics_param_low[idx]))
        _, _, reward_a, *_ = _rollout_step(state, env, pos, 4, cfg, wc, 0, 1.0)
        _, _, reward_b, *_ = _rollout_step(state, alt, pos, 4, cfg, wc, 0, 1.0)
        assert float(reward_a) != float(reward_b), gene
    for gene, idx in [("hazard_decay", 10), ("signal_diffusion", 12), ("signal_decay", 13)]:
        altp = env.physics.at[idx].set(cfg.physics_param_low[idx])
        w0 = replace(baseline, concentration=baseline.concentration.at[3, 3].set(1.0))
        a, _ = _step_world_physics(
            w0, jax.random.PRNGKey(8), _poet_world_config(cfg), env.physics, env.params
        )
        b, _ = _step_world_physics(
            w0, jax.random.PRNGKey(8), _poet_world_config(cfg), altp, env.params
        )
        target = "hazards" if idx == 10 else "concentration"
        assert not np.array_equal(getattr(a, target), getattr(b, target)), gene


def test_physics_determinism_evaluation_run_and_archive(tmp_path):
    cfg = cfg8(physics_genes=True)
    a, b = (
        initialize_environment(cfg, jax.random.PRNGKey(5)),
        initialize_environment(cfg, jax.random.PRNGKey(5)),
    )
    np.testing.assert_array_equal(a.physics, b.physics)
    np.testing.assert_array_equal(
        mutate_environment(a, cfg, jax.random.PRNGKey(6)).physics,
        mutate_environment(b, cfg, jax.random.PRNGKey(6)).physics,
    )
    policy = _init_policy(cfg, jax.random.PRNGKey(2))
    assert evaluate_policy(policy, a, cfg, 17) == evaluate_policy(policy, b, cfg, 17)
    np.testing.assert_allclose(
        evaluate_policy(policy, a, cfg, 17),
        _evaluate_policy_reference(policy, a, cfg, 17),
        rtol=1e-5,
        atol=1e-5,
    )
    one = run_poet(cfg, seed=3, generations=1, out_dir=tmp_path / "a")
    two = run_poet(cfg, seed=3, generations=1, out_dir=tmp_path / "b")
    assert one.to_jsonable()["transfer_history"] == two.to_jsonable()["transfer_history"]
    archive = json.loads((tmp_path / "a" / "environment_archive.json").read_text())
    assert all(
        "physics" in item and "physics_gene_names" in item for item in archive["environments"]
    )
    legacy_dir = tmp_path / "legacy"
    run_poet(cfg8(), seed=3, generations=1, out_dir=legacy_dir)
    legacy = json.loads((legacy_dir / "environment_archive.json").read_text())
    assert all(
        "physics" not in item and "physics_gene_names" not in item
        for item in legacy["environments"]
    )


def test_physics_jit_cache_uses_gene_arrays_as_dynamic_inputs():
    cfg = cfg8(physics_genes=True)
    env = env_for(cfg)
    world, wc = initialize_environment_world(env, cfg, jax.random.PRNGKey(1))
    from hybrid_alife.poet import _step_world_physics_fast

    before = _step_world_physics_fast._cache_size()
    for i in range(4):
        _step_world_physics_fast(
            world,
            jax.random.PRNGKey(4),
            _poet_world_config(cfg),
            env.physics.at[0].set(0.06 + i * 0.01),
            env.params,
        )
    assert _step_world_physics_fast._cache_size() == before


def test_lbm_live_solver_is_trace_time_constant_in_poet_scan():
    cfg = _poet_world_config(cfg8())
    live = replace(cfg, flow_solver="lbm_live")
    legacy_env = initialize_environment(cfg8(), jax.random.PRNGKey(7))
    world, _ = initialize_environment_world(legacy_env, cfg8(), jax.random.PRNGKey(1))

    def scan(w, steps):
        return jax.lax.scan(
            lambda carry, _: (_step_world_jitted(carry, jax.random.PRNGKey(2), live)[0], None),
            w,
            None,
            length=steps,
        )[0]

    one, three = scan(world, 1), scan(world, 3)
    changed = replace(world, flow=jnp.zeros_like(world.flow))
    independent = scan(changed, 1)
    np.testing.assert_array_equal(one.flow, three.flow)
    np.testing.assert_array_equal(one.flow, independent.flow)
