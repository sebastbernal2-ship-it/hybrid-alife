"""The native Avida kernel under jax.jit (jax.pure_callback) against eager native and JAX."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_avida_native import POP_FIELDS, _coupled_setup, _same

from hybrid_alife import poet  # also registers the state dataclasses as pytrees
from hybrid_alife.agents import avida_vm

STEPS = 4
MUTATION = {"point_mutation_prob": 0.05, "insertion_prob": 0.3, "deletion_prob": 0.3,
            "duplication_prob": 0.4}
ENERGY = {"instruction_energy_cost": 0.1, "energy_initial": 3.0, "energy_efficiency": 0.5}


def _fresh(tree):
    # The step functions assign attributes on their inputs; hand each run its own objects.
    return dataclasses.replace(tree)


def _case(**over):
    cfg, world, pop, positions = _coupled_setup(2, 0.5, True, **over)
    pop = avida_vm._ensure_energy_fields(pop, cfg)
    return cfg, world, pop, positions


def _eager(cfg, world, pop, positions, native, coupled=True):
    cfg = dataclasses.replace(cfg, native=native)
    world, pop, nxt = _fresh(world), _fresh(pop), 1000
    for i in range(STEPS):
        pop, nxt = avida_vm.step_avida_population(
            pop, world, cfg, nxt, jax.random.PRNGKey(20 + i), positions if coupled else None)
    return pop, world, nxt


def _jitted(cfg, world, pop, positions, coupled=True):
    cfg = dataclasses.replace(cfg, native=True)

    @jax.jit
    def step(pop, world, nxt, key):
        pop, world = _fresh(pop), _fresh(world)
        pop, nxt = avida_vm.step_avida_population(
            pop, world, cfg, nxt, key, positions if coupled else None)
        return pop, world, nxt

    world, pop, nxt = _fresh(world), _fresh(pop), 1000
    for i in range(STEPS):
        pop, world, nxt = step(pop, world, nxt, jax.random.PRNGKey(20 + i))
    return pop, world, int(nxt)


def _assert_match(a, b, energy):
    (pa, wa, na), (pb, wb, nb) = a, b
    assert na == nb
    for f in POP_FIELDS + (("energy", "energy_in", "heat") if energy else ()):
        assert getattr(pa, f).dtype == getattr(pb, f).dtype, f
        _same(getattr(pa, f), getattr(pb, f), err_msg=f)
    _same(wa.metabolites, wb.metabolites)
    _same(wa.concentration, wb.concentration)


@pytest.mark.parametrize("over", [{}, ENERGY], ids=["plain", "energy"])
def test_jit_native_matches_eager_native_and_jax_without_mutation(over):
    case = _case(**over)
    jit = _jitted(*case)
    _assert_match(jit, _eager(*case, native=True), bool(over))
    _assert_match(jit, _eager(*case, native=False), bool(over))
    assert int((jit[0].lineage_depth > 0).sum()) > 0


def test_jit_native_matches_eager_native_with_mutation():
    # Same keys give the same kernel seeds, so mutated runs agree too.
    case = _case(**MUTATION)
    _assert_match(_jitted(*case), _eager(*case, native=True), False)


def test_jit_native_without_positions():
    case = _case()
    jit = _jitted(*case, coupled=False)
    _assert_match(jit, _eager(*case, native=True, coupled=False), False)
    _assert_match(jit, _eager(*case, native=False, coupled=False), False)


def test_jitted_scan_of_native_steps():
    cfg, world, pop, positions = _case(**ENERGY)
    cfg = dataclasses.replace(cfg, native=True)

    def body(carry, key):
        pop, world, nxt = carry
        pop, world = _fresh(pop), _fresh(world)
        pop, nxt = avida_vm.step_avida_population(pop, world, cfg, nxt, key, positions)
        return (pop, world, nxt), pop.alive.sum()

    @jax.jit
    def run(pop, world):
        keys = jnp.stack([jax.random.PRNGKey(20 + i) for i in range(STEPS)])
        return jax.lax.scan(body, (pop, world, jnp.int32(1000)), keys)

    # A concrete pop may carry a transient attribute that the pytree drops.
    (out_pop, out_world, nxt), alive = run(_fresh(pop), _fresh(world))
    assert alive.shape == (STEPS,)
    _assert_match((out_pop, out_world, int(nxt)),
                  _eager(cfg, world, pop, positions, native=False), True)


def test_vmapped_native_step_matches_separate_runs():
    cfg, world, pop, positions = _case()
    cfg = dataclasses.replace(cfg, native=True)

    def one(key):
        p, w = _fresh(pop), _fresh(world)
        p, _ = avida_vm.step_avida_population(p, w, cfg, 0, key, positions)
        return p, w.metabolites

    keys = jnp.stack([jax.random.PRNGKey(i) for i in range(3)])
    batched, metab = jax.jit(jax.vmap(one))(keys)
    for i in range(3):
        single, single_metab = one(keys[i])
        for f in POP_FIELDS:
            _same(getattr(batched, f)[i], getattr(single, f), err_msg=f)
        _same(metab[i], single_metab)


def test_callback_leaves_its_inputs_untouched():
    cfg, world, pop, positions = _case(**MUTATION)
    cfg = dataclasses.replace(cfg, native=True)
    before = [np.array(x) for x in jax.tree_util.tree_leaves((pop, world))]

    @jax.jit
    def step(pop, world):
        p, w = _fresh(pop), _fresh(world)
        p, _ = avida_vm.step_avida_population(p, w, cfg, 0, jax.random.PRNGKey(0), positions)
        return p, w

    first = step(pop, world)
    second = step(pop, world)
    for a, b in zip(before, jax.tree_util.tree_leaves((pop, world)), strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(*map(jax.tree_util.tree_leaves, (first, second)), strict=True):
        np.testing.assert_array_equal(a, b)


def test_poet_step_runs_native_under_its_own_jit(monkeypatch):
    base = poet._poet_avida_config()
    cfg = poet.POETConfig(avida_enabled=True, rollout_steps=4)
    env = poet.initialize_environment(cfg, jax.random.PRNGKey(0))
    params = poet._init_policy(cfg, jax.random.PRNGKey(1))
    results = []
    for native in (False, True):
        quiet = dataclasses.replace(base, native=native, point_mutation_prob=0.0)
        monkeypatch.setattr(poet, "_poet_avida_config", lambda quiet=quiet: quiet)
        results.append(poet._collect_rollout(
            params, env, cfg, jax.random.PRNGKey(2), stochastic=False, horizon=None))
    (batch_a, pop_a, diag_a), (batch_b, pop_b, diag_b) = results
    for f in POP_FIELDS:
        _same(getattr(pop_a, f), getattr(pop_b, f), err_msg=f)
    _same(batch_a.rewards, batch_b.rewards)
    assert diag_a["avida_feedback"] == pytest.approx(diag_b["avida_feedback"], rel=1e-4)
