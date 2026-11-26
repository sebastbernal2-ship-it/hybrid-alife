import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_avida_native import POP_FIELDS, _coupled_setup, _run_steps, _same

from hybrid_alife.agents import avida_vm
from hybrid_alife.physics import avida_native


def _divide(pop, cfg, dividing_slot, native):
    """Apply only the division phase, with `dividing_slot` holding a full offspring buffer."""
    n = pop.alive.shape[0]
    pop = dataclasses.replace(
        pop, copied=pop.copied.at[dividing_slot, :6].set(pop.genomes[dividing_slot, :6]),
        copied_length=pop.copied_length.at[dividing_slot].set(6))
    mask = jnp.zeros(n, bool).at[dividing_slot].set(True)
    if not native:
        pop._divide_op_mask = mask
        return avida_vm.apply_h_divide(pop, cfg, 1000, jax.random.PRNGKey(12))
    x = {f: np.array(getattr(pop, f), np.int32, order="C") for f in avida_native._INT_FIELDS}
    x["merit"] = np.array(pop.merit, np.float32, order="C")
    alive = np.array(pop.alive, np.uint8)
    avida_native._call_divide(
        avida_native._load(), cfg, x, 1000, 0, alive, np.array(mask, np.uint8), None)
    return dataclasses.replace(pop, alive=alive.astype(bool), **x)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("fix", [False, True])
def test_unmatched_rows_clobber_a_slot_zero_child_unless_fixed(native, fix):
    cfg, _, pop, _ = _coupled_setup(11, 0.0, True, divide_scatter_fix=fix)  # only slot 0 dead
    out = _divide(pop, cfg, 1, native)
    assert int(out.copied_length[1]) == 0  # the parent gave up its buffer either way
    assert bool(out.alive[0]) == fix  # legacy: the offspring is silently lost
    assert (int(out.lineage_id[0]) == 1000) == fix


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("fix", [False, True])
def test_unmatched_rows_undo_a_slot_zero_parent_reset_unless_fixed(native, fix):
    cfg, _, pop, _ = _coupled_setup(11, 0.0, False, divide_scatter_fix=fix)
    pop = dataclasses.replace(pop, alive=pop.alive.at[1].set(False))
    out = _divide(pop, cfg, 0, native)
    assert bool(out.alive[1])  # the child in slot 1 is born either way
    assert int(out.copied_length[0]) == (0 if fix else 6)  # legacy: parent keeps its buffer
    assert bool(jnp.any(out.copied[0] != 0)) == (not fix)


@pytest.mark.parametrize("energy", [{}, {"instruction_energy_cost": 0.1, "energy_initial": 3.0}],
                         ids=["plain", "energy"])
@pytest.mark.parametrize("scatter_fix", [False, True])
def test_native_matches_jax_multiple_steps(scatter_fix, energy):
    scenarios = ((0, 0.5, True), (1, 0.5, False), (2, 0.9, True))
    for seed, dead_frac, kill0 in scenarios:
        cfg, world, pop, positions = _coupled_setup(
            seed, dead_frac, kill0, divide_scatter_fix=scatter_fix, **energy)
        ref, ref_world = _run_steps(cfg, world, pop, positions, native=False, steps=4)
        nat, nat_world = _run_steps(cfg, world, pop, positions, native=True, steps=4)
        for field in POP_FIELDS + (("energy",) if energy else ()):
            _same(getattr(ref, field), getattr(nat, field), err_msg=field)
        _same(ref_world.metabolites, nat_world.metabolites)
        _same(ref_world.concentration, nat_world.concentration)


@pytest.mark.parametrize("native", [False, True])
def test_fixed_slot_zero_birth_conserves_energy(native):
    cfg, world, pop, positions = _coupled_setup(
        14, 0.9, True, divide_scatter_fix=True, instruction_energy_cost=0.01,
        energy_initial=20.0, energy_efficiency=0.0)
    cfg = dataclasses.replace(cfg, metabolite_uptake=0.0, metabolite_deposit=0.0)
    pop = dataclasses.replace(pop, energy=jnp.full((64,), 20.0),
                              energy_in=jnp.zeros(()), heat=jnp.zeros(()))
    total = float(np.asarray(pop.energy).sum())
    out, _ = _run_steps(cfg, world, pop, positions, native=native, steps=1)
    assert bool(out.alive[0])
    np.testing.assert_allclose(float(np.asarray(out.energy).sum()) - total,
                               float(out.energy_in) - float(out.heat), rtol=1e-5, atol=1e-4)
