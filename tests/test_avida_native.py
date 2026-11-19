import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_alife.agents import avida_vm
from hybrid_alife.types import AvidaConfig, WorldConfig
from hybrid_alife.world.env import initialize_world

FIELDS = ("registers", "ip", "read_head", "write_head", "copied", "copied_length",
          "tasks_completed", "last_input_a", "last_input_b")


def _setup(prob):
    cfg = AvidaConfig(
        enabled=True, population_size=64, genome_length=24, max_genome_length=32, registers=3,
        memory_size=8, cycles_per_update=4, point_mutation_prob=prob, insertion_prob=0.0,
        deletion_prob=0.0, max_cycles_per_update=12)
    wcfg = WorldConfig(width=12, height=12, toroidal=True, resource_channels=1, hazard_channels=1,
                       max_agents=4, max_digital_organisms=64, flow_noise_std=0.0,
                       concentration_decay=0.9, concentration_diffusion=0.05)
    world = initialize_world(wcfg, jax.random.PRNGKey(1))
    # Non-trivial concentration so LOAD_ENV inputs differ and logic tasks can score.
    world.concentration = world.concentration + 0.3
    pop = avida_vm.initialize_avida_population(cfg, jax.random.PRNGKey(2))
    positions = jax.random.uniform(jax.random.PRNGKey(3), (64, 2), minval=0, maxval=12)
    return cfg, world, pop, positions


def _clone(pop):
    return dataclasses.replace(pop)


def test_native_matches_jax_without_mutation():
    cfg, world, pop, positions = _setup(0.0)
    key = jax.random.PRNGKey(4)
    budget = avida_vm._cycle_budget(pop.merit, cfg)

    ref = _clone(pop)
    ref._divide_op_mask = jnp.zeros_like(ref.alive)
    for c in range(cfg.max_cycles_per_update):
        ref = avida_vm.vm_cycle(ref, world, cfg, positions, key, budget > c)

    from hybrid_alife.physics import avida_native

    nat = _clone(pop)
    env_a, env_b = avida_vm.sample_env(world, positions, 64)
    div = avida_native.run_update(
        nat, cfg, budget, env_a, env_b, avida_vm._task_reward_vector(cfg), seed=0)

    for f in FIELDS:
        np.testing.assert_array_equal(getattr(ref, f), getattr(nat, f), err_msg=f)
    np.testing.assert_allclose(ref.merit, nat.merit, rtol=1e-5)
    np.testing.assert_array_equal(ref._divide_op_mask, div)
    assert int(div.sum()) > 0  # the seeded replicator template actually divides


def test_native_step_runs_with_mutation_and_divides():
    cfg, world, pop, positions = _setup(0.05)
    cfg = dataclasses.replace(cfg, native=True)
    key = jax.random.PRNGKey(5)
    pop2, nxt = avida_vm.step_avida_population(pop, world, cfg, 1000, key, positions)
    assert nxt == 1000 + 64
    assert int(pop2.copied_length.max()) >= 0 and pop2.genomes.shape == (64, 32)
    assert bool(jnp.all((pop2.genomes >= 0) & (pop2.genomes < avida_vm.INSTRUCTION_COUNT)))


POP_FIELDS = ("genomes", "genome_lengths", "registers", "memory", "ip", "read_head", "write_head",
              "copied", "copied_length", "merit", "age", "alive", "lineage_id", "parent_id",
              "lineage_depth", "tasks_completed", "last_input_a", "last_input_b")


def _coupled_setup(seed, dead_frac, kill_slot0, **cfg_over):
    cfg, world, pop, positions = _setup(0.0)
    cfg = dataclasses.replace(cfg, metabolite_uptake=0.3, metabolite_deposit=0.05, **cfg_over)
    key = jax.random.PRNGKey(seed)
    world.metabolites = jax.random.uniform(key, world.metabolites.shape)
    # Few distinct cells so many organisms share a cell (exercises duplicate scatter-adds).
    positions = jnp.floor(positions * 4.0) / 12.0 + 0.01
    alive = jax.random.bernoulli(jax.random.fold_in(key, 1), 1.0 - dead_frac, (64,))
    alive = alive.at[0].set(not kill_slot0)
    pop = dataclasses.replace(pop, alive=alive)
    return cfg, world, pop, positions


def _run_steps(cfg, world, pop, positions, native, steps=3, seed=9):
    cfg = dataclasses.replace(cfg, native=native)
    world = dataclasses.replace(world)
    pop = _clone(pop)
    nxt = 1000
    for i in range(steps):
        pop, nxt = avida_vm.step_avida_population(
            pop, world, cfg, nxt, jax.random.PRNGKey(seed + i), positions)
    return pop, world


def _same(a, b, err_msg=""):
    """Exact for integer/bool state; float32 fields differ by ~1 ulp across CPUs (summation
    order and FMA use in XLA vs the C++ build), so they get a tight relative tolerance."""
    a, b = np.asarray(a), np.asarray(b)
    if np.issubdtype(a.dtype, np.floating):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6, err_msg=err_msg)
    else:
        np.testing.assert_array_equal(a, b, err_msg=err_msg)


def test_full_native_step_is_bit_exact_without_mutation():
    divided = 0
    for seed, dead_frac, kill0 in ((0, 0.5, True), (1, 0.5, False), (2, 0.9, True),
                                   (3, 0.2, True), (4, 0.0, False)):
        cfg, world, pop, positions = _coupled_setup(seed, dead_frac, kill0)
        ref_pop, ref_world = _run_steps(cfg, world, pop, positions, native=False)
        nat_pop, nat_world = _run_steps(cfg, world, pop, positions, native=True)
        for f in POP_FIELDS:
            _same(getattr(ref_pop, f), getattr(nat_pop, f), err_msg=f"{f} seed={seed}")
        _same(ref_world.metabolites, nat_world.metabolites)
        _same(ref_world.concentration, nat_world.concentration)
        divided += int((ref_pop.lineage_depth > 0).sum())
    assert divided > 0  # births actually happened in the compared runs


def test_native_mutated_division_keeps_invariants():
    cfg, world, pop, positions = _coupled_setup(5, 0.6, False)
    cfg = dataclasses.replace(cfg, point_mutation_prob=0.02, insertion_prob=0.3,
                              deletion_prob=0.3, duplication_prob=0.4, native=True)
    world = dataclasses.replace(world)
    pop = _clone(pop)
    depth_before = np.asarray(pop.lineage_depth)
    ids_before = np.asarray(pop.lineage_id)
    alive_before = np.asarray(pop.alive)
    pop, _ = avida_vm.step_avida_population(
        pop, world, cfg, 1000, jax.random.PRNGKey(3), positions)
    g, lens = np.asarray(pop.genomes), np.asarray(pop.genome_lengths)
    assert g.min() >= 0 and g.max() < avida_vm.INSTRUCTION_COUNT
    assert lens.min() >= cfg.min_genome_length and lens.max() <= cfg.max_genome_length
    born = np.asarray(pop.alive) & ~alive_before
    assert born.sum() > 0
    parents = np.asarray(pop.parent_id)[born]
    depth = np.asarray(pop.lineage_depth)
    assert np.all(np.asarray(pop.lineage_id)[born] >= 1000)
    for child, pid in zip(np.nonzero(born)[0], parents, strict=True):
        parent_slot = int(np.nonzero(ids_before == pid)[0][0])
        assert depth[child] == depth_before[parent_slot] + 1
        assert float(pop.merit[child]) >= 1.0 and int(pop.age[child]) == 1


# ---------------------------------------------------------------------------
# Energy / heat accounting
# ---------------------------------------------------------------------------


def _energy_setup(seed=0, dead_frac=0.4, kill0=True, **cfg_over):
    over = {"instruction_energy_cost": 0.1, "energy_initial": 20.0, "energy_efficiency": 0.5,
            **cfg_over}
    cfg, world, pop, positions = _coupled_setup(seed, dead_frac, kill0)
    return dataclasses.replace(cfg, **over), world, pop, positions


def _energy_total(pop):
    return float(np.asarray(pop.energy, dtype=np.float64).sum())


def _ledger_gap(pop, e0_total):
    return (_energy_total(pop) - e0_total) - (float(pop.energy_in) - float(pop.heat))


def test_energy_is_off_by_default_and_costless_energy_changes_nothing():
    cfg, world, pop, positions = _coupled_setup(0, 0.4, True)
    ref, ref_world = _run_steps(cfg, world, pop, positions, native=False)
    assert ref.energy is None and ref.heat is None and ref.energy_in is None
    # Energy enabled but never binding: identical dynamics on every other field.
    rich = dataclasses.replace(cfg, instruction_energy_cost=1e-9, energy_initial=1e6)
    on, on_world = _run_steps(rich, world, pop, positions, native=False)
    for f in POP_FIELDS:
        np.testing.assert_array_equal(getattr(ref, f), getattr(on, f), err_msg=f)
    np.testing.assert_array_equal(ref_world.metabolites, on_world.metabolites)
    np.testing.assert_array_equal(ref_world.concentration, on_world.concentration)


def test_energy_ledger_matches_independent_counts_without_division():
    cfg, world, pop, positions = _energy_setup(1, 0.3, False, energy_initial=50.0)
    cfg = dataclasses.replace(cfg, metabolite_uptake=0.0, metabolite_deposit=0.0)
    genomes = jnp.where(pop.genomes == avida_vm.Op.H_DIVIDE, avida_vm.Op.NOP, pop.genomes)
    pop = dataclasses.replace(pop, genomes=genomes)
    budget = np.asarray(avida_vm._cycle_budget(pop.merit, cfg))
    alive = np.asarray(pop.alive)
    merit0 = float(np.asarray(pop.merit, dtype=np.float64)[alive].sum())
    for native in (False, True):
        out, _ = _run_steps(cfg, world, pop, positions, native=native, steps=1)
        cycles = float(budget[alive].sum())  # every organism has ample energy
        np.testing.assert_allclose(
            float(out.heat), cfg.instruction_energy_cost * cycles, rtol=1e-4)
        gained = float(np.asarray(out.merit, dtype=np.float64)[alive].sum()) - merit0
        np.testing.assert_allclose(
            float(out.energy_in), cfg.energy_efficiency * gained, rtol=1e-4)
        e0 = float(alive.sum()) * cfg.energy_initial
        assert abs(_ledger_gap(out, e0)) < 1e-3 * max(1.0, float(out.energy_in))


def test_energy_ledger_balances_with_births_and_starvation():
    for native in (False, True):
        cfg, world, pop, positions = _energy_setup(2, 0.5, True, energy_initial=1.5)
        e0 = float(np.asarray(pop.alive).sum()) * cfg.energy_initial
        out, _ = _run_steps(cfg, world, pop, positions, native=native, steps=4)
        energy, alive = np.asarray(out.energy), np.asarray(out.alive)
        assert energy.min() >= 0.0  # payments are clamped to energy on hand
        assert np.all(energy[alive] > 0.0)  # alive implies energy left
        assert np.all(energy[~alive] == 0.0)  # death leaves nothing behind
        assert alive.sum() < 64 and float(out.heat) > 0.0  # starvation and dissipation happened
        assert abs(_ledger_gap(out, e0)) < 1e-3 * max(1.0, float(out.energy_in) + e0)


def test_parent_energy_is_split_with_the_child():
    cfg, world, pop, positions = _energy_setup(3, 0.5, False, energy_initial=50.0,
                                               instruction_energy_cost=1e-6)
    cfg = dataclasses.replace(cfg, metabolite_uptake=0.0, metabolite_deposit=0.0)
    out, _ = _run_steps(cfg, world, pop, positions, native=False, steps=1)
    born = np.asarray(out.alive) & ~np.asarray(pop.alive)
    assert born.sum() > 0
    energy = np.asarray(out.energy)
    # A child holds the half its parent gave up, so it sits near half of the starting energy.
    assert np.all(np.abs(energy[born] - 25.0) < 5.0)


def test_native_and_jax_agree_on_energy_without_mutation():
    scenarios = ((0, 0.4, True, {}), (1, 0.8, True, {"energy_initial": 1.0}),
                 (2, 0.0, False, {"energy_initial": 2.0, "instruction_energy_cost": 0.3}))
    for seed, dead_frac, kill0, over in scenarios:
        cfg, world, pop, positions = _energy_setup(seed, dead_frac, kill0, **over)
        ref, _ = _run_steps(cfg, world, pop, positions, native=False, steps=3)
        nat, _ = _run_steps(cfg, world, pop, positions, native=True, steps=3)
        for f in (*POP_FIELDS, "energy"):
            _same(getattr(ref, f), getattr(nat, f), err_msg=f"{f} seed={seed}")
        np.testing.assert_allclose(float(ref.heat), float(nat.heat), rtol=1e-4)
        np.testing.assert_allclose(float(ref.energy_in), float(nat.energy_in), rtol=1e-4)
def test_native_under_jit_fails_with_a_clear_error():
    import pytest

    cfg, world, pop, positions = _setup(0.0)
    cfg = dataclasses.replace(cfg, native=True)

    @jax.jit
    def step(genomes, merit):
        p = dataclasses.replace(pop, genomes=genomes, merit=merit)
        out, _ = avida_vm.step_avida_population(p, world, cfg, 0, jax.random.PRNGKey(0), positions)
        return out.merit

    with pytest.raises(RuntimeError, match="cannot run under jax.jit"):
        step(pop.genomes, pop.merit)


def test_native_rejects_states_that_would_index_out_of_bounds():
    import pytest

    from hybrid_alife.physics import avida_native

    cfg, world, pop, positions = _setup(0.0)
    bad_len = dataclasses.replace(pop, genome_lengths=pop.genome_lengths.at[3].set(10_000))
    with pytest.raises(ValueError, match="exceed max_genome_length"):
        avida_native.run_update(
            bad_len, cfg, jnp.full((64,), 2, jnp.int32), jnp.zeros(64, jnp.int32),
            jnp.zeros(64, jnp.int32), avida_vm._task_reward_vector(cfg), seed=0)
    one_reg = dataclasses.replace(pop, registers=pop.registers[:, :1])
    with pytest.raises(ValueError, match="at least 2 registers"):
        avida_native.run_update(
            one_reg, cfg, jnp.full((64,), 2, jnp.int32), jnp.zeros(64, jnp.int32),
            jnp.zeros(64, jnp.int32), avida_vm._task_reward_vector(cfg), seed=0)


def test_native_opcode_count_matches_python_vm():
    from hybrid_alife.physics import avida_native

    assert avida_native._load().avida_n_ops() == avida_vm.INSTRUCTION_COUNT


def test_avida_config_validates_energy_and_native_registers():
    import pytest

    base = dict(
        enabled=True, population_size=4, genome_length=8, max_genome_length=16, registers=3,
        memory_size=4, cycles_per_update=2, point_mutation_prob=0.0, insertion_prob=0.0,
        deletion_prob=0.0)
    with pytest.raises(ValueError, match="instruction_energy_cost"):
        AvidaConfig(**base, instruction_energy_cost=-1.0)
    with pytest.raises(ValueError, match="energy_initial"):
        AvidaConfig(**base, instruction_energy_cost=0.1, energy_initial=0.0)
    with pytest.raises(ValueError, match="registers"):
        AvidaConfig(**{**base, "registers": 1}, native=True)
    AvidaConfig(**base, instruction_energy_cost=0.1)  # valid config still constructs


def test_native_divide_and_couple_reject_bad_grids_and_tracers():
    import pytest

    from hybrid_alife.physics import avida_native

    cfg, world, pop, positions = _setup(0.0)
    with pytest.raises(ValueError, match="one row per organism"):
        avida_native.couple(pop, world, dataclasses.replace(cfg, metabolite_uptake=0.1),
                            positions[:10])

    @jax.jit
    def traced(g):
        p = dataclasses.replace(pop, genomes=g)
        avida_native.divide(p, cfg, 0, 1, jnp.zeros(64, bool))
        return p.genomes

    with pytest.raises(RuntimeError, match="cannot run under jax.jit"):
        traced(pop.genomes)
