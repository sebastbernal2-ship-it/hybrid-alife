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


def test_full_native_step_is_bit_exact_without_mutation():
    divided = 0
    for seed, dead_frac, kill0 in ((0, 0.5, True), (1, 0.5, False), (2, 0.9, True),
                                   (3, 0.2, True), (4, 0.0, False)):
        cfg, world, pop, positions = _coupled_setup(seed, dead_frac, kill0)
        ref_pop, ref_world = _run_steps(cfg, world, pop, positions, native=False)
        nat_pop, nat_world = _run_steps(cfg, world, pop, positions, native=True)
        for f in POP_FIELDS:
            np.testing.assert_array_equal(
                getattr(ref_pop, f), getattr(nat_pop, f), err_msg=f"{f} seed={seed}")
        np.testing.assert_array_equal(ref_world.metabolites, nat_world.metabolites)
        np.testing.assert_array_equal(ref_world.concentration, nat_world.concentration)
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
