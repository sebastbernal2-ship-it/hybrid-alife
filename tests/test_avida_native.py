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
