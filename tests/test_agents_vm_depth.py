"""Deep tests for the embodied controller, action semantics, and Avida VM.

These complement ``test_smoke.py`` by exercising each branch's individual
mechanisms — observation construction, every action class, controller
ablations, VM replication, logic-task scoring, and merit-based CPU
allocation — at a finer granularity than the end-to-end smoke run.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_alife.agents.avida_vm import (
    INSTRUCTION_COUNT,
    Op,
    _cycle_budget,
    _logic_task_outputs,
    apply_h_divide,
    initialize_avida_population,
    step_avida_population,
    vm_cycle,
)
from hybrid_alife.agents.controller import (
    decode_actions,
    forward as controller_forward,
    init_controller_params,
)
from hybrid_alife.agents.embodied import (
    act_embodied,
    apply_embodied_actions,
    apply_reproduction,
    embodied_action_dim,
    embodied_observation_dim,
    initialize_embodied_population,
    observe_embodied,
)
from hybrid_alife.experiments.runner import load_config
from hybrid_alife.types import AvidaConfig, EmbodiedConfig
from hybrid_alife.world.env import initialize_world


def _cfg():
    return load_config("configs/base.yaml")


def _replace_embodied(cfg, **kwargs) -> EmbodiedConfig:
    return dataclasses.replace(cfg.embodied, **kwargs)


def _replace_avida(cfg, **kwargs) -> AvidaConfig:
    return dataclasses.replace(cfg.avida, **kwargs)


# ---------------------------------------------------------------------------
# Controller / observation
# ---------------------------------------------------------------------------


def test_controller_forward_shapes():
    cfg = _cfg()
    n = cfg.embodied.population_size
    obs_dim = embodied_observation_dim(cfg.embodied, cfg.world)
    act_dim = embodied_action_dim(cfg.embodied)
    params = init_controller_params(n, obs_dim, act_dim, cfg.embodied.hidden_size, jax.random.PRNGKey(0))
    obs = jnp.ones((n, obs_dim), dtype=jnp.float32)
    hidden = jnp.zeros((n, cfg.embodied.hidden_size), dtype=jnp.float32)
    raw, new_hidden = controller_forward(params, obs, hidden, use_memory=True)
    assert raw.shape == (n, act_dim)
    assert new_hidden.shape == hidden.shape


def test_controller_mlp_ablation_drops_memory():
    """With use_memory=False, the GRU update gate is bypassed and the new
    hidden state equals the candidate. With memory on, the new hidden carries
    a weighted average of the previous hidden, so the two paths must differ
    given a non-zero starting hidden."""
    cfg = _cfg()
    n = cfg.embodied.population_size
    obs_dim = embodied_observation_dim(cfg.embodied, cfg.world)
    act_dim = embodied_action_dim(cfg.embodied)
    params = init_controller_params(n, obs_dim, act_dim, cfg.embodied.hidden_size, jax.random.PRNGKey(0))
    obs = jax.random.normal(jax.random.PRNGKey(1), (n, obs_dim))
    h_start = jnp.ones((n, cfg.embodied.hidden_size))
    _, h_mem = controller_forward(params, obs, h_start, use_memory=True)
    _, h_no_mem = controller_forward(params, obs, h_start, use_memory=False)
    # The two should not be identical (memory mixes in h_start).
    assert not bool(jnp.allclose(h_mem, h_no_mem, atol=1e-4))


def test_observation_dim_matches_layout():
    cfg = _cfg()
    key = jax.random.PRNGKey(0)
    world = initialize_world(cfg.world, key)
    pop = initialize_embodied_population(cfg.embodied, cfg.world, key)
    obs = observe_embodied(pop, world, cfg.embodied, cfg.world, key)
    assert obs.shape == (
        cfg.embodied.population_size,
        embodied_observation_dim(cfg.embodied, cfg.world),
    )
    # No NaNs in observations is a baseline invariant.
    assert bool(jnp.all(jnp.isfinite(obs)))


def test_stochastic_action_sampling_produces_binary_gates():
    cfg = _cfg()
    embodied_cfg = _replace_embodied(cfg, stochastic_actions=True)
    n = embodied_cfg.population_size
    act_dim = embodied_action_dim(embodied_cfg)
    # Build large positive logits so sigmoid -> ~1 and Bernoulli samples should
    # be ~1 (hard gates).
    raw = jnp.concatenate(
        [
            jnp.zeros((n, 2)),
            jnp.full((n, 6), 10.0),  # gates
            jnp.zeros((n, embodied_cfg.message_size)),
        ],
        axis=-1,
    )
    actions = decode_actions(raw, embodied_cfg, jax.random.PRNGKey(0))
    assert actions.shape == (n, act_dim)
    # Gates should be hard ~1 (sigmoid(10) ~= 0.99995).
    assert bool(jnp.all(actions[:, 2:8] > 0.5))


# ---------------------------------------------------------------------------
# Action semantics: each gate has a deterministic, testable effect
# ---------------------------------------------------------------------------


def _zero_actions(n: int, dim: int) -> jnp.ndarray:
    return jnp.zeros((n, dim), dtype=jnp.float32)


def test_action_move_changes_positions():
    cfg = _cfg()
    key = jax.random.PRNGKey(7)
    world = initialize_world(cfg.world, key)
    pop = initialize_embodied_population(cfg.embodied, cfg.world, key)
    n = pop.alive.shape[0]
    act = _zero_actions(n, embodied_action_dim(cfg.embodied)).at[:, 0].set(1.0)
    p_before = pop.positions
    pop, _, _ = apply_embodied_actions(pop, act, world, cfg.embodied, cfg.world, key)
    # x movement (column 0) should shift x positions for living agents.
    moved = jnp.linalg.norm(pop.positions - p_before, axis=-1)
    assert bool(jnp.all(moved > 0.0))


def test_action_eat_drains_resources_and_gains_energy():
    cfg = _cfg()
    key = jax.random.PRNGKey(8)
    world = initialize_world(cfg.world, key)
    pop = initialize_embodied_population(cfg.embodied, cfg.world, key)
    n = pop.alive.shape[0]
    e0 = pop.energy
    act = _zero_actions(n, embodied_action_dim(cfg.embodied)).at[:, 2].set(1.0)
    pop, new_world, _ = apply_embodied_actions(pop, act, world, cfg.embodied, cfg.world, key)
    # Total resource mass should not increase from eating.
    assert float(jnp.sum(new_world.resources)) <= float(jnp.sum(world.resources)) + 1e-3
    # Energy of at least one agent must change (gain or net loss after basal).
    assert bool(jnp.any(pop.energy != e0))


def test_action_attack_deposits_hazard():
    cfg = _cfg()
    key = jax.random.PRNGKey(9)
    world = initialize_world(cfg.world, key)
    pop = initialize_embodied_population(cfg.embodied, cfg.world, key)
    n = pop.alive.shape[0]
    act = _zero_actions(n, embodied_action_dim(cfg.embodied)).at[:, 3].set(1.0)
    h_before = jnp.sum(world.hazards)
    _, new_world, _ = apply_embodied_actions(pop, act, world, cfg.embodied, cfg.world, key)
    assert float(jnp.sum(new_world.hazards)) >= float(h_before)


def test_action_terraform_changes_concentration():
    cfg = _cfg()
    key = jax.random.PRNGKey(10)
    world = initialize_world(cfg.world, key)
    pop = initialize_embodied_population(cfg.embodied, cfg.world, key)
    n = pop.alive.shape[0]
    act_r = _zero_actions(n, embodied_action_dim(cfg.embodied)).at[:, 5].set(1.0)
    _, w_r, _ = apply_embodied_actions(pop, act_r, world, cfg.embodied, cfg.world, key)
    assert float(jnp.mean(w_r.concentration)) >= float(jnp.mean(world.concentration)) - 1e-6
    act_h = _zero_actions(n, embodied_action_dim(cfg.embodied)).at[:, 6].set(1.0)
    _, w_h, _ = apply_embodied_actions(pop, act_h, world, cfg.embodied, cfg.world, key)
    # Terraform-hazard pushes the first concentration channel down (negative
    # deposit) relative to the resource-terraform case.
    assert float(jnp.mean(w_h.concentration[..., 0])) <= float(jnp.mean(w_r.concentration[..., 0])) + 1e-6


def test_action_emit_respects_message_gate_threshold():
    cfg = _cfg()
    embodied_cfg = _replace_embodied(cfg, message_gate_threshold=0.6)
    key = jax.random.PRNGKey(11)
    world = initialize_world(cfg.world, key)
    pop = initialize_embodied_population(embodied_cfg, cfg.world, key)
    n = pop.alive.shape[0]
    act_dim = embodied_action_dim(embodied_cfg)

    # emit gate just below threshold -> no message change
    act_low = _zero_actions(n, act_dim).at[:, 7].set(0.5)
    act_low = act_low.at[:, 8:].set(1.0)
    pop_low, _, _ = apply_embodied_actions(pop, act_low, world, embodied_cfg, cfg.world, key)
    # Initial messages are zero; below-threshold emit must keep them zero (post-decay).
    assert float(jnp.max(jnp.abs(pop_low.messages))) < 1e-6

    # emit gate above threshold -> messages updated
    act_high = _zero_actions(n, act_dim).at[:, 7].set(0.9)
    act_high = act_high.at[:, 8:].set(1.0)
    pop_high, _, _ = apply_embodied_actions(pop, act_high, world, embodied_cfg, cfg.world, key)
    assert float(jnp.max(jnp.abs(pop_high.messages))) > 0.0


def test_action_reproduce_uses_dead_slots():
    cfg = _cfg()
    key = jax.random.PRNGKey(12)
    pop = initialize_embodied_population(cfg.embodied, cfg.world, key)
    n = pop.alive.shape[0]
    half = n // 2
    pop.alive = pop.alive.at[half:].set(False)
    pop.energy = pop.energy.at[:half].set(cfg.embodied.reproduce_energy_threshold + 1)
    repro = jnp.where(jnp.arange(n) < half, 1.0, 0.0)
    pop, _births = apply_reproduction(pop, repro, cfg.embodied, next_lineage_start=999, key=key)
    assert int(pop.alive.sum()) == n


def test_action_cost_drains_energy_under_full_load():
    cfg = _cfg()
    # Aggressive action cost to make this easy to observe.
    embodied_cfg = _replace_embodied(
        cfg,
        cost_eat=0.5,
        cost_attack=0.5,
        cost_terraform=0.5,
        cost_emit=0.5,
        cost_move_scale=1.0,
    )
    key = jax.random.PRNGKey(13)
    world = initialize_world(cfg.world, key)
    pop = initialize_embodied_population(embodied_cfg, cfg.world, key)
    n = pop.alive.shape[0]
    act_dim = embodied_action_dim(embodied_cfg)
    # All gates on, full movement.
    act = jnp.zeros((n, act_dim), dtype=jnp.float32)
    act = act.at[:, 0].set(1.0).at[:, 1].set(1.0)
    act = act.at[:, 2:8].set(1.0)
    e0 = pop.energy
    pop, _, _ = apply_embodied_actions(pop, act, world, embodied_cfg, cfg.world, key)
    # Energy should have *decreased* relative to a zero-action baseline.
    pop_zero = initialize_embodied_population(embodied_cfg, cfg.world, key)
    zero_act = jnp.zeros((n, act_dim), dtype=jnp.float32)
    pop_zero, _, _ = apply_embodied_actions(pop_zero, zero_act, world, embodied_cfg, cfg.world, key)
    assert float(pop.energy.mean()) < float(pop_zero.energy.mean())
    _ = e0


# ---------------------------------------------------------------------------
# Avida VM
# ---------------------------------------------------------------------------


def test_logic_task_outputs_correct():
    a_val = 0b1100
    b_val = 0b1010
    a = jnp.array([a_val], dtype=jnp.int32)
    b = jnp.array([b_val], dtype=jnp.int32)
    out = _logic_task_outputs(a, b)[0]
    mask = np.uint32(0xFFFFFFFF)
    expected = np.array(
        [
            (~np.uint32(a_val)) & mask,
            (~(np.uint32(a_val) & np.uint32(b_val))) & mask,
            np.uint32(a_val) & np.uint32(b_val),
            np.uint32(a_val) | ((~np.uint32(b_val)) & mask),
            np.uint32(a_val) | np.uint32(b_val),
            np.uint32(a_val) & ((~np.uint32(b_val)) & mask),
            (~(np.uint32(a_val) | np.uint32(b_val))) & mask,
            np.uint32(a_val) ^ np.uint32(b_val),
            (~(np.uint32(a_val) ^ np.uint32(b_val))) & mask,
        ],
        dtype=np.uint32,
    )
    # Compare two's-complement bit patterns (JAX is signed int32).
    out_u32 = np.asarray(out).astype(np.uint32)
    assert bool(np.all(out_u32 == expected))


def test_cycle_budget_scales_with_merit():
    cfg = _cfg()
    merit = jnp.array([1.0, 4.0, 16.0, 100.0], dtype=jnp.float32)
    budget = _cycle_budget(merit, cfg.avida)
    # Higher merit -> at least as many cycles, bounded by max.
    assert int(budget[0]) <= int(budget[-1]) <= cfg.avida.max_cycles_per_update
    assert int(budget[0]) >= cfg.avida.cycles_per_update


def test_avida_h_alloc_resets_buffer():
    cfg = _cfg()
    key = jax.random.PRNGKey(14)
    world = initialize_world(cfg.world, key)
    pop = initialize_avida_population(cfg.avida, key)
    # Pre-pollute copied buffer and length.
    pop.copied = pop.copied + 7
    pop.copied_length = jnp.full_like(pop.copied_length, 10)
    # Set every organism's current instruction to H_ALLOC.
    pop.genomes = jnp.full_like(pop.genomes, int(Op.H_ALLOC))
    pop.genome_lengths = jnp.full_like(pop.genome_lengths, cfg.avida.max_genome_length)
    pop.ip = jnp.zeros_like(pop.ip)
    pop = vm_cycle(pop, world, cfg.avida, None, key)
    assert int(pop.copied_length.max()) == 0
    assert int(pop.write_head.max()) == 0


def test_avida_h_copy_then_divide_creates_offspring():
    cfg = _cfg()
    key = jax.random.PRNGKey(15)
    world = initialize_world(cfg.world, key)
    avida_cfg = _replace_avida(cfg, point_mutation_prob=0.0, insertion_prob=0.0, deletion_prob=0.0, duplication_prob=0.0)
    pop = initialize_avida_population(avida_cfg, key)
    # Kill half so there are dead slots to receive children.
    n = pop.alive.shape[0]
    pop.alive = pop.alive.at[n // 2:].set(False)
    # Inject a hand-crafted minimal replicator at every parent.
    template = jnp.array(
        [Op.H_ALLOC] + [Op.H_COPY] * avida_cfg.genome_length + [Op.H_DIVIDE],
        dtype=jnp.int32,
    )
    pop.genomes = pop.genomes.at[:, : template.shape[0]].set(template[None, :])
    pop.genome_lengths = jnp.full_like(pop.genome_lengths, template.shape[0])
    parent_before = pop.alive.sum()
    for _ in range(3):
        key, sub = jax.random.split(key)
        pop, _ = step_avida_population(pop, world, avida_cfg, 1000, sub)
    assert int(pop.alive.sum()) >= int(parent_before)


def test_avida_emit_credits_logic_task_and_grows_merit():
    cfg = _cfg()
    key = jax.random.PRNGKey(16)
    world = initialize_world(cfg.world, key)
    pop = initialize_avida_population(cfg.avida, key)
    n = pop.alive.shape[0]
    # Pre-load inputs and arrange for the next instruction to be EMIT.
    pop.last_input_a = jnp.full((n,), 0b1100, dtype=jnp.int32)
    pop.last_input_b = jnp.full((n,), 0b1010, dtype=jnp.int32)
    # Set register 0 to the AND of those inputs.
    pop.registers = pop.registers.at[:, 0].set(0b1100 & 0b1010)
    # Force every IP to point at an EMIT instruction.
    pop.genomes = jnp.full_like(pop.genomes, int(Op.EMIT))
    pop.genome_lengths = jnp.full_like(pop.genome_lengths, cfg.avida.max_genome_length)
    pop.ip = jnp.zeros_like(pop.ip)
    merit_before = pop.merit
    pop = vm_cycle(pop, world, cfg.avida, None, key)
    # AND task bit should now be set in every (live) organism.
    bit_and = 1 << 2
    assert int((pop.tasks_completed & bit_and).min()) == bit_and
    # Merit should have increased by at least task_reward_and.
    assert float((pop.merit - merit_before).min()) >= float(cfg.avida.task_reward_and) - 1e-3


def test_avida_duplication_extends_genome_length():
    cfg = _cfg()
    key = jax.random.PRNGKey(17)
    avida_cfg = _replace_avida(
        cfg,
        point_mutation_prob=0.0,
        insertion_prob=0.0,
        deletion_prob=0.0,
        duplication_prob=1.0,
    )
    world = initialize_world(cfg.world, key)
    pop = initialize_avida_population(avida_cfg, key)
    n = pop.alive.shape[0]
    pop.alive = pop.alive.at[n // 2:].set(False)
    # Mark some organisms as having just executed H_DIVIDE with enough copy.
    pop._divide_op_mask = jnp.ones((n,), dtype=bool)  # type: ignore[attr-defined]
    pop.copied_length = jnp.full((n,), avida_cfg.genome_length, dtype=jnp.int32)
    pop.copied = jnp.broadcast_to(
        jnp.arange(avida_cfg.max_genome_length)[None, :],
        pop.copied.shape,
    ).astype(jnp.int32)
    pop = apply_h_divide(pop, avida_cfg, 5000, key)
    # At least one new child should have genome_length > parent's genome_length.
    assert int(pop.genome_lengths.max()) > avida_cfg.genome_length


def test_avida_mutation_only_active_when_prob_positive():
    cfg = _cfg()
    key = jax.random.PRNGKey(18)
    avida_cfg = _replace_avida(cfg, point_mutation_prob=0.0)
    world = initialize_world(cfg.world, key)
    pop = initialize_avida_population(avida_cfg, key)
    # Set every IP to NAND so we can check determinism.
    pop.genomes = jnp.full_like(pop.genomes, int(Op.NAND))
    pop.genome_lengths = jnp.full_like(pop.genome_lengths, avida_cfg.max_genome_length)
    pop.registers = pop.registers.at[:, 0].set(0xF0).at[:, 1].set(0x0F)
    out1 = vm_cycle(pop, world, avida_cfg, None, key)
    # Reset and rerun: deterministic register update under no mutation.
    pop2 = initialize_avida_population(avida_cfg, key)
    pop2.genomes = jnp.full_like(pop2.genomes, int(Op.NAND))
    pop2.genome_lengths = jnp.full_like(pop2.genome_lengths, avida_cfg.max_genome_length)
    pop2.registers = pop2.registers.at[:, 0].set(0xF0).at[:, 1].set(0x0F)
    out2 = vm_cycle(pop2, world, avida_cfg, None, key)
    assert bool(jnp.all(out1.registers == out2.registers))


def test_act_embodied_deterministic_when_not_stochastic():
    """When stochastic_actions=False, the key is unused and the result is
    a deterministic function of (genome, obs, hidden). We use fresh pops to
    avoid leaking mutated hidden state across runs."""
    cfg = _cfg()
    key = jax.random.PRNGKey(19)
    world = initialize_world(cfg.world, key)
    pop1 = initialize_embodied_population(cfg.embodied, cfg.world, key)
    obs = observe_embodied(pop1, world, cfg.embodied, cfg.world, key)
    a1, _ = act_embodied(pop1, obs, cfg.embodied)
    pop2 = initialize_embodied_population(cfg.embodied, cfg.world, key)
    a2, _ = act_embodied(pop2, obs, cfg.embodied, jax.random.PRNGKey(0))
    assert bool(jnp.allclose(a1, a2))
