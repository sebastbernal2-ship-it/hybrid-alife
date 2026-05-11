"""Smoke tests covering world, embodied actions, Avida VM replication, runner,
metrics, checkpoints, and archives. CPU-feasible.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_alife.agents.avida_vm import (
    INSTRUCTION_COUNT,
    initialize_avida_population,
    step_avida_population,
    vm_cycle,
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
from hybrid_alife.evolution.archives import MapElitesArchive, NoveltyArchive
from hybrid_alife.evolution.selection import (
    reseed_avida,
    reseed_embodied,
    select_and_mutate_embodied,
    tournament_select,
)
from hybrid_alife.experiments.runner import (
    initialize_sim,
    load_config,
    run_experiment,
    step_sim,
)
from hybrid_alife.logging.jsonl import JsonlWriter, read_jsonl
from hybrid_alife.metrics.core import collect_full_metrics
from hybrid_alife.replay.checkpoint import (
    load_checkpoint,
    restore_states,
    save_checkpoint,
)
from hybrid_alife.world.env import (
    consume_from_grid,
    gather_patch,
    initialize_world,
    positions_to_grid,
    scatter_add_to_grid,
    step_world,
)
from hybrid_alife.world.sensing import sample_sixth_sense, sixth_sense_dim


def _cfg():
    return load_config("configs/base.yaml")


# ----- world ----------------------------------------------------------------


def test_world_init_shapes():
    cfg = _cfg()
    key = jax.random.PRNGKey(0)
    world = initialize_world(cfg.world, key)
    h, w = cfg.world.height, cfg.world.width
    assert world.terrain.shape == (h, w, 4)
    assert world.resources.shape == (h, w, cfg.world.resource_channels)
    assert world.flow.shape == (h, w, 2)
    assert world.shear.shape == (h, w, 2)
    assert world.shear_grad.shape == (h, w, 2)
    assert world.enrichment.shape == (h, w, 1)
    assert world.lift.shape == (h, w, 2)
    assert world.metabolites.shape == (h, w, cfg.world.metabolite_channels)


def test_step_world_runs():
    cfg = _cfg()
    state = initialize_sim(cfg)
    out = step_world(state, cfg.world)
    assert int(out.world.time) == 1


def test_positions_to_grid_within_bounds():
    pos = jnp.array([[0.0, 0.0], [0.999, 0.999], [0.5, 0.25]])
    grid = positions_to_grid(pos, 10, 10)
    assert int(grid[0, 0]) == 0
    assert int(grid[1, 0]) == 9


def test_gather_patch_shape():
    cfg = _cfg()
    world = initialize_world(cfg.world, jax.random.PRNGKey(0))
    pos = jnp.array([[0.1, 0.2], [0.9, 0.5]])
    patch = gather_patch(world.resources, pos, 2, cfg.world.height, cfg.world.width, True)
    side = 2 * 2 + 1
    assert patch.shape == (2, side * side * cfg.world.resource_channels)


def test_scatter_and_consume():
    cfg = _cfg()
    world = initialize_world(cfg.world, jax.random.PRNGKey(0))
    pos = jnp.array([[0.1, 0.1]])
    alive = jnp.array([True])
    val = jnp.ones((1, cfg.world.resource_channels))
    new_field = scatter_add_to_grid(world.resources, pos, val, alive, cfg.world.height, cfg.world.width)
    assert new_field.shape == world.resources.shape
    _, take = consume_from_grid(world.resources, pos, 0.5, alive, cfg.world.height, cfg.world.width)
    assert take.shape == (1,)


# ----- sixth sense ----------------------------------------------------------


def test_six_sense_shape_and_blind_ablation():
    cfg = _cfg()
    key = jax.random.PRNGKey(0)
    world = initialize_world(cfg.world, key)
    pop = initialize_embodied_population(cfg.embodied, cfg.world, key)
    sense = sample_sixth_sense(
        world, pop.positions, pop.alive, pop.messages, cfg.world, cfg.embodied, key
    )
    assert sense.shape == (pop.positions.shape[0], sixth_sense_dim(cfg.world))

    from hybrid_alife.types import EmbodiedConfig

    blind_kwargs = {f.name: getattr(cfg.embodied, f.name) for f in cfg.embodied.__dataclass_fields__.values()}
    blind_kwargs["blind"] = True
    blind_cfg = EmbodiedConfig(**blind_kwargs)
    blind = sample_sixth_sense(world, pop.positions, pop.alive, pop.messages, cfg.world, blind_cfg, key)
    assert bool(jnp.all(blind == 0.0))


# ----- embodied actions -----------------------------------------------------


def test_embodied_actions_all_paths():
    cfg = _cfg()
    key = jax.random.PRNGKey(0)
    world = initialize_world(cfg.world, key)
    pop = initialize_embodied_population(cfg.embodied, cfg.world, key)
    obs = observe_embodied(pop, world, cfg.embodied, cfg.world, key)
    assert obs.shape == (cfg.embodied.population_size, embodied_observation_dim(cfg.embodied, cfg.world))
    actions, pop = act_embodied(pop, obs, cfg.embodied)
    assert actions.shape == (cfg.embodied.population_size, embodied_action_dim(cfg.embodied))
    n = cfg.embodied.population_size
    forced = jnp.zeros((n, embodied_action_dim(cfg.embodied)))
    # Exercise every action class at least once.
    for i, j in enumerate([0, 1, 2, 3, 4, 5, 6, 7]):
        if i < n:
            forced = forced.at[i, j].set(1.0)
    pop, new_world, repro_gate = apply_embodied_actions(
        pop, forced, world, cfg.embodied, cfg.world, key
    )
    assert pop.positions.shape == (n, 2)
    assert pop.action_history.shape == (n, cfg.embodied.action_history_len)
    assert new_world.resources.shape == world.resources.shape


def test_reproduction_into_dead_slots():
    cfg = _cfg()
    key = jax.random.PRNGKey(1)
    pop = initialize_embodied_population(cfg.embodied, cfg.world, key)
    n = cfg.embodied.population_size
    half = n // 2
    pop.alive = pop.alive.at[half:].set(False)
    pop.energy = pop.energy.at[:half].set(cfg.embodied.reproduce_energy_threshold + 5)
    repro_gate = jnp.ones((n,))
    new_pop = apply_reproduction(pop, repro_gate, cfg.embodied, next_lineage_start=1000, key=key)
    assert int(new_pop.alive.sum()) >= half


# ----- Avida VM -------------------------------------------------------------


def test_avida_self_replication_eventually_makes_offspring():
    cfg = _cfg()
    key = jax.random.PRNGKey(2)
    world = initialize_world(cfg.world, key)
    pop = initialize_avida_population(cfg.avida, key)
    pop.alive = pop.alive.at[16:].set(False)
    parent_lineage_before = pop.lineage_id
    lc = 1000
    for _ in range(20):
        key, sub = jax.random.split(key)
        pop, lc = step_avida_population(pop, world, cfg.avida, lc, sub)
    diff = bool(jnp.any(pop.lineage_id != parent_lineage_before))
    assert diff


def test_avida_vm_cycle_runs_all_ops():
    cfg = _cfg()
    key = jax.random.PRNGKey(3)
    world = initialize_world(cfg.world, key)
    pop = initialize_avida_population(cfg.avida, key)
    sweep = jnp.arange(INSTRUCTION_COUNT)
    repeats = (cfg.avida.max_genome_length + INSTRUCTION_COUNT - 1) // INSTRUCTION_COUNT
    full = jnp.tile(sweep, (repeats,))[: cfg.avida.max_genome_length]
    pop.genomes = jnp.broadcast_to(full[None, :], pop.genomes.shape).astype(jnp.int32)
    pop.genome_lengths = jnp.full_like(pop.genome_lengths, cfg.avida.max_genome_length)
    for _ in range(INSTRUCTION_COUNT * 2):
        key, sub = jax.random.split(key)
        pop = vm_cycle(pop, world, cfg.avida, None, sub)
    assert pop.registers.shape == (cfg.avida.population_size, cfg.avida.registers)


# ----- evolution & archives -------------------------------------------------


def test_tournament_select_picks_high_fitness():
    fitness = jnp.array([0.0, 10.0, 1.0, 5.0])
    key = jax.random.PRNGKey(0)
    sel = tournament_select(fitness, 4, key)
    assert int(sel[0]) == 1


def test_select_and_mutate_embodied_runs():
    cfg = _cfg()
    key = jax.random.PRNGKey(4)
    pop = initialize_embodied_population(cfg.embodied, cfg.world, key)
    pop.alive = pop.alive.at[:4].set(False)
    pop = select_and_mutate_embodied(pop, pop.energy, cfg.embodied, cfg.evolution, key)
    assert bool(jnp.all(pop.alive))


def test_reseed_recovers_extinction():
    cfg = _cfg()
    key = jax.random.PRNGKey(5)
    pop = initialize_embodied_population(cfg.embodied, cfg.world, key)
    pop.alive = pop.alive.at[:].set(False)
    pop = reseed_embodied(pop, pop.energy, cfg.embodied, cfg.world, key)
    assert int(pop.alive.sum()) == cfg.embodied.population_size

    av = initialize_avida_population(cfg.avida, key)
    av.alive = av.alive.at[:].set(False)
    av = reseed_avida(av, 0.1, key)
    assert int(av.alive.sum()) == cfg.avida.population_size


def test_novelty_archive_returns_distances():
    arc = NoveltyArchive(capacity=8, k=2)
    arc.add_batch(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    nov = arc.novelty(np.array([[0.5, 0.5]], dtype=np.float32))
    assert nov.shape == (1,) and nov[0] > 0.0


def test_map_elites_archive_coverage():
    me = MapElitesArchive(bins=4)
    descriptors = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], dtype=np.float32)
    fitness = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    me.update(descriptors, fitness)
    assert me.coverage > 0


# ----- metrics --------------------------------------------------------------


def test_collect_full_metrics_keys():
    cfg = _cfg()
    state = initialize_sim(cfg)
    metrics = collect_full_metrics(state)
    for key in [
        "embodied_alive_frac",
        "avida_alive_frac",
        "action_entropy",
        "message_energy",
        "comm_usage_rate",
        "enrichment_separation",
        "coordinated_behavior_index",
    ]:
        assert key in metrics


# ----- jsonl, checkpoint, runner --------------------------------------------


def test_jsonl_roundtrip(tmp_path: Path):
    p = tmp_path / "m.jsonl"
    with JsonlWriter(p) as w:
        w.write({"a": 1, "b": 2.5})
        w.write({"a": 3, "b": 4.5})
    rows = read_jsonl(p)
    assert rows == [{"a": 1, "b": 2.5}, {"a": 3, "b": 4.5}]


def test_checkpoint_roundtrip(tmp_path: Path):
    cfg = _cfg()
    state = initialize_sim(cfg)
    ck = save_checkpoint(tmp_path / "ck.pkl", state, {"seed": 1})
    payload = load_checkpoint(ck)
    world, embodied, avida = restore_states(payload)
    assert world.flow.shape == state.world.flow.shape
    assert embodied is not None and avida is not None


def test_runner_end_to_end(tmp_path: Path):
    cfg = _cfg()
    # frozen dataclass -- bypass to redirect outputs
    object.__setattr__(cfg, "output_dir", str(tmp_path))
    state = run_experiment(cfg)
    assert state.generation == cfg.evolution.generations - 1
    metrics = read_jsonl(tmp_path / cfg.run_name / "metrics.jsonl")
    assert len(metrics) > 0


def test_step_sim_returns_pair():
    cfg = _cfg()
    state = initialize_sim(cfg)
    state2, lc = step_sim(state, cfg, 100)
    assert state2.step == 1
    assert lc >= 100
