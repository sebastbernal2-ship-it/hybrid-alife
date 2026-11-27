from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from hybrid_alife.agents.embodied import embodied_observation_dim
from hybrid_alife.ecosystem import ecosystem_step, initialize_ecosystem, load_ecosystem_config
from hybrid_alife.ecosystem_study import load_checkpoint, Registry, save_checkpoint
from hybrid_alife.experiments.runner import step_sim
from hybrid_alife.types import EmbodiedConfig
from hybrid_alife.world.compact_senses import COMPACT_SENSE_DIM, compact_observation

CONFIG = "configs/ecosystem.yaml"


def _small(seed=0):
    cfg, params = load_ecosystem_config(CONFIG)
    cfg = replace(cfg, seed=seed, embodied=replace(cfg.embodied, population_size=64))
    return cfg, replace(params, initial_food=400)


def test_ecosystem_knobs_default_to_legacy_behaviour():
    defaults = {f: EmbodiedConfig.__dataclass_fields__[f].default for f in (
        "compact_senses", "controller_init_scale", "move_scale",
        "senescence_age", "senescence_cost",
    )}
    assert defaults == {
        "compact_senses": False, "controller_init_scale": 1.0, "move_scale": 0.05,
        "senescence_age": 0, "senescence_cost": 0.0,
    }
    cfg, _ = _small()
    legacy = replace(cfg.embodied, compact_senses=False)
    assert embodied_observation_dim(cfg.embodied, cfg.world) == COMPACT_SENSE_DIM
    assert embodied_observation_dim(legacy, cfg.world) > 100


def test_compact_senses_point_at_food_and_blind_zeroes_the_outside_world():
    cfg, params = _small()
    eco = initialize_ecosystem(cfg, replace(params, initial_food=0))
    pop = eco.sim.embodied
    pop.positions = jnp.full_like(pop.positions, 0.5)  # cell (32, 32)
    food = jnp.zeros((64, 64, 1)).at[35, 32, 0].set(1.0)  # three rows further on
    world = replace(eco.sim.world, resources=food, hazards=jnp.zeros((64, 64, 1)))
    obs = np.asarray(compact_observation(pop, world, cfg.embodied, cfg.world))
    assert obs[0, 0] > 0.4 and abs(obs[0, 1]) < 0.2
    assert np.all(obs[:, 6:9] == 0.0)
    blind = replace(cfg.embodied, blind=True)
    assert np.all(np.asarray(compact_observation(pop, world, blind, cfg.world))[:, :9] == 0)


def test_compiled_step_matches_the_runner_step():
    cfg, params = _small()
    # No reproduction and no environment changes: compare the physics itself.
    eco = initialize_ecosystem(cfg, params)
    reference = initialize_ecosystem(cfg, params)
    ref_sim, _ = step_sim(reference.sim, cfg, 64, allow_reproduction=False)
    ecosystem_step(eco, allow_reproduction=False)
    for name in ("positions", "energy", "alive", "hidden"):
        np.testing.assert_allclose(
            np.asarray(getattr(eco.sim.embodied, name)),
            np.asarray(getattr(ref_sim.embodied, name)),
            atol=1e-5,
        )


def test_senescence_costs_old_bodies_energy():
    cfg, params = _small()
    energies = []
    for age in (0, 1500):
        eco = initialize_ecosystem(cfg, params)
        eco.sim.embodied.age = jnp.full((64,), age, jnp.int32)
        ecosystem_step(eco, allow_reproduction=False)
        energies.append(np.asarray(eco.sim.embodied.energy))
    alive = (energies[0] > 0) & (energies[1] > 0)
    expected = cfg.embodied.senescence_cost * (1500 - cfg.embodied.senescence_age)
    np.testing.assert_allclose((energies[0] - energies[1])[alive], expected, atol=1e-4)


def test_births_come_only_from_living_parents_and_events_account_for_everyone(tmp_path):
    cfg, params = _small(seed=3)
    cfg = replace(cfg, embodied=replace(cfg.embodied, reproduce_energy_threshold=20.0))
    eco = initialize_ecosystem(cfg, params)
    registry = Registry(eco)
    start = int(np.asarray(eco.sim.embodied.alive).sum())
    births = deaths = 0
    for _ in range(40):
        alive = np.asarray(eco.sim.embodied.alive).copy()
        events = ecosystem_step(eco)
        registry.update(eco, events)
        dead = {slot for slot, _ in events.deaths}
        for child, parent in events.births:
            assert alive[parent] and parent not in dead
            assert not alive[child] or child in dead
        births += len(events.births)
        deaths += len(events.deaths)
    now = np.asarray(eco.sim.embodied.alive)
    assert births > 0
    assert int(now.sum()) == start + births - deaths
    living = {i for i, row in registry.rows.items() if row[3] is None}
    assert living == set(np.asarray(eco.sim.embodied.lineage_id)[now].tolist())
    save_checkpoint(tmp_path / "c.pkl", eco, registry)
    restored, _ = load_checkpoint(tmp_path / "c.pkl")
    a, b = ecosystem_step(eco), ecosystem_step(restored)
    assert a.births == b.births and a.deaths == b.deaths
    np.testing.assert_array_equal(
        np.asarray(eco.sim.embodied.positions), np.asarray(restored.sim.embodied.positions)
    )
