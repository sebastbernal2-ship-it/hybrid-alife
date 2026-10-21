from __future__ import annotations

import numpy as np

from hybrid_alife.experiments.latent_law_embodied import embodied_law_config
from hybrid_alife.vivarium import (
    VivariumConfig,
    initialize_vivarium,
    render_frame,
    season_name,
    vivarium_step,
)


def _config() -> VivariumConfig:
    sim = embodied_law_config(
        seed=0,
        generations=1,
        population=8,
        steps_per_generation=1,
        size=12,
        proprioception=True,
    )
    return VivariumConfig(
        sim=sim, season_length=2, generation_length=0, fertility_patch=4
    )


def test_seasons_wrap() -> None:
    cfg = _config()
    state = initialize_vivarium(cfg)
    assert season_name(state) == "spring"
    state = vivarium_step(state, cfg)
    assert season_name(state) == "spring"
    state = vivarium_step(state, cfg)
    assert season_name(state) == "summer"
    state = vivarium_step(state, cfg)
    state = vivarium_step(state, cfg)
    assert season_name(state) == "autumn"


def test_corpse_decays_and_releases_nutrients() -> None:
    cfg = _config()
    state = initialize_vivarium(cfg)
    state.corpses = [(0.5, 0.5, 10.0)]
    state = vivarium_step(state, cfg)
    resources = np.asarray(state.sim.world.resources)
    assert resources[6, 6, 0] >= 0.29
    assert state.corpses
    assert state.corpses[0][2] < 10.0


def test_death_leaves_a_corpse() -> None:
    cfg = _config()
    state = initialize_vivarium(cfg)
    state.sim.embodied.energy = state.sim.embodied.energy.at[0].set(0.0)
    state.sim.embodied.alive = state.sim.embodied.alive.at[0].set(True)
    state = vivarium_step(state, cfg)
    assert state.deaths >= 1
    assert len(state.corpses) >= 1


def test_render_writes_a_frame(tmp_path) -> None:
    cfg = _config()
    state = initialize_vivarium(cfg)
    frame = render_frame(state, cfg, size=220)
    out = tmp_path / "frame.png"
    frame.save(out)
    assert out.exists()
    assert out.stat().st_size > 0
