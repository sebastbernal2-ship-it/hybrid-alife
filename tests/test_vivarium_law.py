from __future__ import annotations

import numpy as np

from hybrid_alife.experiments.latent_law_embodied import embodied_law_config
from hybrid_alife.vivarium import (
    VivariumConfig,
    initialize_vivarium,
    reference_alignment,
    set_uniform_law,
)


def _state():
    sim = embodied_law_config(
        seed=0,
        generations=1,
        population=4,
        steps_per_generation=1,
        size=8,
        proprioception=True,
    )
    return initialize_vivarium(
        VivariumConfig(sim=sim, season_length=4, generation_length=0)
    )


def test_set_uniform_law_present_and_absent() -> None:
    state = _state()
    state = set_uniform_law(state, (1.0, 0.0))
    flow = np.asarray(state.sim.world.flow)
    assert np.allclose(flow[..., 0], 0.2, atol=1e-5)
    assert np.allclose(flow[..., 1], 0.0, atol=1e-5)
    state = set_uniform_law(state, (0.0, 0.0))
    assert np.allclose(np.asarray(state.sim.world.flow), 0.0)


def test_reference_alignment_signs() -> None:
    state = _state()
    state.sim.embodied.alive = np.ones((4,), dtype=bool)
    state.sim.embodied.last_action = state.sim.embodied.last_action.at[:, 0].set(1.0)
    state.sim.embodied.last_action = state.sim.embodied.last_action.at[:, 1].set(0.0)
    assert np.isclose(reference_alignment(state, (1.0, 0.0)), 1.0)
    assert np.isclose(reference_alignment(state, (-1.0, 0.0)), -1.0)
    assert np.isclose(reference_alignment(state, (0.0, 1.0)), 0.0)
