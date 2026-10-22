from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

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


def test_d1_arm_config_carries_the_mechanisms() -> None:
    import argparse

    from run_vivarium_law import build_config

    args = argparse.Namespace(
        arm="d1",
        population=8,
        world=12,
        season_length=4,
    )
    config = build_config(0, args)
    assert config.sim.embodied.linear_policy is True
    assert config.sim.embodied.unit_flow_features is True
    assert config.sim.embodied.rank_replacement is True
    assert config.fitness_mode == "min_alignment"

    args.arm = "natural"
    natural = build_config(0, args)
    assert natural.sim.embodied.linear_policy is False
    assert natural.fitness_mode == "energy"


def test_vivarium_min_alignment_selection_runs() -> None:
    from dataclasses import replace

    from hybrid_alife.vivarium import vivarium_step

    base = embodied_law_config(
        seed=0,
        generations=1,
        population=8,
        steps_per_generation=1,
        size=12,
        proprioception=True,
    )
    sim = replace(
        base,
        embodied=replace(
            base.embodied,
            body_morphology=True,
            linear_policy=True,
            unit_flow_features=True,
            rank_replacement=True,
        ),
    )
    config = VivariumConfig(
        sim=sim,
        season_length=4,
        generation_length=2,
        fitness_mode="min_alignment",
        fitness_steps=2,
    )
    state = initialize_vivarium(config)
    for _ in range(4):
        state = vivarium_step(state, config)
    assert np.isfinite(np.asarray(state.sim.embodied.energy)).all()
