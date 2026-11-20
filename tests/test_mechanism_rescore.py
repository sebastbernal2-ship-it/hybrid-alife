from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import measure_monopole_mechanism as mm  # noqa: E402

from hybrid_alife.experiments.latent_law_embodied import evaluate_frozen_alignment  # noqa: E402
from hybrid_alife.experiments.runner import initialize_sim  # noqa: E402

DIRECTION = mm.DIAGONALS[1]


def _setup(fluid: str):
    config = mm.law_config(0, mm.CONDITIONS[fluid])
    genomes = {
        k: np.asarray(v) for k, v in initialize_sim(config).embodied.genomes.items()
    }
    return config, genomes


def test_official_rescore_matches_the_study_score_and_shadows_do_not_perturb_the_rollout():
    config, genomes = _setup("E")
    official = evaluate_frozen_alignment(
        None, config, DIRECTION, steps=mm.EVAL_STEPS,
        genomes={k: jnp.asarray(v) for k, v in genomes.items()},
    )["mean_alignment"]
    with_shadows = mm.rollout(config, genomes, DIRECTION, shadows=True)
    without = mm.rollout(config, genomes, DIRECTION, shadows=False)
    np.testing.assert_allclose(mm.rescore(with_shadows)["official"], official, atol=1e-9)
    np.testing.assert_allclose(mm.rescore(without)["official"], official, atol=1e-9)


def test_shadow_fluids_superpose_to_the_full_fluid():
    config, genomes = _setup("E")
    recs = mm.rollout(config, genomes, DIRECTION, shadows=True)
    stats = mm.perturbation_stats(recs)
    assert stats["total_perturbation_mean"] > 1e-3  # the monopole really perturbs the flow
    assert stats["superposition_residual_max"] < 0.25
    assert stats["own_wake_mean"] > 0.0


def test_uncoupled_fluid_has_no_wake_and_variants_collapse():
    config, genomes = _setup("B")
    scores = mm.rescore(mm.rollout(config, genomes, DIRECTION, shadows=False))
    assert scores["official"] == scores["noself"]
    np.testing.assert_allclose(scores["official"], scores["law"], atol=1e-6)
    np.testing.assert_allclose(scores["law"], scores["selfonly"], atol=1e-6)


def test_rescore_variants_on_constructed_records():
    law = np.array([0.2, 0.0])
    thrust = np.array([[1.0, 0.0], [0.0, 1.0]])
    rec = mm.Record(
        thrust=thrust, alive=np.array([True, True]),
        flow=np.array([[0.2, 0.2], [0.2, 0.0]]),  # agent 0 pushed sideways by its own wake
        self_pert=np.array([[0.0, 0.2], [0.0, 0.0]]), others_sum=np.zeros((2, 2)), law=law,
        superposition_residual=0.0,
    )
    scores = mm.rescore([rec])
    s2 = 1 / np.sqrt(2)
    np.testing.assert_allclose(scores["official"], (s2 + 0.0) / 2)  # cos(45deg), cos(90deg)
    np.testing.assert_allclose(scores["law"], (1.0 + 0.0) / 2)
    np.testing.assert_allclose(scores["noself"], (1.0 + 0.0) / 2)  # flow - wake = law
    np.testing.assert_allclose(scores["selfonly"], (s2 + 0.0) / 2)
    np.testing.assert_allclose(scores["noself_sum"], (1.0 + 0.0) / 2)  # law + no others
