import numpy as np

from hybrid_alife.experiments.latent_law_embodied import angular_error_deg, flow_estimate_errors


def test_angular_error_known_angles():
    a = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    b = np.array([[1.0, 0.0], [0.0, 3.0], [-1.0, 0.0], [0.0, -1.0]])
    np.testing.assert_allclose(angular_error_deg(a, b), [0.0, 90.0, 180.0, 180.0], atol=1e-6)


def test_angular_error_ignores_zero_vectors_and_scale():
    a = np.array([[0.0, 0.0], [1e-3, 0.0], [5.0, 5.0]])
    b = np.array([[1.0, 0.0], [10.0, 0.0], [0.0, 0.0]])
    out = angular_error_deg(a, b)
    assert np.isnan(out[0]) and np.isnan(out[2]) and abs(out[1]) < 1e-6


def _record(estimate, flow, alive=(True, True), law=(0.2, 0.0)):
    return {
        "estimate": np.asarray(estimate, float),
        "flow": np.asarray(flow, float),
        "alive": np.asarray(alive, bool),
        "law": np.asarray(law, float),
    }


def test_estimate_pairs_previous_estimate_with_next_flow():
    e0 = [[1.0, 0.0], [1.0, 0.0]]
    f1 = [[0.0, 1.0], [1.0, 0.0]]  # agent 0 sees flow turned 90 degrees, agent 1 unchanged
    records = [_record(e0, [[0.2, 0.0]] * 2), _record([[1.0, 0.0]] * 2, f1)]
    out = flow_estimate_errors(records)
    assert out["n"] == 2
    assert abs(out["angle_mean"] - 45.0) < 1e-6  # (90 + 0) / 2
    assert abs(out["frac_over_45"] - 0.5) < 1e-9


def test_dead_agents_and_law_deviation():
    law = (0.2, 0.0)
    flow = [[0.2, 0.0], [0.0, 0.2]]  # second agent: 90 degrees off, |flow-law| = 0.2828
    records = [_record([[1.0, 0.0]] * 2, flow, law=law),
               _record([[1.0, 0.0]] * 2, flow, alive=(True, False), law=law)]
    out = flow_estimate_errors(records)
    assert out["n"] == 1  # only the agent alive at both steps counts
    assert out["law_angle_mean"] == 0.0 and out["magnitude_dev_mean"] == 0.0
    both = flow_estimate_errors([_record([[1.0, 0.0]] * 2, flow, law=law)] * 2)
    assert abs(both["law_angle_mean"] - 45.0) < 1e-6
    assert abs(both["magnitude_dev_mean"] - (0.0 + np.hypot(0.2, 0.2) / 0.2) / 2) < 1e-9
