"""Tests for the synthetic communication benchmark.

These pin the directional behaviour of topsim/posdis/bosdis/channel_capacity
against a controlled task with three reference protocols (compositional,
holistic, random) and two ablations (channel shuffle, channel zero).

If a refactor of :mod:`hybrid_alife.metrics.communication` breaks these
relationships, the benchmark will fail loudly long before downstream
analysis pipelines silently regress.
"""

from __future__ import annotations

import numpy as np
import pytest

from hybrid_alife.metrics.comm_benchmark import (
    PROTOCOLS,
    BenchmarkSpec,
    compositional_protocol,
    enumerate_meanings,
    holistic_protocol,
    random_protocol,
    run_benchmark,
    summarise,
)


@pytest.fixture(scope="module")
def small_spec() -> BenchmarkSpec:
    # 3^3 = 27 meanings * 3 replicates = 81 rows; cheap and stable.
    return BenchmarkSpec(n_attr=3, attr_vocab=3, msg_vocab=12, replicates=3, seed=0)


def test_enumerate_meanings_shape(small_spec):
    m = enumerate_meanings(small_spec)
    assert m.shape == (small_spec.attr_vocab ** small_spec.n_attr * small_spec.replicates,
                       small_spec.n_attr)


def test_compositional_protocol_is_lossless_without_noise(small_spec):
    meanings = enumerate_meanings(small_spec)
    msg = compositional_protocol(meanings, small_spec)
    # Decoding: every position j encodes attribute (j % n_attr) shifted by a
    # known offset, so (msg[:, j] - offset) % msg_vocab must recover that
    # attribute's values.
    for j in range(small_spec.effective_msg_len):
        attr = j % small_spec.n_attr
        offset = (attr * small_spec.attr_vocab) % small_spec.msg_vocab
        recovered = (msg[:, j] - offset) % small_spec.msg_vocab
        assert np.array_equal(recovered, meanings[:, attr])


def test_compositional_maximises_topsim_posdis_bosdis(small_spec):
    res = run_benchmark(small_spec, include_controls=False)
    comp = res["compositional"].metrics
    holi = res["holistic"].metrics
    rand = res["random"].metrics

    # Headline ordering: compositional dominates the structureless baselines
    # by a wide margin on every disentanglement metric.
    for key in ("topsim", "posdis", "bosdis"):
        assert comp[key] > holi[key] + 0.4, (key, comp[key], holi[key])
        assert comp[key] > rand[key] + 0.4, (key, comp[key], rand[key])

    # Compositional should be near the ceiling; tolerate small slack from
    # the topsim subsampler and the MI plug-in estimator.
    assert comp["topsim"] > 0.95
    assert comp["posdis"] > 0.9
    assert comp["bosdis"] > 0.9


def test_holistic_has_high_channel_capacity_but_low_disentanglement(small_spec):
    """Sanity check that channel_capacity and posdis/bosdis are not
    interchangeable — holistic codes still carry information about the
    referent but are not compositional. This is the key reason the memo
    asks for all three metrics rather than capacity alone.
    """
    res = run_benchmark(small_spec, include_controls=False)
    holi = res["holistic"].metrics
    assert holi["channel_capacity"] > 0.5  # 1:1 codebook -> high MI
    assert holi["posdis"] < 0.1
    assert holi["bosdis"] < 0.1


def test_shuffle_collapses_topsim_for_compositional(small_spec):
    res = run_benchmark(small_spec, include_controls=True)
    direct = res["compositional"].metrics
    shuffled = res["compositional"].controls["shuffle"]
    # Channel shuffle should obliterate the meaning<->message pairing.
    assert shuffled["topsim"] < direct["topsim"] - 0.6
    assert shuffled["posdis"] < 0.2
    assert shuffled["bosdis"] < 0.2
    assert shuffled["channel_capacity"] < direct["channel_capacity"]


def test_zero_channel_drives_capacity_and_disentanglement_to_floor(small_spec):
    res = run_benchmark(small_spec, include_controls=True)
    zero = res["compositional"].controls["zero"]
    assert zero["posdis"] == pytest.approx(0.0, abs=1e-9)
    assert zero["bosdis"] == pytest.approx(0.0, abs=1e-9)
    assert zero["channel_capacity"] == pytest.approx(0.0, abs=1e-6)


def test_noise_monotonically_degrades_topsim():
    """Adding symbol-flip noise should lower topsim, never raise it."""
    base = BenchmarkSpec(n_attr=3, attr_vocab=3, msg_vocab=12, replicates=4, seed=1)
    meanings = enumerate_meanings(base)
    clean = compositional_protocol(meanings, base)
    noisy = compositional_protocol(
        meanings,
        BenchmarkSpec(**{**base.__dict__, "noise": 0.4, "seed": 1}),
    )
    from hybrid_alife.metrics.communication import topsim as _topsim
    assert _topsim(meanings, clean) > _topsim(meanings, noisy)


def test_random_protocol_is_at_floor(small_spec):
    meanings = enumerate_meanings(small_spec)
    msg = random_protocol(meanings, small_spec)
    from hybrid_alife.metrics.communication import topsim as _topsim
    assert _topsim(meanings, msg) < 0.4
    # iid random messages should be at posdis/bosdis floor.
    res = run_benchmark(small_spec, protocols=["random"], include_controls=False)
    assert res["random"].metrics["posdis"] < 0.1
    assert res["random"].metrics["bosdis"] < 0.1


def test_protocols_registry_matches_callables():
    assert set(PROTOCOLS) == {"compositional", "holistic", "random"}
    assert PROTOCOLS["compositional"] is compositional_protocol
    assert PROTOCOLS["holistic"] is holistic_protocol
    assert PROTOCOLS["random"] is random_protocol


def test_summarise_is_flat_json_friendly(small_spec):
    res = run_benchmark(small_spec, include_controls=True)
    flat = summarise(res)
    assert set(flat) == {"compositional", "holistic", "random"}
    for proto, metrics in flat.items():
        assert "topsim" in metrics
        assert "shuffle.topsim" in metrics
        assert "zero.posdis" in metrics
        # all values must be JSON-serialisable floats
        for v in metrics.values():
            assert isinstance(v, float)


def test_unknown_protocol_raises(small_spec):
    with pytest.raises(ValueError, match="unknown protocol"):
        run_benchmark(small_spec, protocols=["does-not-exist"])
