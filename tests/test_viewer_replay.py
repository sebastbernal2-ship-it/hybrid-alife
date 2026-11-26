from __future__ import annotations

import gzip
import struct
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from hybrid_alife.viewer.bundle import make_server, standalone
from hybrid_alife.viewer.export import record, viewer_config
from hybrid_alife.viewer.replay import read_replay, write_replay


def test_replay_round_trip_and_alignment(tmp_path):
    extremes = np.array([[-32768, 32767], [32767, -32768], [0, 1]], dtype=np.int16)
    arrays = {
        "delta": (extremes, "delta-shuffle"),
        "bytes": (np.array([0, 1, 255], dtype=np.uint8), "raw"),
        "ints": (np.array([[-1, 0], [2, 2**30]], dtype=np.int32), "raw"),
        "floats": (np.array([1.25, -2.5, 3.0], dtype=np.float32), "raw"),
    }
    path = write_replay(tmp_path / "roundtrip.hvr", {"label": "test"}, arrays)
    meta, result = read_replay(path)
    assert meta == {"label": "test"}
    for name, (expected, _) in arrays.items():
        np.testing.assert_array_equal(result[name], expected)


@pytest.mark.parametrize("array,codec,match", [
    (np.array([1], dtype=np.uint8), "mystery", "unsupported codec"),
    (np.array([1], dtype=np.int64), "raw", "unsupported dtype"),
    (np.array([1], dtype=np.float32), "delta-shuffle", "unsupported codec"),
])
def test_replay_rejects_unsupported_codec_or_dtype(tmp_path, array, codec, match):
    with pytest.raises(ValueError, match=match):
        write_replay(tmp_path / "bad.hvr", {}, {"bad": (array, codec)})


def test_replay_rejects_bad_magic(tmp_path):
    payload = b"NOPE" + struct.pack("<I", 2) + b"{}  "
    path = tmp_path / "bad-magic.hvr"
    path.write_bytes(gzip.compress(payload))
    with pytest.raises(ValueError, match="not an HVR1 replay"):
        read_replay(path)


def test_tiny_periodic_export_is_valid_and_deterministic():
    cfg = viewer_config("periodic", grid=32, population=6, seed=17)
    meta, arrays = record(cfg, "periodic", steps=4, field_every=2)
    assert arrays["velocity"][0].shape == (3, 32, 32, 2)
    assert arrays["signal"][0].shape == (3, 32, 32)
    for name, (values, _) in arrays.items():
        if name not in {"velocity", "signal", "solid"}:
            assert values.shape[0:2] == (5, 6), name
    np.testing.assert_array_equal(arrays["solid"][0], np.zeros((32, 32), np.uint8))
    alive = arrays["alive"][0].astype(bool)
    founder = arrays["founder"][0]
    assert np.all(founder[alive] >= 0)
    assert np.all(founder[alive] < 6)
    assert meta["flow_movement_scale"] == pytest.approx(
        cfg.world.lbm_peak_speed * cfg.world.lbm_substeps / cfg.world.width
    )

    meta2, arrays2 = record(cfg, "periodic", steps=4, field_every=2)
    assert meta == meta2
    assert arrays.keys() == arrays2.keys()
    for name in arrays:
        np.testing.assert_array_equal(arrays[name][0], arrays2[name][0])


@pytest.mark.parametrize("steps,field_every", [(0, 1), (3, 2), (4, 0)])
def test_record_rejects_invalid_steps(steps, field_every):
    cfg = viewer_config("periodic", grid=32, population=6, seed=17)
    with pytest.raises(ValueError, match="need steps"):
        record(cfg, "periodic", steps=steps, field_every=field_every)


def test_standalone_inlines_assets_and_server_is_restricted(tmp_path):
    replay = write_replay(tmp_path / "r.hvr", {}, {})
    html = standalone(replay, tmp_path / "v.html").read_text(encoding="utf-8")
    assert "window.REPLAY_B64" in html
    assert 'src="viewer.js"' not in html and 'href="style.css"' not in html
    with make_server(replay, 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            assert urllib.request.urlopen(f"{base}/replay.hvr").read() == replay.read_bytes()
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(f"{base}/../export.py")
        finally:
            server.shutdown()
