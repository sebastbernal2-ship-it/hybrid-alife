from __future__ import annotations

import gzip
import struct
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from hybrid_alife.physics import ib
from hybrid_alife.viewer.bundle import make_server, standalone
from hybrid_alife.viewer.export import SWIMMERS, export
from hybrid_alife.viewer.overlap import OverlapStats
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


def test_short_scene_export_matches_the_viewer_layout(tmp_path):
    meta, arrays = read_replay(export(tmp_path / "scene.hvr", beats=1, progress=lambda _: None))
    frames = meta["frames"]
    assert frames == meta["beat_steps"] // meta["stride"] + 1
    total = sum(b["count"] for b in meta["bodies"])
    assert arrays["points"].shape == (frames, total, 2)
    assert arrays["velocity"].shape == (frames, meta["ny"] // 2, meta["nx"] // 2, 2)
    # the first body starts at its configured hub, in lattice cells
    hub = arrays["points"][0, 0] / meta["point_scale"]
    np.testing.assert_allclose(hub, SWIMMERS[0][:2], atol=2)
    assert np.any(arrays["velocity"][-1])
    for body in meta["bodies"]:
        assert body["ring"] % 2 == 0 and 1 + body["ring"] < body["count"]
    assert meta["stats"]["frames"] == frames
    assert meta["stats"]["drawn_body_overlap_frames"] == 0
    assert meta["stats"]["drawn_surface_penetration_frames"] == 0


def test_overlap_stats_detect_drawn_overlap_crossing_and_wall_penetration():
    def stats_for(second_x, second_y, heading=0.0):
        bodies = [ib.swimmer((40.0, 20.0)), ib.swimmer((second_x, second_y), heading)]
        structure = ib.Structure.from_bodies(
            bodies, planes=[(0.0, 1.0, 0.5)], wall_range=1.0, wall_k=0.1
        )
        stats = OverlapStats(structure, np.zeros((96, 128), np.uint8))
        stats.frame(structure.points)
        return stats.summary()

    apart = stats_for(40.0, 60.0)
    assert apart["drawn_body_overlap_frames"] == apart["outline_crossing_frames"] == 0
    assert apart["drawn_surface_penetration_frames"] == 0
    crossed = stats_for(25.0, 20.5, np.pi / 2)  # head of one across the tail of the other
    assert crossed["drawn_body_overlap_frames"] == crossed["outline_crossing_frames"] == 1
    assert stats_for(40.0, 2.0)["drawn_surface_penetration_frames"] == 1


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
