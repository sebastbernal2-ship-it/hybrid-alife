from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from serve_vivarium import Controls, FrameBroadcaster, nearest_agent  # noqa: E402

from hybrid_alife.vivarium import encode_jpeg, live_page  # noqa: E402


def test_encode_jpeg_returns_jpeg_bytes() -> None:
    from PIL import Image

    frame = Image.new("RGB", (32, 32), (10, 40, 20))
    data = encode_jpeg(frame)
    assert data.startswith(b"\xff\xd8")
    assert data.endswith(b"\xff\xd9")
    assert len(data) > 100


def test_live_page_references_the_stream_and_stats() -> None:
    page = live_page("Test Vivarium")
    assert "/stream.mjpg" in page
    assert "/stats.json" in page
    assert "/control" in page
    assert "/inspect" in page
    assert 'id="inspect"' in page
    assert 'id="toggle"' in page
    assert 'id="reset"' in page
    assert "Test Vivarium" in page


def test_broadcaster_returns_the_latest_frame() -> None:
    broadcaster = FrameBroadcaster()
    broadcaster.publish(b"frame-a", {"step": 1}, [{"index": 0}])
    serial, frame, stats, agents = broadcaster.latest(-1, timeout=0.1)
    assert serial == 1
    assert frame == b"frame-a"
    assert stats["step"] == 1
    assert agents == [{"index": 0}]


def test_controls_toggle_speed_and_reset() -> None:
    controls = Controls()
    assert controls.apply("toggle", None)["paused"] is True
    assert controls.apply("toggle", None)["paused"] is False
    assert controls.apply(None, "2")["speed"] == 2.0
    assert controls.apply(None, "99")["speed"] == 4.0
    controls.apply("reset", None)
    paused, speed, reset, selected, selected_lineage = controls.snapshot()
    assert reset is True
    assert paused is False
    assert selected is None
    assert selected_lineage is None
    assert controls.snapshot()[2] is False


def test_nearest_agent_wraps_the_seam() -> None:
    agents = [
        {"index": 0, "x": 0.1, "y": 0.1},
        {"index": 1, "x": 0.9, "y": 0.9},
    ]
    assert nearest_agent(agents, 0.95, 0.95)["index"] == 1
    assert nearest_agent(agents, 0.05, 0.05)["index"] == 0
    assert nearest_agent([], 0.5, 0.5) is None
