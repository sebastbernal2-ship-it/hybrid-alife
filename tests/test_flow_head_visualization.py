from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from visualize_flow_head import DIRECTIONS, render_gif, render_mp4, rollout  # noqa: E402

from hybrid_alife.experiments.latent_law_embodied import embodied_law_config  # noqa: E402
from hybrid_alife.experiments.runner import initialize_sim  # noqa: E402


def _frames():
    config = embodied_law_config(
        seed=0,
        generations=1,
        population=4,
        steps_per_generation=1,
        size=8,
        proprioception=True,
        unit_flow_features=True,
        flow_head=True,
    )
    state = initialize_sim(config)
    return rollout(config, state.embodied.genomes, DIRECTIONS["diagonal"], steps=3)


def test_rollout_records_the_expected_arrays() -> None:
    frames = _frames()
    assert len(frames) == 3
    assert frames[0]["positions"].shape == (4, 2)
    assert frames[0]["commanded"].shape == (4, 2)
    assert frames[0]["alignment"].shape == (4,)
    assert np.all(np.isfinite(frames[0]["positions"]))


def test_gif_renderer_writes_a_file(tmp_path) -> None:
    path = render_gif(_frames(), DIRECTIONS["diagonal"], tmp_path / "flow.gif", fps=4)
    assert path.exists()
    assert path.stat().st_size > 0


def test_mp4_renderer_writes_a_file_when_ffmpeg_exists(tmp_path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    path = render_mp4(_frames(), DIRECTIONS["diagonal"], tmp_path / "flow.mp4", fps=4)
    assert path.exists()
    assert path.stat().st_size > 0
