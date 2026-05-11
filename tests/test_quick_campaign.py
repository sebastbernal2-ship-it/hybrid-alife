"""Smoke tests for `scripts/run_quick_campaign.py` (dry-run behavior only).

These tests do NOT execute any experiment cell; they verify that the launcher
plans the matrix, materializes per-cell configs, and writes a manifest in
dry-run mode.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_quick_campaign.py"


def _load_module():
    if "run_quick_campaign" in sys.modules:
        return sys.modules["run_quick_campaign"]
    spec = importlib.util.spec_from_file_location("run_quick_campaign", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_quick_campaign"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_minimal_campaign(tmp_path: Path) -> Path:
    campaign = {
        "name": "tiny",
        "description": "test-only campaign",
        "base_output_dir": str(tmp_path / "out"),
        "configs": ["configs/smoke200.yaml"],
        "seeds": [0, 1],
    }
    p = tmp_path / "campaign.yaml"
    p.write_text(yaml.safe_dump(campaign), encoding="utf-8")
    return p


def test_dry_run_plans_matrix_and_writes_manifest(tmp_path: Path) -> None:
    mod = _load_module()
    campaign_path = _write_minimal_campaign(tmp_path)

    rc = mod.main([
        "--campaign", str(campaign_path),
        "--dry-run",
        "--generations", "3",
    ])
    assert rc == 0

    out_dir = tmp_path / "out"
    manifest_json = out_dir / "manifest.json"
    manifest_md = out_dir / "manifest.md"
    assert manifest_json.exists(), "manifest.json should exist after dry-run"
    assert manifest_md.exists(), "manifest.md should exist after dry-run"

    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert manifest["invocation"]["dry_run"] is True
    assert manifest["invocation"]["generations"] == 3
    cells = manifest["cells"]
    assert len(cells) == 2, "1 config x 2 seeds = 2 cells"
    for cell in cells:
        assert cell["status"] == "dry-run"
        assert cell["return_code"] is None
        # The resolved per-cell config must be on disk and contain overrides.
        cfg_path = Path(cell["cell_config_path"])
        assert cfg_path.exists()
        resolved = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert resolved["seed"] == cell["seed"]
        assert resolved["run_name"] == cell["run_name"]
        assert resolved["evolution"]["generations"] == 3


def test_seeds_override_takes_precedence(tmp_path: Path) -> None:
    mod = _load_module()
    campaign_path = _write_minimal_campaign(tmp_path)
    rc = mod.main([
        "--campaign", str(campaign_path),
        "--dry-run",
        "--seeds", "11", "22", "33",
    ])
    assert rc == 0
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    seeds = sorted({c["seed"] for c in manifest["cells"]})
    assert seeds == [11, 22, 33]


def test_max_runs_marks_remaining_cells_skipped(tmp_path: Path) -> None:
    # Verify cap accounting via the planner + the executor's skip path, but
    # avoid actually running cells by using a no-op python interpreter that
    # exits immediately.
    mod = _load_module()
    campaign_path = _write_minimal_campaign(tmp_path)
    # `python -c "pass"` style: we point --python at a real interpreter and
    # use a run-sim shim that does nothing.
    shim = tmp_path / "noop_run_sim.py"
    shim.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")

    rc = mod.main([
        "--campaign", str(campaign_path),
        "--max-runs", "1",
        "--run-sim", str(shim),
    ])
    assert rc == 0
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    statuses = [c["status"] for c in manifest["cells"]]
    assert statuses.count("ok") == 1
    assert statuses.count("skipped") == 1
