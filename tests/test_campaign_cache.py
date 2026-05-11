"""Tests for the --resume/--force/--workers cache features in run_quick_campaign.

Like the existing dry-run tests, these never invoke a real experiment cell.
They use a no-op `run_sim` shim and seed the output directory with a fake
`metrics.json` to simulate prior completion.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
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


def _campaign(tmp_path: Path) -> Path:
    cfg = {
        "name": "tiny",
        "description": "cache tests",
        "base_output_dir": str(tmp_path / "out"),
        "configs": ["configs/smoke200.yaml"],
        "seeds": [0, 1],
    }
    p = tmp_path / "campaign.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def _noop_shim(tmp_path: Path) -> Path:
    shim = tmp_path / "noop_run_sim.py"
    shim.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
    return shim


def test_hash_config_is_deterministic_and_order_insensitive():
    mod = _load_module()
    a = {"seed": 1, "evolution": {"generations": 3}, "run_name": "x"}
    b = {"run_name": "x", "evolution": {"generations": 3}, "seed": 1}
    assert mod.hash_config(a) == mod.hash_config(b)
    c = {"seed": 2, "evolution": {"generations": 3}, "run_name": "x"}
    assert mod.hash_config(a) != mod.hash_config(c)


def test_resume_skips_cells_with_matching_metrics_and_hash(tmp_path: Path) -> None:
    mod = _load_module()
    campaign_path = _campaign(tmp_path)
    shim = _noop_shim(tmp_path)

    # First run: populate metrics.json artifacts to simulate completion.
    rc = mod.main([
        "--campaign", str(campaign_path),
        "--run-sim", str(shim),
    ])
    assert rc == 0
    out = tmp_path / "out"
    manifest1 = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for cell in manifest1["cells"]:
        # The noop shim doesn't write metrics; fake it now to simulate
        # "previous campaign actually finished".
        (Path(cell["output_dir"]) / cell["run_name"] / "metrics.json").write_text(
            '{"final_fitness": 0.5}', encoding="utf-8",
        )

    # Second run with --resume: every cell should be cached.
    rc = mod.main([
        "--campaign", str(campaign_path),
        "--run-sim", str(shim),
        "--resume",
    ])
    assert rc == 0
    manifest2 = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    statuses = [c["status"] for c in manifest2["cells"]]
    assert statuses == ["cached", "cached"], statuses
    assert all(c["cache_hit"] for c in manifest2["cells"])
    assert manifest2["summary"]["n_cached"] == 2
    assert manifest2["invocation"]["resume"] is True
    # config_hash must be populated for cached cells too.
    assert all(c["config_hash"] for c in manifest2["cells"])


def test_resume_misses_when_metrics_absent(tmp_path: Path) -> None:
    mod = _load_module()
    campaign_path = _campaign(tmp_path)
    shim = _noop_shim(tmp_path)

    # No metrics on disk -> resume cannot hit; cells must run normally.
    rc = mod.main([
        "--campaign", str(campaign_path),
        "--run-sim", str(shim),
        "--resume",
    ])
    assert rc == 0
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    statuses = [c["status"] for c in manifest["cells"]]
    assert statuses.count("ok") == 2
    assert manifest["summary"]["n_cached"] == 0


def test_resume_misses_when_config_hash_differs(tmp_path: Path) -> None:
    mod = _load_module()
    campaign_path = _campaign(tmp_path)
    shim = _noop_shim(tmp_path)

    rc = mod.main([
        "--campaign", str(campaign_path),
        "--run-sim", str(shim),
        "--generations", "5",
    ])
    assert rc == 0
    out = tmp_path / "out"
    for cell in json.loads((out / "manifest.json").read_text(encoding="utf-8"))["cells"]:
        (Path(cell["output_dir"]) / cell["run_name"] / "metrics.json").write_text(
            '{"x": 1}', encoding="utf-8",
        )

    # Same campaign but with different generations override -> hash differs
    # -> resume must NOT hit.
    rc = mod.main([
        "--campaign", str(campaign_path),
        "--run-sim", str(shim),
        "--generations", "7",
        "--resume",
    ])
    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    statuses = [c["status"] for c in manifest["cells"]]
    assert statuses.count("ok") == 2
    assert manifest["summary"]["n_cached"] == 0


def test_force_and_resume_are_mutually_exclusive(tmp_path: Path) -> None:
    mod = _load_module()
    campaign_path = _campaign(tmp_path)
    with pytest.raises(SystemExit):
        mod.main([
            "--campaign", str(campaign_path),
            "--resume",
            "--force",
        ])


def test_manifest_includes_summary_and_hash_fields(tmp_path: Path) -> None:
    mod = _load_module()
    campaign_path = _campaign(tmp_path)
    rc = mod.main([
        "--campaign", str(campaign_path),
        "--dry-run",
    ])
    assert rc == 0
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert "summary" in manifest
    assert manifest["summary"]["n_cells"] == 2
    for cell in manifest["cells"]:
        assert cell["config_hash"], "config_hash must be set after materialization"


def test_workers_flag_is_recorded_in_manifest(tmp_path: Path) -> None:
    mod = _load_module()
    campaign_path = _campaign(tmp_path)
    shim = _noop_shim(tmp_path)
    rc = mod.main([
        "--campaign", str(campaign_path),
        "--run-sim", str(shim),
        "--workers", "2",
    ])
    assert rc == 0
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["invocation"]["workers"] == 2
    # All cells still completed.
    assert manifest["summary"]["n_ok"] == 2
