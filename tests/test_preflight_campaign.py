"""Tests for scripts/preflight_campaign.py.

Covers the happy path against the real bundled campaign plus several
failure paths (missing file, malformed YAML, missing required fields,
unknown source config, unwritable output dir).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "preflight_campaign.py"


def _load_module():
    if "preflight_campaign" in sys.modules:
        return sys.modules["preflight_campaign"]
    spec = importlib.util.spec_from_file_location("preflight_campaign", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["preflight_campaign"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def preflight():
    return _load_module()


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_real_campaign(preflight):
    campaign = REPO_ROOT / "configs" / "campaigns" / "quick_20min.yaml"
    report = preflight.run_preflight(campaign)
    assert report.ok, [c.to_dict() for c in report.checks if not c.ok]
    assert report.run_count == 3 * 2
    assert report.devices, "expected at least one JAX device"
    assert report.recommended_commands


def test_main_returns_zero_on_happy_path(preflight, capsys):
    rc = preflight.main(
        ["--campaign", "configs/campaigns/quick_20min.yaml", "--print-commands"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "RESULT: PASS" in out
    assert "Recommended commands:" in out


def test_main_json_emits_valid_json(preflight, capsys):
    rc = preflight.main(["--campaign", "configs/campaigns/quick_20min.yaml", "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert rc == 0
    assert payload["ok"] is True
    assert payload["run_count"] == 6
    assert any(c["name"] == "jax_devices" for c in payload["checks"])


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_missing_campaign_file(preflight, tmp_path):
    missing = tmp_path / "no_such.yaml"
    report = preflight.run_preflight(missing)
    assert not report.ok
    failed = [c.name for c in report.checks if not c.ok]
    assert "campaign_file_exists" in failed


def test_malformed_yaml(preflight, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: oops\n  bad-indent: [unclosed")
    report = preflight.run_preflight(bad)
    assert not report.ok
    assert any(
        not c.ok and c.name == "campaign_file_parses" for c in report.checks
    )


def test_missing_required_fields(preflight, tmp_path):
    cfg = tmp_path / "no_fields.yaml"
    _write_yaml(cfg, {"name": "only_name"})
    report = preflight.run_preflight(cfg)
    assert not report.ok
    failed = [c.name for c in report.checks if not c.ok]
    assert "campaign_required_fields" in failed


def test_empty_configs_list(preflight, tmp_path):
    cfg = tmp_path / "empty.yaml"
    _write_yaml(cfg, {"name": "x", "configs": [], "seeds": [0]})
    report = preflight.run_preflight(cfg)
    assert not report.ok
    assert any(
        c.name == "campaign_configs_nonempty" and not c.ok for c in report.checks
    )


def test_unknown_source_config(preflight, tmp_path):
    cfg = tmp_path / "bad_src.yaml"
    _write_yaml(
        cfg,
        {
            "name": "x",
            "base_output_dir": str(tmp_path / "out"),
            "configs": ["configs/this_does_not_exist.yaml"],
            "seeds": [0],
        },
    )
    report = preflight.run_preflight(cfg)
    assert not report.ok
    src_check = next(c for c in report.checks if c.name == "source_configs_load")
    assert not src_check.ok
    assert "missing" in src_check.detail


def test_unwritable_output_dir(preflight, tmp_path, monkeypatch):
    if os.geteuid() == 0:
        pytest.skip("root can write anywhere; skipping unwritable-dir check")
    # Point base_output_dir at a path under a file (mkdir will fail).
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    cfg = tmp_path / "unwritable.yaml"
    _write_yaml(
        cfg,
        {
            "name": "x",
            "base_output_dir": str(blocker / "sub"),
            "configs": ["configs/smoke200.yaml"],
            "seeds": [0],
        },
    )
    report = preflight.run_preflight(cfg)
    out_check = next(c for c in report.checks if c.name == "output_dir_writable")
    assert not out_check.ok


def test_run_count_estimation(preflight, tmp_path):
    cfg = tmp_path / "matrix.yaml"
    _write_yaml(
        cfg,
        {
            "name": "x",
            "base_output_dir": str(tmp_path / "out"),
            "configs": ["configs/smoke200.yaml", "configs/ablation_no_comms.yaml"],
            "seeds": [0, 1, 2, 3],
        },
    )
    report = preflight.run_preflight(cfg)
    assert report.run_count == 8


def test_main_nonzero_on_failure(preflight, tmp_path, capsys):
    missing = tmp_path / "nope.yaml"
    rc = preflight.main(["--campaign", str(missing)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "RESULT: FAIL" in out
