#!/usr/bin/env python
"""Preflight checks for hours-long campaign runs.

Verifies that a campaign is ready to launch without actually running any
simulation. Designed to fail fast on misconfigured campaigns so wasted hours
of compute can be avoided.

Checks performed:

  - Campaign YAML exists, parses, and has required fields.
  - Every referenced source config exists and parses via ``load_config``.
  - Base output directory is writable (or can be created and written to).
  - Required helper scripts (``run_sim.py``, ``run_quick_campaign.py``) exist.
  - JAX is importable and reports its devices.
  - Run count is estimated as ``len(configs) * len(seeds)``.
  - Optional ``--print-commands`` prints the exact dry-run + execute commands.

Exit code is 0 when all checks pass, 1 otherwise. The same information is
also emitted as JSON when ``--json`` is supplied, so the script is easy to
wire into CI or other automation.

Examples
--------
    python scripts/preflight_campaign.py \
        --campaign configs/campaigns/quick_20min.yaml

    python scripts/preflight_campaign.py \
        --campaign configs/campaigns/quick_20min.yaml --json --print-commands
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SCRIPTS = ("run_sim.py", "run_quick_campaign.py")
REQUIRED_CAMPAIGN_FIELDS = ("name", "configs", "seeds")


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class PreflightReport:
    campaign_path: str
    checks: list[CheckResult] = field(default_factory=list)
    run_count: int | None = None
    devices: list[str] = field(default_factory=list)
    recommended_commands: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_path": self.campaign_path,
            "ok": self.ok,
            "run_count": self.run_count,
            "devices": self.devices,
            "recommended_commands": self.recommended_commands,
            "checks": [c.to_dict() for c in self.checks],
        }


def _add(report: PreflightReport, name: str, ok: bool, detail: str = "") -> CheckResult:
    result = CheckResult(name=name, ok=ok, detail=detail)
    report.checks.append(result)
    return result


def check_campaign_file(path: Path, report: PreflightReport) -> dict[str, Any] | None:
    if not path.exists():
        _add(report, "campaign_file_exists", False, f"Not found: {path}")
        return None
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        _add(report, "campaign_file_parses", False, f"YAML error: {exc}")
        return None
    if not isinstance(data, dict):
        _add(report, "campaign_file_parses", False, "Top-level YAML is not a mapping")
        return None
    _add(report, "campaign_file_exists", True, str(path))

    missing = [f for f in REQUIRED_CAMPAIGN_FIELDS if f not in data]
    if missing:
        _add(report, "campaign_required_fields", False, f"Missing: {missing}")
        return None
    _add(report, "campaign_required_fields", True, "")

    if not isinstance(data.get("configs"), list) or not data["configs"]:
        _add(report, "campaign_configs_nonempty", False, "configs must be a non-empty list")
        return None
    if not isinstance(data.get("seeds"), list) or not data["seeds"]:
        _add(report, "campaign_seeds_nonempty", False, "seeds must be a non-empty list")
        return None
    _add(report, "campaign_configs_nonempty", True, f"{len(data['configs'])} configs")
    _add(report, "campaign_seeds_nonempty", True, f"{len(data['seeds'])} seeds")
    return data


def check_source_configs(configs: list[str], report: PreflightReport) -> None:
    try:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from hybrid_alife.experiments.runner import load_config
    except Exception as exc:
        _add(report, "load_config_importable", False, f"Could not import load_config: {exc}")
        return
    _add(report, "load_config_importable", True, "")

    bad: list[str] = []
    for cfg_path in configs:
        full = (REPO_ROOT / cfg_path).resolve()
        if not full.exists():
            bad.append(f"missing: {cfg_path}")
            continue
        try:
            load_config(full)
        except Exception as exc:
            bad.append(f"{cfg_path}: {exc}")
    if bad:
        _add(report, "source_configs_load", False, "; ".join(bad))
    else:
        _add(report, "source_configs_load", True, f"{len(configs)} configs OK")


def check_output_dir(base_output_dir: str | None, report: PreflightReport) -> None:
    if not base_output_dir:
        _add(report, "output_dir_specified", False, "base_output_dir missing")
        return
    out = (REPO_ROOT / base_output_dir).resolve()
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _add(report, "output_dir_writable", False, f"mkdir failed: {exc}")
        return
    try:
        with tempfile.NamedTemporaryFile(dir=out, delete=True):
            pass
    except OSError as exc:
        _add(report, "output_dir_writable", False, f"write probe failed: {exc}")
        return
    _add(report, "output_dir_writable", True, str(out))


def check_required_scripts(report: PreflightReport) -> None:
    missing = [s for s in REQUIRED_SCRIPTS if not (REPO_ROOT / "scripts" / s).exists()]
    if missing:
        _add(report, "required_scripts_present", False, f"Missing: {missing}")
    else:
        _add(report, "required_scripts_present", True, ", ".join(REQUIRED_SCRIPTS))


def check_jax(report: PreflightReport) -> None:
    try:
        import jax  # type: ignore
    except Exception as exc:
        _add(report, "jax_importable", False, f"import failed: {exc}")
        return
    _add(report, "jax_importable", True, getattr(jax, "__version__", "?"))
    try:
        devices = [str(d) for d in jax.devices()]
    except Exception as exc:
        _add(report, "jax_devices", False, f"jax.devices() failed: {exc}")
        return
    if not devices:
        _add(report, "jax_devices", False, "no devices reported")
        return
    report.devices = devices
    _add(report, "jax_devices", True, ", ".join(devices))


def estimate_run_count(campaign: dict[str, Any]) -> int:
    return len(campaign.get("configs", [])) * len(campaign.get("seeds", []))


def recommended_commands(campaign_path: Path) -> list[str]:
    if campaign_path.is_absolute():
        try:
            rel = campaign_path.relative_to(REPO_ROOT)
        except ValueError:
            rel = campaign_path
    else:
        rel = campaign_path
    return [
        f"python scripts/run_quick_campaign.py --campaign {rel} --dry-run",
        f"python scripts/run_quick_campaign.py --campaign {rel}",
    ]


def run_preflight(campaign_path: Path) -> PreflightReport:
    report = PreflightReport(campaign_path=str(campaign_path))
    data = check_campaign_file(campaign_path, report)
    check_required_scripts(report)
    check_jax(report)
    if data is None:
        return report
    check_source_configs(list(data["configs"]), report)
    check_output_dir(data.get("base_output_dir"), report)
    report.run_count = estimate_run_count(data)
    report.recommended_commands = recommended_commands(campaign_path)
    return report


def format_text(report: PreflightReport, print_commands: bool) -> str:
    lines = [f"Preflight for {report.campaign_path}"]
    for c in report.checks:
        marker = "OK  " if c.ok else "FAIL"
        suffix = f" -- {c.detail}" if c.detail else ""
        lines.append(f"  [{marker}] {c.name}{suffix}")
    if report.run_count is not None:
        lines.append(f"  estimated run count: {report.run_count}")
    if report.devices:
        lines.append(f"  jax devices: {', '.join(report.devices)}")
    if print_commands and report.recommended_commands:
        lines.append("Recommended commands:")
        for cmd in report.recommended_commands:
            lines.append(f"  $ {cmd}")
    lines.append("RESULT: " + ("PASS" if report.ok else "FAIL"))
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--campaign",
        default="configs/campaigns/quick_20min.yaml",
        help="Path to a campaign YAML (relative paths resolve against repo root).",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON instead of human text.")
    p.add_argument(
        "--print-commands",
        action="store_true",
        help="Include recommended commands in the human output.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = Path(args.campaign)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    report = run_preflight(path)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_text(report, print_commands=args.print_commands))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
