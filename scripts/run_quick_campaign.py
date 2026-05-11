#!/usr/bin/env python
"""Quick experiment campaign launcher.

Orchestrates a (configs x seeds) matrix of single-run experiments without
manual command stitching. For each cell the launcher:

  1. Loads the cell's source YAML config (one of the existing
     `configs/*.yaml` files).
  2. Applies overrides (seed, run_name, output_dir, optional generations).
  3. Writes a fully resolved per-cell config to the cell's output dir.
  4. Invokes `scripts/run_sim.py --config <cell_config>` as a subprocess.

`--dry-run` plans the matrix and writes the manifest without executing any
cell. The manifest records both planned and executed cells (including
return codes, durations, and the exact subprocess commands).

Examples
--------
    # Plan only, no subprocess execution
    python scripts/run_quick_campaign.py \
        --campaign configs/campaigns/quick_20min.yaml --dry-run

    # Bound the matrix to the first 2 cells, override seeds and generations
    python scripts/run_quick_campaign.py \
        --campaign configs/campaigns/quick_20min.yaml \
        --max-runs 2 --seeds 0 1 --generations 5
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_SIM = REPO_ROOT / "scripts" / "run_sim.py"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    """A single (config, seed) cell in the campaign matrix."""

    index: int
    source_config: str
    seed: int
    run_name: str
    output_dir: str
    cell_config_path: str
    command: list[str]
    status: str = "planned"
    return_code: int | None = None
    duration_s: float | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    config_hash: str | None = None
    cache_hit: bool = False


@dataclass
class Campaign:
    name: str
    description: str
    base_output_dir: str
    configs: list[str]
    seeds: list[int]
    generations: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Argument parsing & campaign loading
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Launch a small experiment campaign over (configs x seeds).",
    )
    p.add_argument(
        "--campaign",
        default="configs/campaigns/quick_20min.yaml",
        help="Path to a campaign YAML (see configs/campaigns/quick_20min.yaml).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the matrix and write the manifest; do not execute any cell.",
    )
    p.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Hard cap on the number of cells executed (planned cells beyond "
        "this cap are still listed in the manifest as 'skipped').",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Override the campaign's seed list.",
    )
    p.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Override evolution.generations for every cell.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Override the campaign's base_output_dir.",
    )
    p.add_argument(
        "--run-sim",
        default=str(DEFAULT_RUN_SIM),
        help="Path to the single-run entrypoint (defaults to scripts/run_sim.py).",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for the subprocess invocations.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip cells whose output dir already contains a non-empty "
        "metrics.json/metrics.csv and whose resolved config hash matches.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-execute every cell, ignoring any prior outputs. "
        "Mutually exclusive with --resume.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of cells to execute concurrently (default 1). "
        "Use with care: most cells saturate JAX threads on their own.",
    )
    return p.parse_args(argv)


def load_campaign(path: str | Path) -> Campaign:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    missing = [k for k in ("configs",) if k not in raw]
    if missing:
        raise ValueError(f"Campaign {path} missing required keys: {missing}")
    return Campaign(
        name=raw.get("name", path.stem),
        description=raw.get("description", ""),
        base_output_dir=raw.get("base_output_dir", "outputs/quick_campaign"),
        configs=list(raw["configs"]),
        seeds=list(raw.get("seeds", [0])),
        generations=raw.get("generations"),
        extras={k: v for k, v in raw.items() if k not in {
            "name", "description", "base_output_dir", "configs", "seeds", "generations",
        }},
    )


# ---------------------------------------------------------------------------
# Cell planning
# ---------------------------------------------------------------------------


def _slug(value: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in value).strip("_")


def _resolve(path_like: str | Path) -> Path:
    """Resolve path-like to absolute, anchoring relative paths at the repo root."""
    p = Path(path_like)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _load_source_config(source_path: Path) -> dict[str, Any]:
    with source_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _override_config(
    raw: dict[str, Any],
    seed: int,
    run_name: str,
    output_dir: str,
    generations: int | None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(raw)
    cfg["seed"] = seed
    cfg["run_name"] = run_name
    cfg["output_dir"] = output_dir
    if generations is not None:
        cfg.setdefault("evolution", {})
        cfg["evolution"]["generations"] = generations
    return cfg


def hash_config(cfg: dict[str, Any]) -> str:
    """Deterministic SHA-256 over a resolved per-cell config."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _metrics_present(output_dir: Path, run_name: str) -> Path | None:
    """Return the metrics file proving a cell completed, or None."""
    cell_dir = output_dir / run_name
    for name in ("metrics.json", "metrics.csv"):
        p = cell_dir / name
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _cached_config_hash(cell_cfg_path: Path) -> str | None:
    """Hash of the on-disk resolved config, or None if unreadable."""
    if not cell_cfg_path.exists():
        return None
    try:
        with cell_cfg_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return hash_config(raw)
    except Exception:  # noqa: BLE001
        return None


def is_cache_hit(cell: Cell, expected_hash: str) -> bool:
    """A cell is cached when metrics exist AND the on-disk config hash matches."""
    base_out = Path(cell.output_dir)
    metrics = _metrics_present(base_out, cell.run_name)
    if metrics is None:
        return False
    on_disk = _cached_config_hash(Path(cell.cell_config_path))
    return on_disk == expected_hash


def plan_cells(
    campaign: Campaign,
    seeds: list[int],
    generations: int | None,
    base_output_dir: Path,
    python: str,
    run_sim: str,
) -> list[Cell]:
    cells: list[Cell] = []
    idx = 0
    for cfg_rel in campaign.configs:
        source_path = _resolve(cfg_rel)
        if not source_path.exists():
            raise FileNotFoundError(f"Source config not found: {cfg_rel} (resolved {source_path})")
        raw = _load_source_config(source_path)
        base_name = raw.get("run_name", source_path.stem)
        for seed in seeds:
            run_name = f"{_slug(base_name)}_s{seed}"
            cell_out = base_output_dir / run_name
            cell_cfg_path = cell_out / "cell_config.yaml"
            cells.append(
                Cell(
                    index=idx,
                    source_config=str(cfg_rel),
                    seed=int(seed),
                    run_name=run_name,
                    output_dir=str(base_output_dir),
                    cell_config_path=str(cell_cfg_path),
                    command=[python, run_sim, "--config", str(cell_cfg_path)],
                )
            )
            idx += 1
    return cells


def materialize_cell_config(
    cell: Cell,
    campaign: Campaign,
    generations: int | None,
) -> dict[str, Any]:
    """Write the per-cell resolved YAML config to disk and return the dict."""
    source_path = _resolve(cell.source_config)
    raw = _load_source_config(source_path)
    resolved = _override_config(
        raw,
        seed=cell.seed,
        run_name=cell.run_name,
        output_dir=cell.output_dir,
        generations=generations,
    )
    cell_cfg_path = Path(cell.cell_config_path)
    cell_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cell_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)
    cell.config_hash = hash_config(resolved)
    return resolved


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def execute_cell(cell: Cell) -> None:
    cell.started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    try:
        result = subprocess.run(  # noqa: S603 — command list is launcher-built
            cell.command,
            cwd=str(REPO_ROOT),
            check=False,
        )
        cell.return_code = int(result.returncode)
        cell.status = "ok" if result.returncode == 0 else "failed"
    except Exception as exc:  # noqa: BLE001 — surface any launch error in manifest
        cell.return_code = -1
        cell.status = "error"
        cell.error = repr(exc)
    finally:
        cell.duration_s = time.time() - t0
        cell.finished_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _git_sha() -> str | None:
    try:
        out = subprocess.run(  # noqa: S603,S607
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def build_manifest(
    args: argparse.Namespace,
    campaign: Campaign,
    cells: list[Cell],
    seeds: list[int],
    generations: int | None,
    base_output_dir: Path,
) -> dict[str, Any]:
    return {
        "campaign": {
            "name": campaign.name,
            "description": campaign.description,
            "source_path": str(_resolve(args.campaign)),
        },
        "invocation": {
            "argv": sys.argv,
            "dry_run": bool(args.dry_run),
            "max_runs": args.max_runs,
            "seeds": seeds,
            "generations": generations,
            "out_dir": str(base_output_dir),
            "python": args.python,
            "run_sim": args.run_sim,
            "resume": bool(getattr(args, "resume", False)),
            "force": bool(getattr(args, "force", False)),
            "workers": int(getattr(args, "workers", 1) or 1),
        },
        "summary": {
            "n_cells": len(cells),
            "n_cached": sum(1 for c in cells if c.cache_hit),
            "n_ok": sum(1 for c in cells if c.status == "ok"),
            "n_failed": sum(1 for c in cells if c.status in {"failed", "error"}),
            "n_skipped": sum(1 for c in cells if c.status == "skipped"),
            "total_duration_s": sum(c.duration_s or 0.0 for c in cells),
        },
        "environment": {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "cwd": str(REPO_ROOT),
            "git_sha": _git_sha(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "cells": [cell.__dict__ for cell in cells],
    }


def write_manifest(manifest: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "manifest.json"
    md_path = out_dir / "manifest.md"
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    md_path.write_text(_render_manifest_md(manifest), encoding="utf-8")
    return json_path, md_path


def _render_manifest_md(m: dict[str, Any]) -> str:
    lines: list[str] = []
    c = m["campaign"]
    inv = m["invocation"]
    env = m["environment"]
    lines.append(f"# Campaign: {c['name']}")
    lines.append("")
    if c["description"]:
        lines.append(c["description"].strip())
        lines.append("")
    lines.append(f"- Source: `{c['source_path']}`")
    lines.append(f"- Dry run: `{inv['dry_run']}`")
    lines.append(f"- Seeds: `{inv['seeds']}`")
    lines.append(f"- Generations override: `{inv['generations']}`")
    lines.append(f"- Max runs: `{inv['max_runs']}`")
    lines.append(f"- Out dir: `{inv['out_dir']}`")
    lines.append(f"- Git SHA: `{env['git_sha']}`")
    lines.append(f"- Created: `{env['created_at']}`")
    lines.append("")
    lines.append("## Cells")
    lines.append("")
    lines.append("| # | source_config | seed | run_name | status | rc | duration_s | cache | hash |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for cell in m["cells"]:
        h = (cell.get("config_hash") or "")[:8]
        lines.append(
            f"| {cell['index']} | `{cell['source_config']}` | {cell['seed']} | "
            f"`{cell['run_name']}` | {cell['status']} | {cell['return_code']} | "
            f"{cell['duration_s']} | {cell.get('cache_hit', False)} | `{h}` |"
        )
    lines.append("")
    lines.append("## Commands")
    lines.append("")
    for cell in m["cells"]:
        lines.append(f"- cell {cell['index']}: `{' '.join(cell['command'])}`")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.resume and args.force:
        raise SystemExit("--resume and --force are mutually exclusive")
    campaign = load_campaign(args.campaign)

    seeds = args.seeds if args.seeds is not None else campaign.seeds
    generations = args.generations if args.generations is not None else campaign.generations
    base_out = Path(args.out_dir) if args.out_dir else Path(campaign.base_output_dir)
    if not base_out.is_absolute():
        base_out = REPO_ROOT / base_out

    cells = plan_cells(
        campaign,
        seeds=seeds,
        generations=generations,
        base_output_dir=base_out,
        python=args.python,
        run_sim=str(_resolve(args.run_sim)),
    )

    # Materialize first so cache-hit detection can compare the on-disk hash
    # against the freshly planned hash. This also makes dry-run plans
    # trivially executable later.
    expected_hashes: dict[int, str] = {}
    for cell in cells:
        # Capture the hash that would result from a fresh write, BEFORE
        # overwriting any cached on-disk config (so we can compare).
        source_path = _resolve(cell.source_config)
        raw = _load_source_config(source_path)
        fresh = _override_config(
            raw,
            seed=cell.seed,
            run_name=cell.run_name,
            output_dir=cell.output_dir,
            generations=generations,
        )
        expected_hashes[cell.index] = hash_config(fresh)

        # Check cache BEFORE overwriting the on-disk cell_config.yaml, so a
        # genuine resume detects existing identical configs.
        if args.resume and not args.dry_run:
            if is_cache_hit(cell, expected_hashes[cell.index]):
                cell.status = "cached"
                cell.cache_hit = True
                cell.config_hash = expected_hashes[cell.index]
                cell.duration_s = 0.0
                continue

        materialize_cell_config(cell, campaign, generations)

    if args.dry_run:
        for cell in cells:
            if cell.status != "cached":
                cell.status = "dry-run"
    else:
        executable = [c for c in cells if c.status != "cached"]
        max_runs = args.max_runs if args.max_runs is not None else len(executable)
        to_run = executable[:max_runs]
        skipped_overflow = executable[max_runs:]
        for c in skipped_overflow:
            c.status = "skipped"

        if args.workers and args.workers > 1 and to_run:
            print(f"workers={args.workers} (parallel)")
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(execute_cell, to_run))
        else:
            for i, cell in enumerate(to_run):
                print(f"[{i + 1}/{len(to_run)}] {' '.join(cell.command)}")
                execute_cell(cell)

    cached_n = sum(1 for c in cells if c.cache_hit)
    if cached_n:
        print(f"cache: {cached_n}/{len(cells)} cells skipped (resume)")

    manifest = build_manifest(args, campaign, cells, seeds, generations, base_out)
    json_path, md_path = write_manifest(manifest, base_out)
    print(f"manifest: {json_path}")
    print(f"manifest: {md_path}")

    # Non-zero exit if any cell failed (after dry-run filter).
    failed = [c for c in cells if c.status in {"failed", "error"}]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
