#!/usr/bin/env python
"""Lightweight visualization for a hybrid-alife run directory.

Produces a static PNG gallery plus an `index.html` from whatever artifacts
exist in the run directory. Robust to missing files: any artifact that is
absent or fails to decode is skipped with a note in the report.

Inputs (all optional, any subset):
    - metrics.jsonl         time-series of QD/novelty/lineage/comm/enrichment
    - map_elites.npz        MAP-Elites archive (fitness, filled, [descriptors])
    - novelty_archive.npz   novelty archive (behaviors / descriptors)
    - checkpoint_final.pkl  optional, only used if --include-checkpoint set

Outputs (written under <run_dir>/viz/ by default):
    - timeseries_<group>.png  one panel per metric group
    - map_elites.png          heatmap when map_elites.npz is present
    - novelty_archive.png     scatter when novelty_archive.npz is present
    - index.html              gallery referencing all generated images

Example:
    python scripts/visualize_run.py outputs/runs/smoke200
    python scripts/visualize_run.py outputs/runs/smoke200 --out-dir /tmp/viz
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

METRIC_GROUPS: dict[str, list[str]] = {
    "qd": [
        "qd_score",
        "map_elites_coverage",
        "map_elites_qd_score",
        "archive_size",
    ],
    "novelty": [
        "novelty_archive_size",
        "mean_novelty",
        "novelty_threshold",
    ],
    "lineage": [
        "embodied_mean_lineage_depth",
        "embodied_max_lineage_depth",
        "embodied_lineage_hill1d",
        "avida_lineage_hill1d",
    ],
    "enrichment": [
        "mean_enrichment",
        "enrichment_separation",
        "mean_concentration",
        "mean_metabolite",
    ],
    "communication": [
        "action_entropy",
        "message_energy",
        "comm_usage_rate",
        "coordinated_behavior_index",
    ],
    "population": [
        "embodied_alive_frac",
        "avida_alive_frac",
        "mean_avida_merit",
        "avida_tasks_solved",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("run_dir", type=str, help="Run directory containing metrics/archives.")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: <run_dir>/viz).",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Title for the HTML report (default: run dir name).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=110,
        help="PNG DPI (default: 110).",
    )
    return p.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _x_axis(records: list[dict[str, Any]]) -> tuple[list[float], str]:
    if not records:
        return [], "index"
    if "step" in records[0]:
        return [r.get("step", i) for i, r in enumerate(records)], "step"
    if "generation" in records[0]:
        return [r.get("generation", i) for i, r in enumerate(records)], "generation"
    return list(range(len(records))), "index"


def _values(records: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for r in records:
        v = r.get(key)
        if v is None:
            out.append(float("nan"))
        else:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(float("nan"))
    return out


def _has_any_data(values: list[float]) -> bool:
    return any(not np.isnan(v) for v in values)


def plot_metric_group(
    records: list[dict[str, Any]],
    group: str,
    keys: list[str],
    out_path: Path,
    dpi: int,
) -> Path | None:
    """Plot a panel for one metric group. Returns path if any data was drawn."""
    xs, x_label = _x_axis(records)
    present: list[tuple[str, list[float]]] = []
    for k in keys:
        vals = _values(records, k)
        if _has_any_data(vals):
            present.append((k, vals))
    if not present:
        return None
    n = len(present)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 2.6 * rows), squeeze=False)
    for ax, (key, vals) in zip(axes.flat, present, strict=False):
        ax.plot(xs, vals, lw=1.4)
        ax.set_title(key, fontsize=10)
        ax.set_xlabel(x_label, fontsize=8)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(present):]:
        ax.set_visible(False)
    fig.suptitle(f"{group}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_map_elites(archive_path: Path, out_path: Path, dpi: int) -> Path | None:
    try:
        data = np.load(archive_path, allow_pickle=False)
    except Exception:
        return None
    if "fitness" not in data or "filled" not in data:
        return None
    fitness = np.asarray(data["fitness"], dtype=float)
    filled = np.asarray(data["filled"]).astype(bool)
    if fitness.ndim != 2 or filled.shape != fitness.shape:
        return None
    arr = np.where(filled, fitness, np.nan)
    coverage = float(filled.mean()) if filled.size else 0.0
    qd = float(np.where(filled, fitness, 0.0).sum())
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(arr, cmap="cividis", origin="lower", aspect="auto")
    ax.set_title(f"MAP-Elites — coverage {coverage:.2f}, QD {qd:.1f}")
    ax.set_xlabel("descriptor 1 bin")
    ax.set_ylabel("descriptor 0 bin")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fitness")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_novelty_archive(archive_path: Path, out_path: Path, dpi: int) -> Path | None:
    try:
        data = np.load(archive_path, allow_pickle=False)
    except Exception:
        return None
    candidates = ("behaviors", "descriptors", "points", "archive")
    arr: np.ndarray | None = None
    for key in candidates:
        if key in data:
            arr = np.asarray(data[key], dtype=float)
            break
    if arr is None or arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if arr.shape[1] == 1:
        ax.hist(arr[:, 0], bins=min(40, max(5, arr.shape[0] // 5)), color="steelblue")
        ax.set_xlabel("descriptor 0")
        ax.set_ylabel("count")
    else:
        d0, d1 = arr[:, 0], arr[:, 1]
        ax.scatter(d0, d1, s=12, alpha=0.6, c=np.arange(len(arr)), cmap="viridis")
        ax.set_xlabel("descriptor 0")
        ax.set_ylabel("descriptor 1")
    ax.set_title(f"Novelty archive — n={arr.shape[0]}, dim={arr.shape[1]}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def write_index_html(
    out_dir: Path,
    title: str,
    images: list[tuple[str, Path]],
    notes: list[str],
) -> Path:
    rel_imgs = [(label, img.name) for label, img in images]
    parts: list[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append(f"<title>{html.escape(title)}</title>")
    parts.append(
        "<style>"
        "body{font-family:system-ui,sans-serif;margin:24px;background:#fafafa;color:#222}"
        "h1{margin-bottom:0}"
        ".meta{color:#666;margin-bottom:16px;font-size:0.9em}"
        ".panel{background:white;border:1px solid #ddd;border-radius:6px;"
        "padding:12px;margin:12px 0;box-shadow:0 1px 2px rgba(0,0,0,0.04)}"
        ".panel h2{margin:0 0 8px 0;font-size:1.05em}"
        ".panel img{max-width:100%;height:auto;display:block}"
        ".notes{font-size:0.85em;color:#555}"
        ".notes li{margin:2px 0}"
        "</style></head><body>"
    )
    parts.append(f"<h1>{html.escape(title)}</h1>")
    parts.append(f"<div class='meta'>Generated by scripts/visualize_run.py</div>")
    if not rel_imgs:
        parts.append(
            "<div class='panel'><p>No visualizable artifacts were found in this run.</p></div>"
        )
    for label, fname in rel_imgs:
        parts.append("<div class='panel'>")
        parts.append(f"<h2>{html.escape(label)}</h2>")
        parts.append(f"<img src='{html.escape(fname)}' alt='{html.escape(label)}'>")
        parts.append("</div>")
    if notes:
        parts.append("<div class='panel'><h2>Notes</h2><ul class='notes'>")
        for n in notes:
            parts.append(f"<li>{html.escape(n)}</li>")
        parts.append("</ul></div>")
    parts.append("</body></html>")
    out_path = out_dir / "index.html"
    out_path.write_text("".join(parts), encoding="utf-8")
    return out_path


def visualize_run(run_dir: Path, out_dir: Path, title: str, dpi: int) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    images: list[tuple[str, Path]] = []
    notes: list[str] = []

    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        records = _read_jsonl(metrics_path)
        if records:
            for group, keys in METRIC_GROUPS.items():
                out_path = out_dir / f"timeseries_{group}.png"
                produced = plot_metric_group(records, group, keys, out_path, dpi)
                if produced is not None:
                    images.append((f"Time series — {group}", produced))
                else:
                    notes.append(f"No data for metric group '{group}' (keys: {', '.join(keys)}).")
        else:
            notes.append("metrics.jsonl is present but empty or malformed.")
    else:
        notes.append("metrics.jsonl not found — skipping time-series plots.")

    map_elites_path = run_dir / "map_elites.npz"
    if map_elites_path.exists():
        out_path = out_dir / "map_elites.png"
        produced = plot_map_elites(map_elites_path, out_path, dpi)
        if produced is not None:
            images.append(("MAP-Elites heatmap", produced))
        else:
            notes.append("map_elites.npz present but missing fitness/filled arrays.")
    else:
        notes.append("map_elites.npz not found — skipping MAP-Elites heatmap.")

    novelty_path = run_dir / "novelty_archive.npz"
    if novelty_path.exists():
        out_path = out_dir / "novelty_archive.png"
        produced = plot_novelty_archive(novelty_path, out_path, dpi)
        if produced is not None:
            images.append(("Novelty archive", produced))
        else:
            notes.append("novelty_archive.npz present but no usable descriptors.")
    else:
        notes.append("novelty_archive.npz not found — skipping novelty scatter.")

    index_path = write_index_html(out_dir, title, images, notes)
    return {"index": index_path, "images": [str(p) for _, p in images], "notes": notes}


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"run dir does not exist: {run_dir}")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "viz"
    title = args.title or f"Run visualization — {run_dir.name}"
    result = visualize_run(run_dir, out_dir, title, args.dpi)
    print(f"Wrote {len(result['images'])} image(s) to {out_dir}")
    print(f"Open: {result['index']}")
    for note in result["notes"]:
        print(f"  · {note}")


if __name__ == "__main__":
    main()
