"""Matplotlib utilities for visualizing world fields, trajectories, and metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hybrid_alife.logging.jsonl import read_jsonl
from hybrid_alife.replay.checkpoint import load_checkpoint


def plot_world_fields(checkpoint_path: str | Path, out_path: str | Path) -> Path:
    payload = load_checkpoint(checkpoint_path)
    world = payload["world"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    panels = [
        ("flow_x", world["flow"][..., 0]),
        ("flow_y", world["flow"][..., 1]),
        ("curvature", world["curvature"][..., 0]),
        ("shear_grad_mag", np.linalg.norm(world["shear_grad"], axis=-1)),
        ("enrichment", world["enrichment"][..., 0]),
        ("lift_mag", np.linalg.norm(world["lift"], axis=-1)),
        ("concentration_0", world["concentration"][..., 0]),
        ("metabolite_0", world["metabolites"][..., 0]),
    ]
    for ax, (title, arr) in zip(axes.flat, panels, strict=True):
        im = ax.imshow(arr, cmap="viridis", origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"World fields @ gen {payload['generation']} step {payload['step']}")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_agent_positions(checkpoint_path: str | Path, out_path: str | Path) -> Path:
    payload = load_checkpoint(checkpoint_path)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(payload["world"]["enrichment"][..., 0], cmap="magma", origin="lower")
    if payload["embodied"] is not None:
        pos = payload["embodied"]["positions"]
        alive = payload["embodied"]["alive"].astype(bool)
        h, w = payload["world"]["enrichment"].shape[:2]
        ax.scatter(pos[alive, 1] * w, pos[alive, 0] * h, c="cyan", s=40, edgecolor="white")
    ax.set_title("Agent positions over enrichment")
    ax.set_xticks([])
    ax.set_yticks([])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_metrics(metrics_path: str | Path, out_path: str | Path) -> Path:
    records = read_jsonl(metrics_path)
    if not records:
        return Path(out_path)
    steps = [r["step"] for r in records]
    keys = [
        "embodied_alive_frac",
        "avida_alive_frac",
        "embodied_mean_lineage_depth",
        "action_entropy",
        "message_energy",
        "comm_usage_rate",
        "enrichment_separation",
        "coordinated_behavior_index",
        "mean_avida_merit",
    ]
    fig, axes = plt.subplots(3, 3, figsize=(13, 8))
    for ax, key in zip(axes.flat, keys, strict=True):
        values = [r.get(key, float("nan")) for r in records]
        ax.plot(steps, values, lw=1.3)
        ax.set_title(key)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_map_elites(archive_path: str | Path, out_path: str | Path) -> Path:
    data = np.load(archive_path)
    fitness = data["fitness"]
    filled = data["filled"]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    arr = np.where(filled, fitness, np.nan)
    im = ax.imshow(arr, cmap="cividis", origin="lower")
    ax.set_title(
        f"MAP-Elites coverage {filled.mean():.2f}, QD score {np.where(filled, fitness, 0.0).sum():.1f}"
    )
    ax.set_xlabel("eat-rate bin")
    ax.set_ylabel("speed bin")
    plt.colorbar(im, ax=ax)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
