"""Render a GIF of evolved flow-head organisms following an imposed law.

Trains one seed of the accepted no-immigrant flow-head configuration, then rolls
the evolved genomes out on a held-out diagonal and renders positions, trails, the
imposed flow, and each organism's commanded thrust.

Usage:
    JAX_PLATFORMS=cpu python scripts/visualize_flow_head.py --seed 0
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter  # noqa: E402

from hybrid_alife.agents.embodied import (  # noqa: E402
    act_embodied,
    apply_embodied_actions,
    observe_embodied,
)
from hybrid_alife.experiments.latent_law_embodied import (  # noqa: E402
    _impose_flow,
    _local_flows,
    run_embodied_law_pilot,
)
from hybrid_alife.experiments.runner import initialize_sim  # noqa: E402

CARDINALS = ((0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0))
DIRECTIONS = {
    "diagonal": (0.70710678, 0.70710678),
    "anti_diagonal": (0.70710678, -0.70710678),
    "east": (1.0, 0.0),
    "north": (0.0, 1.0),
}
MAGNITUDE = 0.2


def train(seed: int, generations: int) -> tuple[dict, object]:
    print(f"training seed {seed} for {generations} generations ...", flush=True)
    pilot = run_embodied_law_pilot(
        seed=seed,
        directions=CARDINALS,
        generations=generations,
        population=16,
        steps_per_generation=8,
        size=14,
        proprioception=True,
        unit_flow_features=True,
        flow_head=True,
        rank_replacement=True,
        immigrant_fraction=0.0,
    )
    return pilot["state"].embodied.genomes, pilot["config"]


def rollout(config, genomes, direction, steps: int) -> list[dict]:
    eval_config = replace(
        config,
        embodied=replace(config.embodied, initial_alive_fraction=1.0),
    )
    state = initialize_sim(eval_config)
    state.embodied.genomes = {name: jnp.asarray(v) for name, v in genomes.items()}
    state.avida = None
    state = _impose_flow(state, direction, MAGNITUDE)
    key = jax.random.PRNGKey(0)
    frames: list[dict] = []
    for step in range(steps):
        obs = observe_embodied(
            state.embodied, state.world, eval_config.embodied, eval_config.world, key
        )
        actions, state.embodied = act_embodied(
            state.embodied, obs, eval_config.embodied, key
        )
        positions = np.asarray(state.embodied.positions)
        alive = np.asarray(state.embodied.alive)
        commanded = np.asarray(actions[:, 0:2])
        flow = _local_flows(state)
        move_norm = np.linalg.norm(commanded, axis=-1)
        flow_norm = np.linalg.norm(flow, axis=-1)
        valid = alive & (move_norm > 1e-8) & (flow_norm > 1e-8)
        alignment = np.where(
            valid,
            np.sum(commanded * flow, axis=-1) / np.maximum(move_norm * flow_norm, 1e-8),
            np.nan,
        )
        frames.append(
            {
                "positions": positions,
                "alive": alive,
                "commanded": commanded,
                "flow": flow,
                "alignment": alignment,
                "step": step,
            }
        )
        state.embodied, state.world, _ = apply_embodied_actions(
            state.embodied,
            actions,
            state.world,
            eval_config.embodied,
            eval_config.world,
            key,
        )
        state = _impose_flow(state, direction, MAGNITUDE)
    return frames


def _flow_arrows(ax, direction) -> None:
    unit = np.asarray(direction) / np.linalg.norm(direction)
    grid = np.linspace(0.08, 0.92, 6)
    xs, ys = np.meshgrid(grid, grid)
    ax.quiver(
        xs,
        ys,
        np.full_like(xs, unit[0]),
        np.full_like(ys, unit[1]),
        color="0.55",
        alpha=0.7,
        pivot="mid",
        scale=22,
        width=0.004,
        zorder=1,
    )


def _build_animation(frames: list[dict], direction):
    colors = plt.cm.tab20(np.linspace(0, 1, frames[0]["positions"].shape[0]))
    trails: list[list[np.ndarray]] = [
        [] for _ in range(frames[0]["positions"].shape[0])
    ]
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    fig.patch.set_facecolor("#0e1117")

    def update(index: int):
        frame = frames[index]
        ax.clear()
        ax.set_facecolor("#11161f")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        _flow_arrows(ax, direction)
        positions = frame["positions"]
        for agent in range(positions.shape[0]):
            trails[agent].append(positions[agent])
            trails[agent] = trails[agent][-14:]
        for agent, trail in enumerate(trails):
            if len(trail) > 1:
                arr = np.asarray(trail)
                ax.plot(arr[:, 0], arr[:, 1], color=colors[agent], lw=1.2, alpha=0.6)
        alive = frame["alive"]
        ax.scatter(
            positions[alive, 0],
            positions[alive, 1],
            c=colors[alive],
            s=55,
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
        ax.scatter(
            positions[~alive, 0],
            positions[~alive, 1],
            c="0.4",
            s=30,
            marker="x",
            zorder=4,
        )
        commanded = frame["commanded"]
        ax.quiver(
            positions[:, 0],
            positions[:, 1],
            commanded[:, 0],
            commanded[:, 1],
            color=colors,
            scale=28,
            width=0.005,
            zorder=5,
        )
        finite = frame["alignment"][np.isfinite(frame["alignment"])]
        mean = float(np.mean(finite)) if finite.size else float("nan")
        label = f"{mean:+.2f}" if np.isfinite(mean) else "n/a"
        ax.set_title(
            f"step {frame['step']:02d}   mean alignment {label}",
            color="white",
            fontsize=13,
        )
        return ()

    return fig, FuncAnimation(fig, update, frames=len(frames), interval=50)


def render_gif(frames: list[dict], direction, path: Path, fps: int = 20) -> Path:
    fig, animation = _build_animation(frames, direction)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def render_mp4(frames: list[dict], direction, path: Path, fps: int = 20) -> Path:
    fig, animation = _build_animation(frames, direction)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(
        path, writer=FFMpegWriter(fps=fps, codec="libx264", bitrate=2400)
    )
    plt.close(fig)
    return path


def render_summary(frames: list[dict], direction, path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    fig.patch.set_facecolor("#0e1117")
    for ax in axes.flat:
        ax.set_facecolor("#11161f")
        for spine in ax.spines.values():
            spine.set_color("0.5")
    final = frames[-1]
    ax = axes[0, 0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    _flow_arrows(ax, direction)
    alive = final["alive"]
    ax.scatter(
        final["positions"][alive, 0],
        final["positions"][alive, 1],
        c=plt.cm.tab20(np.linspace(0, 1, alive.shape[0]))[alive],
        s=55,
        edgecolor="white",
        zorder=4,
    )
    ax.quiver(
        final["positions"][:, 0],
        final["positions"][:, 1],
        final["commanded"][:, 0],
        final["commanded"][:, 1],
        color=plt.cm.tab20(np.linspace(0, 1, alive.shape[0])),
        scale=28,
        width=0.005,
        zorder=5,
    )
    ax.set_title("final step: positions and commanded thrust", color="white")

    ax = axes[0, 1]
    means = [np.nanmean(f["alignment"]) for f in frames]
    for agent in range(frames[0]["positions"].shape[0]):
        series = [f["alignment"][agent] for f in frames]
        ax.plot(series, color="0.55", lw=1, alpha=0.6)
    ax.plot(means, color="#4cc9f0", lw=2.4)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("step", color="white")
    ax.set_ylabel("alignment with the imposed law", color="white")
    ax.set_title("per-organism alignment (grey) and mean (blue)", color="white")
    ax.tick_params(colors="white")

    ax = axes[1, 0]
    unit = np.asarray(direction) / np.linalg.norm(direction)
    mean_cmd = np.nanmean(final["commanded"][alive], axis=0)
    mean_cmd = mean_cmd / max(np.linalg.norm(mean_cmd), 1e-8)
    ax.arrow(0, 0, unit[0], unit[1], color="#f72585", width=0.02, head_width=0.09)
    ax.arrow(0, 0, mean_cmd[0], mean_cmd[1], color="#4cc9f0", width=0.012, head_width=0.07)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.set_title("law direction (pink) vs mean thrust (blue)", color="white")
    ax.tick_params(colors="white")

    ax = axes[1, 1]
    values = final["alignment"][np.isfinite(final["alignment"])]
    ax.hist(values, bins=np.linspace(-1, 1, 21), color="#4cc9f0", edgecolor="white")
    ax.set_xlim(-1, 1)
    ax.set_xlabel("alignment", color="white")
    ax.set_ylabel("organisms", color="white")
    ax.set_title("final-step alignment distribution", color="white")
    ax.tick_params(colors="white")

    fig.suptitle("Evolved flow-head organisms following an imposed law", color="white", fontsize=15)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--format", choices=("gif", "mp4", "both"), default="both")
    parser.add_argument("--direction", choices=tuple(DIRECTIONS), default="diagonal")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/flow_head_viz"))
    args = parser.parse_args()

    genomes, config = train(args.seed, args.generations)
    direction = DIRECTIONS[args.direction]
    frames = rollout(config, genomes, direction, args.steps)
    written: list[Path] = []
    if args.format in ("gif", "both"):
        written.append(
            render_gif(frames, direction, args.out_dir / "flow_head.gif", args.fps)
        )
    if args.format in ("mp4", "both"):
        written.append(
            render_mp4(frames, direction, args.out_dir / "flow_head.mp4", args.fps)
        )
    written.append(
        render_summary(frames, direction, args.out_dir / "flow_head_summary.png")
    )
    means = [np.nanmean(f["alignment"]) for f in frames]
    print(f"direction {args.direction} {direction}")
    print(f"mean alignment: first {means[0]:+.2f}, final {means[-1]:+.2f}")
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
