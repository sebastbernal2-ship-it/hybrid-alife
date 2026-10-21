"""A small living-world layer over the embodied simulation.

Seasons modulate food regrowth. Dead organisms leave corpses that decay back into
food. The renderer draws food, hazards, flow, creatures, trails, and corpses.

This is the foundation for a Bibites-like world. The research question is layered
on later as an imposed environmental law, not the other way around.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_alife.evolution.selection import (
    reseed_embodied,
    select_and_mutate_embodied,
)
from hybrid_alife.experiments.runner import initialize_sim, step_sim

SEASONS = ("spring", "summer", "autumn", "winter")
SEASON_COLORS = {
    "spring": (18, 34, 22),
    "summer": (24, 34, 16),
    "autumn": (36, 24, 16),
    "winter": (18, 24, 32),
}


@dataclass
class VivariumConfig:
    sim: object
    season_length: int = 150
    season_regen: tuple[float, float, float, float] = (0.012, 0.018, 0.006, 0.002)
    corpse_energy: float = 2.0
    corpse_decay: float = 0.03
    max_corpses: int = 256
    generation_length: int = 200
    flow_magnitude: float = 0.2
    fertility_fraction: float = 0.35
    fertility_patch: int = 8


@dataclass
class VivariumState:
    sim: object
    season: int = 0
    season_step: int = 0
    corpses: list[tuple[float, float, float]] | None = None
    lineage_counter: int = 0
    births: int = 0
    deaths: int = 0
    fertility: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.corpses is None:
            self.corpses = []


def initialize_vivarium(cfg: VivariumConfig) -> VivariumState:
    state = initialize_sim(cfg.sim)
    h, w = cfg.sim.world.height, cfg.sim.world.width
    patch = max(2, cfg.fertility_patch)
    rng = np.random.default_rng(cfg.sim.seed)
    coarse = rng.random((max(2, h // patch + 1), max(2, w // patch + 1)))
    mask = coarse > (1.0 - cfg.fertility_fraction)
    fertility = np.repeat(np.repeat(mask, patch, axis=0), patch, axis=1)[:h, :w]
    return VivariumState(
        sim=state,
        corpses=[],
        lineage_counter=int(cfg.sim.embodied.population_size),
        fertility=fertility.astype(np.float32),
    )


def season_name(state: VivariumState) -> str:
    return SEASONS[state.season % len(SEASONS)]


def _step_config(cfg: VivariumConfig, season: int):
    regen = cfg.season_regen[season % len(cfg.season_regen)]
    return replace(cfg.sim, world=replace(cfg.sim.world, resource_regen=regen))


def _local_flow(state) -> np.ndarray:
    positions = np.asarray(state.sim.embodied.positions)
    flow = np.asarray(state.sim.world.flow)
    h, w = flow.shape[:2]
    xi = np.clip((positions[:, 0] * w).astype(int), 0, w - 1)
    yi = np.clip((positions[:, 1] * h).astype(int), 0, h - 1)
    return flow[yi, xi]


def vivarium_step(state: VivariumState, cfg: VivariumConfig) -> VivariumState:
    """Advance one step, then season, corpses, and selection."""
    state.season_step += 1
    if state.season_step >= cfg.season_length:
        state.season_step = 0
        state.season = (state.season + 1) % len(SEASONS)

    positions_before = np.asarray(state.sim.embodied.positions).copy()
    alive_before = np.asarray(state.sim.embodied.alive).copy()
    new_sim, lineage_counter = step_sim(
        state.sim, _step_config(cfg, state.season), state.lineage_counter
    )
    alive_after = np.asarray(new_sim.embodied.alive).copy()
    died = alive_before & ~alive_after
    born = ~alive_before & alive_after
    state.births += int(born.sum())
    state.deaths += int(died.sum())

    for index in np.flatnonzero(died):
        state.corpses.append(
            (
                float(positions_before[index, 0]),
                float(positions_before[index, 1]),
                cfg.corpse_energy,
            )
        )
    if len(state.corpses) > cfg.max_corpses:
        state.corpses = state.corpses[-cfg.max_corpses :]

    resources = new_sim.world.resources
    if state.fertility is not None:
        resources = resources * (
            0.02 + 0.98 * jnp.asarray(state.fertility)[..., None]
        )
    kept: list[tuple[float, float, float]] = []
    h, w = resources.shape[:2]
    for x, y, energy in state.corpses:
        release = energy * cfg.corpse_decay
        energy -= release
        xi = int(np.clip(x * w, 0, w - 1))
        yi = int(np.clip(y * h, 0, h - 1))
        resources = resources.at[yi, xi, 0].add(release)
        if energy > 0.02:
            kept.append((x, y, energy))
    state.corpses = kept
    new_sim.world = replace(new_sim.world, resources=jnp.clip(resources, 0.0, 1.0))

    state.sim = new_sim
    state.lineage_counter = lineage_counter

    if cfg.generation_length > 0 and (state.season_step + 1) % cfg.generation_length == 0:
        fitness = jnp.asarray(np.asarray(state.sim.embodied.energy))
        state.sim.rng, k_sel = jax.random.split(state.sim.rng)
        state.sim.embodied = select_and_mutate_embodied(
            state.sim.embodied,
            fitness,
            cfg.sim.embodied,
            cfg.sim.evolution,
            k_sel,
            world_cfg=cfg.sim.world,
        )
        state.sim.rng, k_res = jax.random.split(state.sim.rng)
        state.sim.embodied = reseed_embodied(
            state.sim.embodied,
            fitness,
            cfg.sim.embodied,
            cfg.sim.world,
            k_res,
        )
    return state


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def _lineage_color(lineage: int) -> tuple[int, int, int]:
    palette = np.asarray(
        [
            (235, 87, 87),
            (242, 201, 76),
            (111, 207, 151),
            (86, 204, 242),
            (187, 107, 217),
            (247, 143, 179),
            (255, 159, 28),
            (144, 190, 109),
            (131, 149, 255),
            (255, 122, 89),
        ],
        dtype=np.uint8,
    )
    return tuple(int(v) for v in palette[lineage % len(palette)])


def render_frame(
    state: VivariumState,
    cfg: VivariumConfig,
    *,
    size: int = 800,
    trails: dict[int, list[tuple[float, float]]] | None = None,
):
    """Draw one frame. Returns a Pillow RGB image."""
    from PIL import Image, ImageDraw

    season = season_name(state)
    resources = np.asarray(state.sim.world.resources[..., 0], dtype=np.float32)
    hazards = np.asarray(state.sim.world.hazards[..., 0], dtype=np.float32)
    h, w = resources.shape

    canvas = Image.new("RGB", (size, size), SEASON_COLORS[season])
    draw = ImageDraw.Draw(canvas)
    cell_w = size / w
    cell_h = size / h
    for yi, xi in np.argwhere(resources > 0.45):
        level = float(resources[yi, xi])
        radius = max(1.5, cell_w * (0.20 + 0.18 * level))
        cx = (xi + 0.5) * cell_w
        cy = (yi + 0.5) * cell_h
        color = (58, 132, 72) if level < 0.68 else (86, 196, 108)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
    if hazards.max() > 0:
        for yi, xi in np.argwhere(hazards > 0.45 * float(hazards.max())):
            radius = max(1.5, cell_w * 0.3)
            cx = (xi + 0.5) * cell_w
            cy = (yi + 0.5) * cell_h
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                fill=(150, 56, 56),
            )

    flow = np.asarray(state.sim.world.flow)
    grid = np.linspace(0.06, 0.94, 8)
    for gx in grid:
        for gy in grid:
            xi = int(gx * w)
            yi = int(gy * h)
            u, v = flow[yi, xi]
            draw.line(
                (
                    gx * size,
                    gy * size,
                    gx * size + u * size * 0.06,
                    gy * size + v * size * 0.06,
                ),
                fill=(96, 116, 136),
                width=2,
            )

    positions = np.asarray(state.sim.embodied.positions)
    alive = np.asarray(state.sim.embodied.alive)
    energy = np.asarray(state.sim.embodied.energy)
    lineage = np.asarray(state.sim.embodied.lineage_id)
    last_action = np.asarray(state.sim.embodied.last_action)

    if trails is not None:
        for index in list(trails):
            if not alive[index]:
                trails.pop(index, None)
                continue
            points = trails[index]
            if len(points) < 2:
                continue
            color = _lineage_color(int(lineage[index]))
            for a, b in zip(points[-14:], points[-13:], strict=False):
                if abs(a[0] - b[0]) > 0.2 or abs(a[1] - b[1]) > 0.2:
                    continue
                draw.line(
                    (a[0] * size, a[1] * size, b[0] * size, b[1] * size),
                    fill=tuple(int(0.55 * c) for c in color),
                    width=2,
                )

    for x, y, corpse_energy in state.corpses or []:
        radius = 2 + 4 * max(corpse_energy / cfg.corpse_energy, 0.0)
        draw.ellipse(
            (
                x * size - radius,
                y * size - radius,
                x * size + radius,
                y * size + radius,
            ),
            fill=(120, 110, 100),
            outline=(170, 160, 150),
            width=1,
        )

    initial = max(float(cfg.sim.embodied.initial_energy), 1e-6)
    for index in np.flatnonzero(alive):
        x, y = positions[index]
        radius = 3 + 7 * float(np.sqrt(max(energy[index] / initial, 0.0)))
        color = _lineage_color(int(lineage[index]))
        draw.ellipse(
            (x * size - radius, y * size - radius, x * size + radius, y * size + radius),
            fill=color,
            outline=(235, 240, 245),
            width=1,
        )
        dx, dy = last_action[index, 0], last_action[index, 1]
        draw.line(
            (
                x * size,
                y * size,
                x * size + dx * radius * 2.2,
                y * size + dy * radius * 2.2,
            ),
            fill=(240, 245, 250),
            width=2,
        )
        if trails is not None:
            trails.setdefault(int(index), []).append((float(x), float(y)))
            trails[int(index)] = trails[int(index)][-20:]

    font = _font(max(14, size // 32))
    small = _font(max(12, size // 44))
    population = int(alive.sum())
    mean_energy = float(energy[alive].mean()) if population else 0.0
    flow = _local_flow(state)[alive] if population else np.zeros((0, 2))
    moves = last_action[alive, 0:2]
    move_norm = np.linalg.norm(moves, axis=-1)
    flow_norm = np.linalg.norm(flow, axis=-1)
    valid = (move_norm > 1e-8) & (flow_norm > 1e-8)
    alignment = (
        float(
            np.mean(
                np.sum(moves[valid] * flow[valid], axis=-1)
                / (move_norm[valid] * flow_norm[valid])
            )
        )
        if valid.any()
        else 0.0
    )
    draw.rectangle((0, 0, size, size // 7), fill=(10, 14, 20))
    draw.text((18, 12), f"{season.upper()}   step {state.sim.step}", fill="white", font=font)
    draw.text(
        (18, 12 + size // 30),
        f"population {population}   energy {mean_energy:.1f}   "
        f"births {state.births}   deaths {state.deaths}",
        fill=(210, 220, 230),
        font=small,
    )
    draw.text(
        (18, 12 + 2 * size // 30),
        f"mean thrust alignment {alignment:+.2f}   corpses {len(state.corpses or [])}",
        fill=(180, 210, 235),
        font=small,
    )
    return canvas


def write_video(
    cfg: VivariumConfig,
    *,
    steps: int,
    size: int,
    fps: int,
    out_path: Path,
    warmup: int = 0,
    seed: int = 0,
    poster_path: Path | None = None,
) -> dict:
    """Run the world, stream frames to ffmpeg, and return a small summary."""
    import subprocess

    state = initialize_vivarium(cfg)
    for _ in range(warmup):
        state = vivarium_step(state, cfg)
    state.season = 0
    state.season_step = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{size}x{size}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        str(out_path),
    ]
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    trails: dict[int, list[tuple[float, float]]] = {}
    populations: list[int] = []
    alignments: list[float] = []
    poster = None
    for _ in range(steps):
        state = vivarium_step(state, cfg)
        frame = render_frame(state, cfg, size=size, trails=trails)
        if process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        process.stdin.write(frame.tobytes())
        populations.append(int(np.asarray(state.sim.embodied.alive).sum()))
        if poster is None or len(populations) == steps // 2:
            poster = frame
        alignments.append(0.0)
    process.stdin.close()
    process.wait()
    if poster_path is not None and poster is not None:
        poster_path.parent.mkdir(parents=True, exist_ok=True)
        poster.save(poster_path)
    return {
        "out_path": str(out_path),
        "steps": steps,
        "population_min": min(populations),
        "population_max": max(populations),
        "population_final": populations[-1],
        "births": state.births,
        "deaths": state.deaths,
    }
