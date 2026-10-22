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
    fitness_mode: str = "energy"
    fitness_directions: tuple[tuple[float, float], ...] = (
        (0.0, -1.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (-1.0, 0.0),
    )
    fitness_steps: int = 6


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


def _periodic_fertility(
    h: int, w: int, fraction: float, patch: int, seed: int
) -> np.ndarray:
    """Smooth fertility that wraps seamlessly at the toroidal seam."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = max(1, w // max(2 * patch, 2))
    field = np.zeros((h, w), dtype=np.float64)
    for harmonic in (1, 2, 3, 4, 5, 6):
        kx = base * harmonic
        ky = base * harmonic
        phase_x, phase_y = rng.uniform(0.0, 2.0 * np.pi, 2)
        amp_x, amp_y = rng.uniform(0.5, 1.0, 2)
        field += amp_x * np.sin(2.0 * np.pi * kx * xx / w + phase_x)
        field += amp_y * np.cos(2.0 * np.pi * ky * yy / h + phase_y)
    threshold = np.quantile(field, 1.0 - fraction)
    return (field > threshold).astype(np.float32)


def initialize_vivarium(cfg: VivariumConfig) -> VivariumState:
    state = initialize_sim(cfg.sim)
    h, w = cfg.sim.world.height, cfg.sim.world.width
    fertility = _periodic_fertility(
        h, w, cfg.fertility_fraction, cfg.fertility_patch, cfg.sim.seed
    )
    return VivariumState(
        sim=state,
        corpses=[],
        lineage_counter=int(cfg.sim.embodied.population_size),
        fertility=fertility,
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
        if cfg.fitness_mode == "min_alignment":
            from hybrid_alife.experiments.latent_law_embodied import (
                minimum_direction_alignment,
            )

            fitness = jnp.asarray(
                minimum_direction_alignment(
                    state.sim,
                    cfg.sim,
                    cfg.fitness_directions,
                    steps=cfg.fitness_steps,
                )
            )
        else:
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
    trails: dict[tuple[int, int], list[tuple[float, float]]] | None = None,
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
    grid = np.linspace(0.1, 0.9, 5)
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
                fill=(70, 84, 100),
                width=1,
            )

    positions = np.asarray(state.sim.embodied.positions)
    alive = np.asarray(state.sim.embodied.alive)
    energy = np.asarray(state.sim.embodied.energy)
    lineage = np.asarray(state.sim.embodied.lineage_id)
    last_action = np.asarray(state.sim.embodied.last_action)
    body_radius = np.asarray(
        state.sim.embodied.genomes.get("body_radius", np.ones(len(positions)))
    )
    body_aspect = np.asarray(
        state.sim.embodied.genomes.get("body_aspect", np.ones(len(positions)))
    )

    if trails is not None:
        for key in list(trails):
            index, trail_lineage = key
            if not alive[index] or int(lineage[index]) != trail_lineage:
                trails.pop(key, None)
                continue
            points = trails[key]
            if len(points) < 2:
                continue
            color = _lineage_color(trail_lineage)
            for a, b in zip(points[-10:], points[-9:], strict=False):
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
        radius = (3 + 7 * float(np.sqrt(max(energy[index] / initial, 0.0)))) * float(
            body_radius[index]
        )
        aspect = float(body_aspect[index])
        color = _lineage_color(int(lineage[index]))
        dx, dy = last_action[index, 0], last_action[index, 1]
        if abs(dx) + abs(dy) < 1e-6:
            dx, dy = 1.0, 0.0
        angle = float(np.arctan2(dy, dx))
        a = radius * 1.35 * aspect
        b = radius * 0.8
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                cx = (x + ox) * size
                cy = (y + oy) * size
                if not (-2 * a <= cx <= size + 2 * a):
                    continue
                if not (-2 * a <= cy <= size + 2 * a):
                    continue
                points = []
                for t in np.linspace(0.0, 2.0 * np.pi, 18):
                    ex = a * np.cos(t)
                    ey = b * np.sin(t)
                    points.append(
                        (
                            cx + ex * np.cos(angle) - ey * np.sin(angle),
                            cy + ex * np.sin(angle) + ey * np.cos(angle),
                        )
                    )
                draw.polygon(points, fill=color, outline=(235, 240, 245))
                eye_x = cx + 0.62 * a * np.cos(angle)
                eye_y = cy + 0.62 * a * np.sin(angle)
                draw.ellipse(
                    (eye_x - 2, eye_y - 2, eye_x + 2, eye_y + 2),
                    fill=(18, 20, 24),
                )
        if trails is not None:
            key = (int(index), int(lineage[index]))
            trails.setdefault(key, []).append((float(x), float(y)))
            trails[key] = trails[key][-12:]

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
        f"mean thrust alignment {alignment:+.2f}   corpses {len(state.corpses or [])}   "
        f"mean body {float(np.mean(body_radius[alive])) if population else 1.0:.2f}",
        fill=(180, 210, 235),
        font=small,
    )
    return canvas




def set_uniform_law(
    state: VivariumState, direction: tuple[float, float], magnitude: float = 0.2
) -> VivariumState:
    """Impose a uniform law over the whole world. (0, 0) means no law."""
    vector = np.asarray(direction, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-8:
        vector = vector / norm * magnitude
    else:
        vector = np.zeros(2, dtype=np.float32)
    flow = np.broadcast_to(
        vector, state.sim.world.flow.shape
    ).astype(np.float32)
    state.sim.world = replace(state.sim.world, flow=jnp.asarray(flow))
    return state


def reference_alignment(state: VivariumState, reference: tuple[float, float]) -> float:
    """Mean cosine of the commanded thrust with a reference direction."""
    alive = np.asarray(state.sim.embodied.alive)
    moves = np.asarray(state.sim.embodied.last_action[alive, 0:2])
    if moves.size == 0:
        return float("nan")
    norm = np.linalg.norm(moves, axis=-1)
    valid = norm > 1e-8
    if not valid.any():
        return float("nan")
    unit = np.asarray(reference, dtype=np.float64)
    unit = unit / max(float(np.linalg.norm(unit)), 1e-12)
    return float(np.mean((moves[valid] @ unit) / norm[valid]))


def mean_body_radius(state: VivariumState) -> float:
    alive = np.asarray(state.sim.embodied.alive)
    if not alive.any():
        return float("nan")
    genes = state.sim.embodied.genomes
    body = np.asarray(genes.get("body_radius", np.ones(alive.shape[0])))
    return float(body[alive].mean())


def encode_jpeg(frame, quality: int = 85) -> bytes:
    """Encode a Pillow frame as JPEG bytes for a live stream."""
    import io

    buffer = io.BytesIO()
    frame.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def live_page(title: str = "Vivarium") -> str:
    """A minimal page that displays the MJPEG stream."""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  html, body {{ margin: 0; background: #0b0f14; color: #dfe8f1;
    font-family: system-ui, sans-serif; }}
  .wrap {{ display: flex; flex-direction: column; align-items: center;
    justify-content: center; min-height: 100vh; gap: 12px; }}
  img {{ width: min(92vw, 92vh); height: auto; image-rendering: auto;
    border-radius: 6px; box-shadow: 0 0 40px #0008; }}
  .bar {{ display: flex; gap: 16px; align-items: center; font-size: 14px;
    color: #9fb2c6; }}
  .dot {{ width: 9px; height: 9px; border-radius: 50%; background: #4cc9f0;
    box-shadow: 0 0 10px #4cc9f0; }}
</style>
</head>
<body>
<div class="wrap">
  <img src="/stream.mjpg" alt="live vivarium">
  <div class="bar">
    <span class="dot"></span>
    <span>live</span>
    <span id="stats">loading</span>
  </div>
</div>
<script>
async function tick() {{
  try {{
    const r = await fetch('/stats.json', {{cache: 'no-store'}});
    const s = await r.json();
    document.getElementById('stats').textContent =
      `${{s.season}} · step ${{s.step}} · population ${{s.population}} · ` +
      `births ${{s.births}} · deaths ${{s.deaths}} · body ${{s.mean_body}}`;
  }} catch (e) {{}}
}}
setInterval(tick, 1000); tick();
</script>
</body>
</html>
"""


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
    trails: dict[tuple[int, int], list[tuple[float, float]]] = {}
    populations: list[int] = []
    alignments: list[float] = []
    poster = None
    for step in range(steps):
        state = vivarium_step(state, cfg)
        state.sim.step = step + 1
        frame = render_frame(state, cfg, size=size, trails=trails)
        if process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        process.stdin.write(frame.tobytes())
        populations.append(int(np.asarray(state.sim.embodied.alive).sum()))
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
