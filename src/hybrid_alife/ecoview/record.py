"""Record a window of a real ecosystem run into one compact replay.

Frame ``t`` holds the state after step ``t``: positions and energy of everyone alive,
with the senses, hidden state and outputs the controller used in that step. A newborn
has not acted yet, so its senses, hidden state and outputs are zero in its first frame.
Events are the simulator's own (``ecosystem.StepEvents``).
"""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

import numpy as np

from hybrid_alife.ecosystem import DEATH_CAUSES, Ecosystem, ecosystem_step
from hybrid_alife.ecosystem_study import Registry
from hybrid_alife.physics.spheroid_drag import drag_factors
from hybrid_alife.world.compact_senses import COMPACT_SENSE_LABELS

WEB = Path(__file__).with_name("web")
MAGIC = b"ECO1"
OUTPUT_LABELS = ("move_row", "move_col", "eat", "attack", "reproduce")
# Fixed-point scales (stored integer = value * scale).
SCALE = {"pos": 65535.0, "energy": 5.0, "sense": 63.0, "hidden": 127.0, "output": 127.0,
         "intake": 80.0, "weight": 32.0}
_MAX_CLADES = 8


def _q(value, scale: float, dtype) -> np.ndarray:
    info = np.iinfo(dtype)
    return np.clip(np.rint(np.asarray(value) * scale), info.min, info.max).astype(dtype)


def _ancestor_born_by(rows: dict, ident: int, when: int) -> int:
    while rows[ident][2] > when and rows[ident][0] >= 0:
        ident = rows[ident][0]
    return ident


def assign_clades(rows: dict, living: list[int], start: int) -> tuple[int, dict[int, int]]:
    """Latest anchor step at which the window's first population has at most 8 ancestors."""

    def count(when: int) -> int:
        return len({_ancestor_born_by(rows, i, when) for i in living})

    low, high = 0, start
    while low < high:  # the number of distinct ancestors only grows with the anchor step
        mid = (low + high + 1) // 2
        if count(mid) <= _MAX_CLADES:
            low = mid
        else:
            high = mid - 1
    return low, {i: _ancestor_born_by(rows, i, low) for i in living}


class _Stream:
    def __init__(self) -> None:
        self.parts: dict[str, list[np.ndarray]] = {}

    def add(self, name: str, value: np.ndarray) -> None:
        self.parts.setdefault(name, []).append(value)

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.concatenate(v) for k, v in self.parts.items()}


def record(eco: Ecosystem, registry: Registry, steps: int, history: dict | None = None) -> bytes:
    cfg = eco.cfg
    start = eco.step
    stream = _Stream()
    counts: dict[str, list[int]] = {k: [] for k in ("agent", "food", "birth", "death", "attack")}
    window_ids: list[int] = []
    genes: dict[int, tuple] = {}

    def note_genes() -> None:
        emb = eco.sim.embodied
        ids = np.asarray(emb.lineage_id)
        alive = np.asarray(emb.alive)
        fresh = [i for i in np.flatnonzero(alive) if int(ids[i]) not in genes]
        if not fresh:
            return
        g = {k: np.asarray(v) for k, v in emb.genomes.items()}
        depth = np.asarray(emb.lineage_depth)
        for i in fresh:
            ident = int(ids[i])
            genes[ident] = (
                float(np.clip(g["body_radius"][i], 0.6, 1.8)),
                float(np.clip(g["body_aspect"][i], 0.7, 1.4)),
                int(depth[i]),
                g["w_xh"][i].copy(),
                g["w_act"][i][:, : len(OUTPUT_LABELS)].copy(),
            )
            window_ids.append(ident)

    def food_cells() -> np.ndarray:
        return _q(np.asarray(eco.sim.world.resources[..., 0]).ravel(), 255.0, np.uint8)

    food = food_cells()
    food_first = food.copy()
    hazards = [eco.hazard_pos.copy()]
    note_genes()
    emb = eco.sim.embodied
    first_living = [int(i) for i in np.asarray(emb.lineage_id)[np.asarray(emb.alive)]]
    previous_ids = np.asarray(emb.lineage_id).copy()

    def add_agents(obs, actions, intake, newborn_slots) -> None:
        emb = eco.sim.embodied
        live = np.flatnonzero(np.asarray(emb.alive))
        silent = np.isin(live, list(newborn_slots))[:, None]
        positions = np.asarray(emb.positions)[live]
        stream.add("agent_id", np.asarray(emb.lineage_id)[live].astype(np.int32))
        stream.add("agent_pos", _q(positions, SCALE["pos"], np.uint16).reshape(-1))
        stream.add("agent_energy", _q(np.asarray(emb.energy)[live], SCALE["energy"], np.uint8))
        stream.add("agent_age", np.asarray(emb.age)[live].astype(np.uint16))
        sense = np.where(silent, 0.0, obs[live])
        stream.add("agent_sense", _q(sense, SCALE["sense"], np.int8).reshape(-1))
        hidden = np.asarray(emb.hidden)[live]
        stream.add("agent_hidden", _q(hidden, SCALE["hidden"], np.int8).reshape(-1))
        out = np.where(silent, 0.0, actions[live, : len(OUTPUT_LABELS)])
        stream.add("agent_output", _q(out, SCALE["output"], np.int8).reshape(-1))
        eaten = np.where(silent[:, 0], 0.0, intake[live])
        stream.add("agent_intake", _q(eaten, SCALE["intake"], np.uint8))
        counts["agent"].append(len(live))

    n = cfg.embodied.population_size
    zeros = np.zeros((n, len(COMPACT_SENSE_LABELS)))
    add_agents(zeros, np.zeros((n, 8)), np.zeros(n), range(n))
    for name in ("food", "birth", "death", "attack"):
        counts[name].append(0)

    for _ in range(steps):
        before = np.asarray(eco.sim.embodied.positions).copy()
        events = ecosystem_step(eco)
        registry.update(eco, events)
        note_genes()
        ids = np.asarray(eco.sim.embodied.lineage_id).copy()
        add_agents(events.observation, events.actions, events.eaters,
                   {c for c, _ in events.births})
        now = food_cells()
        changed = np.flatnonzero(now != food)
        food = now
        stream.add("food_cell", changed.astype(np.uint16))
        stream.add("food_level", now[changed])
        counts["food"].append(len(changed))
        births = np.array([[ids[c], previous_ids[p]] for c, p in events.births], np.int32)
        stream.add("birth", births.reshape(-1).astype(np.int32))
        counts["birth"].append(len(events.births))
        dead = np.array(
            [[previous_ids[s], DEATH_CAUSES.index(c)] for s, c in events.deaths], np.int32
        )
        stream.add("death", dead.reshape(-1).astype(np.int32))
        where = _q(before[[s for s, _ in events.deaths]], SCALE["pos"], np.uint16)
        stream.add("death_pos", where.reshape(-1))
        counts["death"].append(len(events.deaths))
        hits = np.array(
            [[previous_ids[a], previous_ids[t]] for a, t in events.attacks], np.int32
        )
        stream.add("attack", hits.reshape(-1).astype(np.int32))
        counts["attack"].append(len(events.attacks))
        previous_ids = ids
        hazards.append(eco.hazard_pos.copy())
        if not np.asarray(eco.sim.embodied.alive).any():
            break

    rows = registry.rows
    anchor, first_clade = assign_clades(rows, first_living, start)
    clade_roots = sorted(set(first_clade.values()),
                         key=lambda c: -sum(v == c for v in first_clade.values()))
    wanted: set[int] = set()
    for ident in window_ids:
        while ident >= 0 and ident not in wanted:
            wanted.add(ident)
            ident = rows[ident][0]
    people = sorted(wanted)
    clade_of = {}
    for ident in window_ids:
        clade_of[ident] = clade_roots.index(_ancestor_born_by(rows, ident, anchor))
    table = np.array(
        [
            [i, rows[i][0], rows[i][2], -1 if rows[i][3] is None else rows[i][3],
             -1 if rows[i][4] is None else DEATH_CAUSES.index(rows[i][4]), rows[i][1]]
            for i in people
        ],
        np.int32,
    )
    # 1-based birth order over the whole run, founders first: the name shown on screen.
    order = {ident: n + 1 for n, ident in enumerate(sorted(rows))}
    arrays = stream.arrays()
    arrays["person"] = table.reshape(-1)
    arrays["person_serial"] = np.array([order[i] for i in people], np.int32)
    arrays["body_id"] = np.array(window_ids, np.int32)
    arrays["body_gene"] = np.array([genes[i][:2] for i in window_ids], np.float32).reshape(-1)
    arrays["body_depth"] = np.array([genes[i][2] for i in window_ids], np.int32)
    arrays["body_clade"] = np.array([clade_of[i] for i in window_ids], np.uint8)
    arrays["brain_in"] = _q([genes[i][3] for i in window_ids], SCALE["weight"], np.int8).reshape(-1)
    arrays["brain_out"] = _q([genes[i][4] for i in window_ids], SCALE["weight"],
                             np.int8).reshape(-1)
    arrays["food_first"] = food_first
    arrays["hazard_pos"] = _q(np.array(hazards), SCALE["pos"], np.uint16).reshape(-1)
    for name, values in counts.items():
        arrays[f"n_{name}"] = np.array(values, np.int32)

    p = eco.params
    along, across = drag_factors(np.array([0.7, 1.0, 1.4]))
    meta = {
        "seed": cfg.seed,
        "start_step": start,
        "frames": len(counts["agent"]),
        "grid": [cfg.world.height, cfg.world.width],
        "hidden": cfg.embodied.hidden_size,
        "senses": list(COMPACT_SENSE_LABELS),
        "outputs": list(OUTPUT_LABELS),
        "causes": list(DEATH_CAUSES),
        "scale": SCALE,
        "capacity": cfg.embodied.population_size,
        "max_energy": cfg.embodied.max_energy,
        "breed_energy": cfg.embodied.reproduce_energy_threshold,
        "senescence_age": cfg.embodied.senescence_age,
        "predation_radius": cfg.embodied.predation_radius,
        # radius where the blob exceeds the 0.2 cut-off in ecosystem._hazard_field
        "hazard_draw_radius": float(p.hazard_radius / 2.0 * np.sqrt(2.0 * np.log(5.0))),
        "hazard_patches": int(p.hazard_patches),
        "clade_anchor_step": int(anchor),
        "clade_roots": [int(c) for c in clade_roots],
        "drag_check": [np.asarray(along).tolist(), np.asarray(across).tolist()],
        "history": history,
    }
    return encode(meta, arrays)


def encode(meta: dict, arrays: dict[str, np.ndarray]) -> bytes:
    index, blobs, offset = {}, [], 0
    for name, array in arrays.items():
        data = np.ascontiguousarray(array).tobytes()
        index[name] = {"dtype": array.dtype.name, "offset": offset, "count": int(array.size)}
        pad = (-len(data)) % 4
        blobs.append(data + b"\0" * pad)
        offset += len(data) + pad
    header = json.dumps({"meta": meta, "arrays": index}, separators=(",", ":")).encode()
    header += b" " * ((-len(header)) % 4)
    body = MAGIC + len(header).to_bytes(4, "little") + header + b"".join(blobs)
    return gzip.compress(body, compresslevel=6, mtime=0)


def decode(blob: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    body = gzip.decompress(blob)
    if body[:4] != MAGIC:
        raise ValueError("not an ECO1 replay")
    size = int.from_bytes(body[4:8], "little")
    header = json.loads(body[8 : 8 + size])
    base = 8 + size
    arrays = {
        name: np.frombuffer(body, spec["dtype"], spec["count"], base + spec["offset"])
        for name, spec in header["arrays"].items()
    }
    return header["meta"], arrays


def standalone(replay: bytes, out: str | Path) -> Path:
    """One self-contained HTML file: no server, no network."""
    html, css, script = (
        (WEB / name).read_text(encoding="utf-8")
        for name in ("index.html", "style.css", "eco.js")
    )
    data = base64.b64encode(replay).decode("ascii")
    for marker, replacement in (
        ("<!--STYLE-->", f"<style>\n{css}</style>"),
        ("<!--SCRIPT-->",
         f'<script>window.REPLAY_B64="{data}";</script>\n<script>\n{script}</script>'),
    ):
        if html.count(marker) != 1:
            raise RuntimeError(f"index.html must contain exactly one {marker!r}")
        html = html.replace(marker, replacement)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
