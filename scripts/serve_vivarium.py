"""Serve the vivarium as a live MJPEG stream.

Open http://127.0.0.1:8765 in a browser to watch the world run.

Zero extra dependencies: Pillow renders, the standard library serves.

Usage:
    JAX_PLATFORMS=cpu python scripts/serve_vivarium.py
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_alife.experiments.latent_law_embodied import embodied_law_config  # noqa: E402
from hybrid_alife.vivarium import (  # noqa: E402
    VivariumConfig,
    encode_jpeg,
    initialize_vivarium,
    live_page,
    render_frame,
    season_name,
    vivarium_step,
)


@dataclass
class Controls:
    paused: bool = False
    speed: float = 1.0
    reset: bool = False
    selected: int | None = None
    selected_lineage: int | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def apply(self, action: str | None, speed: str | None) -> dict:
        with self._lock:
            if action == "toggle":
                self.paused = not self.paused
            elif action == "pause":
                self.paused = True
            elif action == "resume":
                self.paused = False
            elif action == "reset":
                self.reset = True
            if speed is not None:
                try:
                    self.speed = min(max(float(speed), 0.25), 4.0)
                except ValueError:
                    pass
            return {"paused": self.paused, "speed": self.speed}

    def snapshot(self) -> tuple[bool, float, bool, int | None, int | None]:
        with self._lock:
            reset = self.reset
            self.reset = False
            return self.paused, self.speed, reset, self.selected, self.selected_lineage


class FrameBroadcaster:
    """Holds the latest frame and lets stream handlers wait for a new one."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._serial = 0
        self._frame: bytes | None = None
        self._stats: dict = {}
        self._agents: list[dict] = []

    def publish(self, frame: bytes, stats: dict, agents: list[dict]) -> None:
        with self._condition:
            self._serial += 1
            self._frame = frame
            self._stats = stats
            self._agents = agents
            self._condition.notify_all()

    def latest(self, after_serial: int, timeout: float = 5.0):
        with self._condition:
            self._condition.wait_for(
                lambda: self._serial > after_serial, timeout=timeout
            )
            return self._serial, self._frame, dict(self._stats), list(self._agents)


def build_config(args) -> VivariumConfig:
    sim = embodied_law_config(
        args.seed,
        generations=1,
        population=args.population,
        steps_per_generation=1,
        size=args.world,
        proprioception=True,
        initial_alive_fraction=0.6,
    )
    sim = replace(
        sim,
        world=replace(sim.world, curvature_regime="straight"),
        embodied=replace(
            sim.embodied,
            body_morphology=True,
            max_energy=80.0,
            predation_rate=0.35,
            predation_radius=0.04,
            escape_gain=0.03,
            escape_radius=0.06,
        ),
    )
    return VivariumConfig(
        sim=sim, season_length=args.season_length, generation_length=200
    )


def _agents(state) -> list[dict]:
    alive = np.asarray(state.sim.embodied.alive)
    positions = np.asarray(state.sim.embodied.positions)
    lineage = np.asarray(state.sim.embodied.lineage_id)
    energy = np.asarray(state.sim.embodied.energy)
    age = np.asarray(state.sim.embodied.age)
    genes = state.sim.embodied.genomes
    body = np.asarray(genes.get("body_radius", np.ones(alive.shape[0])))
    aspect = np.asarray(genes.get("body_aspect", np.ones(alive.shape[0])))
    return [
        {
            "index": int(i),
            "x": float(positions[i, 0]),
            "y": float(positions[i, 1]),
            "lineage": int(lineage[i]),
            "energy": float(energy[i]),
            "age": int(age[i]),
            "body_radius": float(body[i]),
            "body_aspect": float(aspect[i]),
        }
        for i in np.flatnonzero(alive)
    ]


def nearest_agent(agents: list[dict], x: float, y: float) -> dict | None:
    best = None
    best_dist = float("inf")
    for agent in agents:
        dx = agent["x"] - x
        dy = agent["y"] - y
        dx -= round(dx)
        dy -= round(dy)
        dist = dx * dx + dy * dy
        if dist < best_dist:
            best_dist = dist
            best = agent
    return best


def _stats(state, cfg) -> dict:
    alive = np.asarray(state.sim.embodied.alive)
    genes = state.sim.embodied.genomes
    body = np.asarray(genes.get("body_radius", np.ones(alive.shape[0])))
    return {
        "season": season_name(state),
        "step": int(state.sim.step),
        "population": int(alive.sum()),
        "births": int(state.births),
        "deaths": int(state.deaths),
        "mean_body": round(float(body[alive].mean()) if alive.any() else 1.0, 2),
    }


def sim_loop(
    config,
    args,
    broadcaster: FrameBroadcaster,
    stop: threading.Event,
    controls: Controls,
) -> None:
    state = initialize_vivarium(config)
    for _ in range(args.warmup):
        state = vivarium_step(state, config)
    state.season = 0
    state.season_step = 0
    trails: dict[tuple[int, int], list[tuple[float, float]]] = {}
    step = 0
    while not stop.is_set():
        paused, speed, reset, selected, selected_lineage = controls.snapshot()
        if reset:
            state = initialize_vivarium(config)
            state.season = 0
            state.season_step = 0
            trails.clear()
            step = 0
            continue
        if paused:
            time.sleep(0.05)
            continue
        started = time.monotonic()
        state = vivarium_step(state, config)
        step += 1
        state.sim.step = step
        if selected is not None:
            lineage = np.asarray(state.sim.embodied.lineage_id)
            if selected >= len(lineage) or int(lineage[selected]) != selected_lineage:
                selected = None
        frame = render_frame(
            state, config, size=args.size, trails=trails, selected=selected
        )
        broadcaster.publish(
            encode_jpeg(frame, quality=args.quality),
            _stats(state, config),
            _agents(state),
        )
        interval = 1.0 / max(args.fps * speed, 1e-6)
        elapsed = time.monotonic() - started
        if elapsed < interval:
            time.sleep(interval - elapsed)


class Handler(BaseHTTPRequestHandler):
    broadcaster: FrameBroadcaster
    controls: Controls

    def log_message(self, format, *args) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = live_page().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/inspect"):
            query = parse_qs(urlparse(self.path).query)
            x = float(query.get("x", ["0.5"])[0])
            y = float(query.get("y", ["0.5"])[0])
            _, _, _, agents = self.broadcaster.latest(-1, timeout=0.05)
            agent = nearest_agent(agents, x, y)
            self.controls.selected = agent["index"] if agent else None
            self.controls.selected_lineage = agent["lineage"] if agent else None
            body = json.dumps(agent if agent else {"index": None}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/control"):
            query = parse_qs(urlparse(self.path).query)
            action = query.get("action", [None])[0]
            speed = query.get("speed", [None])[0]
            state = self.controls.apply(action, speed)
            body = json.dumps(state).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/stats.json":
            _, _, stats, _ = self.broadcaster.latest(-1, timeout=0.1)
            body = json.dumps(stats).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = -1
            try:
                while True:
                    serial, frame, _, _ = self.broadcaster.latest(
                        last, timeout=10.0
                    )
                    if frame is None:
                        continue
                    last = serial
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                    )
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            return
        self.send_response(204)
        self.end_headers()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--world", type=int, default=64)
    parser.add_argument("--size", type=int, default=720)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--quality", type=int, default=82)
    parser.add_argument("--season-length", type=int, default=150)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    config = build_config(args)
    broadcaster = FrameBroadcaster()
    controls = Controls()
    stop = threading.Event()
    thread = threading.Thread(
        target=sim_loop,
        args=(config, args, broadcaster, stop, controls),
        daemon=True,
    )
    thread.start()

    Handler.broadcaster = broadcaster
    Handler.controls = controls
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    print(f"Vivarium live at http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
