"""Overlap and penetration of the *drawn* swimmer shapes, measured from simulated points.

The viewer draws the head as the ring of points grown by ``HEAD_PAD`` and the tail as a
ribbon through the tail points with half-width ``tail_half_width``. ``viewer.js`` uses the
same two definitions, so a negative gap here is a visible overlap on screen.
"""

from __future__ import annotations

import numpy as np

HEAD_PAD = 0.6
TAIL_ROOT, TAIL_TIP = 1.15, 0.35


def tail_half_width(count: int) -> np.ndarray:
    return TAIL_ROOT + (TAIL_TIP - TAIL_ROOT) * np.linspace(0, 1, count)


def _densify(p: np.ndarray, radius: np.ndarray, per: int = 5) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(per)[:, None] / per
    pts = (p[None, :-1] * (1 - t[..., None]) + p[None, 1:] * t[..., None]).reshape(-1, 2)
    rad = (radius[None, :-1] * (1 - t) + radius[None, 1:] * t).reshape(-1)
    return np.vstack([pts, p[-1:]]), np.append(rad, radius[-1])


def _inside(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    a, b = polygon, np.roll(polygon, -1, 0)
    px, py = points[:, 0:1], points[:, 1:2]
    span = (a[:, 1] > py) != (b[:, 1] > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = a[:, 0] + (py - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
    return (span & (px < cross)).sum(1) % 2 == 1


def _crossings(a: np.ndarray, b: np.ndarray) -> int:
    """Number of proper intersections between two segment sets [n, 2, 2]."""
    def side(s, q):
        d = s[:, None, 1] - s[:, None, 0]
        r = q[None] - s[:, None, 0]
        return d[..., 0] * r[..., 1] - d[..., 1] * r[..., 0]

    one = side(a, b[:, 0]) * side(a, b[:, 1]) < 0
    two = (side(b, a[:, 0]) * side(b, a[:, 1]) < 0).T
    return int(np.sum(one & two))


class OverlapStats:
    def __init__(self, structure, mask: np.ndarray):
        self.ny, self.nx = mask.shape
        self.index = [np.flatnonzero(structure.body == b) for b in range(len(structure.kinds))]
        self.rings = [int(np.sum((structure.edges == i[0]).any(1))) for i in self.index]
        self.planes, self.discs = structure.planes, structure.discs
        self.frames = 0
        self.overlap_frames = self.crossing_frames = self.surface_frames = 0
        self.body_gap = self.surface_gap = np.inf
        self.point_distance = self.step_wall_gap = np.inf

    def step_report(self, report: dict) -> None:
        """Extremes over every time step (not only recorded frames), from the kernel."""
        self.point_distance = min(self.point_distance, report["min_contact_distance"])
        self.step_wall_gap = min(self.step_wall_gap, report["min_wall_gap"])

    def _shapes(self, X: np.ndarray):
        for idx, ring in zip(self.index, self.rings, strict=True):
            p = X[idx]
            head = p[1 : 1 + ring]
            tail = np.vstack([p[1 + ring // 2], p[1 + ring :]])
            outline, pad = _densify(np.vstack([head, head[:1]]), np.full(ring + 1, HEAD_PAD))
            line, width = _densify(tail, tail_half_width(len(tail)))
            segments = np.concatenate([
                np.stack([head, np.roll(head, -1, 0)], 1), np.stack([tail[:-1], tail[1:]], 1)
            ])
            yield head, np.vstack([outline, line]), np.append(pad, width), segments

    def frame(self, X: np.ndarray) -> None:
        self.frames += 1
        shapes = list(self._shapes(X))
        box = np.array([self.nx, self.ny], float)
        gap_here, surface_here, crossed = np.inf, np.inf, 0
        for i, (head, pts, rad, seg) in enumerate(shapes):
            for nx, ny, c in self.planes:
                surface_here = min(surface_here, np.min(pts @ [nx, ny] - c - rad))
            for cx, cy, r in self.discs:
                d = pts - (cx, cy)
                d -= box * np.rint(d / box)
                surface_here = min(surface_here, np.min(np.hypot(*d.T) - r - rad))
            for head2, pts2, rad2, seg2 in shapes[i + 1 :]:
                shift = box * np.rint((pts2.mean(0) - pts.mean(0)) / box)
                other = pts2 - shift
                dist = np.linalg.norm(pts[:, None] - other[None], axis=2)
                gap = np.min(dist - rad[:, None] - rad2[None])
                if _inside(other, head).any() or _inside(pts, head2 - shift).any():
                    gap = min(gap, -HEAD_PAD)
                gap_here = min(gap_here, gap)
                if gap < 1.0:
                    crossed += _crossings(seg, seg2 - shift)
        self.body_gap = min(self.body_gap, gap_here)
        self.surface_gap = min(self.surface_gap, surface_here)
        self.overlap_frames += gap_here < 0
        self.surface_frames += surface_here < 0
        self.crossing_frames += crossed > 0

    def summary(self) -> dict:
        def number(v):
            return None if not np.isfinite(v) else round(float(v), 3)

        return {
            "frames": self.frames,
            "drawn_body_min_gap": number(self.body_gap),
            "drawn_body_overlap_frames": int(self.overlap_frames),
            "outline_crossing_frames": int(self.crossing_frames),
            "drawn_surface_min_gap": number(self.surface_gap),
            "drawn_surface_penetration_frames": int(self.surface_frames),
            "every_step_min_point_distance": number(self.point_distance),
            "every_step_min_point_wall_gap": number(self.step_wall_gap),
        }
