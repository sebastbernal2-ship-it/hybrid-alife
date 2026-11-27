"""Summarise docs/results/ib_convergence/raw into summary.json (orders and extrapolations).

    python scripts/analyze_ib_convergence.py
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path("docs/results/ib_convergence")


def power_fit(r, s) -> tuple[float, float] | None:
    """(limit, order) of s = limit - c r^-p through three points; None if no 0.05 < p < 4."""
    r, s = np.asarray(r, float), np.asarray(s, float)
    target = (s[1] - s[0]) / (s[2] - s[1])

    def mismatch(p):
        h = r**-p
        return (h[0] - h[1]) / (h[1] - h[2]) - target

    lo, hi = 0.05, 4.0
    if mismatch(lo) * mismatch(hi) > 0:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mismatch(lo) * mismatch(mid) > 0 else (lo, mid)
    h = r**-lo
    c = (s[2] - s[1]) / (h[1] - h[2])
    return float(s[2] + c * h[2]), float(lo)


def main() -> None:
    raw = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "raw").glob("*.json"))}
    summary: dict = {"failed": [k for k, d in raw.items() if d["study"] == "failed"],
                     "swimmer": {}, "sheet": {}}
    for name, d in raw.items():
        if d["study"] == "swimmer":
            # Beats 1-2 are ramp and transient; the spread of the rest is the window error.
            x = np.array(d["stride_xy_per_beat"])[2:, 0]
            summary["swimmer"][name] = {
                "r": d["r"], "stride": float(x.mean()), "beat_to_beat_std": float(x.std(ddof=1)),
                "tip_amplitude": float(np.mean(d["tip_amplitude_per_beat"][2:]))}
        elif d["study"] == "sheet":
            summary["sheet"][name] = {
                k: d[k] for k in ("n", "height", "error", "error_standard_error",
                                  "longitudinal_correction")}
    fits = {}
    for kernel, prefix in ((4, "swimmer_consistent_r"), (3, "kernel3_swimmer_r")):
        rows = sorted((v["r"], v["stride"]) for k, v in summary["swimmer"].items()
                      if k.startswith(prefix))
        out = []
        for triple in combinations(rows, 3):
            fit = power_fit(*zip(*triple, strict=True))
            out.append({"r": [t[0] for t in triple],
                        "limit": fit and fit[0], "order": fit and fit[1]})
        # Fixed first order through each adjacent pair.
        first = [{"r": [a[0], b[0]], "limit": float((b[0] * b[1] - a[0] * a[1]) / (b[0] - a[0]))}
                 for a, b in zip(rows[:-1], rows[1:], strict=True)]
        # limit - a/r - b/r^2: asymptotic order one with a second-order correction.
        two = [{"r": [t[0] for t in triple], "limit": float(np.linalg.solve(
                    [[1, -1 / t[0], -1 / t[0] ** 2] for t in triple],
                    [t[1] for t in triple])[0])} for triple in combinations(rows, 3)]
        fits[f"kernel{kernel}"] = {"free_order_triples": out, "first_order_pairs": first,
                                   "two_term_triples": two}
    summary["swimmer_fits"] = fits
    orders = {}
    for height in (1, 2):
        rows = sorted((v["n"], abs(v["error"]), v["error_standard_error"])
                      for k, v in summary["sheet"].items()
                      if k.startswith("sheet_") and v["height"] == height)
        orders[f"h{height}"] = [
            {"n": [a[0], b[0]], "order": float(np.log(a[1] / b[1]) / np.log(b[0] / a[0])),
             "order_uncertainty": float(np.hypot(a[2] / a[1], b[2] / b[1]) / np.log(b[0] / a[0]))}
            for a, b in zip(rows[:-1], rows[1:], strict=True)]
    summary["sheet_orders"] = orders
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("failed", "swimmer_fits", "sheet_orders")}))
    for k, v in summary["swimmer"].items():
        print(k, round(v["stride"], 4), round(v["beat_to_beat_std"], 4),
              round(v["tip_amplitude"], 3))


if __name__ == "__main__":
    main()
