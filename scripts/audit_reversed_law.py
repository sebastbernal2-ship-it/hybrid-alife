"""Recompute embodied-law verdicts without the reversed-law clause, from per-seed JSONs.

Prints, per file: stored passes, passes on the primary criterion plus an explicit
evolved >= 0.2 floor (what the clause amounts to), passes on the primary criterion alone,
the number of reversed values that differ from the antipodal evolved value (``mismatch``),
and seeds whose verdict flips without the clause as (seed, min_delta, min_evolved).

Usage (from the repo root): python scripts/audit_reversed_law.py
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

MARGIN = FLOOR = 0.2


def main() -> None:
    root = Path("docs/results")
    files = sorted([*root.glob("embodied_law*/**/*.json"), *root.glob("d1_ablation/*.json")])
    total_rows = total_values = 0
    worst = 0.0
    for path in files:
        data = json.loads(path.read_text())
        rows = data if isinstance(data, list) else data.get("results")
        if not isinstance(rows, list) or not rows or "reversed_law" not in rows[0]:
            continue
        gate = {} if isinstance(data, list) else data.get("gate", {})
        absolute = gate.get("min_evolved")
        stored = floored = primary_only = mismatch = 0
        flips = []
        for row in rows:
            evolved = row.get("evolved", row.get("trained"))
            by_direction = {ast.literal_eval(k): v for k, v in evolved.items()}
            for key, value in row["reversed_law"].items():
                x, y = ast.literal_eval(key)
                difference = abs(value - by_direction[(-x, -y)])
                worst = max(worst, difference)
                mismatch += difference > 0
                total_values += 1
            total_rows += 1
            min_delta = min(evolved[k] - row["initial"][k] for k in evolved)
            if absolute is None:
                primary = all(evolved[k] - row["initial"][k] >= MARGIN for k in evolved)
            else:
                primary = min(evolved.values()) >= absolute
            stored += bool(row["passed"])
            primary_only += primary
            floored += primary and min(evolved.values()) >= FLOOR
            if primary != bool(row["passed"]):
                flips.append((row["seed"], round(min_delta, 3), round(min(evolved.values()), 3)))
        print(
            f"{path.relative_to(root)}: n={len(rows)} stored={stored} floor={floored}"
            f" without={primary_only} mismatch={mismatch} flips={flips}"
        )
    print(f"rows={total_rows} values={total_values} max_difference={worst}")


if __name__ == "__main__":
    main()
