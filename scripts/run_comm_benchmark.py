"""Runner for the synthetic communication benchmark.

Reads a config (default ``configs/comm_task.yaml``), scores every protocol,
prints a human-readable table, and writes a JSON report. Acceptance
thresholds in the config define a pass/fail outcome — the script exits 1
on any failure so it can be wired into CI without further glue.

Usage
-----

    python scripts/run_comm_benchmark.py
    python scripts/run_comm_benchmark.py --config configs/comm_task.yaml \\
        --out outputs/comm_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from hybrid_alife.metrics.comm_benchmark import (
    BenchmarkSpec,
    run_benchmark,
    summarise,
)


def _check_acceptance(flat: dict[str, dict[str, float]], thresholds: dict) -> list[str]:
    failures: list[str] = []
    comp = flat.get("compositional", {})
    if (v := comp.get("topsim", 0.0)) < thresholds.get("compositional_topsim_min", 0.0):
        failures.append(f"compositional.topsim {v:.3f} < {thresholds['compositional_topsim_min']}")
    if (v := comp.get("posdis", 0.0)) < thresholds.get("compositional_posdis_min", 0.0):
        failures.append(f"compositional.posdis {v:.3f} < {thresholds['compositional_posdis_min']}")
    if (v := comp.get("bosdis", 0.0)) < thresholds.get("compositional_bosdis_min", 0.0):
        failures.append(f"compositional.bosdis {v:.3f} < {thresholds['compositional_bosdis_min']}")
    if (v := comp.get("shuffle.topsim", 1.0)) > thresholds.get("shuffle_topsim_max", 1.0):
        failures.append(f"compositional.shuffle.topsim {v:.3f} > {thresholds['shuffle_topsim_max']}")
    if (v := comp.get("zero.posdis", 1.0)) > thresholds.get("zero_posdis_max", 1.0):
        failures.append(f"compositional.zero.posdis {v:.6f} > {thresholds['zero_posdis_max']}")
    if (v := comp.get("zero.bosdis", 1.0)) > thresholds.get("zero_bosdis_max", 1.0):
        failures.append(f"compositional.zero.bosdis {v:.6f} > {thresholds['zero_bosdis_max']}")
    if (v := abs(comp.get("zero.channel_capacity", 1.0))) > thresholds.get("zero_capacity_max", 1.0):
        failures.append(
            f"compositional.zero.channel_capacity {v:.6f} > {thresholds['zero_capacity_max']}"
        )
    return failures


def _print_table(flat: dict[str, dict[str, float]]) -> None:
    metric_cols = ["topsim", "posdis", "bosdis", "channel_capacity"]
    print(f"{'protocol':<22} " + " ".join(f"{c:>10}" for c in metric_cols))
    for proto, metrics in flat.items():
        for variant in ("", "shuffle.", "zero."):
            label = proto if not variant else f"{proto} [{variant.rstrip('.')}]"
            row = [metrics.get(f"{variant}{c}", float("nan")) for c in metric_cols]
            print(f"{label:<22} " + " ".join(f"{v:>10.4f}" for v in row))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic communication benchmark")
    parser.add_argument("--config", type=Path, default=Path("configs/comm_task.yaml"))
    parser.add_argument("--out", type=Path, default=None,
                        help="optional JSON output path (default: stdout only)")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    spec_kwargs = cfg.get("benchmark", {})
    spec = BenchmarkSpec(**spec_kwargs)
    protocols = cfg.get("protocols")
    include_controls = bool(cfg.get("include_controls", True))

    results = run_benchmark(spec, protocols=protocols, include_controls=include_controls)
    flat = summarise(results)

    _print_table(flat)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(flat, indent=2, sort_keys=True))
        print(f"\nwrote {args.out}")

    failures = _check_acceptance(flat, cfg.get("acceptance", {}))
    if failures:
        print("\nFAILED acceptance checks:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nall acceptance checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
