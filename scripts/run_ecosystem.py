"""Long ecosystem runs and the common-garden behaviour test.

    python scripts/run_ecosystem.py run --seed 0 --steps 60000 --out outputs/ecosystem
    python scripts/run_ecosystem.py garden --run outputs/ecosystem/seed_0
    python scripts/run_ecosystem.py record --run outputs/ecosystem/seed_0 \\
        --checkpoint 77000 --out outputs/ecosystem/replay.html
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_alife import ecosystem_study as study  # noqa: E402
from hybrid_alife.ecosystem import load_ecosystem_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, default=ROOT / "configs/ecosystem.yaml")
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--steps", type=int, default=60000)
    run.add_argument("--window", type=int, default=2000)
    run.add_argument("--checkpoint", type=int, nargs="*", default=[])
    run.add_argument("--out", type=Path, default=Path("outputs/ecosystem"))
    garden = commands.add_parser("garden")
    garden.add_argument("--run", type=Path, required=True)
    garden.add_argument("--eval-seeds", type=int, default=8)
    garden.add_argument("--steps", type=int, default=300)
    garden.add_argument("--cohort", type=int, default=100)
    rec = commands.add_parser("record", help="record a window from a checkpoint")
    rec.add_argument("--run", type=Path, required=True)
    rec.add_argument("--checkpoint", type=int, required=True)
    rec.add_argument("--steps", type=int, default=3000)
    rec.add_argument("--out", type=Path, required=True, help="standalone HTML to write")
    args = parser.parse_args()

    if args.command == "record":
        from hybrid_alife.ecoview import record

        eco, registry = study.load_checkpoint(args.run / f"checkpoint_{args.checkpoint}.pkl")
        summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
        keys = ("step", "population_mean", "population_min", "population_max",
                "founder_lineages")
        history = {k: [w[k] for w in summary["windows"]] for k in keys}
        history["founders"] = summary["founders"]
        replay = record.record(eco, registry, args.steps, history)
        record.standalone(replay, args.out)
        args.out.with_suffix(".eco").write_bytes(replay)
        print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    elif args.command == "run":
        cfg, params = load_ecosystem_config(args.config)
        cfg = replace(cfg, seed=args.seed)
        out = args.out / f"seed_{args.seed}"
        out.mkdir(parents=True, exist_ok=True)
        study.long_run(cfg, params, args.steps, args.window, out, set(args.checkpoint))
    else:
        saved = pickle.loads((args.run / "genomes.pkl").read_bytes())
        result = study.common_garden(
            saved, eval_seeds=args.eval_seeds, steps=args.steps, cohort=args.cohort
        )
        (args.run / "garden.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        for name, row in result["cohorts"].items():
            print(name, {k: round(float(np.mean(v)), 4) for k, v in row.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
