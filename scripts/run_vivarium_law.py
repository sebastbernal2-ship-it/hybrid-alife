"""Run the research question inside the living world.

Matched vivarium worlds with a uniform law present in two opposite directions,
and one law-absent control. The law is only physically coupled: no alignment
reward. The readout is the population's commanded thrust alignment.

Usage:
    JAX_PLATFORMS=cpu python scripts/run_vivarium_law.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_alife.experiments.latent_law_embodied import embodied_law_config  # noqa: E402
from hybrid_alife.vivarium import (  # noqa: E402
    VivariumConfig,
    initialize_vivarium,
    mean_body_radius,
    reference_alignment,
    season_name,
    set_uniform_law,
    vivarium_step,
)

LAWS = {
    "law_positive": (1.0, 0.0),
    "law_negative": (-1.0, 0.0),
    "law_absent": (0.0, 0.0),
}
REFERENCE = (1.0, 0.0)
GATE_MARGIN = 0.2


def build_config(seed: int, args) -> VivariumConfig:
    sim = embodied_law_config(
        seed,
        generations=1,
        population=args.population,
        steps_per_generation=1,
        size=args.world,
        proprioception=True,
        initial_alive_fraction=0.6,
        flow_alignment_reward=0.0,
        displacement_reward=0.0,
    )
    if args.arm == "d1":
        sim = replace(
            sim,
            world=replace(sim.world, curvature_regime="straight"),
            embodied=replace(
                sim.embodied,
                body_morphology=True,
                linear_policy=True,
                unit_flow_features=True,
                rank_replacement=True,
                immigrant_fraction=0.0,
            ),
        )
        return VivariumConfig(
            sim=sim,
            season_length=args.season_length,
            generation_length=200,
            fitness_mode="min_alignment",
            fitness_steps=6,
        )
    sim = replace(
        sim,
        world=replace(sim.world, curvature_regime="straight"),
        embodied=replace(sim.embodied, body_morphology=True),
    )
    return VivariumConfig(
        sim=sim, season_length=args.season_length, generation_length=200
    )


def run_condition(seed: int, condition: str, args) -> dict:
    config = build_config(seed, args)
    state = initialize_vivarium(config)
    law = LAWS[condition]
    # Present conditions are scored against their own law, so a law-follower is
    # positive in both directions. The absent control is scored against +x, the
    # direction the positive law would have had.
    reference = law if law != (0.0, 0.0) else REFERENCE
    state = set_uniform_law(state, law)
    records = []
    for step in range(args.steps):
        state = vivarium_step(state, config)
        state.sim.step = step + 1
        state = set_uniform_law(state, law)
        if (step + 1) % args.record_every == 0:
            records.append(
                {
                    "step": step + 1,
                    "alignment": reference_alignment(state, reference),
                    "population": int(np.asarray(state.sim.embodied.alive).sum()),
                    "births": int(state.births),
                    "deaths": int(state.deaths),
                    "mean_body": mean_body_radius(state),
                    "season": season_name(state),
                }
            )
    tail = records[-max(1, len(records) // 4) :]
    return {
        "condition": condition,
        "seed": seed,
        "mean_alignment_tail": float(np.nanmean([r["alignment"] for r in tail])),
        "population_tail": float(np.nanmean([r["population"] for r in tail])),
        "births": int(state.births),
        "deaths": int(state.deaths),
        "mean_body_tail": float(np.nanmean([r["mean_body"] for r in tail])),
        "records": records,
    }


def plot(results: list[dict], out_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "law_positive": "#4cc9f0",
        "law_negative": "#f72585",
        "law_absent": "#8d99ae",
    }
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for condition in LAWS:
        rows = [r for r in results if r["condition"] == condition]
        if not rows:
            continue
        steps = [rec["step"] for rec in rows[0]["records"]]
        series = np.asarray(
            [[rec["alignment"] for rec in row["records"]] for row in rows]
        )
        mean = np.nanmean(series, axis=0)
        low = np.nanmin(series, axis=0)
        high = np.nanmax(series, axis=0)
        ax.plot(steps, mean, color=colors[condition], lw=2, label=condition)
        ax.fill_between(steps, low, high, color=colors[condition], alpha=0.15)
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("commanded thrust alignment with +x")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Do the organisms exploit the imposed law?")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--world", type=int, default=64)
    parser.add_argument("--season-length", type=int, default=150)
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--arm", choices=("natural", "d1"), default="natural")
    parser.add_argument("--out", type=Path, default=Path("docs/results/vivarium_law"))
    args = parser.parse_args()

    results = []
    for seed in args.seeds:
        for condition in LAWS:
            row = run_condition(seed, condition, args)
            results.append(row)
            print(
                f"seed {seed} {condition}: tail alignment "
                f"{row['mean_alignment_tail']:+.3f} "
                f"population {row['population_tail']:.0f}",
                flush=True,
            )

    by_seed: dict[int, dict[str, float]] = {}
    for row in results:
        by_seed.setdefault(row["seed"], {})[row["condition"]] = row["mean_alignment_tail"]
    seeds_passed = 0
    for values in by_seed.values():
        absent = values["law_absent"]
        if (
            values["law_positive"] >= absent + GATE_MARGIN
            and values["law_negative"] >= absent + GATE_MARGIN
        ):
            seeds_passed += 1

    summary = {
        "status": "pilot_only",
        "seeds": args.seeds,
        "steps": args.steps,
        "population": args.population,
        "world": args.world,
        "record_every": args.record_every,
        "arm": args.arm,
        "gate": {"margin": GATE_MARGIN, "absent_reference": REFERENCE},
        "seeds_passed": seeds_passed,
        "seeds_total": len(by_seed),
        "per_seed": {str(k): v for k, v in by_seed.items()},
        "results": results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(results, args.out / "alignment_over_time.png")
    print(
        f"\n{seeds_passed}/{len(by_seed)} seeds pass "
        f"(both law directions beat absent by {GATE_MARGIN})"
    )
    print(f"wrote {args.out / 'summary.json'}")
    print(f"wrote {args.out / 'alignment_over_time.png'}")
    return 0 if seeds_passed == len(by_seed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
