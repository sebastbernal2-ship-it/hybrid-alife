"""Sweep predation and escape rates for a stable living world.

Runs a grid of worlds and reports the tail population, death rate, energy, and
body size. The output is a JSON summary and a balance map.

Usage:
    JAX_PLATFORMS=cpu python scripts/run_vivarium_sweep.py
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
    vivarium_step,
)

RATES = (0.0, 0.2, 0.35, 0.5, 0.7)
GAINS = (0.0, 0.03, 0.06)


def build_config(seed: int, rate: float, gain: float, args) -> VivariumConfig:
    sim = embodied_law_config(
        seed,
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
            predation_rate=rate,
            predation_radius=0.04,
            escape_gain=gain,
            escape_radius=0.06,
        ),
    )
    return VivariumConfig(sim=sim, season_length=args.season_length, generation_length=200)


def run_cell(seed: int, rate: float, gain: float, args) -> dict:
    config = build_config(seed, rate, gain, args)
    state = initialize_vivarium(config)
    populations: list[float] = []
    energies: list[float] = []
    bodies: list[float] = []
    for step in range(args.steps):
        state = vivarium_step(state, config)
        state.sim.step = step + 1
        if (step + 1) % args.record_every == 0:
            alive = np.asarray(state.sim.embodied.alive)
            populations.append(float(alive.sum()))
            energy = np.asarray(state.sim.embodied.energy)[alive]
            energies.append(float(energy.mean()) if alive.any() else 0.0)
            bodies.append(mean_body_radius(state) if alive.any() else 1.0)
    tail = max(1, len(populations) // 4)
    return {
        "seed": seed,
        "predation_rate": rate,
        "escape_gain": gain,
        "tail_population": float(np.mean(populations[-tail:])),
        "tail_energy": float(np.nanmean(energies[-tail:])),
        "tail_body": float(np.nanmean(bodies[-tail:])),
        "births": int(state.births),
        "deaths": int(state.deaths),
        "death_rate": float(state.deaths) / max(args.steps, 1),
        "final_population": int(np.asarray(state.sim.embodied.alive).sum()),
    }


def aggregate(results: list[dict]) -> list[dict]:
    cells: dict[tuple[float, float], list[dict]] = {}
    for row in results:
        cells.setdefault((row["predation_rate"], row["escape_gain"]), []).append(row)
    rows = []
    for (rate, gain), group in sorted(cells.items()):
        rows.append(
            {
                "predation_rate": rate,
                "escape_gain": gain,
                "tail_population": float(np.mean([r["tail_population"] for r in group])),
                "tail_energy": float(np.mean([r["tail_energy"] for r in group])),
                "tail_body": float(np.mean([r["tail_body"] for r in group])),
                "death_rate": float(np.mean([r["death_rate"] for r in group])),
                "final_population": float(np.mean([r["final_population"] for r in group])),
                "seeds": len(group),
            }
        )
    return rows


def plot(rows: list[dict], out_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pop = np.zeros((len(RATES), len(GAINS)))
    death = np.zeros_like(pop)
    for row in rows:
        i = RATES.index(row["predation_rate"])
        j = GAINS.index(row["escape_gain"])
        pop[i, j] = row["tail_population"]
        death[i, j] = row["death_rate"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, data, title, cmap in (
        (axes[0], pop, "tail population", "viridis"),
        (axes[1], death, "death rate per step", "magma"),
    ):
        image = ax.imshow(data, origin="lower", cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(GAINS)), [str(g) for g in GAINS])
        ax.set_yticks(range(len(RATES)), [str(r) for r in RATES])
        ax.set_xlabel("escape gain")
        ax.set_ylabel("predation rate")
        ax.set_title(title)
        for i in range(len(RATES)):
            for j in range(len(GAINS)):
                ax.text(
                    j,
                    i,
                    f"{data[i, j]:.1f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Vivarium balance sweep")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--world", type=int, default=64)
    parser.add_argument("--season-length", type=int, default=150)
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("docs/results/vivarium_balance"))
    args = parser.parse_args()

    results = []
    for rate in RATES:
        for gain in GAINS:
            for seed in args.seeds:
                row = run_cell(seed, rate, gain, args)
                results.append(row)
                print(
                    f"rate {rate:.2f} escape {gain:.2f} seed {seed}: "
                    f"pop {row['tail_population']:.0f} death {row['death_rate']:.3f}",
                    flush=True,
                )
    rows = aggregate(results)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(
            {"grid": {"rates": RATES, "gains": GAINS}, "rows": rows, "runs": results},
            indent=2,
        )
        + "\n"
    )
    plot(rows, args.out / "balance_map.png")
    print("\nrate   gain   pop   death   energy   body")
    for row in rows:
        print(
            f"{row['predation_rate']:.2f}   {row['escape_gain']:.2f}   "
            f"{row['tail_population']:.0f}   {row['death_rate']:.3f}   "
            f"{row['tail_energy']:.1f}   {row['tail_body']:.2f}"
        )
    print(f"wrote {args.out / 'summary.json'}")
    print(f"wrote {args.out / 'balance_map.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
