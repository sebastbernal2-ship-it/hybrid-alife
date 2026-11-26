"""Run the opt-in physics evolution study (docs/results/physics_evolution_study/).

One JSON per (condition, seed). The loop mirrors ``run_experiment`` (selection and
reseeding included) without its checkpoints. Births are counted from slots whose
lineage id changed and that are alive, not from ``apply_reproduction``'s match count,
which also counts children lost to the legacy slot-0 scatter.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "physics_evolution_study.yaml"

THERMO = {"embodied": {"thermo_efficiency": 0.5}}
SIGNAL = {"embodied": {"chemical_signaling": True, "signal_content": True}}
STOKES = {"embodied": {"stokes_mechanics": True, "aspect_drag": True}}
AVIDA_ENERGY = {"avida": {"instruction_energy_cost": 0.002}}  # double the base cost
LBM = {"world": {"flow_solver": "lbm_live", "lbm_agent_drag": 0.5,
                 "lbm_agent_coupling": "dipole"}}
LBM_ONEWAY = {"world": {"flow_solver": "lbm_live", "lbm_agent_drag": 0.0}}


def _merge(*parts: dict) -> dict:
    out: dict = {}
    for part in parts:
        for section, values in part.items():
            out.setdefault(section, {}).update(values)
    return out


# The baseline has divide_scatter_fix on (set in the base config).
CONDITIONS = {
    "baseline": {},
    "thermo": THERMO,
    "signaling": SIGNAL,
    "stokes": STOKES,
    "avida_energy": AVIDA_ENERGY,
    "lbm_dipole": LBM,
    "combined": _merge(THERMO, SIGNAL, STOKES, AVIDA_ENERGY, LBM),
    # Confound controls
    "ledger": {"embodied": {"energy_ledger": True}},
    "legacy_divide": {"avida": {"divide_scatter_fix": False}},
    "lbm_oneway": LBM_ONEWAY,
    "avida_nocost": {"avida": {"instruction_energy_cost": 0.0}},
}


def build_config(condition: str, seed: int, override: dict | None = None):
    from hybrid_alife.experiments.runner import load_config

    cfg = load_config(BASE)
    changes = _merge(override or {}, CONDITIONS[condition])
    parts = {k: dataclasses.replace(getattr(cfg, k), **v) for k, v in changes.items()}
    return dataclasses.replace(cfg, seed=seed, **parts)


def run_one(condition: str, seed: int, override: dict | None = None) -> dict:
    import jax
    import numpy as np

    from hybrid_alife.evolution.selection import (
        reseed_avida,
        reseed_embodied,
        select_and_mutate_embodied,
    )
    from hybrid_alife.experiments.runner import (
        initialize_sim,
        step_sim,
        validate_runtime_invariants,
    )

    cfg = build_config(condition, seed, override)
    state = initialize_sim(cfg)
    counter = cfg.embodied.population_size + cfg.avida.population_size
    gens, per = cfg.evolution.generations, cfg.evolution.steps_per_generation
    window_start = (gens * per * 3) // 4
    emitted = alive_steps = 0
    records = []

    def snapshot(pop):
        return np.array(pop.alive), np.array(pop.lineage_id)

    def turnover(before, pop):
        alive, ids = snapshot(pop)
        births = int((alive & (ids != before[1])).sum())
        return births, int(before[0].sum()) + births - int(alive.sum())

    for generation in range(gens):
        state.generation = generation
        rec = dict.fromkeys(("emb_births", "emb_deaths", "av_births", "av_deaths"), 0)
        for _ in range(per):
            emb_before, av_before = snapshot(state.embodied), snapshot(state.avida)
            state, counter = step_sim(state, cfg, counter)
            for prefix, before, pop in (
                ("emb", emb_before, state.embodied), ("av", av_before, state.avida)
            ):
                births, deaths = turnover(before, pop)
                rec[f"{prefix}_births"] += births
                rec[f"{prefix}_deaths"] += deaths
            if state.step > window_start:
                # Agents that acted this step (alive before it).
                acted = emb_before[0]
                gate = np.asarray(state.embodied.last_action)[:, 7]
                emitted += int((acted & (gate > cfg.embodied.message_gate_threshold)).sum())
                alive_steps += int(acted.sum())
        validate_runtime_invariants(state, cfg)
        alive = np.asarray(state.embodied.alive)
        av_alive = np.asarray(state.avida.alive)

        def mean(values, mask):
            return float(np.asarray(values)[mask].mean()) if mask.any() else None

        rec.update(
            emb_alive_end=int(alive.sum()),
            av_alive_end=int(av_alive.sum()),
            emb_lineage_depth=mean(state.embodied.lineage_depth, alive),
            av_lineage_depth=mean(state.avida.lineage_depth, av_alive),
            av_genome_length=mean(state.avida.genome_lengths, av_alive),
            av_merit=mean(state.avida.merit, av_alive),
            body_radius=mean(
                np.clip(np.asarray(state.embodied.genomes["body_radius"]), 0.6, 1.8), alive
            ),
            body_aspect=mean(
                np.clip(np.asarray(state.embodied.genomes["body_aspect"]), 0.7, 1.4), alive
            ),
            emb_energy=mean(state.embodied.energy, alive),
        )
        records.append(rec)

        # End of generation, as in run_experiment.
        fitness = state.embodied.energy
        state.rng, k_sel = jax.random.split(state.rng)
        state.embodied = select_and_mutate_embodied(
            state.embodied, fitness, cfg.embodied, cfg.evolution, k_sel
        )
        state.rng, k_re = jax.random.split(state.rng)
        if cfg.evolution.reseed_on_extinction:
            state.embodied = reseed_embodied(
                state.embodied, fitness, cfg.embodied, cfg.world, k_re
            )
            state.rng, k_av = jax.random.split(state.rng)
            state.avida = reseed_avida(
                state.avida, fraction=0.1, key=k_av, energy=cfg.avida.energy_initial
            )
        state.embodied_births_this_gen = state.embodied_deaths_this_gen = 0

    totals = {
        k: sum(r[k] for r in records)
        for k in ("emb_births", "emb_deaths", "av_births", "av_deaths")
    }
    final = records[-1]
    return {
        "condition": condition,
        "seed": seed,
        "override": override or {},
        **totals,
        "emb_min_alive_end": min(r["emb_alive_end"] for r in records),
        "av_min_alive_end": min(r["av_alive_end"] for r in records),
        "emit_rate_final_quarter": emitted / alive_steps if alive_steps else None,
        **{k: final[k] for k in (
            "emb_alive_end", "av_alive_end", "emb_lineage_depth", "av_lineage_depth",
            "av_genome_length", "av_merit", "body_radius", "body_aspect", "emb_energy",
        )},
        "generations": records,
    }


def _job(args):
    condition, seed, override, out = args
    result = run_one(condition, seed, override)
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    return condition, seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", default=["baseline"], choices=list(CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--override", type=json.loads, default=None, help="calibration only")
    args = parser.parse_args()
    jobs = []
    for condition in args.conditions:
        (args.out / condition).mkdir(parents=True, exist_ok=True)
        jobs += [
            (condition, s, args.override, str(args.out / condition / f"seed_{s}.json"))
            for s in args.seeds
        ]
    with mp.get_context("spawn").Pool(min(args.workers, 8)) as pool:
        for condition, seed in pool.imap_unordered(_job, jobs):
            print(condition, seed, flush=True)


if __name__ == "__main__":
    main()
