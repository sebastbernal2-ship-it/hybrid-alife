"""Tests proving the tiny-CPU lineage_growth config produces real births.

Acceptance criteria (mission `sprint/lineage-growth-fast`):
  * max_lineage_depth_embodied(final) > 0
  * total embodied_births_per_generation across the run > 0
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp

from hybrid_alife.agents.embodied import (
    apply_reproduction,
    initialize_embodied_population,
)
from hybrid_alife.experiments.runner import load_config, run_experiment


def test_apply_reproduction_increments_lineage_depth_and_counts_births():
    """Unit-level guard: a forced reproduction event sets parent_id, child depth=1, and
    apply_reproduction reports a positive birth count."""
    cfg = load_config("configs/lineage_growth.yaml")
    import jax

    pop = initialize_embodied_population(cfg.embodied, cfg.world, jax.random.PRNGKey(0))
    # Force half the population alive at high energy and gate them open.
    n = pop.alive.shape[0]
    half = n // 2
    pop.alive = jnp.arange(n) < half
    pop.energy = jnp.where(pop.alive, 100.0, 0.0)
    pop.parent_id = -jnp.ones((n,), dtype=jnp.int32)
    pop.lineage_depth = jnp.zeros((n,), dtype=jnp.int32)
    repro_gate = jnp.where(pop.alive, 1.0, 0.0)
    new_pop, births = apply_reproduction(
        pop, repro_gate, cfg.embodied, next_lineage_start=1000, key=jax.random.PRNGKey(1)
    )
    assert births > 0, f"expected >0 births, got {births}"
    # At least one child slot now has depth == 1 and a valid parent_id link.
    depths = jnp.asarray(new_pop.lineage_depth)
    parent_ids = jnp.asarray(new_pop.parent_id)
    assert int(depths.max()) >= 1
    # Children created should have parent_id pointing into an originally-alive lineage_id.
    child_mask = depths >= 1
    assert int(child_mask.sum()) >= 1
    assert int((parent_ids[child_mask] >= 0).sum()) >= 1


def test_lineage_growth_smoke_run_births_and_depth(tmp_path):
    """Integration guard: a controlled run of configs/lineage_growth.yaml produces
    max lineage depth > 0 AND positive total births reported in metrics.jsonl."""
    cfg = load_config("configs/lineage_growth.yaml")
    # Redirect outputs to a tmp dir so tests stay hermetic.
    cfg = cfg.__class__(
        seed=cfg.seed,
        run_name="lineage_growth_test",
        output_dir=str(tmp_path),
        world=cfg.world,
        embodied=cfg.embodied,
        avida=cfg.avida,
        evolution=cfg.evolution,
        logging=cfg.logging,
    )
    state = run_experiment(cfg)

    assert state.embodied is not None
    max_depth = int(state.embodied.lineage_depth.max())
    assert max_depth > 0, f"max lineage depth should be >0, got {max_depth}"

    metrics_path = Path(tmp_path) / "lineage_growth_test" / "metrics.jsonl"
    assert metrics_path.exists()
    total_births = 0
    saw_any_record = False
    for line in metrics_path.read_text().splitlines():
        rec = json.loads(line)
        saw_any_record = True
        total_births = max(total_births, int(rec.get("embodied_births_per_generation", 0)))
    assert saw_any_record
    assert total_births > 0, f"expected >0 births across the run, got {total_births}"
