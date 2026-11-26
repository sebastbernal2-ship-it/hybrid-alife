"""Cumulative energy ledger for embodied agents (``EmbodiedConfig.energy_ledger``).

``EmbodiedPopulationState.ledger`` holds running scalar totals. Every entry is accumulated
from the term that changes energy, never from the observed change in total energy, so
comparing the ledger with the actual energies (``audit_energy_ledger``) is a real check.

Identity: sum(energy) == initial + intake + external_work + injected - heat.

  intake        energy assimilated by eating
  external_work flow-alignment and displacement rewards (signed): work supplied from outside
                the modelled system, since the flow loses nothing when agents are paid
  injected      newborn endowments and end-of-generation resets by selection or reseeding
                (signed); neither is funded by another organism
  heat          costs + damage + predation_loss + cap_loss + reproduction_cost
                + discarded - shortfall
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# Entries that leave organisms as heat. "shortfall" is subtracted: it is the part of the
# booked losses that an organism could not pay (energy below zero is reset to zero).
HEAT_TERMS = (
    "costs", "damage", "predation_loss", "cap_loss", "reproduction_cost", "discarded",
)
INPUT_TERMS = ("intake", "external_work", "injected")
SIGNED_HEAT_TERMS = ("shortfall",)
LEDGER_KEYS = ("initial", *INPUT_TERMS, *HEAT_TERMS, *SIGNED_HEAT_TERMS)


def new_ledger(energy) -> dict:
    """Zeroed totals; ``initial`` is the total energy when accounting starts."""
    ledger = {k: jnp.zeros((), dtype=jnp.float32) for k in LEDGER_KEYS}
    ledger["initial"] = jnp.sum(energy).astype(jnp.float32)
    return ledger


def book(ledger: dict, **amounts) -> dict:
    """Return a new ledger with ``amounts`` (arrays or scalars, summed) added."""
    out = dict(ledger)
    for name, value in amounts.items():
        out[name] = out[name] + jnp.sum(value).astype(jnp.float32)
    return out


def dissipated_heat(ledger: dict) -> float:
    """Total energy dissipated as heat since accounting started."""
    total = sum(float(np.asarray(ledger[k])) for k in HEAT_TERMS)
    return total - float(np.asarray(ledger["shortfall"]))


def audit_energy_ledger(pop) -> dict[str, float]:
    """Compare the energy organisms hold with what the ledger says they should hold."""
    if pop.ledger is None:
        raise ValueError("no ledger: enable EmbodiedConfig.energy_ledger")
    totals = {k: float(np.asarray(v)) for k, v in pop.ledger.items()}
    heat = dissipated_heat(pop.ledger)
    expected = totals["initial"] + sum(totals[k] for k in INPUT_TERMS) - heat
    held = float(np.asarray(pop.energy, dtype=np.float64).sum())
    return {**totals, "heat": heat, "held": held, "expected": expected,
            "residual": held - expected}


def book_reset(pop, energy_before) -> None:
    """Book an external overwrite of organism energy (selection, reseeding)."""
    if pop.ledger is not None:
        pop.ledger = book(pop.ledger, injected=pop.energy - energy_before)


def heat_metrics(state) -> dict[str, float]:
    """Heat metrics for whichever ledgers exist; empty (legacy keys unchanged) otherwise."""
    out: dict[str, float] = {}
    if state.embodied is not None and state.embodied.ledger is not None:
        out["embodied_dissipated_heat"] = dissipated_heat(state.embodied.ledger)
    if state.avida is not None and state.avida.heat is not None:
        out["avida_dissipated_heat"] = float(np.asarray(state.avida.heat))
    return out
