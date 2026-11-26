"""Cumulative energy ledger for embodied agents (``EmbodiedConfig.energy_ledger``).

``EmbodiedPopulationState.ledger`` holds running scalar totals. Every entry is accumulated
from the term that changes energy, never from the observed change in total energy, so
comparing the ledger with the actual energies (``audit_energy_ledger``) is a real check.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# Entries that leave organisms as heat. "shortfall" is subtracted: it is the part of the
# booked losses that an organism could not pay (energy below zero is reset to zero).
HEAT_TERMS = ("costs", "damage")
SIGNED_HEAT_TERMS = ("shortfall",)
LEDGER_KEYS = ("initial", *HEAT_TERMS, *SIGNED_HEAT_TERMS)


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


def heat_metrics(state) -> dict[str, float]:
    """Heat metrics for whichever ledgers exist; empty (legacy keys unchanged) otherwise."""
    out: dict[str, float] = {}
    if state.embodied is not None and state.embodied.ledger is not None:
        out["embodied_dissipated_heat"] = dissipated_heat(state.embodied.ledger)
    if state.avida is not None and state.avida.heat is not None:
        out["avida_dissipated_heat"] = float(np.asarray(state.avida.heat))
    return out
