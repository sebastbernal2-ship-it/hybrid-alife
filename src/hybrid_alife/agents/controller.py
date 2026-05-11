"""Per-agent neural controller as a clean JAX pytree.

This module isolates the controller architecture from the lifecycle and
world-coupling code in `embodied.py`. The controller is a vectorized
per-population GRU (or MLP when `use_memory=False`) where each individual
owns its own parameter tensors so that mutation acts directly on the genome.

Design goals
------------
- *Pure*: every function is a pure mapping over JAX arrays / pytrees so we
  can mutate, vmap, and (later) jit without surprises.
- *Compatible*: the genome dict and forward-pass shapes are unchanged from
  the previous implementation. Existing checkpoints continue to load.
- *Optional richness*: stochastic action sampling, message gating thresholds,
  and per-action temperature are all controlled by config — defaults keep
  the deterministic behavior unchanged.

Genome layout
~~~~~~~~~~~~~
``w_xz, w_hz, b_z`` — update-gate weights
``w_xh, w_hh, b_h`` — candidate-state weights
``w_act, b_act``    — action head (linear projection from hidden -> action)

Action vector (length ``8 + message_size``)::

  [0, 1]   continuous movement in [-1, 1]
  [2..7]   six action gates in [0, 1] (eat, attack, reproduce,
           terraform-resource, terraform-hazard, emit-message)
  [8..]    message payload (linear)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hybrid_alife.types import EmbodiedConfig, PRNGKey


def init_controller_params(
    n: int, obs_dim: int, action_dim: int, hidden: int, key: PRNGKey
) -> dict[str, jax.Array]:
    """Allocate per-agent GRU/MLP parameters with small Gaussian init."""
    keys = jax.random.split(key, 7)
    return {
        "w_xz": 0.1 * jax.random.normal(keys[0], (n, obs_dim, hidden)),
        "w_hz": 0.1 * jax.random.normal(keys[1], (n, hidden, hidden)),
        "b_z": jnp.zeros((n, hidden)),
        "w_xh": 0.1 * jax.random.normal(keys[2], (n, obs_dim, hidden)),
        "w_hh": 0.1 * jax.random.normal(keys[3], (n, hidden, hidden)),
        "b_h": jnp.zeros((n, hidden)),
        "w_act": 0.1 * jax.random.normal(keys[4], (n, hidden, action_dim)),
        "b_act": 0.01 * jax.random.normal(keys[5], (n, action_dim)),
    }


def forward(
    genomes: dict[str, jax.Array],
    obs: jax.Array,
    hidden: jax.Array,
    use_memory: bool,
) -> tuple[jax.Array, jax.Array]:
    """Run one controller step.

    Returns ``(raw_action_logits, new_hidden)``. The caller is responsible for
    turning logits into bounded actions (e.g. tanh/sigmoid/sample).
    """
    z = jax.nn.sigmoid(
        jnp.einsum("no,noh->nh", obs, genomes["w_xz"])
        + jnp.einsum("nh,nhk->nk", hidden, genomes["w_hz"])
        + genomes["b_z"]
    )
    cand = jnp.tanh(
        jnp.einsum("no,noh->nh", obs, genomes["w_xh"])
        + jnp.einsum("nh,nhk->nk", hidden, genomes["w_hh"])
        + genomes["b_h"]
    )
    new_hidden = (1.0 - z) * hidden + z * cand
    if not use_memory:
        # MLP ablation: do not retain recurrent state; controller is a pure
        # feedforward map of obs -> action via the candidate hidden state.
        new_hidden = cand
    raw = jnp.einsum("nh,nha->na", new_hidden, genomes["w_act"]) + genomes["b_act"]
    return raw, new_hidden


def decode_actions(
    raw: jax.Array,
    cfg: EmbodiedConfig,
    key: PRNGKey,
) -> jax.Array:
    """Map raw logits to a bounded action vector.

    - movement: tanh on logits / temperature
    - gates: sigmoid (probability). When ``cfg.stochastic_actions`` is true,
      we Bernoulli-sample the gates so the action is genuinely stochastic.
    - message: linear, gated by the emit gate (we leave gating to the
      caller because the gate also drives world updates).
    """
    move = jnp.tanh(raw[..., 0:2] / jnp.maximum(cfg.action_temperature, 1e-6))
    gate_logits = raw[..., 2:8] / jnp.maximum(cfg.action_temperature, 1e-6)
    gate_probs = jax.nn.sigmoid(gate_logits)
    if cfg.stochastic_actions:
        sample = jax.random.bernoulli(key, gate_probs).astype(jnp.float32)
        # Soft surrogate so downstream cost models see a smooth signal:
        gates = sample * gate_probs + (1.0 - sample) * (gate_probs * 0.0)
        # Hard-binarized gates are useful, so emit a hard {0,1} that retains
        # the probability magnitude via multiplication.
        gates = jnp.where(sample > 0.5, gate_probs, jnp.zeros_like(gate_probs))
    else:
        gates = gate_probs
    msg = raw[..., 8:]
    return jnp.concatenate([move, gates, msg], axis=-1)
