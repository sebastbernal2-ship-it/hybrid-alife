"""Synthetic communication benchmark for compositionality metrics.

A controlled mini-task where meanings live on a small attribute grid and a
known protocol maps them to messages. The point is not to train agents — it
is to give the metrics in :mod:`hybrid_alife.metrics.communication` a target
they should certify as compositional (or not), so a regression in
``topsim``/``posdis``/``bosdis``/``channel_capacity`` is loud.

Three reference protocols are provided:

- ``"compositional"`` — each attribute is encoded in its own message position
  with an attribute-specific symbol offset. This is the maximally
  compositional protocol on the grid and should drive all three
  disentanglement metrics toward 1.0.
- ``"holistic"``  — every distinct meaning gets a unique random message that
  shares no structure with neighbouring meanings. Topsim collapses;
  posdis/bosdis sit near zero.
- ``"random"``    — messages are sampled iid from the vocabulary regardless
  of meaning. All metrics should be at floor.

The benchmark also exposes the canonical ablations used in the science memo:
channel shuffle and channel zero. ``run_benchmark`` returns a dict suitable
for direct comparison in tests and for dumping to JSON from a runner script.

No agent simulation, no JAX, no I/O — this module is intentionally cheap so
it can run inside the unit-test suite and as a smoke check in CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .communication import (
    bosdis,
    channel_capacity,
    posdis,
    shuffle_channel,
    topsim,
    zero_channel,
)


@dataclass(frozen=True)
class BenchmarkSpec:
    """Grid of meanings to enumerate.

    n_attr attributes each drawn from ``range(attr_vocab)``. The grid is
    fully enumerated (cardinality ``attr_vocab ** n_attr``) and optionally
    replicated ``replicates`` times to give the metric estimators enough
    samples — topsim subsamples internally but posdis/bosdis benefit from
    multiple draws per meaning under noisy protocols.
    """

    n_attr: int = 3
    attr_vocab: int = 4
    msg_vocab: int = 8
    msg_len: int | None = None  # defaults to n_attr for compositional protocol
    replicates: int = 4
    noise: float = 0.0  # symbol-flip probability on emitted messages
    seed: int = 0

    @property
    def effective_msg_len(self) -> int:
        return self.msg_len if self.msg_len is not None else self.n_attr


def enumerate_meanings(spec: BenchmarkSpec) -> np.ndarray:
    """Return every attribute combination, tiled ``replicates`` times."""
    grids = np.meshgrid(
        *[np.arange(spec.attr_vocab) for _ in range(spec.n_attr)], indexing="ij"
    )
    flat = np.stack([g.ravel() for g in grids], axis=-1)  # (V**n_attr, n_attr)
    return np.tile(flat, (spec.replicates, 1))


def _apply_noise(messages: np.ndarray, spec: BenchmarkSpec, rng: np.random.Generator) -> np.ndarray:
    if spec.noise <= 0.0:
        return messages
    flip = rng.random(messages.shape) < spec.noise
    rand = rng.integers(0, spec.msg_vocab, size=messages.shape)
    return np.where(flip, rand, messages)


def compositional_protocol(meanings: np.ndarray, spec: BenchmarkSpec) -> np.ndarray:
    """One symbol per attribute, with a per-position offset so positions do
    not collide. Requires ``msg_vocab >= attr_vocab * n_attr`` to be lossless;
    when smaller the protocol still encodes information but topsim drops.
    """
    rng = np.random.default_rng(spec.seed)
    n, n_attr = meanings.shape
    msg_len = spec.effective_msg_len
    msg = np.zeros((n, msg_len), dtype=np.int64)
    for j in range(msg_len):
        attr = j % n_attr
        offset = (attr * spec.attr_vocab) % spec.msg_vocab
        msg[:, j] = (meanings[:, attr] + offset) % spec.msg_vocab
    return _apply_noise(msg, spec, rng)


def holistic_protocol(meanings: np.ndarray, spec: BenchmarkSpec) -> np.ndarray:
    """Each unique meaning -> a random message; no compositional structure."""
    rng = np.random.default_rng(spec.seed + 17)
    msg_len = spec.effective_msg_len
    unique, inverse = np.unique(meanings, axis=0, return_inverse=True)
    codebook = rng.integers(0, spec.msg_vocab, size=(unique.shape[0], msg_len))
    msg = codebook[inverse]
    return _apply_noise(msg, spec, rng)


def random_protocol(meanings: np.ndarray, spec: BenchmarkSpec) -> np.ndarray:
    """Messages independent of meanings — the floor case."""
    rng = np.random.default_rng(spec.seed + 99)
    msg_len = spec.effective_msg_len
    return rng.integers(0, spec.msg_vocab, size=(meanings.shape[0], msg_len)).astype(np.int64)


PROTOCOLS = {
    "compositional": compositional_protocol,
    "holistic": holistic_protocol,
    "random": random_protocol,
}


@dataclass
class BenchmarkResult:
    protocol: str
    metrics: dict[str, float]
    controls: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "protocol": self.protocol,
            "metrics": self.metrics,
            "controls": self.controls,
        }


def _score(meanings: np.ndarray, messages: np.ndarray) -> dict[str, float]:
    # Pick a referent column for channel_capacity; the first attribute is
    # the canonical "what is being referred to" by convention.
    referent = meanings[:, 0]
    return {
        "topsim": topsim(meanings, messages),
        "posdis": posdis(meanings, messages),
        "bosdis": bosdis(meanings, messages),
        "channel_capacity": channel_capacity(messages, referent),
    }


def run_benchmark(
    spec: BenchmarkSpec | None = None,
    protocols: list[str] | None = None,
    include_controls: bool = True,
) -> dict[str, BenchmarkResult]:
    """Run all requested protocols and return their metric dicts.

    Each protocol is scored both directly and under the two ablations
    (channel shuffle, channel zero) so callers can verify the metrics move
    in the expected direction.
    """
    spec = spec or BenchmarkSpec()
    protocols = protocols or list(PROTOCOLS)
    meanings = enumerate_meanings(spec)
    rng = np.random.default_rng(spec.seed + 1)
    out: dict[str, BenchmarkResult] = {}
    for name in protocols:
        if name not in PROTOCOLS:
            raise ValueError(f"unknown protocol {name!r}; valid: {sorted(PROTOCOLS)}")
        messages = PROTOCOLS[name](meanings, spec)
        result = BenchmarkResult(protocol=name, metrics=_score(meanings, messages))
        if include_controls:
            result.controls["shuffle"] = _score(meanings, shuffle_channel(messages, rng))
            result.controls["zero"] = _score(meanings, zero_channel(messages))
        out[name] = result
    return out


def summarise(results: dict[str, BenchmarkResult]) -> dict[str, dict[str, float]]:
    """Flatten run_benchmark output to a JSON-friendly dict-of-dicts."""
    flat: dict[str, dict[str, float]] = {}
    for name, res in results.items():
        flat[name] = dict(res.metrics)
        for ctrl_name, ctrl_metrics in res.controls.items():
            for k, v in ctrl_metrics.items():
                flat[name][f"{ctrl_name}.{k}"] = v
    return flat
