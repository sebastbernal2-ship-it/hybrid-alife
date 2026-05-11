"""Emergent-communication compositionality metrics.

Implements the three field-standard metrics from the memo:

- **topsim** (topographic similarity): Spearman correlation between meaning-
  distance and message-distance matrices (Brighton & Kirby, 2006).
- **posdis** (positional disentanglement): per-position mutual-information
  gap (Chaabouni et al., 2020, arXiv:2004.09124).
- **bosdis** (bag-of-symbols disentanglement): symbol-count MI gap.

Plus helpers for the channel-shuffle and channel-zero ablations recommended
by the memo. Inputs are NumPy arrays so this module can be called from any
analysis script regardless of whether the agents are using JAX.

Conventions
-----------
- meanings: int array of shape (N, n_attr), each column ranges over a small
  discrete attribute alphabet (e.g. nearest-resource-direction).
- messages: int array of shape (N, msg_len), discretised tokens. If the
  agents emit continuous messages, the caller is expected to quantise first.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Distance and rank helpers
# ---------------------------------------------------------------------------


def _hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a[:, None, :] != b[None, :, :]).sum(axis=-1).astype(np.float64)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    rx = _rank(x)
    ry = _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    if denom <= 0:
        return 0.0
    return float((rx * ry).sum() / denom)


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(a.size)
    return ranks


# ---------------------------------------------------------------------------
# Information theory
# ---------------------------------------------------------------------------


def _entropy(x: np.ndarray) -> float:
    _, counts = np.unique(x, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def _mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    # Joint and marginal histograms over integer-coded variables.
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    xy = np.stack([x, y], axis=-1)
    _, joint_counts = np.unique(xy, axis=0, return_counts=True)
    p_xy = joint_counts / joint_counts.sum()
    _, xc = np.unique(x, return_counts=True)
    _, yc = np.unique(y, return_counts=True)
    p_x = xc / xc.sum()
    p_y = yc / yc.sum()
    h_x = -np.sum(p_x * np.log(p_x + 1e-12))
    h_y = -np.sum(p_y * np.log(p_y + 1e-12))
    h_xy = -np.sum(p_xy * np.log(p_xy + 1e-12))
    return float(h_x + h_y - h_xy)


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------


def topsim(meanings: np.ndarray, messages: np.ndarray, sample: int = 256) -> float:
    """Spearman rank correlation between meaning- and message-distance vectors.

    Subsamples to `sample` rows for tractability — topsim on large populations
    is O(N^2) and the memo only requires a representative estimate.
    """
    meanings = np.atleast_2d(meanings).astype(np.int64)
    messages = np.atleast_2d(messages).astype(np.int64)
    n = meanings.shape[0]
    if n < 4:
        return 0.0
    if n > sample:
        idx = np.random.default_rng(0).choice(n, size=sample, replace=False)
        meanings = meanings[idx]
        messages = messages[idx]
    dm = _hamming(meanings, meanings)
    dmsg = _hamming(messages, messages)
    iu = np.triu_indices_from(dm, k=1)
    return _spearman(dm[iu], dmsg[iu])


def posdis(meanings: np.ndarray, messages: np.ndarray) -> float:
    """Positional disentanglement (Chaabouni et al., 2020).

    For each message position j, find the two attributes with highest MI(s_j;
    a_i); posdis is the mean over j of (MI_best - MI_second) / H(s_j).
    """
    meanings = np.atleast_2d(meanings).astype(np.int64)
    messages = np.atleast_2d(messages).astype(np.int64)
    if messages.shape[0] < 4 or meanings.shape[1] == 0:
        return 0.0
    gaps = []
    for j in range(messages.shape[1]):
        s_j = messages[:, j]
        h_sj = _entropy(s_j)
        if h_sj <= 0:
            continue
        mis = np.array(
            [_mutual_information(s_j, meanings[:, i]) for i in range(meanings.shape[1])]
        )
        if mis.size < 2:
            mis = np.concatenate([mis, [0.0]])
        order = np.argsort(mis)[::-1]
        gap = (mis[order[0]] - mis[order[1]]) / h_sj
        gaps.append(gap)
    if not gaps:
        return 0.0
    return float(np.mean(gaps))


def bosdis(meanings: np.ndarray, messages: np.ndarray, vocab_size: int | None = None) -> float:
    """Bag-of-symbols disentanglement (Chaabouni et al., 2020).

    Like posdis but on per-symbol counts (order-free); useful for permutation-
    invariant protocols.
    """
    meanings = np.atleast_2d(meanings).astype(np.int64)
    messages = np.atleast_2d(messages).astype(np.int64)
    if messages.shape[0] < 4 or meanings.shape[1] == 0:
        return 0.0
    if vocab_size is None:
        vocab_size = int(messages.max()) + 1
    # Bag-of-symbols representation: (N, vocab_size)
    bag = np.zeros((messages.shape[0], vocab_size), dtype=np.int64)
    for k in range(vocab_size):
        bag[:, k] = (messages == k).sum(axis=1)
    gaps = []
    for k in range(vocab_size):
        c_k = bag[:, k]
        h_ck = _entropy(c_k)
        if h_ck <= 0:
            continue
        mis = np.array(
            [_mutual_information(c_k, meanings[:, i]) for i in range(meanings.shape[1])]
        )
        if mis.size < 2:
            mis = np.concatenate([mis, [0.0]])
        order = np.argsort(mis)[::-1]
        gap = (mis[order[0]] - mis[order[1]]) / h_ck
        gaps.append(gap)
    if not gaps:
        return 0.0
    return float(np.mean(gaps))


def channel_capacity(messages: np.ndarray, referents: np.ndarray) -> float:
    """Lower bound on I(message; referent) by treating each as a discrete variable.

    `messages` may be 1D or 2D — for 2D we hash rows to a single integer label.
    """
    messages = np.atleast_2d(messages).astype(np.int64)
    if messages.shape[1] > 1:
        flat = np.ascontiguousarray(messages).view(np.dtype((np.void, messages.dtype.itemsize * messages.shape[1])))
        flat = np.unique(flat, return_inverse=True)[1]
    else:
        flat = messages.flatten()
    return _mutual_information(flat, np.asarray(referents).astype(np.int64).flatten())


def shuffle_channel(messages: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a channel-shuffle ablation: rows permuted so message no longer
    pairs with its meaning. Recommended baseline for `cue leakage` check."""
    rng = rng or np.random.default_rng(0)
    out = np.array(messages, copy=True)
    rng.shuffle(out, axis=0)
    return out


def zero_channel(messages: np.ndarray) -> np.ndarray:
    return np.zeros_like(messages)


def comm_summary(meanings: np.ndarray, messages: np.ndarray) -> dict[str, float]:
    """Compose the three headline compositionality metrics."""
    return {
        "topsim": topsim(meanings, messages),
        "posdis": posdis(meanings, messages),
        "bosdis": bosdis(meanings, messages),
    }
