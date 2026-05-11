# Synthetic Communication Benchmark

A controlled mini-task that verifies the communication metrics
(`topsim`, `posdis`, `bosdis`, `channel_capacity`) respond in the correct
direction on known inputs. It is **not** an agent training experiment — it
exists to keep the metric implementations honest.

## What is being measured

Meanings are enumerated over a small attribute grid (default: 3 attributes
each from a 4-symbol alphabet, replicated 4×). Three reference protocols
map meanings to messages with known properties:

| protocol         | expected topsim | expected posdis/bosdis | channel capacity |
|------------------|-----------------|------------------------|------------------|
| `compositional`  | ≈ 1.0           | ≈ 1.0                  | high             |
| `holistic`       | low             | near zero              | high (1:1 codes) |
| `random`         | floor           | floor                  | floor            |

Two ablations are applied to every protocol:

- **channel shuffle** — rows of the message matrix are permuted, decoupling
  meaning from message. Should collapse topsim and channel capacity.
- **channel zero** — all message symbols replaced with 0. Should drive
  posdis, bosdis, and channel capacity to floor (≈ 0).

The pairing of high channel capacity with low posdis/bosdis under the
holistic protocol is the headline reason the science memo asks for **all
three** disentanglement metrics rather than capacity alone.

## Running it

```bash
python scripts/run_comm_benchmark.py --out outputs/comm_benchmark.json
```

The script reads `configs/comm_task.yaml`, scores every protocol, prints a
table, optionally writes JSON, and exits non-zero if any acceptance
threshold fails. Threshold values live in the config under `acceptance:`.

The same checks are pinned by `tests/test_comm_benchmark.py`, which runs in
under a second.

## Language guardrail

This benchmark is a **functional regression test for the metric code**, not
evidence about emergent communication. When reporting results from agent
experiments, please observe:

- A high `topsim`/`posdis`/`bosdis` on the synthetic benchmark says the
  metric implementations are wired correctly. It does **not** mean any
  agent population has developed a language, protolanguage, or
  compositional protocol.
- Phrases such as "the agents developed a language" or "evidence of
  compositionality emerged" must be reserved for results from real
  simulation runs, scored against a neutral-shadow baseline (see
  `experiments/transfer.py`), and only after the shuffle/zero ablations
  rule out cue leakage.
- For agent results, always report the **adaptive** metric (run minus
  matched shadow) and the channel-ablation deltas alongside the headline
  number. A `topsim` that does not move when the channel is shuffled is
  measuring something other than communication.
- The compositionality metrics are sample-size sensitive. Topsim
  subsamples internally; posdis and bosdis use a plug-in MI estimator that
  is biased upward on small samples. Note the sample size used.

In short: this benchmark verifies the ruler. It does not measure anything
about the agents.
