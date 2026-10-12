"""Small, seeded benchmark for latent-field use and intervention response.

The benchmark is a contextual bandit, not a world-observation adapter. A hidden
binary field determines the correct action. Direct access reveals that field;
indirect access reveals a noisy proxy. Reward feedback lets a one-weight learner
adapt when the field-to-action relation is reversed.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from statistics import mean, pstdev

SCHEMA_VERSION = 1
_METRICS = (
    "field_use_gain",
    "intervention_adaptation",
    "pre_intervention_accuracy",
    "post_intervention_accuracy",
)


def _accuracy(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _evaluate_mode(
    rng: random.Random,
    *,
    access: str,
    repetitions: int,
    episodes: int,
    intervention_at: int,
    indirect_noise: float,
) -> dict[str, float]:
    field_gains: list[float] = []
    adaptations: list[float] = []
    pre_scores: list[float] = []
    post_scores: list[float] = []
    for _ in range(repetitions):
        weight = 1.0
        pre: list[float] = []
        post: list[float] = []
        no_access: list[float] = []
        for episode in range(episodes):
            latent = rng.choice((-1, 1))
            relation = 1 if episode < intervention_at else -1
            target = latent * relation
            cue = latent
            if access == "indirect" and rng.random() < indirect_noise:
                cue = -cue
            prediction = 1 if weight * cue >= 0 else -1
            reward = float(prediction == target)
            if not reward:
                weight += 2 * target * cue
            (pre if episode < intervention_at else post).append(reward)
            no_access.append(float(rng.choice((-1, 1)) == target))

        split = max(1, len(post) // 2)
        early_post = _accuracy(post[:split])
        late_post = _accuracy(post[split:])
        field_gains.append(_accuracy(pre) - _accuracy(no_access[:intervention_at]))
        adaptations.append(max(0.0, late_post - early_post))
        pre_scores.append(_accuracy(pre))
        post_scores.append(_accuracy(post))

    return {
        "field_use_gain": mean(field_gains),
        "intervention_adaptation": mean(adaptations),
        "pre_intervention_accuracy": mean(pre_scores),
        "post_intervention_accuracy": mean(post_scores),
    }


def run_evaluation(
    *,
    seed: int = 0,
    repetitions: int = 20,
    episodes: int = 100,
    intervention_at: int = 50,
    indirect_noise: float = 0.25,
) -> dict:
    """Run a reproducible direct-versus-indirect field-discovery evaluation."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if episodes < 2:
        raise ValueError("episodes must be at least 2")
    if not 1 <= intervention_at < episodes:
        raise ValueError("intervention_at must be between 1 and episodes - 1")
    if not 0.0 <= indirect_noise <= 1.0:
        raise ValueError("indirect_noise must be between 0 and 1")

    direct_rng = random.Random(seed)
    indirect_rng = random.Random(seed ^ 0x5EED)
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "repetitions": repetitions,
        "episodes": episodes,
        "intervention_at": intervention_at,
        "indirect_noise": indirect_noise,
        "access_modes": {
            "direct": _evaluate_mode(
                direct_rng,
                access="direct",
                repetitions=repetitions,
                episodes=episodes,
                intervention_at=intervention_at,
                indirect_noise=indirect_noise,
            ),
            "indirect": _evaluate_mode(
                indirect_rng,
                access="indirect",
                repetitions=repetitions,
                episodes=episodes,
                intervention_at=intervention_at,
                indirect_noise=indirect_noise,
            ),
        },
    }


def summarize_results(results: Iterable[dict]) -> dict:
    """Summarize evaluation runs with deterministic mean and population spread."""
    runs = list(results)
    if not runs:
        raise ValueError("at least one evaluation result is required")
    for result in runs:
        if result.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported field-discovery schema version")
        if set(result.get("access_modes", {})) != {"direct", "indirect"}:
            raise ValueError("each result must contain direct and indirect access modes")

    access_modes = {}
    for mode in ("direct", "indirect"):
        access_modes[mode] = {}
        for metric in _METRICS:
            values = [float(run["access_modes"][mode][metric]) for run in runs]
            access_modes[mode][metric] = {
                "mean": mean(values),
                "std": pstdev(values),
                "n": len(values),
            }
    return {"schema_version": SCHEMA_VERSION, "run_count": len(runs), "access_modes": access_modes}
