from __future__ import annotations

import jax
import numpy as np

from hybrid_alife.experiments.latent_law_embodied import (
    embodied_law_config,
    evaluate_frozen_alignment,
)
from hybrid_alife.experiments.latent_law_embodied_pg import (
    collect_episode,
    init_trainable_genomes,
    single_agent_config,
    train_policy_gradient,
)


def _tiny_config():
    return embodied_law_config(
        seed=0,
        generations=1,
        population=2,
        steps_per_generation=4,
        size=8,
        proprioception=True,
    )


def test_collect_episode_shapes() -> None:
    config = _tiny_config()
    genomes = init_trainable_genomes(config)
    batch = collect_episode(
        genomes,
        config,
        seed=0,
        direction=(1.0, 0.0),
        steps=4,
        sigma=0.4,
        key=jax.random.PRNGKey(0),
    )
    assert batch["observations"].shape[0] == 4
    assert batch["actions"].shape == (4, 2)
    assert batch["hiddens"].shape[0] == 4
    assert batch["rewards"].shape == (4,)
    assert np.all(np.isfinite(np.asarray(batch["actions"])))


def test_training_moves_alignment_up_on_the_training_direction() -> None:
    config = _tiny_config()
    trained, initial = train_policy_gradient(
        config,
        directions=((1.0, 0.0),),
        updates=30,
        episodes_per_update=4,
        steps=4,
        learning_rate=0.05,
        sigma=0.4,
    )
    eval_config = single_agent_config(config)
    trained_eval = evaluate_frozen_alignment(
        None, eval_config, (1.0, 0.0), steps=6, genomes=trained
    )
    initial_eval = evaluate_frozen_alignment(
        None, eval_config, (1.0, 0.0), steps=6, genomes=initial
    )
    assert trained_eval["mean_alignment"] > initial_eval["mean_alignment"] + 0.1
