from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from hybrid_alife.experiments.latent_law_embodied import (
    embodied_law_config,
    minimum_direction_alignment,
    run_embodied_law_pilot,
)
from hybrid_alife.experiments.runner import initialize_sim

CARDINALS = ((0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0))


def _config():
    return embodied_law_config(
        seed=0,
        generations=1,
        population=2,
        steps_per_generation=1,
        size=10,
        proprioception=True,
        linear_policy=True,
        unit_flow_features=True,
        initial_alive_fraction=1.0,
    )


def _hand_set_genomes(config):
    state = initialize_sim(config)
    pop = state.embodied
    obs_dim = pop.genomes["w_linear"].shape[1]
    action_dim = pop.genomes["w_linear"].shape[2]
    weights = jnp.zeros((2, obs_dim, action_dim))
    bias = jnp.zeros((2, action_dim))
    # Slot 0 follows the unit flow, which is the last feature pair.
    weights = weights.at[0, -2, 0].set(1.0).at[0, -1, 1].set(1.0)
    # Slot 1 pushes +x regardless of the law.
    bias = bias.at[1, 0].set(1.0)
    pop.genomes = {"w_linear": weights, "b_linear": bias}
    return state


def test_min_alignment_fitness_prefers_a_flow_follower() -> None:
    config = _config()
    state = _hand_set_genomes(config)
    fitness = minimum_direction_alignment(state, config, CARDINALS, steps=8)
    fitness = np.asarray(fitness)
    assert fitness.shape == (2,)
    assert fitness[0] > 0.8, fitness
    assert fitness[1] < -0.5, fitness
    assert fitness[0] > fitness[1] + 1.0


def test_min_alignment_pilot_runs() -> None:
    result = run_embodied_law_pilot(
        seed=0,
        directions=CARDINALS,
        generations=2,
        population=4,
        steps_per_generation=1,
        size=8,
        proprioception=True,
        linear_policy=True,
        unit_flow_features=True,
        fitness_mode="min_alignment",
        fitness_steps=4,
    )
    assert result["status"] == "pilot_only"
    assert np.isfinite(result["training_alignment"]).all()
