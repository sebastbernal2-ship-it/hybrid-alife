from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_alife.agents.embodied import apply_embodied_actions, embodied_action_dim
from hybrid_alife.experiments.latent_law_embodied import embodied_law_config
from hybrid_alife.experiments.runner import initialize_sim
from hybrid_alife.world.env import step_world
from hybrid_alife.world.sensing import crowding_context

SIZE = 21


def _config(chemical: bool):
    base = embodied_law_config(
        seed=0, generations=1, population=2, steps_per_generation=1, size=SIZE,
        proprioception=True)
    world = replace(base.world, concentration_decay=1.0, concentration_diffusion=0.2,
                    flow_advection_strength=0.0, flow_noise_std=0.0, toroidal=True)
    emb = replace(base.embodied, chemical_signaling=chemical, message_gate_threshold=0.0)
    return replace(base, world=world, embodied=emb)


def _emit(cfg):
    state = initialize_sim(cfg)
    pop = state.embodied
    cell = (SIZE // 2 + 0.5) / SIZE
    pop.positions = jnp.array([[cell, cell], [0.05, 0.05]], dtype=jnp.float32)
    pop.alive = jnp.array([True, False])
    pop.energy = jnp.full((2,), 20.0)
    state.world.concentration = jnp.zeros_like(state.world.concentration)
    actions = jnp.zeros((2, embodied_action_dim(cfg.embodied)), dtype=jnp.float32)
    actions = actions.at[:, 7].set(1.0)  # emit
    pop, world, _ = apply_embodied_actions(
        pop, actions, state.world, cfg.embodied, cfg.world, jax.random.PRNGKey(0))
    state.world = world
    return state, world


def test_emission_deposits_into_signal_channel_only_when_enabled():
    _, world = _emit(_config(True))
    assert float(world.concentration[..., 1].sum()) == 1.0
    assert float(world.concentration[..., 0].sum()) == 0.0
    _, legacy = _emit(_config(False))
    assert float(legacy.concentration.sum()) == 0.0


def test_signal_respects_light_cone_and_conserves_mass():
    cfg = _config(True)
    state, _ = _emit(cfg)
    centre = SIZE // 2
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    dist = np.maximum(abs(yy - centre), abs(xx - centre))  # 5-point stencil: <=1 cell/step
    for t in range(1, 6):
        state = step_world(state, cfg.world)
        sig = np.asarray(state.world.concentration[..., 1])
        assert sig[dist > t].max() == 0.0, f"signal outran diffusion at step {t}"
        assert abs(sig.sum() - 1.0) < 1e-5
    assert sig[centre, centre + 5] >= 0.0 and sig[centre, centre] < 1.0  # spread, not teleported


def test_instant_broadcast_is_removed_from_crowding_context():
    cfg = _config(True)
    pos = jnp.full((2, 2), 0.5)
    alive = jnp.array([True, True])
    msgs = jnp.ones((2, 4))
    crowd = crowding_context(pos, alive, msgs, jnp.zeros((SIZE, SIZE, 2)), SIZE, SIZE)
    assert float(crowd[0, 2]) > 0.0  # legacy path: neighbour in same cell is heard instantly
    state = initialize_sim(cfg)
    from hybrid_alife.world.sensing import sample_sixth_sense

    pop = state.embodied
    pop.positions, pop.alive, pop.messages = pos, alive, msgs
    off = replace(cfg.embodied, chemical_signaling=False)
    zero_noise = replace(cfg.world, sensory_noise_std=0.0)
    key = jax.random.PRNGKey(0)
    on = sample_sixth_sense(state.world, pos, alive, msgs, zero_noise, cfg.embodied, key)
    legacy = sample_sixth_sense(state.world, pos, alive, msgs, zero_noise, off, key)
    assert float(on[0, -1]) == 0.0 and float(legacy[0, -1]) > 0.0
