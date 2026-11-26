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
    emb = replace(
        base.embodied, chemical_signaling=chemical, message_gate_threshold=0.0,
        hidden_field_groups=())  # the law config hides concentration; the signal must be visible
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


def test_chemical_signaling_rejects_configs_where_the_signal_is_unobservable():
    import pytest

    from hybrid_alife.agents.embodied import initialize_embodied_population

    cfg = _config(True)
    key = jax.random.PRNGKey(0)
    initialize_embodied_population(cfg.embodied, cfg.world, key)  # valid config works
    for kwargs, match in (
        ({"blind": True}, "observable"),
        ({"hidden_field_groups": ("concentration",)}, "observable"),
        ({"hidden_field_groups": ("all",)}, "observable"),
        ({"use_comms": False}, "use_comms"),
        ({"signal_deposit": -1.0}, "signal_deposit"),
    ):
        with pytest.raises(ValueError, match=match):
            initialize_embodied_population(replace(cfg.embodied, **kwargs), cfg.world, key)
    # Legacy broadcast is unaffected by the same settings.
    off = replace(cfg.embodied, chemical_signaling=False, blind=True)
    initialize_embodied_population(off, cfg.world, key)


def test_other_hidden_groups_leave_the_signal_observable():
    from hybrid_alife.agents.embodied import initialize_embodied_population

    cfg = _config(True)
    emb = replace(cfg.embodied, hidden_field_groups=("flow", "metabolites"))
    initialize_embodied_population(emb, cfg.world, jax.random.PRNGKey(0))


def test_a_neighbour_senses_the_diffused_signal_through_the_sixth_sense():
    from hybrid_alife.world.sensing import sample_sixth_sense

    cfg = _config(True)
    state, _ = _emit(cfg)
    cell = (SIZE // 2 + 0.5) / SIZE
    near = (SIZE // 2 + 1.5) / SIZE  # one cell away along axis 0
    positions = jnp.array([[cell, cell], [near, cell]], dtype=jnp.float32)
    alive = jnp.array([True, True])
    msgs = jnp.zeros((2, cfg.embodied.message_size))
    world_cfg = replace(cfg.world, sensory_noise_std=0.0)
    key = jax.random.PRNGKey(0)
    before = sample_sixth_sense(state.world, positions, alive, msgs, world_cfg, cfg.embodied, key)
    assert float(before[1, 7]) == 0.0  # nothing has reached the neighbour yet
    state = step_world(state, cfg.world)
    after = sample_sixth_sense(state.world, positions, alive, msgs, world_cfg, cfg.embodied, key)
    np.testing.assert_allclose(after[1, 7], 0.2, rtol=1e-4)  # one diffusion step, coeff 0.2
    assert float(after[0, 7]) > 0.0
    hidden = replace(cfg.embodied, hidden_field_groups=("concentration",))
    blind = sample_sixth_sense(state.world, positions, alive, msgs, world_cfg, hidden, key)
    assert float(blind[1, 7]) == 0.0  # why construction rejects this combination


def test_a_receiver_decodes_message_content_after_diffusion_and_advection():
    from hybrid_alife.world.sensing import sample_sixth_sense, sixth_sense_dim

    base = _config(True)
    world_cfg = replace(base.world, concentration_channels=4, sensory_noise_std=0.0,
                        concentration_decay=0.95, flow_advection_strength=0.5)
    emb = replace(base.embodied, signal_content=True, message_size=3)
    cfg = replace(base, world=world_cfg, embodied=emb)
    cell = (SIZE // 2 + 0.5) / SIZE
    listener = jnp.array([[cell, cell], [(SIZE // 2 + 2.5) / SIZE, cell]], dtype=jnp.float32)
    start = 8 + int(cfg.embodied.probe_enabled)
    assert sixth_sense_dim(world_cfg) == 13
    for message in ([1.0, 0.0, 0.0], [0.2, -0.7, 0.4], [-0.5, 0.5, 1.0]):
        state = initialize_sim(cfg)
        pop = state.embodied
        pop.positions = jnp.array([[cell, cell], [0.05, 0.05]], dtype=jnp.float32)
        pop.alive = jnp.array([True, False])
        state.world.concentration = jnp.zeros_like(state.world.concentration)
        state.world.flow = jnp.broadcast_to(jnp.array([0.4, -0.2]), state.world.flow.shape)
        actions = jnp.zeros((2, embodied_action_dim(cfg.embodied)), dtype=jnp.float32)
        actions = actions.at[0, 7].set(1.0).at[0, start : start + 3].set(jnp.array(message))
        _, state.world, _ = apply_embodied_actions(
            pop, actions, state.world, cfg.embodied, world_cfg, jax.random.PRNGKey(0))
        for _ in range(4):
            state = step_world(state, world_cfg)
        sense = sample_sixth_sense(
            state.world, listener, jnp.array([True, True]),
            jnp.zeros((2, cfg.embodied.message_size)), world_cfg, cfg.embodied,
            jax.random.PRNGKey(0))
        # Layout: curvature 1, shear_grad 2, enrichment 1, lift 2, then concentration.
        heard = np.asarray(sense[1, 6:10])
        assert heard[0] == 0.0  # channel 0 is not part of the message
        got = heard[1:]
        assert np.linalg.norm(got) > 1e-4
        # Diffusion and advection attenuate every channel alike, so direction survives.
        np.testing.assert_allclose(
            got / np.linalg.norm(got), np.array(message) / np.linalg.norm(message), atol=1e-4)


def test_extra_channels_run_end_to_end_and_content_is_validated():
    import pytest

    from hybrid_alife.agents.embodied import (
        initialize_embodied_population,
        observe_embodied,
    )
    from hybrid_alife.experiments.runner import step_sim, validate_runtime_invariants

    base = _config(True)
    world_cfg = replace(base.world, concentration_channels=4)
    emb = replace(base.embodied, signal_content=True, message_size=3)
    cfg = replace(base, world=world_cfg, embodied=emb)
    base = replace(base, embodied=replace(base.embodied, message_size=3))
    state = initialize_sim(cfg)
    default_state = initialize_sim(base)
    side = 2 * cfg.embodied.obs_radius + 1

    def width(s, c):
        key = jax.random.PRNGKey(0)
        return observe_embodied(s.embodied, s.world, c.embodied, c.world, key).shape[1]

    assert width(state, cfg) - width(default_state, base) == 2 * side * side + 2
    for _ in range(2):
        state, _ = step_sim(state, cfg, 2)
    validate_runtime_invariants(state, cfg)
    key = jax.random.PRNGKey(0)
    for emb, wc, match in (
        (replace(cfg.embodied, chemical_signaling=False), world_cfg, "chemical_signaling"),
        (replace(cfg.embodied, message_size=2), world_cfg, "message_size"),
    ):
        with pytest.raises(ValueError, match=match):
            initialize_embodied_population(emb, wc, key)
    with pytest.raises(ValueError, match="concentration_channels"):
        replace(world_cfg, concentration_channels=1)
