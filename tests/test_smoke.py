from hybrid_alife.experiments.runner import initialize_sim, load_config, step_sim


def test_smoke_step():
    cfg = load_config("configs/base.yaml")
    state = initialize_sim(cfg)
    next_state = step_sim(state, cfg)
    assert next_state.step == 1
    assert next_state.world.resources.shape[0] == cfg.world.height

