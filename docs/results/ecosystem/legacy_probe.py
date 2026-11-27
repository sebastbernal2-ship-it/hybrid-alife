"""Does the legacy physics-study setup sustain turnover without generation selection?"""

import dataclasses as dc

from hybrid_alife.experiments.runner import initialize_sim, load_config, step_sim

base = load_config("configs/physics_evolution_study.yaml")
base = dc.replace(
    base,
    avida=dc.replace(base.avida, enabled=False),
    embodied=dc.replace(base.embodied, energy_ledger=True),
)
for seed in range(3):
    cfg = dc.replace(base, seed=seed)
    state, counter, births, deaths = initialize_sim(cfg), 32, 0, 0
    for step in range(1, 3001):
        state, counter = step_sim(state, cfg, counter)
        if step % 500 == 0:
            alive = state.embodied.alive
            print(
                "seed", seed, "step", step, "alive", int(alive.sum()),
                "births", state.embodied_births_this_gen - births,
                "deaths", state.embodied_deaths_this_gen - deaths,
                flush=True,
            )
            births = state.embodied_births_this_gen
            deaths = state.embodied_deaths_this_gen
