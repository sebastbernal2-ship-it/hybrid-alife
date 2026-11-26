"""Core dataclasses and typed state containers for hybrid-alife."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import jax

Array = jax.Array
PRNGKey = Array


class Branch(IntEnum):
    EMBODIED = 0
    AVIDA = 1


class EmbodiedActionIndex(IntEnum):
    MOVE_X = 0
    MOVE_Y = 1
    EAT = 2
    ATTACK = 3
    REPRODUCE = 4
    TERRAFORM_RESOURCE = 5
    TERRAFORM_HAZARD = 6
    EMIT_MESSAGE = 7


@dataclass(frozen=True)
class WorldConfig:
    width: int
    height: int
    toroidal: bool
    resource_channels: int
    hazard_channels: int
    max_agents: int
    max_digital_organisms: int
    flow_noise_std: float
    concentration_decay: float
    concentration_diffusion: float
    drifting: bool = False
    drift_speed: float = 0.02
    sensory_noise_std: float = 0.05
    resource_regen: float = 0.005
    hazard_decay: float = 0.99
    metabolite_channels: int = 2
    metabolite_decay: float = 0.95
    metabolite_diffusion: float = 0.05
    # ---- proxy microfluidics extensions ----
    # Curvature regime: "default" (legacy sin/cos), "straight", "spiral",
    # "serpentine", "obstacle", "random_smooth".
    curvature_regime: str = "default"
    curvature_amplitude: float = 1.0
    # Wavenumber for serpentine / spiral patterns (cycles across the domain).
    curvature_wavenumber: float = 2.0
    # Wall repulsion strength applied to the inertial-lift proxy near domain
    # boundaries (only active when toroidal=False); 0 disables.
    wall_repulsion_strength: float = 0.5
    # Strength of Dean-like secondary flow added on top of the primary flow.
    dean_secondary_strength: float = 0.3
    # Coupling between flow and concentration: per-step advection coefficient
    # in [0, 1]; 0 disables advection.
    flow_advection_strength: float = 0.25
    # Optional flow advection of resources. Zero preserves legacy dynamics.
    resource_advection_strength: float = 0.0
    # Smooth temporal drift parameters (sinusoidal rotation + low-frequency
    # modulation, in addition to the existing Gaussian flow_noise_std term).
    drift_rotation_rate: float = 0.01
    drift_modulation_freq: float = 0.02
    # If True, occupancy pressure subtracts from the enrichment field so that
    # crowded cells get an enrichment penalty (more biologically faithful to
    # crowding-induced margination decline).
    enrichment_uses_occupancy: bool = True
    # "proxy" keeps the stylised Dean-like flow; "lbm" solves channel flow with the
    # native D2Q9 lattice-Boltzmann kernel (obstacle if curvature_regime == "obstacle").
    flow_solver: str = "proxy"
    # "lbm_live" (flow_solver): a stateful LBM fluid advanced `lbm_substeps` steps per world
    # step, flow = lattice velocity / lbm_peak_speed. lbm_agent_drag > 0 makes agents push
    # the fluid with Stokes drag (two-way coupling); 0 leaves the fluid unaffected.
    lbm_viscosity: float = 0.1
    lbm_peak_speed: float = 0.05
    lbm_substeps: int = 10
    lbm_agent_drag: float = 0.0
    # "monopole": net force on the fluid (thrust treated as external). "dipole": a force-free
    # swimmer, two equal and opposite point forces lbm_dipole_length cells apart along the
    # swim direction (pusher: reaction ahead of the body, thrust behind). Net force is zero.
    lbm_agent_coupling: str = "monopole"
    # "periodic": wall-free box carrying a uniform law (see LiveLBM.uniform); "channel" default.
    lbm_box: str = "channel"
    # Periodic box only: relaxation time (LBM steps) of the mean flow toward the imposed law.
    lbm_law_relax_steps: float = 0.0
    lbm_dipole_length: float = 2.0
    # Chemical species in `concentration`. Channel 0 carries terraforming and the Avida
    # coupling; channels 1.. are free for chemical signals. Changes observation widths.
    concentration_channels: int = 2

    def __post_init__(self) -> None:
        if self.concentration_channels < 2:
            raise ValueError("concentration_channels must be >= 2")
        if self.flow_solver not in ("proxy", "lbm", "lbm_live"):
            raise ValueError(f"unknown flow_solver={self.flow_solver!r} (proxy, lbm or lbm_live)")
        if self.lbm_agent_coupling not in ("monopole", "dipole"):
            raise ValueError(
                f"unknown lbm_agent_coupling={self.lbm_agent_coupling!r} (monopole or dipole)"
            )
        if self.flow_solver != "lbm_live":
            return
        if self.lbm_viscosity <= 0.0:
            raise ValueError("lbm_viscosity must be > 0 (relaxation time tau = 3 nu + 0.5)")
        if not 0.0 < self.lbm_peak_speed <= 0.3:
            raise ValueError("lbm_peak_speed must be in (0, 0.3] lattice units (low Mach)")
        if self.lbm_substeps < 1:
            raise ValueError("lbm_substeps must be >= 1")
        if self.lbm_agent_drag < 0.0 or self.lbm_dipole_length <= 0.0:
            raise ValueError("lbm_agent_drag must be >= 0 and lbm_dipole_length > 0")
        if min(self.height, self.width) < 3:
            raise ValueError("lbm_live needs a world of at least 3x3 cells")
        # These would be silently ignored: the live fluid replaces the flow each step.
        if self.drifting or self.flow_noise_std != 0.0:
            raise ValueError("lbm_live ignores drifting and flow_noise_std; set them to off/0")


@dataclass(frozen=True)
class EmbodiedConfig:
    enabled: bool
    population_size: int
    obs_radius: int
    hidden_size: int
    message_size: int
    genome_mutation_std: float
    genome_mutation_prob: float
    initial_energy: float
    reproduce_energy_threshold: float
    reproduction_energy_cost: float
    basal_metabolic_cost: float
    use_memory: bool = True
    use_comms: bool = True
    blind: bool = False
    # Hide named world fields from the observation while keeping their dynamics.
    # Empty means that all existing fields remain visible.
    hidden_field_groups: tuple[str, ...] = ()
    shuffle_sixth_sense: bool = False
    # Spatially translate observed flow vectors while preserving their values.
    shuffle_flow_observation: bool = False
    true_sixth_sense: bool = True
    action_history_len: int = 16
    # ---- sensing uncertainty / delay modes (from world-sixth-sense-depth) ----
    sense_delayed: bool = False
    sense_delay_steps: int = 1
    sense_multiplicative_noise_std: float = 0.0
    sense_dropout_frac: float = 0.0
    # ---- controller-level knobs (from agents-vm-depth) ----
    stochastic_actions: bool = False
    action_temperature: float = 1.0
    message_gate_threshold: float = 0.5
    cost_move_scale: float = 0.5
    cost_eat: float = 0.01
    cost_attack: float = 0.08
    cost_reproduce: float = 0.05
    cost_terraform: float = 0.03
    cost_emit: float = 0.02
    metabolite_deposit_scale: float = 1.0
    hazard_deposit_scale: float = 1.0
    flow_movement_scale: float = 0.0
    # Optional reward for moving with the hidden local flow direction.
    flow_alignment_reward: float = 0.0
    # Append previous actual movement and previous commanded thrust to the
    # observation so an organism can infer hidden advection from its own motion.
    proprioception: bool = False
    # Reward dot(actual_displacement, unit_flow). Disabled by default.
    displacement_reward: float = 0.0
    # Apply a small mutation to survivors each generation, not only to dead slots.
    mutate_survivors: bool = False
    # Flat linear controller: raw = obs @ w_linear + b_linear, no GRU.
    linear_policy: bool = False
    # Append the unit flow estimate, (previous_velocity - previous_thrust) / norm,
    # so the flow signal is order one and scale-free. Needs proprioception.
    unit_flow_features: bool = False
    # Generational replacement by fitness rank: keep the top elite_fraction and
    # replace the rest with mutated copies of the elite, every generation.
    rank_replacement: bool = False
    # Fraction of the bottom slots that receive fresh random genomes (immigrants)
    # instead of mutated elite copies. Needs rank_replacement.
    immigrant_fraction: float = 0.0
    # Four-parameter movement head: raw_move = w_flow @ flow_features + b_flow.
    # Gates and messages are zero. Needs unit_flow_features or raw_flow_features.
    flow_head: bool = False
    # Append the raw flow estimate (previous_velocity - previous_thrust), not
    # normalized. Used to ablate the unit-flow normalization.
    raw_flow_features: bool = False
    # Morphology genes: body radius and aspect. Off by default so the D1 studies
    # keep their exact dynamics.
    body_morphology: bool = False
    # Stokes-drag mechanics: actions are thrust, drag grows with body radius, so swim speed
    # is thrust / radius and movement cost is the dissipated power (radius * speed^2).
    # Normalised so radius 1 at full thrust matches the legacy linear move cost.
    stokes_mechanics: bool = False
    # With stokes_mechanics and body_morphology: body_aspect gives anisotropic spheroid
    # drag (physics.spheroid_drag); the body axis follows the previous step's thrust.
    aspect_drag: bool = False
    # Messages travel as a diffusing chemical signal (concentration channel 1) instead of
    # an instantaneous same-cell broadcast; speed, range, and decay follow the world's
    # concentration diffusion/decay/advection.
    # The signal is an intensity (emit gate x signal_deposit), not message content; the emitter
    # also senses its own deposit; with concentration_decay == 1 it accumulates without bound.
    chemical_signaling: bool = False
    signal_deposit: float = 1.0
    # With chemical_signaling: each signal channel k >= 1 receives
    # signal_deposit * message[k - 1] instead of a bare intensity in channel 1.
    signal_content: bool = False
    # Thermodynamic closure when > 0: fraction of eaten mass converted to energy; the rest
    # returns to the world as metabolite waste, and shared-cell eating cannot overdraw. The
    # closure covers eating only: energy_ledger tracks rewards, predation and max_energy;
    # Avida metabolite uptake is outside both.
    thermo_efficiency: float = 0.0
    # Keep cumulative energy totals (dissipated heat etc.) on the population state; see
    # metrics.energy_ledger. Bookkeeping only, except that reproduction then uses a
    # duplicate-free scatter (the legacy one can overwrite slot 0 and skip its payment).
    energy_ledger: bool = False
    # Flow head only: a constant eat gate, so a movement-only brain can survive in
    # a world where it must eat. 0 keeps the legacy flow-head behavior.
    flow_head_eat_gate: float = 0.0
    # Append the local food gradient so a movement head can add a food-seeking
    # reflex. Off by default.
    food_gradient_features: bool = False
    food_gradient_gain: float = 0.0
    # World balance: soft energy ceiling (0 = off) and predation.
    max_energy: float = 0.0
    predation_rate: float = 0.0
    predation_radius: float = 0.04
    # Fraction of the energy drained from prey that the attacker receives; the rest is
    # dissipated. 1 keeps the legacy lossless transfer.
    predation_efficiency: float = 1.0
    # Prey escape reflex: move away from the nearest neighbour inside the radius.
    escape_gain: float = 0.0
    escape_radius: float = 0.06
    # Optional active-probe gate. Disabled by default to preserve legacy shapes.
    probe_enabled: bool = False
    probe_cost: float = 0.02
    probe_signal_scale: float = 1.0
    # Controller architecture: "gru" (default) or "mlp" for ablations.
    controller_arch: str = "gru"
    # Fraction of slots initialized alive. Setting < 1.0 leaves dead slots
    # available so reproduction has somewhere to place children right away.
    initial_alive_fraction: float = 1.0
    # Reproduce-gate threshold (sigmoid output must exceed this to trigger a
    # birth). Lower it for "births-friendly" tiny CPU configs.
    repro_gate_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.aspect_drag and not (self.stokes_mechanics and self.body_morphology):
            raise ValueError("aspect_drag needs stokes_mechanics and body_morphology")
        if not 0.0 <= self.predation_efficiency <= 1.0:
            raise ValueError("predation_efficiency must be in [0, 1]")
        # Efficiency above 1 would deposit negative waste, i.e. create matter and energy.
        if not 0.0 <= self.thermo_efficiency <= 1.0:
            raise ValueError("thermo_efficiency must be in [0, 1] (0 disables the closure)")


@dataclass(frozen=True)
class AvidaConfig:
    enabled: bool
    population_size: int
    genome_length: int
    max_genome_length: int
    registers: int
    memory_size: int
    cycles_per_update: int
    point_mutation_prob: float
    insertion_prob: float
    deletion_prob: float
    # Extended VM knobs. All defaults preserve existing behaviour numerically.
    min_genome_length: int = 4
    duplication_prob: float = 0.0
    # Merit-based CPU-cycle allocation: each organism gets cycles_per_update *
    # (merit / merit_floor)^merit_cycle_exponent extra cycles, clamped.
    merit_cycle_exponent: float = 0.5
    merit_floor: float = 1.0
    max_cycles_per_update: int = 32
    # Per-task merit rewards (single-shot, deduplicated).
    task_reward_not: float = 1.0
    task_reward_nand: float = 1.0
    task_reward_and: float = 2.0
    task_reward_orn: float = 2.0
    task_reward_or: float = 3.0
    task_reward_andn: float = 3.0
    task_reward_nor: float = 4.0
    task_reward_xor: float = 4.0
    task_reward_equ: float = 5.0
    # Branch coupling: digital organisms read/produce world metabolites.
    metabolite_uptake: float = 0.0
    metabolite_deposit: float = 0.0
    # Run the whole update (cycles, division, metabolite coupling) in the native C++
    # kernel (make -C native). Not supported under jax.jit.
    native: bool = False
    # Opt-in energy economy, enabled when instruction_energy_cost > 0. Every executed
    # instruction costs this much energy, merit gained by an instruction is converted to
    # energy at energy_efficiency, an organism with no energy stops executing and dies at
    # the end of the update, and a dividing parent hands half its energy to its child.
    # Energy that is paid out is dissipated as heat (AvidaPopulationState.heat).
    instruction_energy_cost: float = 0.0
    energy_initial: float = 10.0
    energy_efficiency: float = 0.5

    def __post_init__(self) -> None:
        if self.instruction_energy_cost < 0.0:
            raise ValueError("instruction_energy_cost must be >= 0 (0 disables the economy)")
        if self.instruction_energy_cost > 0.0:
            if self.energy_initial <= 0.0:
                raise ValueError("energy_initial must be > 0 when the energy economy is enabled")
            if self.energy_efficiency < 0.0:
                raise ValueError("energy_efficiency must be >= 0")
        if self.native and self.registers < 2:
            raise ValueError("AvidaConfig.native needs registers >= 2")


@dataclass(frozen=True)
class EvolutionConfig:
    generations: int
    steps_per_generation: int
    elite_fraction: float
    tournament_size: int
    extinction_min_population: int
    novelty_enabled: bool
    map_elites_enabled: bool = False
    map_elites_bins: int = 8
    novelty_k: int = 5
    novelty_archive_size: int = 256
    reseed_on_extinction: bool = True


@dataclass(frozen=True)
class LoggingConfig:
    metrics_every: int
    checkpoint_every_generations: int
    replay_every_generations: int


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    run_name: str
    output_dir: str
    world: WorldConfig
    embodied: EmbodiedConfig
    avida: AvidaConfig
    evolution: EvolutionConfig
    logging: LoggingConfig


@dataclass
class WorldState:
    """Shared 2D proxy-physics substrate.

    Shapes:
      terrain: [H, W, T]
      resources: [H, W, R]
      hazards: [H, W, Z]
      flow: [H, W, 2]
      curvature: [H, W, 1]
      shear: [H, W, 2]
      shear_grad: [H, W, 2]
      enrichment: [H, W, 1]
      lift: [H, W, 2]
      concentration: [H, W, C]
      metabolites: [H, W, M]
      occupancy: [H, W]
      time: scalar int
    """

    terrain: Array
    resources: Array
    hazards: Array
    flow: Array
    curvature: Array
    shear: Array
    shear_grad: Array
    enrichment: Array
    lift: Array
    concentration: Array
    metabolites: Array
    occupancy: Array
    time: Array


@dataclass
class EmbodiedPopulationState:
    positions: Array  # [N, 2] continuous in [0, 1)
    energy: Array
    age: Array
    alive: Array
    hidden: Array
    messages: Array
    genomes: dict[str, Array]
    lineage_id: Array
    parent_id: Array
    lineage_depth: Array
    last_action: Array  # [N, action_dim]
    action_history: Array  # [N, K] int discretized indices
    behavior_descriptor: Array  # [N, 2] for novelty/map-elites
    # Flow signal returned by the previous active probe, or None for legacy states.
    probe_result: Array | None = None
    # Previous-step actual movement and commanded thrust, or None when disabled.
    previous_velocity: Array | None = None
    previous_thrust: Array | None = None
    # Predation events for the renderer: attacker mask and target index.
    last_attack: Array | None = None
    last_attack_target: Array | None = None
    # Cumulative energy totals (metrics.energy_ledger), None unless energy_ledger is on.
    ledger: dict[str, Array] | None = None
    # [N, 2] body symmetry axis (unit, or zero before the first thrust) under aspect_drag.
    body_axis: Array | None = None


@dataclass
class AvidaPopulationState:
    genomes: Array
    genome_lengths: Array
    registers: Array
    memory: Array
    ip: Array
    read_head: Array
    write_head: Array
    copied: Array
    copied_length: Array
    merit: Array
    age: Array
    alive: Array
    lineage_id: Array
    parent_id: Array
    lineage_depth: Array
    # Bitfield of logic tasks completed by each organism (NOT, NAND, AND, ...).
    # We store as an int32 with up to 9 bits set, matching the 9 reward fields.
    tasks_completed: Array | None = None
    # Last env IO input pair (per organism) used to detect logic-task outputs.
    last_input_a: Array | None = None
    last_input_b: Array | None = None
    # Energy economy (None unless AvidaConfig.instruction_energy_cost > 0). `energy` is per
    # organism; `energy_in` and `heat` are scalar running totals, never reset, so that
    # sum(energy) - sum(energy at start) == energy_in - heat holds across births and deaths.
    energy: Array | None = None
    energy_in: Array | None = None
    heat: Array | None = None


@dataclass
class SimState:
    generation: int
    step: int
    rng: PRNGKey
    world: WorldState
    embodied: EmbodiedPopulationState | None
    avida: AvidaPopulationState | None
    metrics: dict[str, Any] = field(default_factory=dict)
    # Birth/death counters accumulated across the current generation. Reset
    # to zero by the runner at the end of each generation.
    embodied_births_this_gen: int = 0
    embodied_deaths_this_gen: int = 0
    # Live LBM fluid state (physics.lbm.LiveLBM) when flow_solver == "lbm_live".
    lbm: Any = None
