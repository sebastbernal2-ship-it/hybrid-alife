# Hybrid Forager Environment Contract

This contract defines the bounded ecology used by the paired PPO-POET pilot.

## Organism state

The focal organism is a single continuous-position forager in a fixed two-dimensional toroidal grid.

The controller receives an eight-value local observation.

The observation contains position, terrain, resource, hazard, energy, goal distance, and the shared chemical signal.

The organism starts with energy `1.0` and a position at the grid centre.

The organism is alive while energy is above zero.

## Environment genes

Each environment genome has six scalar genes and one terrain map.

- `goal_x` and `goal_y` define the success target.
- `resource_regeneration` scales the initial resource field.
- `hazard_rate` scales local hazard exposure.
- `flow_x` and `flow_y` add a constant flow offset to movement.
- `terrain` defines spatial structure that affects terrain, resources, hazards, and observations.

The implementation exposes one-gene intervention checks.

A valid intervention must change an observable field or the focal trajectory under a fixed random seed.

Terrain validity uses a spatial roughness measure rather than only a terrain mean.

## Resources and hazards

Resource cells hold one bounded resource channel.

Action `4` or `5` gathers up to forty percent of the local resource.

Gathered resource increases energy and decreases the world resource at that cell.

Hazard is sampled at the local cell.

Hazard reduces energy and reward.

The world update advances flow and chemical fields before each organism action.

## Lifecycle

The organism pays a fixed energy cost and a movement cost on every step.

Interaction action `6` pays an additional cost and emits a local chemical signal.

Reproduction action `7` is valid when the organism has enough energy.

Reproduction pays an energy cost and deposits a small resource pulse.

The pilot records reproduction events as bounded offspring events.

The focal organism is reset to the start state after success or failure so fixed-length PPO rollouts remain comparable.

## Interactions and coupling

The controller moves the focal position used by the Avida population.

Interaction and reproduction actions deposit a controller signal into the shared concentration field.

Avida reads and writes shared metabolite and concentration fields.

The next controller observation includes those fields.

The pilot records both controller-to-Avida events and Avida-to-controller field changes.

This is bidirectional field coupling, not a claim that the two populations form an open-ended evolutionary system.

## Success and failure

Success occurs when the focal organism reaches the goal with positive energy.

Failure occurs when energy reaches zero.

Success receives a bounded bonus.

Failure receives a bounded penalty.

Resource gain, interaction, hazard damage, energy cost, and chemical coupling are recorded as ecology diagnostics.

## Admission and pilot scope

Candidate admission still requires the configured raw-reward lower bound for compatibility.

Each candidate also records matched random, fixed-policy, and hand-coded baseline scores.

The pilot is a bounded three-seed validity check.

It tests causal ecology, archive diagnostics, and coupling diagnostics.

It does not establish open-ended evolution.
