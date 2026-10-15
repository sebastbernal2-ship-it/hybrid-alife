# Selective causal inquiry protocol

This document owns the evidence contract for the hybrid-alife causal-inquiry program.
`docs/RESEARCH_ALIGNMENT.md` owns the original research question and claim boundary.
This document turns that question into testable stages and controls.
It does not make consciousness claims.

## Operational claim

An artificial organism behaviorally exploits and adapts to an imposed environmental regularity through interaction.
The organism may use a regularity without an explicit symbolic representation.
The regularity may be exposed through ordinary consequences or a declared revelation treatment.

PPO-POET archive transfer is a supporting question.
It asks whether archive contents improve held-out adaptation rather than only increasing search exposure.

## Evidence stages

### Stage 0: environment calibration

Run `scripts/run_awareness_benchmark.py` with the fixed world ladder before expensive training.
The validity gate must pass for solvable, surface-shift, reversal, deceptive, impossible, and non-identifiable worlds.
This stage validates environment mechanics only.

### Stage 1: controller-backed ladder

Train the recurrent controller on training seeds and evaluate it on separate seeds.
Keep world labels out of observations.
Record the controller genome, training seeds, evaluation seeds, and all per-episode actions.
The current bounded trainer is calibration evidence, not simulator evidence.

### Stage 2: simulator bridge

Connect the ladder's hidden-rule semantics to real simulator rollouts without exposing hidden rules.
The bridge must preserve probe cost, targeted probes, reversal timing, nuisance surface changes, deceptive cues, impossible outcomes, and non-identifiability.
A field-discovery result alone cannot substitute for this bridge because field direction and hidden-rule inquiry are different tasks.

The explicit hidden-rule task wrapper is now owned by `src/hybrid_alife/experiments/causal_task.py`.
It separates `agent_observation` from `evaluator_labels` and supports targeted probes, abstention, reversals, deceptive cues, impossible outcomes, and non-identifiability.
The bounded learned-controller pilot is owned by `src/hybrid_alife/experiments/causal_controller.py` and runs with `scripts/run_causal_controller_pilot.py`.
It supports the mutation baseline and a stochastic policy-gradient trainer with entropy regularization.
It compares the learned controller with oracle, random, abstain, and cue-only controls.
The `reward` objective is the baseline.
The `inquiry_shaped` objective is an evaluator-shaped ablation that adds informative-probe and post-probe exploitation terms during training selection.
The `value_of_information` objective uses only episode returns to reward positive post-probe return gain, penalize negative gain, and penalize repeated probes.
It does not receive hidden-rule labels, but it remains a training-selection ablation rather than evidence of autonomous inquiry.
The policy-gradient pilot currently over-probes and fails impossible-world abstention, so it does not pass the selective-inquiry gate.
The contract audit in `tests/test_causal_contract_audit.py` passes for impossible-world expected rewards, deceptive-cue separation, probe timing and cost, and one-step policy-gradient direction.
The policy-gradient trainer now uses a batch timestep value baseline.
Its recurrent loss uses `jax.lax.scan`, `jax.value_and_grad`, and `jax.jit`.
The task supports a declared probe accuracy from 0.5 to 1.0, so probe evidence can be noisy without exposing the hidden rule.
The baseline pilot still over-probes and fails impossible-world abstention.
This localizes the remaining failure to the learning objective or credit assignment rather than an immediate task-contract defect.
Pilot artifacts remain diagnostic and are marked `pilot_only`.
The real field simulator remains a separate proxy environment.
The proxy bundle is run with `scripts/run_causal_rollout.py`.
It records agent-visible observations separately from evaluator-only flow labels and reports probe cost, alignment update, reversal latency, cue controls, and held-out surface transfer.
These are proxy metrics because the field simulator is not yet wired to the explicit hidden-rule task.

### Stage 3: preregistered campaign

Run at least ten seeds per condition and an independent replication only after the controller bridge passes its pilot gates.
Freeze the code commit, world generator, controller architecture, observation schema, action mapping, seed schedule, primary endpoint, and stopping rule before the campaign.

## Primary metrics

- Pre-probe prediction accuracy.
- Post-probe prediction accuracy.
- Selective probe value, measured as improvement minus probe cost.
- Probe rate by expected information value.
- Reversal adaptation latency.
- Cumulative regret after reversal.
- Held-out surface-shift transfer.
- Deceptive-cue resistance.
- Impossible-world abstention and false-structure rate.
- Non-identifiable targeted-probe advantage.

Reward is a secondary metric.
A reward improvement without the matching prediction, probe, transfer, and abstention evidence does not support the primary claim.

## Required controls

- Scripted causal oracle.
- No-signal and random-action baselines.
- Random-probe controller.
- Reward-only controller without inquiry metrics.
- Direct-observation upper bound.
- Shuffled probe outcomes.
- Cue-scrambled and surface-randomized worlds.
- Memory-disabled controller.
- Held-out rules and held-out nuisance surfaces.

The controls must distinguish causal model use from reward improvement, extra training, memorization, and visible-cue exploitation.

## Success gate

A claim requires all of the following:

1. The Stage 0 validity gate passes.
2. The learned controller beats no-signal and random-probe baselines on pre- and post-probe prediction.
3. Probe frequency increases with measured information value and decreases when probe cost exceeds that value.
4. The controller adapts after reversal faster than the memory-disabled and reward-only controls.
5. Surface-shift transfer remains above the preregistered null threshold.
6. Deceptive-cue performance remains above cue-only behavior.
7. Impossible worlds do not produce systematic structure or unnecessary probing.
8. Non-identifiable worlds show a targeted-probe advantage.
9. The effect survives ten seeds and an independent replication with confidence intervals excluding the null.
10. The full artifact audit passes.

Failure of any item yields `insufficient_evidence`, not a revised threshold.

## Interpretation limits

The protocol supports behavioral claims about selective causal inquiry under these worlds and controls.
It does not establish consciousness, subjective experience, general intelligence, open-endedness, or human-like agency.
PPO-POET fixed-policy transfer does not establish open-ended coevolution.

## Current status

- Stage 0 calibration exists and passes its current validity checks.
- Stage 1 now has a deterministic recurrent-controller adapter and bounded trainer.
- Stage 2 has a bounded flow-proxy rollout adapter, but the explicit hidden-rule bridge is not complete.
- The HiperGator archive-control campaign is a separate supporting result and cannot replace Stage 2.
- Adaptation-latency analysis for the archive campaign must be verified before it is reported as a primary metric.

## Reproducibility requirements

Every result must preserve the source commit, complete configuration, seed schedule, machine and backend labels, dependency versions, cache state, per-episode traces, aggregate statistics, and SHA256 checksums.
A different machine is an independent computational replication only when the source and seed contract match and the backend differences are recorded.

No result is promoted from calibration or preliminary evidence without the complete declared campaign and an independent verification pass.
