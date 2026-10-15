# Finalization gate

This file owns the closure rules for the selective-causal-inquiry study.
It prevents a behavioral result from being promoted into a consciousness claim.

## Claims

The study has three separate claim levels.

### Claim A: selective causal inquiry

An agent probes when expected information value exceeds cost, uses the result, adapts after reversal, transfers across surface changes, rejects deceptive cues, and abstains in impossible worlds.

This claim is behavioral.

### Claim B: PPO-POET archive value

Archive-enabled PPO-POET improves held-out transfer over the preregistered controls.

This claim is about archive-mediated transfer.
It does not establish Claim A.

### Claim C: consciousness

The agent has subjective experience.

This claim is not established by the current protocol.
Behavioral success alone cannot prove it.
A consciousness claim would require a separate theory, mechanism, intervention, and validation program.

## Current status

- Task contract: passed local audit.
- Learned controller: failed the current selective-inquiry pilot.
- Policy-gradient controller: over-probed and failed impossible-world abstention.
- Field simulator: proxy evidence only.
- PPO-POET archive campaign: incomplete.
- Scientific status: `insufficient_evidence`.

## Closure requirements for Claim A

Do not close Claim A until every item passes.

- The hidden-rule task is fixed before the confirmatory run.
- Agent observations contain no evaluator-only labels.
- The learned policy beats reward-matched random, abstain, cue-only, and probe-control baselines.
- Probe rate is higher in solvable worlds than impossible worlds.
- Post-probe performance improves by a preregistered amount.
- Reversal recovery meets the preregistered latency bound.
- Deceptive-cue performance exceeds cue-only performance.
- Held-out surface transfer remains positive.
- Repeated probes do not explain the result.
- At least 10 seeds are used per condition.
- A second independent replication uses a distinct seed schedule and cleared cache.
- The primary effect has a confidence interval and meets the preregistered threshold.
- Raw artifacts, code SHA, configuration, and checksums are archived.

If any item fails, report `insufficient_evidence`.
Do not retune the threshold after seeing the result.

## Closure requirements for Claim B

Use the owner file `docs/POET_SUCCESS_CRITERIA.md`.
Its strong gate remains authoritative.
A passing Claim B result does not change the status of Claim A or Claim C.

## Closure rule for Claim C

The current study must not report Claim C as proven.
At most, a future result may motivate a separate consciousness research program.
Any such program must define its theory and measurement bridge before collecting confirmatory data.

## Allowed final language

Before all Claim A gates pass:

> The system provides an evaluated testbed for selective causal inquiry, but current learned agents do not demonstrate the behavior reliably.

After all Claim A gates pass:

> The tested agent demonstrates selective causal inquiry under the declared conditions.

Neither statement supports a consciousness claim.
