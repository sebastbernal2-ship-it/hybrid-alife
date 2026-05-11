# Long-Run Operator Runbook

Operational guide for executing multi-hour Hybrid-ALife campaigns safely.
This runbook is the **single source of truth** for everything that happens
*around* a long-running campaign: preflight, environment, detached sessions,
resume/force decisions, worker counts, disk hygiene, recovery from partial
failures, criteria for aborting a bad campaign, and exact end-to-end
command recipes for the three canonical run sizes.

The mechanics of the launcher itself (what `run_quick_campaign.py` does,
flags, manifest schema) live in
[`experiment_campaign_quickstart.md`](experiment_campaign_quickstart.md).
The mechanics of cache layers live in
[`campaign_cache_acceleration.md`](campaign_cache_acceleration.md).
This file assumes you have read at least the TL;DR of each.

> **Audience.** Whoever is babysitting the run. Could be the original
> author, a continuation chat (see
> [`CONTINUATION_HANDOFF.md`](CONTINUATION_HANDOFF.md)), or a future
> operator picking up a sprint cold.

## Table of contents

1. [Preflight checklist](#1-preflight-checklist)
2. [JAX & cache environment](#2-jax--cache-environment)
3. [Session management — tmux / nohup](#3-session-management--tmux--nohup)
4. [Resume vs. force policy](#4-resume-vs-force-policy)
5. [Worker selection](#5-worker-selection)
6. [Disk hygiene](#6-disk-hygiene)
7. [Failure recovery](#7-failure-recovery)
8. [When to stop a bad campaign](#8-when-to-stop-a-bad-campaign)
9. [Artifact packaging](#9-artifact-packaging)
10. [Command recipes](#10-command-recipes)

---

## 1. Preflight checklist

Run this **every time** before launching anything that will execute for
more than ~10 minutes. The cost is seconds; the cost of catching a typo
after 90 minutes of wasted compute is hours.

```bash
# 1a. Confirm tree is clean (or at least known).
git status

# 1b. Make sure tests still pass on the current commit.
pytest -q                                    # expect 104+ passing

# 1c. Run the dedicated preflight checker (no simulation, fail-fast).
python scripts/preflight_campaign.py \
    --campaign configs/campaigns/quick_20min.yaml \
    --print-commands

# 1d. Generate a dry-run manifest and read it before launching for real.
python scripts/run_quick_campaign.py \
    --campaign configs/campaigns/quick_20min.yaml \
    --dry-run \
    --out-dir outputs/_preflight
cat outputs/_preflight/manifest.md
```

The preflight script verifies, without launching JAX work:

- Campaign YAML parses, has `name`, `configs`, `seeds`.
- Every referenced source config exists and loads via `load_config`.
- The base output directory is writable.
- `scripts/run_sim.py` and `scripts/run_quick_campaign.py` exist.
- JAX imports and reports a device.
- The estimated cell count is `len(configs) * len(seeds)`.

Exit code 0 = green light. **Do not launch a long run on exit code 1.**
Add `--json` for CI-friendly output.

### Sanity smoke before the real launch

For anything longer than 2 hours, also run a single-cell smoke to confirm
the inner loop is healthy on the current commit:

```bash
python scripts/run_sim.py --config configs/base.yaml      # 2 gens, ~seconds
```

If the smoke fails, **do not** continue — the campaign launcher will fail
the same way but you will have wasted a tmux session figuring it out.

---

## 2. JAX & cache environment

These environment variables must be exported **in the same shell** that
launches the campaign (or its tmux/nohup session). JAX reads them once at
import time.

```bash
# JAX persistent compilation cache. Single biggest knob for warm runs.
export JAX_COMPILATION_CACHE_DIR="$PWD/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1

# Force CPU if you do not want to accidentally grab a busy GPU.
export JAX_PLATFORMS=cpu

# Make Python warnings fail loud rather than spam the log.
export PYTHONWARNINGS=default

# Optional: pin thread counts on shared boxes so JAX does not eat the host.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"
export OMP_NUM_THREADS=4
```

Notes:

- `.jax_cache` lives inside the repo so it survives between sessions but
  is `.gitignore`-able. **Blow it away (`rm -rf .jax_cache`) after JAX or
  driver upgrades** — stale entries lead to confusing crashes.
- On a fresh machine, the first cell will pay the full XLA compile cost
  (tens of seconds to a few minutes). Subsequent cells with the same HLO
  graph hit the cache in <1 s.
- `JAX_PLATFORMS=cpu` is recommended unless you have explicitly profiled
  a GPU run. CI runs all CPU.

### Where the cache stops helping

The cache keys on JAX version, XLA backend, and the HLO graph. Things
that **change the HLO graph** (and so miss the cache):

- Changing `world.size`, `population_size`, or any other shape-affecting
  config key.
- Upgrading JAX, jaxlib, or the underlying driver.
- Switching `JAX_PLATFORMS` (CPU vs GPU produce different HLOs).

Things that **do not** change the HLO graph and therefore stay warm:

- Changing `seed`, `evolution.generations`, or anything that flows through
  Python-side state without re-tracing.
- Changing `run_name` / `output_dir`.

---

## 3. Session management — tmux / nohup

Anything that runs longer than your patience for an SSH connection must
be in a detached session. Pick **one** of the two patterns below — do not
mix them.

### Pattern A: tmux (preferred when you want to look at it later)

```bash
tmux new -d -s alife-campaign "
  cd $PWD && \
  source .venv/bin/activate && \
  export JAX_COMPILATION_CACHE_DIR=\"\$PWD/.jax_cache\" && \
  export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0 && \
  export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 && \
  python scripts/run_quick_campaign.py \
      --campaign configs/campaigns/quick_20min.yaml \
      --resume \
      2>&1 | tee outputs/quick_campaign/run.log
"

# Reattach later
tmux attach -t alife-campaign

# Kill if you must
tmux kill-session -t alife-campaign
```

Use `tee` so the log is on disk *and* on the tty. Detached tmux survives
SSH disconnects and machine logouts; it does **not** survive reboots —
for that, use a systemd unit or screen + cron.

### Pattern B: nohup (preferred for fire-and-forget, no interactive review)

```bash
nohup bash -c '
  source .venv/bin/activate
  export JAX_COMPILATION_CACHE_DIR="$PWD/.jax_cache"
  export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
  export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
  python scripts/run_quick_campaign.py \
      --campaign configs/campaigns/quick_20min.yaml \
      --resume
' > outputs/quick_campaign/run.log 2>&1 &

echo $! > outputs/quick_campaign/run.pid

# Tail later
tail -f outputs/quick_campaign/run.log

# Kill if you must
kill "$(cat outputs/quick_campaign/run.pid)"
```

Always write the PID file. Recovering a runaway nohup without one is
painful (`pgrep -fa run_quick_campaign` works but is fragile).

### What **not** to do

- Do **not** launch a long run in a foreground SSH session. Network blip
  → SIGHUP → lost compute.
- Do **not** redirect stdout to `/dev/null` "to keep things quiet". The
  log is your only forensic evidence when something goes wrong.
- Do **not** background with bare `&` and no `nohup`. SIGHUP will still
  reach the process on logout.

---

## 4. Resume vs. force policy

`run_quick_campaign.py` exposes three mutually-exclusive resumption modes
(see [`campaign_cache_acceleration.md`](campaign_cache_acceleration.md)
for the cache semantics):

| Mode | When to use |
|---|---|
| (none / default) | First-ever launch of a brand-new campaign. |
| `--resume` | Re-launching after any interruption (crash, kill, reboot, partial wall-clock). |
| `--force` | After fixing a bug in `src/` that the config hash cannot detect. |

### Decision tree

```
Did I change a config file (configs/*.yaml, campaign YAML)?
├── Yes → cache hash will differ, cells re-run automatically. Use --resume.
│
└── No.
    │
    Did I change anything in src/ that affects simulation output?
    ├── Yes → use --force. Hash cannot see source changes.
    │
    └── No.
        │
        Did a prior partial run leave outputs on disk?
        ├── Yes → use --resume. Completed cells will be skipped, failed
        │          ones will re-run from scratch.
        │
        └── No → omit both flags (or use --resume; it is a no-op when
                 there is nothing to skip).
```

### What `--resume` does and does not do

A cell is **considered complete and skipped** by `--resume` only when:

- Its resolved `cell_config.yaml` exists *and* its content hash matches the
  freshly planned one, **and**
- A non-empty `metrics.json` or `metrics.csv` exists in the cell output dir.

If either condition fails, the cell runs normally. The manifest will record
`status: "cached"` (with `duration_s: 0.0`) for skipped cells, so resumed
campaigns remain fully audited.

### When **not** to resume

- The campaign output dir was previously used for a *different* campaign.
  Choose a fresh `--out-dir` or move the old one aside.
- You suspect a partial cell wrote a corrupt `metrics.json` (e.g., the cell
  crashed mid-write). In that case, `rm -rf <cell_dir>` and resume —
  resume cannot tell the difference between a clean and a corrupt
  `metrics.json`.

---

## 5. Worker selection

`--workers N` runs `N` cells concurrently via `concurrent.futures`.

### Default

`--workers 1`. This is the safe default. Most cells already saturate the
JAX thread pool by themselves, so adding parallel workers usually trades
wall-clock for thrashing.

### When to consider `--workers > 1`

All of the following must be true:

- The host has demonstrable spare CPU during a single-cell run
  (`top` shows `%CPU < 100 * num_cores * 0.6`).
- You have at least `N * 2 GB` of free RAM.
- You are not paying a per-cell JAX compile cost (i.e., the cache is
  warm, or you have accepted that each worker will independently compile
  the first time it sees a new HLO).
- You are willing to accept noisier per-cell timings.

A practical sweet spot on an 8-core CPU box with a warm cache is
`--workers 2`. Anything beyond that has rarely paid off in this codebase.

### When **not** to use `--workers > 1`

- On GPUs (each worker takes the same JAX device and they will fight).
- On the first launch of a campaign (each worker recompiles independently).
- For the overnight recipe — sequential gives cleaner failure modes and
  the campaign is already large enough that worker speedup is a smaller
  fraction of total time.

---

## 6. Disk hygiene

A multi-hour campaign generates **a lot** of files. Plan for it.

### Pre-launch

```bash
df -h .                                       # confirm > 20 GB free for overnight
du -sh outputs/                               # see what is already there
du -sh .jax_cache/ 2>/dev/null                # may be multi-GB on a busy repo
```

A rough per-cell footprint (default smoke200-class config):

- `metrics.jsonl` 1–5 MB
- `metrics.json` < 100 KB
- `cell_config.yaml` < 10 KB
- `checkpoint_gen*.pkl` 1–20 MB each (depends on `checkpoint_every`)
- `novelty_archive.npz`, `map_elites.npz` 100 KB – 5 MB each
- `report.md`, `plots/*.png` <2 MB (if `generate_report.py` is run)

For a 12 × 5 (configs × seeds) overnight campaign with checkpoints every
generation, plan for **5–15 GB** of output.

### Mid-run

- `du -sh outputs/<campaign>/` every few hours.
- If `df -h .` drops below 2 GB **free**, abort, see §8.

### Post-run cleanup

```bash
# Keep only final checkpoints, prune intermediate ones.
find outputs/<campaign>/ -name 'checkpoint_gen*.pkl' \
    ! -name 'checkpoint_gen_final.pkl' \
    ! -name 'checkpoint_final.pkl' \
    -delete

# Compress per-cell logs (they are very repetitive).
gzip outputs/<campaign>/*/metrics.jsonl

# The JAX cache is not in the campaign dir but can be reclaimed safely.
# It will be re-warmed on the next run.
rm -rf .jax_cache
```

Do **not** delete `manifest.json`, `cell_config.yaml`, `metrics.json`,
`metrics.csv`, the final checkpoint, or the report — those are the audit
trail.

---

## 7. Failure recovery

### Failure: process died mid-campaign

1. Inspect the tail of the log (`tail -100 outputs/<campaign>/run.log`)
   and the manifest:
   ```bash
   python -c "import json; m=json.load(open('outputs/<campaign>/manifest.json')); \
              print(m['summary']); \
              [print(c['run_name'], c['status'], c.get('returncode')) for c in m['cells']]"
   ```
2. If the cause was OOM / disk full / external (host reboot), free the
   resource and **re-launch with `--resume`** to the same `--out-dir`.
3. If the cause was a code bug, fix it on a sprint branch, commit, and
   **re-launch with `--force`** — the config hash will not have changed
   so `--resume` would incorrectly skip the already-broken cells.

### Failure: one cell failed, others are still queued

This is the normal mode for a healthy run with one bad cell. The launcher
records `status: "failed"` for the broken cell, continues with the rest,
and exits non-zero at the end.

1. Re-launch with `--resume` once the campaign finishes — `--resume` will
   skip the cells that succeeded and retry the failed one only.
2. If the failure is deterministic (same cell fails again), pull its
   `cell_config.yaml` aside and reproduce with `run_sim.py --config <cell_config>`
   for fast local debugging.

### Failure: corrupt `metrics.json` (partial write before crash)

`--resume` cannot detect a corrupt JSON. Symptoms: cell shows `status: "cached"`
in the resumed manifest but downstream analysis fails to parse.

```bash
# Verify each cell's metrics.json is valid JSON.
for f in outputs/<campaign>/*/metrics.json; do
    python -c "import json,sys; json.load(open('$f'))" 2>/dev/null \
        || echo "BAD: $f"
done
# Remove bad cells' output dirs and resume.
```

### Failure: ran out of disk mid-cell

1. **Stop the campaign immediately** (see §8).
2. Free space: clear `.jax_cache`, prune intermediate checkpoints in
   completed cells (§6), or move `outputs/` to a larger volume and
   symlink it back.
3. The half-written cell is unsafe — `rm -rf` its directory.
4. Re-launch with `--resume`.

### Failure: host rebooted

tmux sessions do not survive reboot.

1. SSH back in. `tmux ls` will show nothing.
2. The manifest on disk records exactly what completed.
3. Start a fresh tmux session and re-launch with the **same** `--out-dir`
   and `--resume`.

---

## 8. When to stop a bad campaign

It is almost always cheaper to abort a misconfigured multi-hour run early
than to let it finish and try to salvage the output. Stop the campaign
when **any** of these are true:

- **Disk free < 2 GB**, and the campaign is still creating outputs.
- **The first 2 completed cells crashed with the same traceback.** Three
  more cells will not help; fix the bug and resume with `--force`.
- **Cell wall-clock is wildly off** from the dry-run estimate (e.g.,
  10x slower). Likely a thermal / contention / OS-level issue —
  diagnose before continuing.
- **The wrong campaign YAML was passed.** Look at the manifest — if it
  lists configs you did not intend to sweep, kill it.
- **`metrics.jsonl` for a completed cell is empty or NaN-laden.** Either
  the metric writer is broken on the current commit or a config is
  pathological. Stop and reproduce on a single cell.
- **The neutral shadow has drifted** from the live config (different
  sensing mode, mutation rate). Bedau ratios will be meaningless;
  fix the shadow before paying for more cells.
  See [`CONTINUATION_HANDOFF.md`](CONTINUATION_HANDOFF.md) §"Open
  scientific risks" item 5.

### How to stop

```bash
# tmux pattern
tmux send-keys -t alife-campaign C-c        # SIGINT, gives launcher a chance to write manifest
sleep 5
tmux kill-session -t alife-campaign         # only if SIGINT did not return

# nohup pattern
kill -INT "$(cat outputs/<campaign>/run.pid)"
sleep 5
kill -KILL "$(cat outputs/<campaign>/run.pid)"   # only if SIGINT did not return
```

Prefer SIGINT (Ctrl-C) over SIGKILL: the launcher's signal handler will
flush the manifest before exiting, so the next `--resume` knows where it
stopped. SIGKILL leaves the manifest stale.

---

## 9. Artifact packaging

When a campaign finishes (or has been deliberately stopped at a known
boundary), package the artifacts so a future operator — or a new chat —
can reproduce headline numbers without rerunning the campaign.

```bash
CAMPAIGN_DIR=outputs/quick_campaign
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

# 1. Validate the manifest before packaging.
python -c "
import json
m = json.load(open('$CAMPAIGN_DIR/manifest.json'))
s = m['summary']
print('cells', s['n_cells'], 'ok', s['n_ok'], 'failed', s['n_failed'])
assert s['n_failed'] == 0, 'failed cells present — investigate before packaging'
"

# 2. Regenerate per-cell reports.
for cell in $CAMPAIGN_DIR/*/; do
    [ -f "$cell/metrics.jsonl" ] || continue
    python scripts/generate_report.py "$cell" || true
done

# 3. Bundle.
tar --exclude='checkpoint_gen[0-9]*.pkl' \
    -czf "${CAMPAIGN_DIR}_${STAMP}.tar.gz" \
    "$CAMPAIGN_DIR/"

# 4. Record the commit, env, and verify checksum.
git rev-parse HEAD > "${CAMPAIGN_DIR}_${STAMP}.commit"
python -V >>  "${CAMPAIGN_DIR}_${STAMP}.commit"
sha256sum "${CAMPAIGN_DIR}_${STAMP}.tar.gz" \
    > "${CAMPAIGN_DIR}_${STAMP}.tar.gz.sha256"
```

The tarball is the audit artifact. It excludes intermediate checkpoints
(which are reproducible from final state + seed) but keeps:

- `manifest.json`, `manifest.md`
- Every cell's `cell_config.yaml`, `metrics.json`/`metrics.jsonl`,
  `metrics.csv`, archives, and final checkpoint
- Every cell's `report.md` and plots

Always store the commit SHA alongside. Without it, the tarball is
unreproducible.

---

## 10. Command recipes

Three canonical run sizes. Each is a single block you can paste into a
fresh shell after preflight.

### 10a. 20-minute sanity sweep

Use this before any longer run, or to validate a sprint branch end-to-end.

```bash
# Preflight (mandatory).
python scripts/preflight_campaign.py \
    --campaign configs/campaigns/quick_20min.yaml \
    --print-commands || exit 1

# Environment (idempotent, safe to re-export).
export JAX_COMPILATION_CACHE_DIR="$PWD/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_PLATFORMS=cpu

# Run in the foreground — 20 minutes is short enough you can babysit.
python scripts/run_quick_campaign.py \
    --campaign configs/campaigns/quick_20min.yaml \
    --resume \
    2>&1 | tee outputs/quick_campaign/run.log

# Verify and report.
python -c "import json; m=json.load(open('outputs/quick_campaign/manifest.json')); print(m['summary'])"
```

Expected: 6 cells (3 configs × 2 seeds), each ~3 minutes on a CPU box.
Exit code 0 = clean run.

### 10b. 2-hour ablation matrix

Use this when validating a sprint that touches simulation behavior.

```bash
# Preflight against a longer campaign YAML (create one if it does not
# exist; see configs/campaigns/quick_20min.yaml as the template).
python scripts/preflight_campaign.py \
    --campaign configs/campaigns/ablation_2h.yaml \
    --print-commands || exit 1

# Environment.
export JAX_COMPILATION_CACHE_DIR="$PWD/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_PLATFORMS=cpu

# Run in tmux — 2 hours is too long to babysit a tty.
tmux new -d -s alife-2h "
  source .venv/bin/activate && \
  export JAX_COMPILATION_CACHE_DIR=\"\$PWD/.jax_cache\" && \
  python scripts/run_quick_campaign.py \
      --campaign configs/campaigns/ablation_2h.yaml \
      --resume \
      2>&1 | tee outputs/ablation_2h/run.log
"

# Status check every ~30 min.
tmux attach -t alife-2h         # detach with Ctrl-b d

# When done, generate per-cell reports and verify.
for d in outputs/ablation_2h/*/; do
    [ -f "$d/metrics.jsonl" ] && python scripts/generate_report.py "$d" || true
done
python -c "import json; m=json.load(open('outputs/ablation_2h/manifest.json')); print(m['summary'])"
```

A 2-hour campaign typically sweeps 4 configs × 3 seeds at smoke200-class
generation budgets, plus the paired neutral shadow. Use `--include-shadow`
on the underlying `run_ablation_matrix.py` if you go that route directly
instead of the campaign launcher.

### 10c. Overnight (8–12 hour) campaign

Use this for headline numbers you intend to publish or hand off.

```bash
# Preflight + dry-run review — read the manifest before you commit overnight.
python scripts/preflight_campaign.py \
    --campaign configs/campaigns/overnight.yaml \
    --print-commands || exit 1
python scripts/run_quick_campaign.py \
    --campaign configs/campaigns/overnight.yaml \
    --dry-run \
    --out-dir outputs/_overnight_preflight
less outputs/_overnight_preflight/manifest.md   # READ THIS

# Confirm disk and free RAM.
df -h .                                          # need ≥ 25 GB free
free -h                                          # need ≥ 4 GB free

# Environment.
export JAX_COMPILATION_CACHE_DIR="$PWD/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4

# Sequential workers — fewer surprising failure modes overnight.
mkdir -p outputs/overnight
tmux new -d -s alife-overnight "
  source .venv/bin/activate && \
  export JAX_COMPILATION_CACHE_DIR=\"\$PWD/.jax_cache\" && \
  export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0 && \
  export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 && \
  export JAX_PLATFORMS=cpu && \
  python scripts/run_quick_campaign.py \
      --campaign configs/campaigns/overnight.yaml \
      --workers 1 \
      --resume \
      2>&1 | tee outputs/overnight/run.log
"

# Drop the PID so you can find the process tomorrow even if tmux dies.
tmux list-panes -t alife-overnight -F '#{pane_pid}' > outputs/overnight/tmux.pid

# In the morning:
tmux attach -t alife-overnight
python -c "import json; m=json.load(open('outputs/overnight/manifest.json')); print(m['summary'])"

# Package per §9.
```

Expected size: 20–60 cells. Plan for `n_failed > 0` and a `--resume`
re-launch in the morning. **Always** use `--resume` (not `--force`) on
the morning re-launch unless you have explicitly changed `src/`.

---

## See also

- [`experiment_campaign_quickstart.md`](experiment_campaign_quickstart.md)
  — launcher flags, manifest schema, dry-run semantics.
- [`campaign_cache_acceleration.md`](campaign_cache_acceleration.md)
  — full detail on the JAX compile cache and the run-skip cache.
- [`CONTINUATION_HANDOFF.md`](CONTINUATION_HANDOFF.md) — repo state,
  open scientific risks, exact commands to reproduce known validation.
- [`scientific_validation.md`](scientific_validation.md) — acceptance
  criteria and language guardrails. A long run that violates these
  guardrails is a bad run regardless of how cleanly it finished.
