# `blocks_per_window` Step-Down Playbook

Operator-facing guide for converting outer-step wall-clock wins (P0–P4)
into real outer-steps-per-chain-window throughput by ratcheting the
`blocks_per_window` hparam down in stages, with a 24-hour soak gate
between each step.

**Owner**: validator operator. **Audience**: anyone running a Hone
validator on subnet 5. **Related plan section**: P3b in
`/Users/carro/.cursor/plans/hone_outer-step_throughput_*.plan.md`.

---

## Purpose

> *The chain-window lever: `blocks_per_window` is just a hparam, not a
> chain constant. The hard floor is 1 block ≈ 12s.*
> — Hone Outer-Step Throughput plan, "Realistic outcome targets"

`blocks_per_window` controls the wall-clock duration of a single
training window. One Bittensor block lands roughly every 12 seconds, so
the window length is `blocks_per_window × 12 s`. Today the validator
ships with `blocks_per_window = 30` (~360 s = 6 min). That number was
set conservatively to fit the slowest path of the pre-P0 pipeline.

After phases P0–P4 land — foreach outer optimizer, 2-bit value packing,
RDA merge, K_quorum, parallel outer step — the slowest path no longer
needs 360 s of wall-clock. The hparam should follow:

```
30 → 15 → 10 → 6
```

Each step roughly doubles the number of outer optimizer steps the
network executes per real-world hour. This is the lever that turns
"the outer step finishes in half the wall-clock" into "we make twice as
much real training progress per day."

The hard floor is `blocks_per_window = 6`. Going lower is bounded by
two independent constraints:

1. **Chain weight-setting overhead.** Validators call `set_weights`
   once every `windows_per_weights` windows. At
   `blocks_per_window = 6` and `windows_per_weights = 2` that's already
   ~5× more `set_weights` traffic per hour than the baseline
   `(30, 3)` config, which is the load Bittensor weight extrinsics
   are sized for.
2. **Inner-step cadence.** Templar Covenant-72B uses H = 30; Streaming
   and Decoupled DiLoCo use H = 24 with τ = 2 fragment overlap. At
   `blocks_per_window = 6` and `inner_steps = 6` we're already at the
   per-fragment cadence those networks publish; the H budget per outer
   step doesn't have any room left to give without also adjusting
   `windows_per_weights`.

---

## When to step down

Do not step down unless **every** gate below holds for at least 24
hours of stable operation since the last step-down (or, on the first
step, since the most recent SIGTERM-thrash fix from P-1).

| Gate | Source | Pass criterion |
|---|---|---|
| Stability | `pm2 logs vali` + `journalctl` | No SIGTERM, no gather timeouts > 5/h, no OOM kills since last step-down. |
| Wall-clock headroom | dashboard `/network` Outer-Step Throughput panel → `timing_window_total` | Sliding median over last 24h `< 0.5 × blocks_per_window × 12 s`. (If your current `blocks_per_window` is 30, that's < 180 s.) |
| Validation perplexity | `windowMetrics.lossOwnAfter` (and benchmark eval rows when present) | 7-day trend flat or improving. A clear regression — for example, ≥ 1% perplexity climb week-over-week — disqualifies the step. |
| Wall-clock-cut behaviors enabled | `hone/hparams/hparams.json` | At least **4 of 7** operator-gated behaviors live: `fragmented_uploads`, `pack_values_2bit`, `token_weighted_aggregation`, `rda_merge`, `eval_single_forward`, `parallel_outer_step`, and `gather_min_quorum ≥ 4`. Step-downs without the underlying wall-clock wins land you in the danger zone where the validator can't finish a window in time. |

The standalone tool
`hone/validator/blocks_per_window_stepdown_check.py` automates the
first three of those checks against the hone-api ingest. The behaviors
gate is a quick visual scan of `hparams.json`; the tool deliberately
doesn't read it because that file is the operator's source of truth and
should never be cross-checked against itself.

---

## Step-down sequence

Recommend `inner_steps` and `max_inner_steps` move in lockstep with
`blocks_per_window` so the inner-step:chain-block ratio stays at 1:1
(one inner step per chain block, matching the Streaming DiLoCo
cadence). Recommend `windows_per_weights` drop from 3 → 2 starting at
step 2 so chain weight-setting cadence stays roughly constant in
real-world time even as windows shrink.

| Step | `blocks_per_window` | `inner_steps` | `max_inner_steps` | `windows_per_weights` | Wall-clock per window | Notes |
|---|---|---|---|---|---|---|
| 0 (current baseline) | 30 | 30 | 40 | 3 | ~360 s | Shipped default. Holds the conservative pre-P0 timing budget. |
| 1 | 15 | 15 | 20 | 3 | ~180 s | First halving. Confirm `set_weights` extrinsic latency stays clean — at `(15, 3)` the validator now sets weights every ~9 min instead of ~18 min. |
| 2 | 10 | 10 | 13 | 2 | ~120 s | Drop `windows_per_weights` to 2 here so weight-setting cadence (~4 min) doesn't double-frequency vs step 1 (~9 min). `max_inner_steps = 13` gives the soft cap a 30% headroom factor over the typical inner-step count. |
| 3 (floor) | 6 | 6 | 8 | 2 | ~72 s | Hard floor. Do not go lower without first raising `windows_per_weights ≥ 3`. At `(6, 2)` we set weights every ~2.4 min — already 2.5× the pre-P3b cadence. |

> **Don't go below 6.** 6 inner steps per outer step is at parity with
> Decoupled DiLoCo's per-fragment cadence and below Templar's H = 30.
> The next regime change (`blocks_per_window = 3` etc.) would require
> revisiting H and the chain weight-set frequency in the same change,
> which is bigger than this playbook covers.

---

## How to step down (step 0 → step 1)

```bash
# 1. Edit hparams in place. The validator reloads hparams from disk on
#    restart, so all three substitutions land atomically when pm2
#    spawns the next process.
sed -i '' 's/"blocks_per_window": 30/"blocks_per_window": 15/' hone/hparams/hparams.json
sed -i '' 's/"inner_steps": 30/"inner_steps": 15/' hone/hparams/hparams.json
sed -i '' 's/"max_inner_steps": 40/"max_inner_steps": 20/' hone/hparams/hparams.json

# 2. Confirm the diff is what you expect before restarting.
git diff hone/hparams/hparams.json

# 3. Hot-restart the validator.
pm2 restart vali

# 4. Verify the new values are live in the process — both the hparams
#    log line on startup and the dashboard's blocks_per_window field
#    on the next-window report should show 15.
pm2 logs vali --lines 200 | rg '(blocks_per_window|inner_steps)'

# 5. Soak. Leave the validator running for at least 24 hours and watch
#    the /network Outer-Step Throughput panel. Look for:
#      - timing_window_total median holding < 0.5 × blocks_per_window × 12 s
#        (i.e. < ~90 s at step 1; the panel's stacked area chart makes
#        regressions obvious)
#      - outer_steps_per_chain_window tile climbing
#      - lossOwnAfter trending flat or down vs the prior 24h
```

For step 2, replace `15 → 10`, `20 → 13`, **and** `"windows_per_weights": 3 → "windows_per_weights": 2`. For step 3, replace `10 → 6` and `13 → 8` (`windows_per_weights` stays at 2).

The standalone gate-check tool can be run before each restart:

```bash
HONE_API_URL=https://api.hone.training \
CURRENT_BPW=30 \
python hone/validator/blocks_per_window_stepdown_check.py
# exit 0 → all gates pass, safe to step down
# exit 1 → at least one gate failed, do not step down
# exit 2 → could not fetch data (network, auth, env var)
```

Run it again 24 hours after the restart, with `CURRENT_BPW` updated to
the new value, before scheduling the next step-down.

---

## How to revert

Revert immediately if **any** of these triggers fires within 24h of a
step-down:

- Validation perplexity (`lossOwnAfter` sliding median) regresses by
  ≥ 1% vs the 24h preceding the step-down.
- Gather timeouts (windows where `gatherSuccessRate < 0.5`) per hour
  grow to ≥ 1.5× the pre-step-down baseline.
- A new SIGTERM/OOM kill cadence appears that wasn't present at the
  previous `blocks_per_window`.
- `timing_window_total` median climbs above `0.7 × blocks_per_window ×
  12 s` for 6+ consecutive windows — the validator is no longer
  fitting in its allotted budget.

```bash
# Restore the previous (blocks_per_window, inner_steps, max_inner_steps)
# triplet. Adjust the numbers below to match whatever step you are
# reverting from. Example: rolling step 1 → step 0.
sed -i '' 's/"blocks_per_window": 15/"blocks_per_window": 30/' hone/hparams/hparams.json
sed -i '' 's/"inner_steps": 15/"inner_steps": 30/' hone/hparams/hparams.json
sed -i '' 's/"max_inner_steps": 20/"max_inner_steps": 40/' hone/hparams/hparams.json
git diff hone/hparams/hparams.json
pm2 restart vali
pm2 logs vali --lines 200 | rg '(blocks_per_window|inner_steps)'
```

After reverting, leave the validator running on the previous step for
at least 24 hours before considering whether to retry the step-down.
File a note in your run log describing which trigger fired and the
window range where it was observed; the next attempt should be
hypothesis-driven (e.g. "perplexity regressed because we hadn't
landed `eval_single_forward` yet — retry after that flag flips on").

---

## Risk register

> *Risk #8 in the plan: Bittensor weight-setting overhead grows with
> shorter windows.*
> — Hone Outer-Step Throughput plan

Concretely: at `blocks_per_window = 6` and `windows_per_weights = 2`,
the validator emits ~25 `set_weights` calls per hour, vs ~5 at the
shipped default of `(30, 3)`. Bittensor extrinsics are not free and not
infinitely scalable. Two mitigations are baked into the sequence
above:

1. **Hard floor at `blocks_per_window = 6`.** This playbook's table
   stops there. Going lower requires a separate plan revision.
2. **Step `windows_per_weights` from 3 → 2 at step 2.** This keeps the
   weight-set cadence at roughly the 4-minute scale instead of letting
   it accelerate with every halving. Without this adjustment, step 3
   would set weights every ~72 s, which the chain has never been
   stress-tested against from a single validator.

Other secondary risks worth watching during the soak:

- **Inner-optimizer state churn.** Each window restart resets / decays
  inner-optimizer state per `reset_inner_optimizer_per_window`. More
  windows per hour means more resets per hour. Watch the
  `Generalization Gap` panel for drift after each step-down.
- **Catchup amplification.** Shorter windows mean a peer that misses
  one window falls relatively further behind in inner-step count. Keep
  `sync_max_steps_behind` proportional to the new `inner_steps` (it
  defaults to 2 inner steps; halving `inner_steps` halves the absolute
  catchup tolerance, which is the intended behavior).
- **R2 read amplification under fragmented uploads** (P1). 24 fragments
  × 20 peers per window × N windows per hour. Verify R2 GET P95
  latency in the dashboard's Outer-Step Throughput panel before
  combining a step-down with `fragmented_uploads = true`.

---

## Quick reference: what the dashboard tells you

The `/network` page surfaces three signals you'll watch through every
step-down:

- **Outer-Step Throughput panel → Outer Steps / Chain Window tile.**
  Should climb monotonically across step-downs. When it crosses
  ≥ 2.0 and `blocks_per_window > 6`, the tile shows a muted-mint
  "step-down eligible" hint — that's the same gate this playbook
  encodes, surfaced at a glance.
- **Outer-Step Throughput panel → Effective Tokens / Second tile.**
  Should also climb. If wall-clock shrinks but tokens/sec doesn't,
  something is dropping work on the floor (peer quorum, gather
  timeouts, single-forward eval not landing) — investigate before the
  next step.
- **Validator Anti-Overfitting Check panel → Generalization Gap.**
  Should stay near zero. A persistent positive gap after a step-down
  means inner-loop overfitting — a hard signal to revert.
