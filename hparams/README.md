# `hone/hparams/`

This directory holds the JSON hparam files that the validator and miner
load at startup. JSON has no comment syntax, so this README annotates
the operator-relevant fields here instead of inline.

## Layered config

Files merge in this order (later wins):

1. `DEFAULT_HPARAMS` (compiled into `hone/src/hone/hparams.py`)
2. `hparams.json` — the base config; defines `model_size`
3. `{model_size}.json` — per-architecture overrides (e.g. `8B-A1B.json`)
4. `hparams-local-run.json` — only when `--local` is passed; never committed

For the field-by-field reference, see
[`hone/docs/hparams-reference.md`](../docs/hparams-reference.md).

## Operator-tunable fields

These are the small set of fields you should expect to change in
production over the lifetime of a run. Everything else in `hparams.json`
is a research / architecture knob — touch it and rebenchmark.

### `blocks_per_window` (integer, hard floor 6)

> **Operator-tunable per the P3b step-down playbook.** See
> [`hone/docs/blocks-per-window-stepdown.md`](../docs/blocks-per-window-stepdown.md)
> for the full procedure, gate criteria, and revert path.

Length of one training window in Bittensor blocks (~12 s each). Wall-clock
budget per window is `blocks_per_window × 12 s`. Today's shipped value
is `30` (~360 s); the documented step-down sequence is `30 → 15 → 10 →
6`, **with a 24-hour soak gate between every step**. Do not go below 6
without coordinating a corresponding `windows_per_weights` bump.

When you change this, change `inner_steps`, `max_inner_steps`, and (at
step 2 onwards) `windows_per_weights` in lockstep — see the table in
the playbook.

Run [`hone/validator/blocks_per_window_stepdown_check.py`](../validator/blocks_per_window_stepdown_check.py)
before each step-down to confirm the gates pass.

### `inner_steps`, `max_inner_steps` (integers)

How many local optimizer steps a miner takes per chain window before
uploading. Move proportionally with `blocks_per_window` so the
inner-step:chain-block ratio stays at 1:1 (Streaming DiLoCo cadence).
Step-down playbook table: `(30, 40) → (15, 20) → (10, 13) → (6, 8)`.

### `windows_per_weights` (integer)

How many windows pass between `set_weights` extrinsics. Drop from 3 to
2 at step 2 of the playbook so chain weight-set cadence stays roughly
constant in real wall-clock time as windows shrink. Going lower
multiplies Bittensor weight-set traffic per hour and is the load
constraint that gates lower `blocks_per_window` values.

### Behavior gates (booleans, default `false`)

These flip on operator-controlled features once the corresponding plan
phase ships. Default-false means the network is shipped in a known-safe
configuration; operators turn them on individually after a soak. The
P3b step-down gate-check tool requires at least 4 of these 7 to be on
before recommending a step-down.

| Field | Phase | What flipping it on does |
|---|---|---|
| `pack_values_2bit` | P0b | Bit-pack n_bins=4 quantized values to true 2 bits on the wire (~1.4× wire reduction at parity). |
| `fragmented_uploads` | P1 | Switch from monolithic upload to balanced bin-packed fragments (Decoupled DiLoCo §C). |
| `fragment_upload_concurrency` | P1 | Cap on in-flight `gradient-frag*` PUTs per window. `1` = serial (today's default); `2-4` = bounded `asyncio.gather`. Setting too high (`>~4`) re-creates the 2026-05-03 chaos-mode bandwidth-contention regression where each fragment got 5-10× slower. |
| `rda_merge` | P2a | Radial-Directional Averaging for non-embedding params during outer-step merge. |
| `token_weighted_aggregation` | P2b | Weight peer gradients by their reported `c_tokens × c_tokens / c_steps` (requires `c_tokens_clamp_enabled = true`, which is default-on). |
| `gather_min_quorum` (≥ 4) | P3 | Fire outer step at quorum K instead of waiting for all 20 peers; pair with `gather_safety_cap = 8`. |
| `parallel_outer_step` | P4 | Run `outer_step` on a 2nd CUDA stream concurrently with peer evaluation. |
| `eval_single_forward` | P5b | Replace 4-forward eval with single-forward delta-loss (requires P5a chain-bound seed binding). |

### Always-on guardrails (booleans, default `true`)

| Field | Guards against |
|---|---|
| `c_tokens_clamp_enabled` | Validator-derived clamp on miner-reported `c_tokens` so token-weighted aggregation can't be gamed. |
| `gather_safety_cap` (integer 8) | Per-peer weight cap during merge: a single Byzantine peer cannot exceed `1 / gather_safety_cap` of the merged update L2 norm. |

## Deferred / experimental fields

These exist in `hparams.json` to keep the schema stable but are not
hooked up in production yet. Don't enable without reading the matching
plan section.

- `count_sketch_buckets`, `count_sketch_min_weight` — soft Count Sketch
  fingerprint signal for P5b score weighting.
- `num_fragments` — number of streaming-upload fragments when
  `fragmented_uploads = true`.
- `fragment_upload_concurrency` — when `1` (default) fragment PUTs
  are issued serially. When `>1`, an `asyncio.Semaphore(N)` caps the
  in-flight stream count. R2 per-miner upload throughput is the
  binding constraint, not connection latency, so values above ~4
  typically degrade total wallclock. Validate against the live
  per-fragment p50/p95 from the dashboard's "Per-fragment upload"
  hint row before raising past 2.

## Out of scope here

Architecture fields (`hidden_size`, `num_hidden_layers`, MoE expert
sizing, etc.), optimizer choice / scheduler settings, FSDP sharding,
and dataloader tuning all live in `{model_size}.json` and are documented
in [`hone/docs/hparams-reference.md`](../docs/hparams-reference.md).
None of those should be touched as part of an ops procedure — they're
research surface, not operator surface.
