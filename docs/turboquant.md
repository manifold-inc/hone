# TurboQuant Codec (P6) — Operator Guide

TurboQuant is a dense scalar gradient codec that replaces Hone's
top-K + 8-bit-uniform compression pipeline with a Hadamard rotation
+ Beta-Lloyd-Max scalar codebook + outlier handling. When the
distributional assumption holds, TurboQuant trades the same wire
budget for ~2-3× lower reconstruction MSE than the 4-bin uniform
scheme it would replace, lifting the per-window goodput of every
peer on the gather budget.

> **Status: DARK CODE.** The codec ships behind a hard-off
> `hparams.turboquant_enabled` flag and does NOT execute on the
> default training path. The flag must NOT be flipped on in
> production until BOTH gates below pass on a real run. Until then
> the legacy P0b 2-bit-pack path remains authoritative.

## Why a hard gate?

The codec's Lloyd-Max codebook is fit against Beta(d/2, d/2). That
distribution is the asymptotic marginal of a coordinate of a uniform
random unit vector — which is what the post-Hadamard rotation
*should* produce IF the input gradient is reasonably isotropic. EF
tensors that have been accumulated across many windows of biased
gradient flow may not satisfy this assumption: the Beta KS test fails,
the codebook is mismatched, and reconstruction silently degrades
without an operator-visible failure mode. The validator scores would
drift, the model would learn slightly worse, and we'd discover the
problem from a 0.5% perplexity regression two weeks later.

The audit prevents that. Run it FIRST against real production EF.

## Reference

- Paper: Lukashevich, Karbasi, Khosravifard et al., *"TurboQuant:
  Online Vector Quantization with Optimal Distortion Rate"* (2024).
- Hone implementation:
  - `hone/src/hone/turboquant_audit.py` — Hadamard rotation,
    Beta KS-test, per-class aggregation. Importable; no side effects.
  - `hone/src/hone/turboquant.py` — codec proper; quantize /
    dequantize / Beta-Lloyd-Max codebook with on-disk cache.
  - `hone/neurons/miner.py` — SIGUSR2 hook that dumps
    `self.error_feedback` for offline analysis.
  - `hone/src/hone/neurons.py::prepare_gradient_dict` — the codec
    branch (gated on `hparams.turboquant_enabled`).
  - `hone/validator/turboquant_audit_cli.py` — P6a gate CLI.
  - `hone/validator/turboquant_ab_compare.py` — P6b A/B gate CLI.

## P6a — Coordinate-distribution audit (hard gate)

The audit decides whether the per-coordinate distribution of post-
Hadamard EF tensors matches the Beta(d/2, d/2) the codec assumes,
PER tensor class (embedding / router / moe_expert / attention / mlp /
other). If even one class fails, the codec must stay disabled — that
class would silently corrupt under the codec.

### 1. Snapshot a real production EF

The miner registers a SIGUSR2 handler at startup that dumps the
current `self.error_feedback` dict to
`/tmp/ef_snapshot_uid<U>_w<W>.pt` when triggered. The handler is a
no-op until the signal fires; zero footprint on normal runs.

On the miner host, with the miner already running and at least one
real (non-warmup) outer step completed:

```bash
# Find the miner process.
pgrep -f "neurons/miner" | head -1

# Trigger the dump.
kill -USR2 <miner_pid>

# Look in the miner's stdout / pm2 logs for:
#   [P6a] dumped EF snapshot to /tmp/ef_snapshot_uid<U>_w<W>.pt (<N> keys)

# OR locate the most-recent snapshot directly:
ls -lt /tmp/ef_snapshot_uid*_w*.pt | head -1
```

### 2. Run the audit

The CLI runs the KS test against Beta(d/2, d/2) for every tensor in
the dump, aggregates per class, and exits 0 iff every class clears
`PASS_RATE_THRESHOLD` (default 80%).

```bash
python hone/validator/turboquant_audit_cli.py \
    /tmp/ef_snapshot_uid42_w12345.pt \
    --json /tmp/ef_audit_uid42_w12345.json \
    --verbose
```

The human summary on stdout:

```
class            count  passed  pass_rate    mean_p   median_p
--------------------------------------------------------------
attention          240     228      0.950    0.4123     0.4502
embedding            2       2      1.000    0.3812     0.3812
mlp                480     421      0.877    0.3110     0.2891
moe_expert        2880    2521      0.875    0.2780     0.2654
other               16      11      0.688    0.1230     0.0987
router              16       0      0.000    0.0001     0.0001

worst-class pass rate = 0.000 (gate threshold = 0.800)
FAIL: P6a coordinate-distribution gate did NOT pass.
Failing classes: other, router. Do NOT enable hparams.turboquant_enabled —
the codec would silently corrupt these tensor classes.
```

In this example router gradients are heavy-tailed / non-Beta and the
codec is unsafe — the operator must NOT proceed to P6b.

If the audit passes (exit code 0, pass rate ≥ threshold for every
class), proceed to P6b.

## P6b — A/B perplexity gate

After P6a passes, the codec is *plausibly* safe but not yet *proven*
safe. The A/B gate runs the codec end-to-end at 1.4B for 200 outer
steps and confirms the perplexity / per-UID-score distribution did
not drift past the per-phase tolerance.

### 1. Capture two sibling runs

Train two 1.4B miners for 200 outer steps with everything identical
except `turboquant_enabled`:

```jsonc
// hparams/1.4B-tq-off.json
{
  "turboquant_enabled": false,
  // ...everything else from 1.4B.json
}

// hparams/1.4B-tq-on.json
{
  "turboquant_enabled": true,
  "turboquant_bits": 4,
  "turboquant_outlier_top_k": 32,
  "turboquant_q_prod_mode": true,
  // ...everything else from 1.4B.json
}
```

The validator's score / perplexity dump should be a JSON list:

```json
[
  {
    "window": 12345,
    "perplexity": 14.32,
    "scores": {"42": 0.912, "17": 0.876, ...}
  },
  ...
]
```

### 2. Run the A/B compare

```bash
python hone/validator/turboquant_ab_compare.py \
    --tq-off scores_tq_off_1.4b.json \
    --tq-on  scores_tq_on_1.4b.json
```

Both gates must pass:

* Validation perplexity regression ≤ 0.5% (TQ-on / TQ-off ≤ 1.005).
* Spearman per-UID-score correlation ≥ 0.85 (TurboQuant must not
  silently re-rank peers).

Sample passing output:

```
PASS: both A/B gates cleared. TurboQuant codec is perplexity-neutral
and does not re-rank peers.
TQ-off perplexity = 14.321
TQ-on  perplexity = 14.298  (-0.161% vs off; threshold 0.500%)
Spearman rho = 0.917 (p=1e-12) on n=18 UIDs (threshold 0.85)
```

If either gate fails, the script exits 1 with diagnostic guidance
(try a higher `turboquant_bits`, raise `turboquant_outlier_top_k`,
or revisit the audit if the EF distribution has drifted since P6a
ran).

## Enabling TurboQuant in production

ONLY after BOTH gates above have passed:

1. P6a audit (`turboquant_audit_cli.py`) on a recent real EF
   snapshot exited 0.
2. P6b A/B (`turboquant_ab_compare.py`) on a 200-step 1.4B sibling
   run exited 0.

…flip the hparam:

```jsonc
// hparams/hparams.json
{
  "turboquant_enabled": true,
  "turboquant_bits": 4,
  "turboquant_outlier_top_k": 32,
  "turboquant_q_prod_mode": true,
  // ...
}
```

This MUST land alongside the validator-side TurboQuant decoder PR.
The legacy `cname + {'idxs','vals','quant_params'}` keys are NOT
populated in the TurboQuant branch (the payload moves to
`cname + 'tq_*'` keys); a TurboQuant-unaware validator that gets a
TurboQuant-encoded gradient will simply skip the param, breaking
upload / outer-step convergence for the affected miners. Coordinate
the rollout.

## Hparams reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `turboquant_enabled` | bool | `false` | **Master switch.** Default OFF. Flip true ONLY after both gates pass and the validator decoder is shipped. |
| `turboquant_bits` | int | `4` | Total bits per coordinate budget. `main_bits` is `b - 1` in prod mode, `b` in mse mode; outlier coords use `min(b + 1, 8)` bits. |
| `turboquant_outlier_top_k` | int | `32` | Number of top-\|coordinate\| entries that get the higher-resolution outlier codebook (paper §4.3). Set to 0 to disable. |
| `turboquant_q_prod_mode` | bool | `true` | When `true`, spend one bit on a 1-bit residual sign correction (TurboQuant Q_prod mode). Recommended for cross-peer aggregation; the inner-product mode reduces aggregated reconstruction bias. |

## Why we audit BEFORE shipping

The original P6 plan intended to run TurboQuant *after* the existing
top-K compressor — quantize the top-K kept values via Beta-Lloyd-Max.
That fails because top-K leaves only ~32 values per chunk of 64,
breaking the Beta(d/2, d/2) distribution Lloyd-Max assumes (you've
hand-selected the high-magnitude tail of the distribution; nothing
about that is Beta-shaped). The reframed plan rotates the FULL d-dim
EF first, then quantizes, then tops-K on the codebook indices — but
this only works if the post-rotation EF is genuinely Beta-distributed.
The P6a audit verifies that empirical claim BEFORE the operator
commits engineering effort to the validator-side decoder + the A/B
soak. If the audit fails, P6 is deferred at zero blast radius.

## Wire-format inflation incident (2026-05-03) and follow-up fix

**Incident.** With `turboquant_enabled: true` on the 8B-A1B production
hparams, miner upload payloads inflated to ~38.7 GB per window
(p50 = 1.45 GB × 24 fragments per `logs/miner-*-out.log` "Uploaded …
24 fragments" lines). The expected legacy P0b payload is ~7 GB.
The 5.5× inflation is the actual root cause of why miner upload
windows exceed the validator's 70 s gather budget — not a parallelism
issue.

**Mitigation (deployed).** Reverted `turboquant_enabled` to `false`
in `hparams/hparams.json`. Drops wire payload from ~38.7 GB → ~7 GB
per window per miner (5.5×). Validator-side decode is unaffected
because the existing `outer_step` and eval paths are codec-agnostic
(they dispatch on the presence of `cname + "tq_idxs"` per-param;
when miners stop emitting that key they fall through to the legacy
top-K decode that the rest of the system has been running on
forever).

**Per-suspect byte accounting** for the audit (full report in chat
transcript [Hone gradient payload audit](TODO-uuid)):

The TurboQuant payload writes three keys per encoded param of
size `d`:

| Key | Type | Length | Bytes per param |
|-----|------|--------|-----------------|
| `tq_idxs` | int64 | `(d * topk) // target_chunk` ≈ `d/2` | **`4d`** |
| `tq_codes` | uint8 | same as `tq_idxs` | **`0.5d`** |
| `tq_meta.sign_bits` | uint8 | full `d` (prod mode only) | **`d`** |
| `tq_meta` other | mixed | small (centroids, outliers) | ~500 B |

Total: **`5.5d` + ~500 B per param**. Across the 8B-A1B model
(~8.6B trainable params, MoE-dominated) this lands at ~47 GB on
paper, ~38.7 GB observed (the ~9 GB shortfall comes from
`torch.save` framing on uint8 arrays plus a handful of tiny
layer-norm params where the meta dict overhead dominates).

The legacy P0b path with `pack_values_2bit=true,
quantization_bins=4` ships:

| Key | Bytes per param |
|-----|-----------------|
| `idxs` (12-bit packed) | `0.75d` (50% density × 1.5 B per kept idx) |
| `vals` (2-bit packed) | `0.125d` (50% density × 0.25 B per kept val) |
| `quant_params` | ~50 B per param (5- or 7-tuple with a 16-byte lookup tensor) |

Total: **`0.875d`**. Across 8.6B params this is ~7.5 GB — matches
the observed legacy baseline before the TurboQuant flip.

**Follow-up (re-enable TurboQuant later).** Three wire-format
changes are required before flipping `turboquant_enabled` back on.
Each MUST land alongside its matching decoder update or it
silently corrupts every TurboQuant peer's gradient.

1. **Pack `tq_idxs` as 12-bit.** Change in
   `hone/src/hone/neurons.py:361`:

   ```python
   gradient[cname + "tq_idxs"] = topk_idx.to("cpu")
   ```

   to use `pack_12bit_indices` exactly like the legacy path
   does at `hone/src/hone/compress.py:612`. Note that
   `topk_idx` indexes into the FULL flat `d_flat`-dim rotated
   vector, so the packer must accept indices up to
   `max(d_flat) - 1` ≈ 100M (the largest single 8B-A1B param).
   The current `pack_12bit_indices` caps at 4096 (12-bit max);
   we'll need either a 28-bit packing variant (`(d_flat * topk)
   // target_chunk`-size index space, 4 B per index instead of
   8 B → 2× saving, contributes ~13 GB → ~6.5 GB) OR
   chunk-then-pack like the legacy path so each chunk's
   indices fit in 12 bits (matches the `0.75d` legacy budget,
   contributes 0.75d ≈ 6.5 GB → ~1.6 GB).

   Saves ~19 GB at the larger packing, ~25 GB at chunk-then-pack.

   Decoder updates required:
   - `hone/neurons/validator.py:5380+` (FU1 decode) — unpack
     before the `batch_decompress_turboquant` call.
   - `hone/src/hone/neurons.py:1284-1335` and
     `hone/src/hone/neurons.py:1055-1080` (outer_step TQ
     decode branch) — same.
   - `hone/src/hone/neurons.py:2671-2687`
     (`check_uid_index_overlap`) — currently SKIPS TQ params;
     no change needed unless we want overlap detection on
     the new wire format.

2. **1-bit-pack `tq_meta['sign_bits']`.** It's currently uint8
   (1 byte per coordinate of the full `d`-dim rotated vector).
   At b=1 it would be 0.125d. The pack/unpack helpers can mirror
   `pack_2bit_values` / `unpack_2bit_values` in
   `hone/src/hone/compress.py:167+`. Saves ~7 GB more (drops
   d → d/8 on the sign-bits contribution).

   Decoder update: `hone/src/hone/turboquant.py:500-510`
   (`dequantize_turboquant` consumes `sign_bits` directly).

3. **Pack `tq_codes` to `main_bits`.** Currently uint8 (1 byte
   per coord) but `main_bits` is `b - 1 = 3` in prod mode. Could
   pack 3-bit (every 8 coords → 3 bytes). Saves ~1.6 GB more.

   Decoder update: same call site as (1).

After all three fixes, payload would be approximately:
- `tq_idxs` (12-bit chunked): 0.75d
- `tq_codes` (3-bit packed): 0.1875d
- `tq_meta.sign_bits` (1-bit packed): 0.125d
- meta overhead: ~500 B per param

Total: **~1.06d ≈ 9 GB** for 8B-A1B — within parity of the legacy
path's ~7 GB and worth re-enabling once tested.

The mitigation is reversible: flipping the hparam back to `true`
restores the bloated wire format. Do not flip `true` again until
**all three** wire-format fixes above have shipped (and ideally
been A/B-tested via `turboquant_ab_compare.py` for cross-peer
quality regression).

## Troubleshooting

**`PASS_RATE_THRESHOLD` is too aggressive for my snapshot.**
Override with `--pass-rate-threshold 0.6` and re-run; the operator
contract is still that the failing class is per-class, not aggregate.
A `0.6` threshold means "60% of tensors in every class must clear KS"
which is workable when the snapshot was taken very early in training.

**The CLI says `kstest` is missing.**
scipy is a hard dep of `hone` (see `pyproject.toml`); `pip install
scipy` then re-run. The codec's Lloyd-Max fit also needs scipy.

**`SIGUSR2` does nothing — no snapshot file appears.**
The handler logs to the miner's stdout when it fires. Check the pm2
logs for `[P6a] dumped EF snapshot` or `[P6a] EF snapshot handler
failed`. The handler is registered ONLY on the master rank; if you
have multi-rank PP, target the rank-0 process. On macOS the handler
also installs only on the main thread; targeting a worker thread
silently no-ops.

**The cache dir `hone/src/hone/_turboquant_cache/` keeps growing.**
Each `(d, b)` pair adds ~1.6KB; even probing every distinct param
shape on an 8B-A1B model produces <100KB total. Safe to delete the
dir at any time — the codec rebuilds entries on demand.
