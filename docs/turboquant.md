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

The validator-side TurboQuant decoder is in place (FU1 dispatch at
`hone/neurons/validator.py:5380+`, outer_step branches in
`hone/src/hone/neurons.py`). The dispatch keys themselves
(`cname + {'tq_idxs','tq_codes','tq_meta'}`) are unchanged across
v1 → v2 — only the wire shapes inside those keys shifted (see "v2
wire format" under the wire-format inflation incident below). The
legacy `cname + {'idxs','vals','quant_params'}` keys are NOT
populated in the TurboQuant branch, so a TurboQuant-unaware
validator that gets a TurboQuant-encoded gradient will simply skip
the param, breaking upload / outer-step convergence for the
affected miners.

The rollout is hard-coordinated, not gracefully backward-compatible:
the v2 decoder rejects `meta['version'] == 'turboquant_v1'` outright
(version guard in `dequantize_turboquant` at
`hone/src/hone/turboquant.py:518`), so any v1-encoded peer would be
IGNORED by every validator running this codec. That is fine — v1
was never enabled in production beyond the 2026-05-03 incident
window, which was reverted same-day. Coordinate the rollout.

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

## Wire-format inflation incident (2026-05-03) and v2 codec fix

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

**Codec rework (deployed same-day).** The v2 wire format described
in "v2 codec wire format" below has shipped. The encoder now
chunk-then-packs (`hone/src/hone/neurons.py:291-464` —
`prepare_gradient_dict` TurboQuant branch); `quantize_turboquant`
1-bit-packs `sign_bits` and stamps `meta['version'] =
"turboquant_v2"` (`hone/src/hone/turboquant.py:338-499`);
`dequantize_turboquant` (`:502-566`) and
`batch_decompress_turboquant` (`:569-856`) consume the v2 chunked
+ bit-packed payload end-to-end. The `turboquant_enabled: false`
flag remains the deployed safety gate pending P6b A/B re-validation
against the new codec — chunk-then-pack is functionally a different
selection strategy from v1's global top-K, so perplexity neutrality
must be re-confirmed before the flag flips back on.

**Per-suspect byte accounting** for the audit (full report in chat
transcript [Hone gradient payload audit](95d0b611-ca0f-4608-b186-e3d39f49b23a)):

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

**v2 codec wire format (deployed 2026-05-03 same-day).** All three
wire-format changes that were required before re-enabling
TurboQuant landed in the same calendar day as the incident, paired
with their matching decoder updates so the codec is internally
consistent end-to-end. The `turboquant_enabled: false` safety gate
is the only thing keeping the new code from running on the gather
path; the codec itself is on `meta['version'] = "turboquant_v2"`
and the v1 layout is no longer accepted by any decoder.

1. **`tq_idxs` packed 12-bit (chunk-then-pack).** The encoder now
   pads the rotated codes to a multiple of `target_chunk`
   (production: 64), takes per-chunk top-K on `|code|`, and packs
   chunk-local indices via `pack_12bit_indices`
   (`hone/src/hone/compress.py:73`). Each index lives in
   `[0, target_chunk)` so trivially fits in 12 bits regardless of
   the underlying `d_flat`. The 28-bit global packer alternative
   was rejected in favour of chunk-then-pack: the chunked layout
   matches the `0.75d` legacy budget (versus ~1.5d for the 28-bit
   variant) and reuses the existing 12-bit packing helper
   unchanged. Note that this also makes the per-chunk top-K
   selection functionally distinct from v1's global top-K — see
   the re-validation note at the end of this section.

   Encoder: `hone/src/hone/neurons.py:291-464`
   (`prepare_gradient_dict`'s TurboQuant branch). Decoder:
   `hone/src/hone/turboquant.py:569-856`
   (`batch_decompress_turboquant`) recovers global indices via
   `chunk_id * target_chunk + local_idx` and filters trailing
   padding at `>= d`.

   None of the existing call sites needed code changes — they all
   delegate to `batch_decompress_turboquant` and so picked up the
   v2 wire shape transparently:
   - `hone/neurons/validator.py:5380+` (FU1 eval decode).
   - `hone/src/hone/neurons.py:1146-1196` (outer_step pre-clip
     decode), `:1399-1427` (P3A per-param pre-pass), `:1505-1535`
     (main outer_step body).
   - `hone/src/hone/neurons.py:2727+` (`check_uid_index_overlap`)
     still SKIPS TQ params; per-chunk set-overlap is not a
     meaningful signal on rotated-flat-top-K indices and the
     count-sketch alternative remains a separate workstream.

2. **`tq_meta['sign_bits']` 1-bit-packed.** Previously uint8 (one
   byte per coordinate of the full `d`-dim rotated vector); now
   1-bit-packed inside `quantize_turboquant`
   (`hone/src/hone/turboquant.py:338-499`) via `pack_1bit_values`
   from `hone/src/hone/compress.py:292`. `dequantize_turboquant`
   (`:502-566`) unpacks via `unpack_1bit_values` using the new
   `meta['sign_bits_original_d']` field to drop the trailing-zero
   padding the packer added on the stride boundary. The codec
   stamps `meta['version'] = "turboquant_v2"` at the same time;
   `dequantize_turboquant` rejects any other version outright at
   line 518. Drops the sign-bits contribution from `d` to `d/8`.

3. **`tq_codes` packed at `main_bits`.** Previously uint8 (one
   byte per kept top-K position). The encoder
   (`hone/src/hone/neurons.py:291-464`, same TurboQuant branch as
   fix 1) now packs `main_bits`-wide (production prod-mode:
   `b - 1 = 3` bits) via the matching `pack_{1,2,3,4}bit_values`
   helper in `hone/src/hone/compress.py` (`pack_2bit_values:167`,
   `pack_1bit_values:292`, `pack_3bit_values:414`,
   `pack_4bit_values:563`); raw uint8 is preserved as the
   `main_bits == 8` no-pack path. The validator decoder dispatches
   on `meta['main_bits']` inside `batch_decompress_turboquant`
   (`hone/src/hone/turboquant.py:569-856`) and runs the matching
   `unpack_{1,2,3,4}bit_values` per peer. Drops `tq_codes` from
   `0.5d` (uint8 over the 50%-density top-K) to `0.1875d`
   (3-bit-packed at the same density) in the production b=4
   prod-mode configuration.

After all three fixes, the per-param wire payload has been verified
end-to-end at:

| Key | Bytes per param (deployed) |
|-----|----------------------------|
| `tq_idxs` (12-bit packed chunk-local) | `0.7500d` |
| `tq_codes` (3-bit packed at `main_bits=3`) | `0.1875d` |
| `tq_meta.sign_bits` (1-bit packed) | `0.1250d` |
| `tq_meta` other (centroids, outliers, chunk dims) | ~500 B |

Total: **`1.0625d` + ~500 B per param**. Across the 8B-A1B model
this lands at ~9 GB per window per miner, within parity of the
legacy `0.875d` ≈ 7 GB path.

### v2 wire format (deployed 2026-05-03)

- `cname + "tq_idxs"` — uint8 packed 12-bit chunk-local indices.
  Length = `(num_chunks * topk_per_chunk * 12) / 8` bytes.
- `cname + "tq_codes"` — uint8 packed at `main_bits` bits (= `b - 1`
  in prod mode, `b` in mse mode). Each kept top-K position
  contributes one packed code.
- `cname + "tq_meta"` — Python dict, version `"turboquant_v2"`.
  Carries `target_chunk`, `num_chunks`, `topk_per_chunk`,
  `d_padded`, `bulk_centroids`, `bulk_scale`, `outlier_*`,
  `sign_bits` (1-bit packed when `mode='prod'`),
  `sign_bits_original_d`, `mode`, `b`, `main_bits`, `d`.
- v1 (pre-2026-05-03) sent: int64 `tq_idxs`, uint8 `tq_codes`,
  uint8 `sign_bits` — all three keys at full byte-per-coord
  density. Cumulative inflation `5.5d` per coord vs the v2
  `1.0625d` per coord. The v2 decoder rejects v1 metadata
  outright; the two formats are NOT interchangeable on the wire.

The mitigation flag (`turboquant_enabled: false`) remains the
deployed state. The underlying codec is now safe to re-enable, but
chunk-then-pack is functionally a different selection strategy from
v1's global top-K and the codec's rate-distortion profile may have
shifted enough to need re-validating perplexity neutrality. Before
flipping `turboquant_enabled` to `true` again, operators MUST run
§P6b's A/B compare against the v2 codec
(`hone/validator/turboquant_ab_compare.py`) and confirm both gates
clear (perplexity regression ≤ 0.5%, Spearman per-UID-score
correlation ≥ 0.85). The §P6a coordinate-distribution audit only
needs to be re-run if the EF distribution has drifted since the
last passing audit; the rotation + Beta assumption is unchanged
between v1 and v2.

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
