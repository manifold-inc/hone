# Compression Tuning Guide

## Overview

Hone uses a DeMo-based compression pipeline to reduce gradient communication bandwidth by roughly 100x while maintaining convergence. The pipeline is adapted from the DeMo paper (Ryabinin et al., ICLR 2026) and extended with 12-bit index packing and configurable quantization.

In decentralized training, every peer must upload compressed gradients to shared storage each window. The compression ratio directly determines how much bandwidth each peer needs and how fast windows can turn over. The defaults are conservative — there is room to push compression harder depending on your bandwidth constraints and tolerance for gradient approximation error.

## Pipeline Steps

The compression pipeline processes each parameter tensor independently, in order:

1. **Momentum accumulation** — The raw gradient is added to a persistent error-feedback buffer with exponential decay (`momentum_decay`). This buffer accumulates gradient signal across windows, ensuring that information lost to sparsification in one window is preserved and eventually communicated.

2. **DCT transform** — The accumulated buffer is reshaped into 2D blocks of size `target_chunk x target_chunk` and transformed with a blockwise Discrete Cosine Transform. DCT concentrates gradient energy into fewer low-frequency coefficients, making the subsequent top-k selection far more effective. DCT basis matrices are precomputed at initialization for all tensor shapes in the model.

3. **Top-k sparsification** — Only the `topk_compression` largest values per chunk (by magnitude) are retained. Everything else is zeroed. This is the primary source of compression — a chunk of 4096 elements reduced to 32 values is a 128x reduction before index/value encoding.

4. **12-bit index packing** — The indices of selected values are packed into a 12-bit representation (pairs of indices packed into 3 bytes). This gives a 25% size reduction over int16 indices and supports chunk sizes up to 4096 elements (12-bit max = 4095).

5. **Value quantization** — Selected values are quantized. The number of bins and range (in standard deviations) are configurable. With `quantization_bins: 4`, values are mapped to 2-bit codes via a lookup table; with `quantization_bins: 256`, full 8-bit quantization is used. The quantization parameters (min, scale, codebook) are stored alongside the quantized values for exact dequantization.

6. **Momentum subtraction** — The communicated values (after dequantization) are subtracted from the error-feedback buffer, scaled by `momentum_subtraction_alpha`. This closes the error-feedback loop: the buffer retains the residual that wasn't communicated, which will be accumulated into the next window's transmission.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `topk_compression` | `32` | Number of top values to keep per chunk |
| `target_chunk` | `64` | Chunk size for DCT blocking |
| `use_dct` | `true` | Enable DCT transform before top-k |
| `momentum_decay` | `0.999` | Momentum buffer decay coefficient |
| `momentum_subtraction_alpha` | `0.2` | Fraction of communicated values subtracted from buffer |
| `quantization_bins` | `4` | Number of quantization bins (4 = 2-bit, 256 = 8-bit) |
| `quantization_range` | `6` | Quantization range in standard deviations |

These are set in `hparams/hparams.json` at the top level.

## Tuning Guide

### Bandwidth vs. quality tradeoff

**`topk_compression`** is the primary knob. Lower values mean less bandwidth and more gradient information lost per window. The DeMo paper demonstrates that k=2 works at 1B scale with DCT enabled. The default of 32 is conservative and suitable for most setups. If you are bandwidth-constrained, try k=8 or k=4 — the error-feedback mechanism ensures that lost information is eventually communicated in subsequent windows, so aggressive sparsification degrades convergence rate but does not cause divergence.

**`use_dct`** should always be enabled. DCT significantly improves compression quality at the same k value, especially for low k. Without DCT, the top-k selection operates on raw gradient values, which are typically diffuse across the chunk. With DCT, energy is concentrated into a few large coefficients, so top-k captures a much larger fraction of the total gradient norm. The compute cost of the DCT is negligible (precomputed basis matrix multiplication).

**`target_chunk`** controls the DCT block size. 64 is the standard from the DeMo paper. Larger chunks provide a better DCT basis (more frequency resolution) but increase compute. Smaller chunks reduce the maximum index value, potentially allowing tighter index packing. In practice, 64 works well across model sizes from 120M to 35B.

### Momentum tuning

**`momentum_decay`** of 0.999 works best in combination with momentum subtraction. High decay means the buffer accumulates more gradient history, which improves the quality of the top-k selection (the buffer converges toward the true gradient direction over multiple windows). Lower decay (e.g. 0.9) forgets faster and can be more responsive to distribution shifts in the data, but generally hurts convergence.

**`momentum_subtraction_alpha`** of 0.2 means only 20% of the communicated gradient values are subtracted from the error-feedback buffer. The DeMo paper finds 0.2 optimal. Full subtraction (alpha=1.0) evolves the top-k element set too quickly between windows — the selected indices change dramatically each step, which prevents the error-feedback mechanism from converging. Partial subtraction stabilizes the selected set while still allowing gradual evolution.

### Compression ratio calculation

For a 2D weight matrix with `target_chunk=64`:

- Each chunk has 64 x 64 = 4096 elements (stored as bf16 = 8192 bytes uncompressed)
- Top-k=32 keeps 32 values + 32 indices
- With 12-bit index packing: 32 indices = 48 bytes (pairs packed into 3-byte groups)
- With 2-bit quantization (4 bins): 32 values = 8 bytes + small quantization params overhead
- Effective compression: ~8192 / ~60 = roughly 130x per chunk

With the default `quantization_bins: 4` and `topk_compression: 32`, the overall pipeline achieves approximately 100x compression on the full model.

## Outer Optimizer

Aggregated compressed gradients from all peers are decompressed and applied to the model via the outer optimizer. The current configuration uses Nesterov momentum SGD:

- **Learning rate**: 0.4 (`outer_learning_rate` in hparams)
- **Momentum**: 0.9
- **Nesterov acceleration**: enabled

The outer step applies updates one parameter at a time to minimize peak GPU memory — the full decompressed gradient for the entire model is never materialized simultaneously. This is important for large MoE models where total parameter count can be 10x+ the active parameter count.

## Streaming Upload

The `prepare_gradient_buckets()` function splits compressed gradients into N buckets (default 4) for incremental upload during training. Instead of uploading all gradients at end-of-window in a single burst, parameters are distributed across buckets round-robin and each bucket is uploaded as it becomes ready.

This reduces peak upload bandwidth by N-fold and allows upload to overlap with the next window's forward/backward passes. The buckets are reassembled on the receiving end using the `bucket_idx` and `num_buckets` metadata attached to each bucket.

```python
from hone.neurons import prepare_gradient_buckets

buckets = prepare_gradient_buckets(gradient, num_buckets=4)
for bucket in buckets:
    comms.upload(bucket)
```

For miners on consumer internet connections (50-100 Mbps upload), streaming with 4 buckets typically eliminates upload as the training bottleneck.
