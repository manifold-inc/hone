# Genesis MoE

## Scaling decentralized training beyond dense models

Dense models have a ceiling. Past a certain parameter count, the compute required to train them grows linearly with model size — every token touches every parameter. Mixture-of-Experts breaks this constraint. A model with 8 experts and top-2 routing has 4x the parameters of its dense equivalent but roughly the same training FLOPs per token. You get the capacity without the compute tax.

The question was never whether MoE works — it does, spectacularly, from GShard to Mixtral to DeepSeek-V3. The question was whether it works *decentralized*. Over unreliable networks, with untrusted peers, using compressed gradients exchanged through object storage.

It does.

## Architecture

Genesis MoE extends the existing LoopLM architecture with a straightforward modification: the SwiGLU MLP in each transformer block is replaced with a gated mixture of independent SwiGLU experts.

- **8 experts per layer**, top-2 routing
- **Shared attention** — the self-attention layers remain dense and identical across all tokens
- **Per-token gating** — a learned linear gate selects the two most relevant experts for each token
- **Load balancing** — an auxiliary loss penalizes uneven expert utilization, preventing routing collapse

The config is backward-compatible. Setting `use_moe: false` (the default) gives you the same dense LoopLM as before. MoE can be enabled per-layer or globally, allowing hybrid architectures where early layers are dense and later layers are sparse.

## Why this matters for decentralized training

MoE changes the math on what's trainable over the internet. A 1.4B dense model requires every peer to process all 1.4B parameters per step. A 1.4B-active MoE model with 8 experts has ~5.6B total parameters but only activates 1.4B per token. The model is dramatically more capable without increasing the per-step compute burden on individual miners.

The compression pipeline handles MoE naturally. Each expert's gradients compress independently through the existing DeMo pipeline — momentum accumulation, DCT transform, top-k sparsification, 12-bit index packing, 8-bit quantization. The per-expert gradient structure actually plays well with top-k: expert gradients are naturally sparse (most tokens don't route to any given expert), so the top-k selection captures a larger fraction of the useful signal.

The load balancing auxiliary loss is critical in the decentralized setting. Without it, routing collapse concentrates all tokens on one or two experts, which wastes capacity and creates gradient imbalance across peers. The auxiliary loss keeps routing distributed, ensuring that all experts receive meaningful gradient signal from every peer.

## Pipeline parallelism: training models larger than a single GPU

MoE alone scales parameter count, but the model still needs to fit somewhere during training. For models that exceed single-node memory, we've implemented pipeline parallelism with ResBM (Residual Bottleneck Models) for activation compression.

The problem with pipeline parallelism over the internet is bandwidth. A standard pipeline stage boundary transmits activations of shape `(batch, seq_len, hidden_dim)` — for a 2048-dim model with sequence length 4096, that's 64 MiB per microbatch per boundary in bf16. At consumer internet speeds (80 Mbps), that's a 6-second stall per microbatch. Training would be completely communication-bound.

ResBM solves this with learned bottleneck layers at pipeline boundaries:

- **128x activation compression** — hidden_dim 2048 compresses to bottleneck_dim 16
- **Identity-preserving residual path** — the bottleneck operates alongside the residual stream, not on it, so gradient flow through the identity path is unimpeded
- **~3.3% parameter overhead** — the encoder/decoder pairs add negligible parameters
- **End-to-end trainable** — bottleneck weights train with the same optimizer as the rest of the model, no special constrained optimization needed

With 128x compression, that 64 MiB activation becomes 0.5 MiB — transferable in ~50ms at 80 Mbps. Pipeline parallelism becomes practical over consumer internet.

## Compression improvements

Alongside MoE and pipeline parallelism, we've tuned the DeMo-based gradient compression:

- **DCT enabled** — the Discrete Cosine Transform concentrates gradient energy into fewer coefficients before top-k selection, improving compression quality at the same bandwidth
- **Top-k reduced from 64 to 32** — with DCT compensating for quality, we halve the per-window upload with no convergence penalty
- **Nesterov momentum on the outer optimizer** — the outer SGD step that applies aggregated gradients now uses Nesterov acceleration (momentum=0.9), matching the configuration used by DiLoCo and improving convergence of the federated update
- **Streaming gradient upload** — instead of uploading all compressed gradients at end-of-window, parameters are split into buckets and uploaded incrementally during training, reducing peak bandwidth

## What's next

The infrastructure is in place. The immediate path forward:

1. **Ablation runs** — dense vs MoE at matched active parameters, measuring convergence per token and per FLOP
2. **Pipeline parallelism validation** — multi-node training with ResBM boundaries over real internet links
3. **Expert parallelism** — distributing different experts across GPUs within a node for larger expert counts
4. **Scaling to 7B+ active parameters** — MoE with pipeline parallelism removes the single-node constraint entirely

The system trains dense and MoE with the same codebase, the same compression pipeline, the same scoring mechanism. The only difference is a config flag. That's the point — making the architecture decision orthogonal to the distributed training infrastructure, so we can scale model capacity without scaling complexity.
