# Pipeline Parallelism in Hone

## Overview

Pipeline parallelism (PP) splits a model's layers across multiple machines, enabling training of models that exceed a single node's memory capacity. In standard PP, each stage sends full activations to the next stage during the forward pass and full gradients back during the backward pass.

In decentralized training, inter-node bandwidth is the bottleneck. Standard PP sends full-dimensional activations between stages, which is impractical when nodes are connected over consumer internet rather than InfiniBand. A single activation transfer can take seconds at typical upload speeds, stalling the entire pipeline.

Hone solves this with **ResBM** (Residual Bottleneck Models), which compress activations by 128x at pipeline boundaries. This reduces inter-stage communication from hundreds of megabytes to single-digit megabytes, making PP feasible over ordinary internet connections.

## How ResBM Works

ResBM places a lightweight bottleneck encoder/decoder pair at each pipeline stage boundary. The key insight is that the bottleneck operates **alongside** the residual stream, not on it. This is critical for training stability -- the residual path is never compressed, so gradient flow through the identity shortcut is unimpeded.

At each boundary:

1. A **bottleneck encoder** compresses activations from `hidden_dim` to `bottleneck_dim` (e.g., 2048 -> 16) via a two-layer network with SiLU activation.
2. An **identity projection** (rectangular identity matrix) truncates the residual to `bottleneck_dim` dimensions.
3. The encoder output and truncated residual are summed and transmitted.
4. On the receiving stage, a **bottleneck decoder** expands back from `bottleneck_dim` to `hidden_dim`.
5. A matching **identity projection** zero-pads the compressed residual back to `hidden_dim`.
6. The decoder output and zero-padded residual are summed, recovering the full-dimensional hidden state.

The identity projection uses a rectangular identity matrix: truncation when going from high to low dimension, zero-padding when going from low to high. This preserves the identity property of the residual connection (ResBM paper, Eq. 6) and ensures that at initialization (when bottleneck weights are near-zero), the stage boundary behaves approximately as an identity function.

The encoder/decoder pairs add roughly **3.3% parameter overhead** to the model.

### Bandwidth Math

**Without ResBM**, the activation size at each pipeline boundary is:

```
activation_bytes = batch_size * seq_len * hidden_dim * 2   (bf16 = 2 bytes per element)
```

For a model with `hidden_dim=8192`, `seq_len=4096`, `batch_size=4`:

```
4 * 4096 * 8192 * 2 = 256 MiB per boundary
```

**With 128x ResBM** (`bottleneck_dim = 8192 / 128 = 64`):

```
4 * 4096 * 64 * 2 = 2 MiB per boundary
```

At a typical 1 Gbps internet connection:

| | Size | Transfer time |
|---|------|---------------|
| Without ResBM | 256 MiB | ~2.0 seconds |
| With 128x ResBM | 2 MiB | ~16 ms |

The same compression applies to gradients flowing backward, so both the forward and backward pass benefit equally.

## Architecture: Two Nodes, One Miner

Consider a 400B-parameter MoE model split across two 8xH200 nodes:

```
Node 1 (Stage 0)                    Node 2 (Stage 1)
+----------------------------+      +----------------------------+
| embedding                  |      | layers 40-79               |
| layers 0-39                |      | RMSNorm                    |
|                            |      | lm_head                    |
| 8x H200 GPUs (FSDP)       |      | 8x H200 GPUs (FSDP)       |
+----------------------------+      +----------------------------+
         |                                    |
         +--- TCP + ResBM (~2 MiB) ----------+
```

**Key points:**

- **Both nodes register as the same Bittensor miner.** They share the same wallet and hotkey, presenting as a single UID to the network. The subnet sees one miner producing gradients, not two half-miners.
- **Intra-node communication** uses NVLink (900 GB/s) for FSDP's all-gather and reduce-scatter across the 8 local GPUs.
- **Inter-node communication** uses TCP with ResBM compression. Only compressed activations (~2 MiB) and gradients traverse the internet link.
- Each node runs FSDP independently across its 8 GPUs to shard its stage's parameters.

## Launch Commands

Both nodes run `torchrun` with the same wallet credentials and complementary `--pp-stage` flags.

**Node 1 (Stage 0):**

```bash
torchrun --nproc_per_node=8 neurons/miner.py \
  --wallet.name my_wallet \
  --wallet.hotkey my_miner \
  --netuid 268 \
  --pp-stage 0 \
  --pp-num-stages 2 \
  --pp-peer node2-ip:50051
```

**Node 2 (Stage 1):**

```bash
torchrun --nproc_per_node=8 neurons/miner.py \
  --wallet.name my_wallet \
  --wallet.hotkey my_miner \
  --netuid 268 \
  --pp-stage 1 \
  --pp-num-stages 2 \
  --pp-peer node1-ip:50051
```

The `--pp-peer` flag points each stage to its counterpart. In a multi-stage setup (3+ stages), each stage points to the next stage's address.

## Per-Window Training Flow with PP

Within each training window, the PP flow proceeds as follows:

1. **Checkpoint loading.** Each node loads only its stage's layers from the latest checkpoint. Stage 0 loads the embedding and layers 0-39; Stage 1 loads layers 40-79, the final norm, and the language model head.

2. **Forward pass.** Stage 0 processes its layers, then the output boundary encodes the activations via ResBM (hidden_dim -> bottleneck_dim). The compressed tensor (~2 MiB) is sent over TCP to Stage 1.

3. **Forward continuation.** Stage 1 receives the compressed activation, decodes it via its input boundary (bottleneck_dim -> hidden_dim), processes its layers, and computes the loss.

4. **Backward pass.** Gradients flow in reverse. Stage 1 computes gradients through its layers and its input boundary decoder, compresses the gradient via the output boundary encoder on Stage 0's side, and sends ~2 MiB back over TCP. Stage 0 receives, decodes, and backpropagates through its layers.

5. **Microbatch pipelining.** A 1F1B (one-forward-one-backward) schedule is used to keep both stages busy. While Stage 1 processes the forward pass for microbatch N, Stage 0 is already running the forward pass for microbatch N+1.

6. **DeMo compression.** At the end of the window, each stage independently runs DeMo (Decoupled Momentum) compression on its own parameters. This is purely local -- no cross-stage coordination is needed.

7. **Gradient upload.** Both stages upload their compressed gradients to R2 storage under the same UID. The validator and other miners see a single coherent gradient update.

## Configuration

Enable pipeline parallelism in `hparams/hparams.json`:

```json
"pipeline": {
    "enabled": true,
    "num_stages": 2,
    "bottleneck_dim": 16
}
```

### Pipeline Config Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable pipeline parallelism. When false, the model runs on a single node. |
| `num_stages` | int | `4` | Number of pipeline stages. The model's layers are divided evenly across stages. |
| `bottleneck_dim` | int | `16` | Bottleneck dimension for ResBM compression. The compression ratio is `hidden_dim / bottleneck_dim`. For `hidden_dim=2048` and `bottleneck_dim=16`, the ratio is 128x. |

### Tied embeddings are not supported with PP

The model's input `embed_tokens` and output `lm_head` projections live on
*different* stages (stage 0 and the last stage), running in *different*
processes. They cannot share a single `nn.Parameter` across that
process boundary, so models trained with `tie_embeddings: true` would
silently end up with two independent copies of the embedding weight that
diverge under per-stage inner-optimizer updates. The validator runs the
un-carved model with the weights actually tied and would only see stage
0's gradient contribution, so the divergence would be permanent.

`Trainer._init_pp_model` raises a `RuntimeError` at startup if `pipeline.num_stages > 1`
and the model config has `tie_embeddings: true`. To use PP, set
`tie_embeddings: false` in the model config (this is already the case for
`8B-A1B.json` and `35B-A3B.json`). To keep tied embeddings, set
`pipeline.num_stages: 1` and let FSDP handle parallelism within a node.

## Scaling Examples

| Model | Total Params | Active Params | Nodes | GPUs/Node | PP Stages | Activation per boundary |
|-------|-------------|---------------|-------|-----------|-----------|------------------------|
| 1.4B Dense | 1.4B | 1.4B | 1 | 1-8 | 1 (no PP) | N/A |
| Qwen MoE | ~35B | ~3B | 1-2 | 8 | 1-2 | 2 MiB @ 128x |
| 400B-A33B MoE | ~400B | ~33B | 2-4 | 8 | 2-4 | 2 MiB @ 128x |

For the dense 1.4B model, a single node with 1-8 GPUs suffices and PP is unnecessary. The MoE models have large total parameter counts (due to expert weights) but modest active parameter counts, and PP lets miners split the full parameter set across nodes while keeping inter-node traffic manageable.

## Validator with PP

The validator must hold the full model to evaluate miner gradients. There are three options:

1. **Single large node (preferred).** The validator does not train, so it has lower peak memory pressure than miners. A node with sufficient aggregate GPU memory can hold the full model with FSDP sharding alone, no PP needed.

2. **Validator-side PP (not yet implemented).** The validator could use its own PP setup to split the model across nodes for evaluation. This adds complexity to the evaluation loop and is not currently supported.

3. **CPU offloading.** Use `offload_optimizer_states: true` and layer-by-layer CPU offloading to fit the model on a smaller node. This is slower but functional for validation workloads where throughput is less critical.

## Current Status

The ResBM bottleneck layers (`BottleneckEncoder`, `BottleneckDecoder`, `IdentityProjection`) and the pipeline stage abstraction (`PipelineStage`, `PipelineStageBoundary`) are implemented in `hone/src/hone/pipeline.py`. The `create_pipeline_stages` function partitions a model's layers into stages with ResBM boundaries.

**Implemented:**
- ResBM encoder/decoder with identity residual projections
- Pipeline stage partitioning with automatic layer distribution
- Near-identity weight initialization for bottleneck layers
- Forward pass through stages with encode/decode at boundaries

**Not yet built:**
- Inter-node TCP transport for activation/gradient transfer
- Stage-aware checkpoint loading (loading only a stage's subset of layers)
- 1F1B microbatch scheduling across nodes
- CLI flags (`--pp-stage`, `--pp-num-stages`, `--pp-peer`)

Currently, PP stages must run on GPUs within the same node, using `CUDA_VISIBLE_DEVICES` to partition GPUs between stages. True multi-node PP requires the transport layer.
