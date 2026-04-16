# Mixture-of-Experts Training Guide

## Overview

Hone supports Mixture-of-Experts (MoE) as a drop-in replacement for the dense LoopLM architecture. The modification is straightforward: the SwiGLU MLP in each transformer block is replaced with multiple independent SwiGLU expert MLPs and a learned linear router.

Each token is routed to the top-k experts (default k=2) based on router logits. Total parameters scale with `num_experts`, but active parameters per token remain fixed — a model with 8 experts and top-2 routing has roughly 4x the total parameters of its dense counterpart but the same training FLOPs per token.

An optional shared expert processes all tokens unconditionally alongside the routed experts, providing baseline capacity regardless of routing quality.

Setting `use_moe: false` (the default) gives you the standard dense LoopLM. MoE can be enabled globally or per-layer via the `moe_layers` field, allowing hybrid architectures where early layers are dense and later layers are sparse.

## Configuration

### Option 1: Small MoE (1.4B-moe)

Set `model_size` in `hparams/hparams.json`:

```json
"model_size": "1.4B-moe"
```

This loads `hparams/1.4B-moe.json`: 8 experts, top-2 routing, same 2048-dim 24-layer architecture as the dense 1.4B model. Total params ~5.6B, active ~1.4B per token.

### Option 2: Qwen-style MoE (35B-A3B)

```json
"model_size": "35B-A3B"
```

This loads `hparams/35B-A3B.json`: 256 experts, top-8 routing, shared expert, 40 layers, aggressive GQA (2 KV heads). Inspired by Qwen3.5-35B-A3B. Total params ~35B, active ~3B per token.

### Option 3: Local override

Create `hparams/hparams-local-run.json` and run with `--local`:

```json
{
    "use_moe": true,
    "num_experts": 8,
    "moe_top_k": 2,
    "moe_aux_loss_coeff": 0.01
}
```

This merges your overrides on top of the base model config, so you only need to specify the fields you want to change.

## MoE Config Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_moe` | bool | `false` | Master toggle for MoE |
| `num_experts` | int | `8` | Number of expert MLPs per layer |
| `moe_top_k` | int | `2` | Experts activated per token |
| `moe_intermediate_size` | int | `null` | FFN dimension per expert (`null` = use dense `intermediate_size`) |
| `shared_expert_intermediate_size` | int | `null` | FFN dim for always-active shared expert (`null` = no shared expert) |
| `moe_layers` | list[int] | `null` | Which layers use MoE (`null` = all layers) |
| `moe_aux_loss_coeff` | float | `0.01` | Coefficient for load-balancing auxiliary loss |

## How MoE Interacts with Training

The model forward pass returns `(logits, aux_loss)` when MoE is active, or plain `logits` when running dense. The training loop handles both:

```python
if isinstance(model_output, tuple):
    logits, aux_loss = model_output
else:
    logits = model_output
    aux_loss = None

loss = compute_loss(logits, labels)
if aux_loss is not None:
    loss = loss + aux_loss
```

The auxiliary loss is a load-balancing term that penalizes uneven expert utilization. Without it, routing collapses — all tokens concentrate on one or two experts, wasting capacity and creating gradient imbalance across peers. The coefficient `moe_aux_loss_coeff` controls its weight relative to the cross-entropy loss. The default of 0.01 is standard; the 35B-A3B config uses 0.001 because with 256 experts the per-expert imbalance signal is naturally smaller.

Each expert's gradients compress independently through the DeMo pipeline. Expert gradients are naturally sparse (most tokens don't route to any given expert), so top-k sparsification captures a larger fraction of the useful signal compared to dense layers.

## Running MoE

Same launch commands as dense training. Change `model_size` in `hparams/hparams.json` and run:

```bash
# Single GPU miner
python neurons/miner.py \
    --wallet.name <wallet> --wallet.hotkey <hotkey> \
    --netuid 268

# Multi-GPU miner with FSDP
torchrun --nproc_per_node=8 neurons/miner.py \
    --wallet.name <wallet> --wallet.hotkey <hotkey> \
    --netuid 268

# Validator (same as dense)
torchrun --nproc_per_node=4 neurons/validator.py \
    --wallet.name <wallet> --wallet.hotkey <hotkey> \
    --netuid 268
```

No changes to the compression pipeline, scoring, or communication layer. The architecture decision is orthogonal to the distributed training infrastructure.

## Design Decisions

### Why top-k routing

Top-k routing with a learned linear gate is the standard for good reason. Hash routing (deterministic assignment) prevents the model from learning which experts are relevant to which tokens. Random routing destroys specialization entirely. Learned top-k routing lets the model develop genuine expert specialization — different experts learn different skills — while keeping the routing decision differentiable through the softmax over gate logits.

### Why shared experts help

A shared expert processes every token regardless of what the router decides. This provides baseline capacity even when routing is imperfect — early in training, before the router has learned meaningful assignments, the shared expert carries the model. In the Qwen/DeepSeek pattern, the shared expert acts as a residual path: it handles common, general-purpose computation while routed experts specialize in less frequent patterns. The 35B-A3B config uses a shared expert with `shared_expert_intermediate_size: 512`.

### Why small expert FFN dims work

The 35B-A3B config uses 256 experts with `moe_intermediate_size: 512` — each expert is tiny. This follows the Qwen/DeepSeek finding: many small experts outperform fewer large ones at the same total parameter count. Small experts specialize more tightly, and with enough of them (256), the top-k selection (k=8) can compose fine-grained capabilities. The combinatorial space of 8-of-256 selections is vastly richer than 2-of-8.

### Why Muon works well with MoE

The Muon optimizer maintains high-rank gradient representations through its Newton-Schulz orthogonalization step, which prevents the low-rank collapse that AdamW can exhibit on sparsely-activated parameters. In MoE, each expert sees only a fraction of tokens per step, so gradient estimates are inherently noisier and lower-rank than in dense models. Muon's orthogonalization counteracts this, maintaining effective gradient signal even for infrequently-activated experts. In practice, this translates to 48-52% faster convergence compared to AdamW on MoE architectures at matched compute budgets.
