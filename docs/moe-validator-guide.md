# MoE Validator Guide

## Overview

Running a validator with MoE works identically to dense validation from an operational perspective -- same commands, same scoring mechanism. The validator loads the same MoE model architecture as the miners, evaluates gradients by measuring loss before/after applying each contribution, and sets weights accordingly.

The key difference is hardware: MoE models have more total parameters than their dense equivalents, so the validator needs more GPU memory.

## Hardware Requirements

### For 1.4B-moe validation

- 2-4x GPUs with 48GB+ VRAM each
- 64GB+ system RAM
- FSDP across GPUs

### For qwen-moe validation

- 4-8x GPUs with 80GB+ VRAM each (A100 80GB or H100/H200)
- 128GB+ system RAM
- FSDP across all GPUs
- The validator holds the FULL model (all 256 experts per layer, all 40 layers) since it needs to evaluate any miner's contribution

## Setup

### 1. Match model_size to the network

The validator MUST use the same `model_size` as the miners:

```json
"model_size": "1.4B-moe"
```

or

```json
"model_size": "qwen-moe"
```

If `model_size` doesn't match the miners, gradient shapes won't align and evaluation will fail.

### 2. Set FSDP sharding

```json
"fsdp": {
    "dp_shard": 4,
    "compile": true,
    "mixed_precision": "bfloat16"
}
```

### 3. Environment variables

Same `.env` as always.

## Launching

### 1.4B-moe validator (4 GPUs)

```bash
torchrun --nproc_per_node=4 neurons/validator.py \
  --wallet.name my_wallet \
  --wallet.hotkey my_validator_hotkey \
  --netuid 268
```

### qwen-moe validator (8 GPUs)

```bash
torchrun --nproc_per_node=8 neurons/validator.py \
  --wallet.name my_wallet \
  --wallet.hotkey my_validator_hotkey \
  --netuid 268
```

### With gradient storage

```bash
torchrun --nproc_per_node=8 neurons/validator.py \
  --wallet.name my_wallet \
  --wallet.hotkey my_validator_hotkey \
  --netuid 268 \
  --store-gathers
```

## MoE-Specific Validation Behavior

### Evaluation flow

The evaluation loop is identical to dense:

1. Gather compressed gradients from miners.
2. For each miner: save state, eval loss before, apply gradient, eval loss after, restore state.
3. Score based on loss improvement.

The model returns `(logits, aux_loss)` during evaluation, but only logits are used for scoring -- the aux_loss is training-only.

### Memory during evaluation

The validator evaluates one miner at a time. For each evaluation:

- Save entire model state (all experts) to CPU.
- Apply one miner's gradient.
- Forward pass on eval data.
- Restore saved state.

With MoE, the saved state is larger because it includes all expert parameters. The system uses `offload_gather_results` to move gathered data to pinned CPU memory between evaluations to manage GPU memory.

### Gradient validation

The validator checks gradient shapes and indices before applying. With MoE, each expert parameter has its own gradient -- the validator validates these independently. The index overlap check also runs per-parameter, detecting miners that copy each other's expert gradients.

## Running Validator and Miners on the Same Machine

For testing with 8 GPUs, you can split them:

```bash
# Validator on GPUs 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  --master_port=29500 \
  neurons/validator.py \
  --wallet.name my_wallet \
  --wallet.hotkey validator_hotkey \
  --netuid 268 --test

# Miner 1 on GPU 4
CUDA_VISIBLE_DEVICES=4 python neurons/miner.py \
  --wallet.name my_wallet --wallet.hotkey miner1 --netuid 268 --test

# Miner 2 on GPU 5
CUDA_VISIBLE_DEVICES=5 python neurons/miner.py \
  --wallet.name my_wallet --wallet.hotkey miner2 --netuid 268 --test
```

Set `dp_shard=4` in `hparams.json` for the validator, and the single-GPU miners will ignore `dp_shard` since `world_size=1`.

When using `CUDA_VISIBLE_DEVICES`, the GPUs are renumbered starting from 0 within each process. The validator sees GPUs 0-3 (which are physical GPUs 0-3), and each miner sees GPU 0 (which is its assigned physical GPU).

## Troubleshooting

### OOM during evaluation

- Enable `offload_gather_results` (default true).
- Reduce `validator_sample_micro_bs`.
- Add more GPUs with FSDP.
- The qwen-moe model with 256 experts x 40 layers is very memory-intensive.

### Slow evaluation

- Each miner evaluation requires a full model save/restore cycle.
- With MoE, save/restore takes longer due to more parameters.
- Reduce `uids_per_window` to evaluate fewer miners per window.
- Reduce `gather_peer_count` to gather fewer miners.

### Shape mismatch errors

- Ensure `model_size` matches between validator and miners.
- If miners are running dense and validator is running MoE (or vice versa), gradient shapes will differ and evaluation will fail.
- All participants on the subnet must agree on `model_size`.
