# MoE Miner Guide

## Overview

This guide covers running a miner with a Mixture-of-Experts model instead of the default dense model. MoE scales model capacity by using multiple expert MLPs per layer with a learned router -- each token only activates a subset of experts, so compute stays manageable while total parameters grow significantly.

Two MoE configs are available out of the box:

- **1.4B-moe**: 8 experts, top-2 routing. ~5.6B total params, ~1.4B active. Same compute as dense 1.4B.
- **qwen-moe**: 256 experts, top-8 routing, shared expert, 40 layers. Inspired by Qwen3.5-35B-A3B. ~35B total params, ~3B active.

## Hardware Requirements

### For 1.4B-moe (8 experts, top-2)

- 1x GPU with 48GB+ VRAM (A100 40GB is tight, 80GB is comfortable)
- Or 2-4x GPUs with FSDP
- ~5.6B params at bf16 = ~11.2 GB weights + optimizer states

### For qwen-moe (256 experts, top-8, 40 layers)

- 4-8x GPUs with 80GB+ VRAM each (A100 80GB or H100/H200)
- ~35B params at bf16 = ~70 GB weights + optimizer states
- FSDP across all GPUs is required

### General

- 64GB+ system RAM (128GB+ for qwen-moe)
- 500+ Mbps internet
- 200GB+ SSD

## Setup

### 1. Choose your MoE config

Edit `hparams/hparams.json` and change `model_size`:

For the small MoE:

```json
"model_size": "1.4B-moe"
```

For the Qwen-style MoE:

```json
"model_size": "qwen-moe"
```

### 2. Adjust FSDP sharding

Set `dp_shard` to match your GPU count:

```json
"fsdp": {
    "dp_shard": 8,
    "compile": true,
    "mixed_precision": "bfloat16"
}
```

### 3. Environment variables

Same `.env` setup as dense miner -- R2 credentials, wallet, HF token.

## Launching

### 1.4B-moe on a single GPU

```bash
python neurons/miner.py \
  --wallet.name my_wallet \
  --wallet.hotkey my_miner_hotkey \
  --netuid 5
```

### 1.4B-moe on 2 GPUs (FSDP)

```bash
torchrun --nproc_per_node=2 neurons/miner.py \
  --wallet.name my_wallet \
  --wallet.hotkey my_miner_hotkey \
  --netuid 5
```

### qwen-moe on 8 GPUs (FSDP)

```bash
torchrun --nproc_per_node=8 neurons/miner.py \
  --wallet.name my_wallet \
  --wallet.hotkey my_miner_hotkey \
  --netuid 5
```

### Test mode

```bash
python neurons/miner.py \
  --wallet.name my_wallet \
  --wallet.hotkey my_miner_hotkey \
  --netuid 5 \
  --test
```

## MoE-Specific Behavior

### Auxiliary loss

MoE adds a load-balancing auxiliary loss to the training objective. This prevents routing collapse where all tokens go to the same few experts. The coefficient is controlled by `moe_aux_loss_coeff` (default 0.01 for 1.4B-moe, 0.001 for qwen-moe). You'll see this as a small addition to the reported training loss.

### Gradient compression

Each expert's gradients compress independently through the DeMo pipeline. Expert gradients tend to be naturally sparse (most tokens don't route to any given expert), so compression quality is often better than dense models at the same top-k.

### Memory considerations

MoE uses significantly more memory than dense models at the same active parameter count because all experts must be stored. For 256 experts with FFN dim 512 and model dim 2048:

- Per expert: gate_proj (2048x512) + up_proj (2048x512) + down_proj (512x2048) = ~6.3M params
- 256 experts per layer x 40 layers = ~64B params just in experts
- FSDP sharding is essential

### Shared expert

The qwen-moe config includes a shared expert that processes all tokens unconditionally. This provides baseline capacity even when routing is suboptimal. It's a separate MLP that runs on every token alongside the routed experts.

## Custom MoE Configurations

You can create a custom config by making a new JSON file in `hparams/`:

Example `hparams/custom-moe.json`:

```json
{
    "tokenizer_name": "google/gemma-2-2b",
    "hidden_size": 2048,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "intermediate_size": 5504,
    "max_position_embeddings": 4096,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-5,
    "vocab_size": 256000,

    "use_moe": true,
    "num_experts": 16,
    "moe_top_k": 4,
    "moe_intermediate_size": 1024,
    "shared_expert_intermediate_size": 1024,
    "moe_aux_loss_coeff": 0.005
}
```

Then set `"model_size": "custom-moe"` in `hparams.json`.

### Hybrid dense/MoE

You can make only certain layers use MoE while keeping others dense:

```json
{
    "use_moe": true,
    "num_experts": 8,
    "moe_top_k": 2,
    "moe_layers": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
}
```

This makes layers 0-5 and 18-23 use dense MLPs, and layers 6-17 use MoE. This is useful for memory savings (early/late layers often don't benefit as much from MoE).

## Troubleshooting

### OOM

- Reduce `micro_batch_size` first (try 2 or 1).
- Add more GPUs with FSDP.
- Use `moe_intermediate_size` to reduce per-expert FFN size.
- Use `moe_layers` to limit which layers are MoE.

### Slow training

- MoE forward pass is slower than dense due to expert dispatch overhead.
- `torch.compile` (`fsdp.compile: true`) helps significantly.
- Reduce `num_experts` if compute-bound.
- The qwen-moe config with 256 experts can be slow without optimized kernels.

### Routing collapse

If all tokens route to the same experts, increase `moe_aux_loss_coeff` (try 0.02 or 0.05). Check wandb logs for per-expert utilization if available.
