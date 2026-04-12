# Hone — LoopLM Distributed Training on Bittensor

Incentivised distributed training of **Looped Language Models** (LoopLM) on Bittensor. Miners train a shared LoopLM model locally and upload compressed gradients; validators aggregate, score, and apply them. The architecture reproduces [Ouro](https://arxiv.org/abs/2510.25741) — a recurrent transformer where a shared stack of layers is applied multiple times, with a learned exit gate for adaptive computation depth.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Running a Miner](#running-a-miner)
- [Running a Validator](#running-a-validator)
- [Multi-GPU Training (FSDP)](#multi-gpu-training-fsdp)
- [Hyperparameters](#hyperparameters)
- [LoopLM Model](#looplm-model)
- [Training Stages](#training-stages)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

---

## Overview

### How It Works

1. **Miners** train the LoopLM model on sharded data for a fixed number of inner steps each window
2. Gradients are compressed via top-k sparsification and uploaded to object storage (R2)
3. **Validators** gather compressed gradients from miners, aggregate them, and apply an outer SGD step
4. Validators score miners based on gradient quality (loss improvement, index overlap detection) and set on-chain weights
5. All nodes stay in sync via chain-window pacing and checkpoint loading

### What is LoopLM?

A standard decoder-only transformer whose layer stack is applied **T_max times** recurrently (weight-tied). At each recurrent step:

- An **LM head** produces next-token logits
- An **exit gate** predicts a halting probability

This yields 2-3x parameter efficiency: a 1.4B LoopLM matches 4B dense models on reasoning benchmarks.

---

## Architecture

```
                         Bittensor Chain
                    (registration, weights, stake)
                              |
              +---------------+---------------+
              |                               |
         +---------+                    +-----------+
         |  MINER  |  x N               | VALIDATOR |
         |         |                    |           |
         | 1. Load data shard          | 1. Gather compressed
         | 2. inner_steps() x30        |    gradients from miners
         | 3. Compress gradients       | 2. Aggregate + outer_step
         | 4. Upload to R2             | 3. Score miners
         | 5. Gather peers' grads      | 4. Set on-chain weights
         | 6. outer_step (SGD)         | 5. Upload checkpoint
         +---------+                    +-----------+
              |                               |
              +---------- R2 Storage ---------+
                   (gradients, checkpoints,
                    peer lists, datasets)
```

### Inner / Outer Loop

- **Inner loop** (local, per-miner): Standard LM training with the LoopLM loss (entropy-regularized, adaptive gate, or SFT depending on training stage). AdamW or Muon optimizer.
- **Outer loop** (network-wide): Compressed gradients from all miners are aggregated and applied via SGD. This is the decentralised training step.

---

## Installation

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (8GB+ VRAM for 1.4B, 16GB+ for 2.6B)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install

```bash
git clone <repo-url>
cd hone

# with uv (recommended)
uv sync

# or with pip
pip install -e .
```

### Create Wallets

```bash
btcli wallet new_coldkey --wallet.name default
btcli wallet new_hotkey --wallet.name default --wallet.hotkey miner
btcli wallet new_hotkey --wallet.name default --wallet.hotkey validator
```

### Register on Subnet

```bash
btcli subnet register --netuid <NETUID> --wallet.name default --wallet.hotkey <miner|validator>
```

---

## Running a Miner

The miner trains the LoopLM model locally each chain window, compresses and uploads gradients, then gathers and applies peer gradients.

### Single-GPU

```bash
python neurons/miner.py \
    --netuid <NETUID> \
    --wallet.name default \
    --wallet.hotkey miner \
    --device cuda \
    --amp-dtype bf16
```

### Multi-GPU (torchrun)

```bash
torchrun --nproc_per_node=8 neurons/miner.py \
    --netuid <NETUID> \
    --wallet.name default \
    --wallet.hotkey miner \
    --device cuda \
    --amp-dtype bf16
```

### All Miner Options

| Flag | Default | Description |
|------|---------|-------------|
| `--netuid` | `268` | Bittensor subnet UID |
| `--device` | `cuda` | Training device |
| `--amp-dtype` | `bf16` | Mixed precision: `bf16` or `fp16` |
| `--actual-batch-size` | (from hparams) | Override batch size |
| `--project` | `hone` | WandB project name |
| `--debug` | off | Enable debug logging |
| `--trace` | off | Enable trace-level logging |
| `--test` | off | Test mode: use all peers without filtering |
| `--local` | off | Local mode: use local-run hparams override |
| `--store-gathers` | off | Upload gathered gradients to R2 |
| `--profile-iters` | `0` | Torch profiler active iterations (0 = disabled) |
| `--profile-dir` | `./log/profiler` | Profiler trace output directory |

Plus all standard `bt.subtensor`, `bt.wallet`, and `bt.logging` arguments.

---

## Running a Validator

The validator gathers compressed gradients from miners, evaluates gradient quality, applies outer optimisation steps, scores miners, and sets on-chain weights.

### Single-GPU

```bash
python neurons/validator.py \
    --netuid <NETUID> \
    --wallet.name default \
    --wallet.hotkey validator \
    --device cuda
```

### Multi-GPU (torchrun)

```bash
torchrun --nproc_per_node=4 neurons/validator.py \
    --netuid <NETUID> \
    --wallet.name default \
    --wallet.hotkey validator \
    --device cuda
```

### All Validator Options

| Flag | Default | Description |
|------|---------|-------------|
| `--netuid` | `5` | Bittensor subnet UID |
| `--device` | `cuda` | Device for model and gradient ops |
| `--project` | `hone` | WandB project name |
| `--debug` | off | Enable debug logging |
| `--trace` | off | Enable trace-level logging |
| `--test` | off | Test mode: use all peers without filtering |
| `--local` | off | Local mode: use local-run hparams override |
| `--store-gathers` | off | Upload gathered gradients to R2 |
| `--profile-iters` | `0` | Torch profiler active iterations (0 = disabled) |
| `--profile-dir` | `./log/profiler` | Profiler trace output directory |

Plus all standard `bt.subtensor`, `bt.wallet`, and `bt.logging` arguments.

---

## Multi-GPU Training (FSDP)

Hone uses PyTorch FSDP2 for multi-GPU training. FSDP shards model parameters across GPUs at the `DecoderLayer` boundary.

### Launch with torchrun

```bash
# 8-GPU miner
torchrun --nproc_per_node=8 neurons/miner.py \
    --netuid <NETUID> \
    --wallet.name default \
    --wallet.hotkey miner

# 4-GPU validator
torchrun --nproc_per_node=4 neurons/validator.py \
    --netuid <NETUID> \
    --wallet.name default \
    --wallet.hotkey validator
```

The `fsdp.dp_shard` value in `hparams.json` controls the FSDP shard degree. It must divide your GPU count evenly.

---

## Hyperparameters

All hyperparameters are loaded from `hparams/` with a layered merge:

1. `DEFAULT_HPARAMS` (built-in defaults)
2. `hparams/hparams.json` (base config, must define `model_size`)
3. `hparams/{model_size}.json` (model architecture: `1.4B.json` or `2.6B.json`)
4. `hparams/hparams-local-run.json` (optional, for `--local` flag)

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_size` | `1.4B` | Architecture config to load (`1.4B` or `2.6B`) |
| `sequence_length` | `4096` | Token sequence length |
| `batch_size` | `192` | Micro-batches per gradient accumulation |
| `inner_steps` | `30` | Local training steps per chain window |
| `outer_learning_rate` | `0.4` | SGD learning rate for outer (aggregate) step |
| `t_max` | `4` | Recurrent steps per forward pass |
| `kl_beta` | `0.05` | KL divergence coefficient (Stage I) |
| `training_stage` | `pretrain` | Active training stage: `pretrain`, `gate`, or `sft` |
| `gate_k` | `50.0` | Sigmoid slope for gate training (Stage II) |
| `gate_gamma` | `0.005` | Improvement threshold for gate training (Stage II) |
| `topk_compression` | `64` | Top-k sparsity for gradient compression |
| `momentum_decay` | `0.95` | Error-feedback momentum decay |

### Model Configurations

**1.4B** (`hparams/1.4B.json`):

| Parameter | Value |
|-----------|-------|
| Layers | 24 |
| Hidden size | 2048 |
| Attention heads | 16 |
| KV heads | 16 (MHA) |
| FFN intermediate | 5504 |
| Vocab size | 49152 |
| RoPE theta | 10000 |

**2.6B** (`hparams/2.6B.json`):

| Parameter | Value |
|-----------|-------|
| Layers | 48 |
| Hidden size | 2048 |
| Attention heads | 16 |
| KV heads | 16 (MHA) |
| FFN intermediate | 5504 |
| Vocab size | 49152 |
| RoPE theta | 10000 |

Both use the SmolLM2 49,152-token vocabulary (`HuggingFaceTB/SmolLM2-135M` tokenizer).

---

## LoopLM Model

The model (`src/hone/model.py`) is a self-contained implementation matching the [official Ouro checkpoint](https://huggingface.co/ByteDance/Ouro-1.4B) for weight-loading compatibility.

### Architecture

```
Input tokens
    |
    v
[Embedding]  (49152 -> 2048)
    |
    v
+-- Recurrent Loop (T_max iterations) -----------+
|                                                  |
|   for each layer in [DecoderLayer x N]:         |
|       Sandwich Norm:                             |
|         input_layernorm -> Attention -> input_layernorm_2 + residual
|         post_attention_layernorm -> MLP -> post_attention_layernorm_2 + residual
|                                                  |
|   RMSNorm                                        |
|   LM Head  -> step_logits[t]                     |
|   Exit Gate -> step_gate_logits[t]               |
|                                                  |
+--------------------------------------------------+
    |
    v
LoopLMOutput(step_logits, step_gate_logits, final_hidden)
```

### Key Design Decisions

- **Sandwich normalization**: Pre-norm AND post-norm on both attention and FFN, critical for recurrent-depth training stability (Geiping et al.)
- **Exit gate outputs raw logits**: Sigmoid is applied downstream in loss functions, matching the official Ouro convention
- **Weight naming**: Parameter names match `ByteDance/Ouro-1.4B` (`q_proj`, `k_proj`, `gate_proj`, `up_proj`, `down_proj`, `input_layernorm`, etc.) for direct weight loading

---

## Training Stages

The LoopLM paper defines three training stages, controlled by `hparams.training_stage`:

### Stage I: Pre-training (`"pretrain"`)

Entropy-regularized objective where the loss is the expected task loss weighted by the exit distribution, minus an entropy bonus:

```
L = sum_t p(t|x) * CE(t) - beta * H(p)
```

- `p(t|x)` is derived from the exit gate via a survival function (Eq. 3 in the paper)
- `beta` (`kl_beta`) prevents collapse to always using T_max
- This is the primary pre-training objective

### Stage II: Adaptive Gate Training (`"gate"`)

The LM parameters are frozen; only the exit gate is trained:

- Per-step loss improvement `I_t = max(0, L_{t-1} - L_t)` is computed
- An ideal continuation label `w_t = sigmoid(k * (I_t - gamma))` is derived
- The gate is trained via BCE to predict when to stop looping
- `gate_k` (50.0) and `gate_gamma` (0.005) control the sharpness and threshold

### Stage III: Supervised Fine-Tuning (`"sft"`)

Standard cross-entropy on the final recurrent step's logits. Used after pre-training for instruction tuning or domain adaptation.

---

## Environment Variables

### Required (R2 Object Storage)

These must be set for gradient exchange and dataset access:

```bash
# Gradient bucket
R2_GRADIENTS_ACCOUNT_ID=...
R2_GRADIENTS_BUCKET_NAME=...
R2_GRADIENTS_READ_ACCESS_KEY_ID=...
R2_GRADIENTS_READ_SECRET_ACCESS_KEY=...
R2_GRADIENTS_WRITE_ACCESS_KEY_ID=...
R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY=...

# Aggregator bucket
R2_AGGREGATOR_ACCOUNT_ID=...
R2_AGGREGATOR_BUCKET_NAME=...
R2_AGGREGATOR_READ_ACCESS_KEY_ID=...
R2_AGGREGATOR_READ_SECRET_ACCESS_KEY=...

# Dataset
DATASET_BINS_PATH=...
```

### Optional

```bash
# WandB logging
WANDB_API_KEY=your_key_here

# HuggingFace token (for gated tokenizers)
HF_TOKEN=your_token_here

# Dataset bucket list override (JSON array)
R2_DATASET_BUCKET_LIST='[{"account_id": "...", ...}]'
```

### .env File

Create a `.env` file in the project root. All variables are loaded automatically via `python-dotenv`:

```bash
cp .env.example .env
# edit .env with your credentials
```

---

## Troubleshooting

### NCCL Errors on Multi-GPU

```bash
# Ensure NCCL can find all GPUs
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # if no InfiniBand

# Verify GPU visibility
python -c "import torch; print(torch.cuda.device_count())"
```

### Out of Memory

- Reduce `batch_size` or `micro_batch_size` in `hparams.json`
- Enable optimizer state offloading: `"offload_optimizer_states": true`
- Use `--amp-dtype bf16` (default) for mixed precision
- For the 2.6B model, 8 GPUs with FSDP is recommended

### Checkpoint Loading Failures

- Ensure the model size in `hparams.json` matches the checkpoint
- Check R2 credentials are correct
- Verify network connectivity to the aggregator bucket

### Validator Not Setting Weights

- Ensure sufficient stake on the validator hotkey
- Check that the validator UID is registered on the subnet
- Wait for the rate-limiting window between weight updates

### Miner Gradients Not Being Accepted

- Verify R2 write credentials are configured
- Check that `topk_compression` matches the network standard
- Ensure the miner's model is in sync (checkpoint loaded correctly)

---

## Project Structure

```
hone/
├── pyproject.toml                 # Package config (no torchtitan dependency)
├── hparams/
│   ├── hparams.json               # Base training config
│   ├── 1.4B.json                  # 1.4B model architecture
│   └── 2.6B.json                  # 2.6B model architecture
├── neurons/
│   ├── base_node.py               # Async lifecycle, chain block listener
│   ├── trainer.py                 # LoopLM training loop + optimizers
│   ├── miner.py                   # Miner node (train + compress + upload)
│   └── validator.py               # Validator node (gather + score + weights)
└── src/hone/
    ├── model.py                   # LoopLM model (matches Ouro checkpoint)
    ├── loss.py                    # Stage I/II/SFT loss functions
    ├── hparams.py                 # Hyperparameter loading
    ├── chain.py                   # Bittensor chain interaction
    ├── comms.py                   # R2 object storage + gradient exchange
    ├── compress.py                # Top-k gradient compression + 12-bit packing
    ├── config.py                  # Environment / bucket configuration
    ├── dataset.py                 # Dataset management
    ├── distributed.py             # NCCL / FSDP helpers
    ├── checkpoint.py              # Distributed checkpointing
    ├── neurons.py                 # outer_step, prepare_gradient_dict
    ├── logging.py                 # Rich + Loki logging
    └── muon/                      # Muon optimizer (Newton-Schulz)
```

---

## License

MIT. See [LICENSE](LICENSE).
