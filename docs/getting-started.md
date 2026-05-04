# Getting Started with Hone

## Prerequisites

- **Python** 3.10+
- **PyTorch** 2.4+ with CUDA support
- **NVIDIA GPU** — H100/H200 recommended, A100 minimum for full-size models. Smaller configs (`120M`) work on consumer GPUs for testing.
- **Bittensor wallet** with a registered hotkey on the target subnet
- **R2/S3-compatible storage credentials** — provided in the `.env` file
- **Git LFS** (if cloning checkpoints)

## Installation

```bash
cd hone
pip install -e .
```

This installs the `hone` package and all dependencies (PyTorch, Bittensor, einops, wandb, etc.) in editable mode.

## Configuration

Hone uses a layered hyperparameter merge system. Each layer overrides the one below it:

```
Priority (highest to lowest):
  4. hparams/hparams-local-run.json   (optional, used with --local)
  3. hparams/{model_size}.json        (architecture-specific)
  2. hparams/hparams.json             (main config, must set model_size)
  1. DEFAULT_HPARAMS in code          (lowest priority fallback)
```

### Key fields in `hparams/hparams.json`

| Field | Description | Default |
|-------|-------------|---------|
| `model_size` | Which architecture config to load (e.g. `"1.4B"`, `"1.4B-moe"`, `"35B-A3B"`) | Required |
| `sequence_length` | Context length per sample | 4096 |
| `micro_batch_size` | Batch size per forward pass | 4 |
| `inner_steps` | Number of inner optimizer steps per window | 30 |
| `blocks_per_window` | Bittensor blocks per training window | 26 |
| `optimizer.type` | Inner optimizer: `"muon"` or `"adamw"` | `"muon"` |
| `fsdp.dp_shard` | Number of FSDP shards (must match `--nproc_per_node`) | 4 |
| `topk_compression` | Top-k values kept per chunk in DeMo compression | 32 |
| `use_dct` | Enable DCT transform in compression pipeline | true |

### Architecture configs (`hparams/{model_size}.json`)

These files define model dimensions: `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, and MoE settings when applicable.

### Local overrides

Create `hparams/hparams-local-run.json` to override any setting for local development without modifying the shared config. This file is loaded when running with `--local`.

## Environment Variables

Copy the `.env` template and fill in your credentials:

```bash
cp .env.example .env
```

Required variables:

```bash
# R2 gradient bucket
R2_GRADIENTS_ACCOUNT_ID=<your-account-id>
R2_GRADIENTS_BUCKET_NAME=<your-bucket>
R2_GRADIENTS_READ_ACCESS_KEY_ID=<read-key>
R2_GRADIENTS_READ_SECRET_ACCESS_KEY=<read-secret>
R2_GRADIENTS_WRITE_ACCESS_KEY_ID=<write-key>
R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY=<write-secret>

# R2 aggregator bucket
R2_AGGREGATOR_ACCOUNT_ID=<your-account-id>
R2_AGGREGATOR_BUCKET_NAME=<your-bucket>
R2_AGGREGATOR_READ_ACCESS_KEY_ID=<read-key>
R2_AGGREGATOR_READ_SECRET_ACCESS_KEY=<read-secret>
R2_AGGREGATOR_WRITE_ACCESS_KEY_ID=<write-key>
R2_AGGREGATOR_WRITE_SECRET_ACCESS_KEY=<write-secret>

# Dataset bucket
R2_DATASET_ACCOUNT_ID=<your-account-id>
R2_DATASET_BUCKET_NAME=<your-bucket>
R2_DATASET_READ_ACCESS_KEY_ID=<read-key>
R2_DATASET_READ_SECRET_ACCESS_KEY=<read-secret>

# HuggingFace token (for downloading gated tokenizers)
HF_TOKEN=<your-hf-token>

# Optional: Weights & Biases
WANDB_API_KEY=<your-wandb-key>

# Optional: Dashboard telemetry
DASHBOARD_API_URL=https://api.hone.training
DASHBOARD_API_KEY=<your-api-key>
```

## Running the Validator

### Single GPU

```bash
python neurons/validator.py \
  --wallet.name <wallet> \
  --wallet.hotkey <hotkey> \
  --netuid 268
```

### Multi-GPU (FSDP)

```bash
torchrun --nproc_per_node=4 neurons/validator.py \
  --wallet.name <wallet> \
  --wallet.hotkey <hotkey> \
  --netuid 268
```

`--nproc_per_node` must match the `fsdp.dp_shard` value in `hparams.json`. If your config says `"dp_shard": 4`, use `--nproc_per_node=4`.

## Running the Miner

### Single GPU

```bash
python neurons/miner.py \
  --wallet.name <wallet> \
  --wallet.hotkey <hotkey> \
  --netuid 268
```

### Multi-GPU (FSDP)

```bash
torchrun --nproc_per_node=8 neurons/miner.py \
  --wallet.name <wallet> \
  --wallet.hotkey <hotkey> \
  --netuid 268
```

As with the validator, ensure `--nproc_per_node` matches `fsdp.dp_shard`.

## Test Mode

Use `--test` to bypass peer filtering and chain registration checks. This is useful for local development and debugging:

```bash
python neurons/miner.py \
  --wallet.name <wallet> \
  --wallet.hotkey <hotkey> \
  --netuid 268 \
  --test
```

Test mode allows a node to train and upload gradients without needing active peers or a registered hotkey on the subnet.

## Monitoring

### Weights & Biases

Set `WANDB_API_KEY` in `.env` to enable automatic logging of loss curves, gradient norms, compression stats, and scoring metrics.

### Dashboard

A live dashboard is available at [hone.training](https://hone.training). It shows subnet-wide training progress, per-miner scores, and model loss over time.

### Console Logging

Increase log verbosity with:

```bash
--debug     # detailed step-by-step logging
--trace     # maximum verbosity (includes tensor shapes, timing, R2 ops)
```

## Available Model Configs

Set `model_size` in `hparams/hparams.json` to one of the following:

| `model_size` | Type | Layers | Hidden | Heads | KV Heads | Params |
|-------------|------|--------|--------|-------|----------|--------|
| `120M` | Dense | 12 | 384 | 6 | 6 | ~120M |
| `1.4B` | Dense | 24 | 2048 | 16 | 16 | ~1.4B |
| `2.6B` | Dense | 48 | 2048 | 16 | 16 | ~2.6B |
| `1.4B-moe` | MoE | 24 | 2048 | 16 | 16 | ~5.6B total, ~1.4B active |
| `35B-A3B` | MoE | 40 | 2048 | 16 | 2 | ~35B total, ~3B active |

### Dense models

Standard Llama-style decoders with SwiGLU MLP, GQA, and RoPE. `1.4B` is the default starting point.

### `1.4B-moe`

Same backbone as the dense 1.4B but with 8 experts per MLP layer, top-2 routing. Total parameter count is roughly 5.6B, but only ~1.4B parameters are active per token.

### `35B-A3B`

Fine-grained MoE in the style of Qwen/DeepSeek. 256 experts with top-8 routing and a shared expert. Uses GQA with 2 KV heads. Each expert has a small FFN dimension (512), and the shared expert also uses dim 512. Total parameters ~35B, active ~3B per token.

### `120M`

A small dense model for development and testing. Runs comfortably on a single consumer GPU.
