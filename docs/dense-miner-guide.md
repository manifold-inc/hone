# Dense Miner Guide

## Overview

A miner contributes gradient updates to the network by training a dense LoopLM model locally and uploading compressed gradients each window. The validator evaluates your contributions and assigns a score that determines your share of rewards.

The default dense model is 1.4B parameters (24 layers, 2048 hidden dim, SwiGLU MLP, GQA attention with 16 heads, vocab size 256k via the Gemma-2 tokenizer). It fits comfortably on a single GPU with 24GB+ VRAM, and scales to multi-GPU with FSDP.

## Hardware Requirements

### Minimum

- 1x NVIDIA GPU with 24GB+ VRAM (A10G, RTX 4090, A100)
- 32GB+ system RAM
- 100+ Mbps internet (upload matters -- you send compressed gradients each window)
- 100GB+ SSD for datasets and checkpoints

### Recommended

- 1x NVIDIA A100 80GB or H100/H200
- 64GB+ system RAM
- 500+ Mbps internet
- For multi-GPU: 2-8x GPUs with NVLink (enables FSDP)

## Prerequisites

1. Python 3.10+
2. PyTorch 2.4+ with CUDA
3. A registered Bittensor wallet with a hotkey on subnet 268
4. R2/S3 storage credentials (provided by the Hone team for gradient and dataset buckets)

## Installation

```bash
git clone <repo-url>
cd hone
pip install -e .
```

## Configuration

### Environment variables

Create a `.env` file in the `hone/` root directory. The system expects separate R2 credentials for gradient storage, the aggregator bucket, and the dataset bucket:

```bash
# Gradient bucket (R2/S3-compatible)
R2_GRADIENTS_ACCOUNT_ID=<your_account_id>
R2_GRADIENTS_BUCKET_NAME=<bucket_name>
R2_GRADIENTS_READ_ACCESS_KEY_ID=<read_key>
R2_GRADIENTS_READ_SECRET_ACCESS_KEY=<read_secret>
R2_GRADIENTS_WRITE_ACCESS_KEY_ID=<write_key>
R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY=<write_secret>

# Aggregator bucket
R2_AGGREGATOR_ACCOUNT_ID=<your_account_id>
R2_AGGREGATOR_BUCKET_NAME=aggregator
R2_AGGREGATOR_READ_ACCESS_KEY_ID=<read_key>
R2_AGGREGATOR_READ_SECRET_ACCESS_KEY=<read_secret>
R2_AGGREGATOR_WRITE_ACCESS_KEY_ID=<write_key>
R2_AGGREGATOR_WRITE_SECRET_ACCESS_KEY=<write_secret>

# Dataset
R2_DATASET_ACCOUNT_ID=<your_account_id>
R2_DATASET_BUCKET_NAME=hone-dataset
R2_DATASET_READ_ACCESS_KEY_ID=<read_key>
R2_DATASET_READ_SECRET_ACCESS_KEY=<read_secret>
DATASET_BINS_PATH=tokenized/

# Optional: Weights & Biases
WANDB_API_KEY=

# Optional: HuggingFace (for gated tokenizers)
HF_TOKEN=

# Dashboard telemetry
DASHBOARD_API_URL=https://api.hone.training
DASHBOARD_API_KEY=<your_api_key>
```

Contact the Hone team for the shared read credentials. Write credentials are unique to your miner.

### Hyperparameters

The network-wide hyperparameters live in `hparams/hparams.json`. All miners and validators must use the same file to stay in consensus. Key fields for the dense model:

| Field | Default | Description |
|-------|---------|-------------|
| `model_size` | `"1.4B"` | Must match the model config in `hparams/1.4B.json` |
| `inner_steps` | `30` | Optimizer steps per window |
| `batch_size` | `192` | Sequences per inner step (per miner) |
| `micro_batch_size` | `4` | Gradient accumulation micro-batch size |
| `sequence_length` | `4096` | Context length |
| `topk_compression` | `32` | DeMo top-k sparsification ratio |
| `blocks_per_window` | `26` | Blockchain blocks per training window (~5-6 min) |

For multi-GPU, set `fsdp.dp_shard` to match your GPU count:

```json
"fsdp": {
    "dp_shard": 4,
    "compile": true,
    "mixed_precision": "bfloat16"
}
```

Do not modify other hparams unless the network has coordinated a change. Mismatched hparams will cause your gradients to be rejected.

## Launching

### Single GPU

```bash
python neurons/miner.py \
  --wallet.name my_wallet \
  --wallet.hotkey my_miner_hotkey \
  --netuid 268
```

### Multi-GPU (FSDP)

```bash
torchrun --nproc_per_node=4 neurons/miner.py \
  --wallet.name my_wallet \
  --wallet.hotkey my_miner_hotkey \
  --netuid 268
```

Match `--nproc_per_node` to `fsdp.dp_shard` in `hparams.json`.

### Test mode (local development)

```bash
python neurons/miner.py \
  --wallet.name my_wallet \
  --wallet.hotkey my_miner_hotkey \
  --netuid 268 \
  --test
```

The `--test` flag bypasses peer filtering and uses all UIDs from the metagraph. Useful for local debugging.

### Local mode (laptop-scale)

```bash
python neurons/miner.py \
  --wallet.name my_wallet \
  --wallet.hotkey my_miner_hotkey \
  --netuid 268 \
  --local
```

The `--local` flag uses a toy-size model that fits on a laptop GPU. Useful for testing the full loop without production hardware.

### Additional CLI flags

| Flag | Description |
|------|-------------|
| `--debug` | Enable debug-level logging |
| `--trace` | Enable trace-level logging (very verbose) |
| `--store-gathers` | Store gathered gradients in R2 for other nodes to sync from |
| `--local-data` | Use only local dataset files; skip R2 downloads and shard swapping |
| `--actual-batch-size N` | Override the batch size defined in hparams |
| `--amp-dtype bf16/fp16` | Mixed-precision type (default: bf16) |
| `--project NAME` | Wandb project name (default: templar) |
| `--profile-iters N` | Enable PyTorch profiler for N iterations per trace (0 = disabled) |
| `--profile-dir PATH` | Directory to save profiler traces (default: `./log/profiler`) |

## What Happens During Training

Each training window corresponds to 26 blockchain blocks (~5-6 minutes). The miner repeats this cycle:

1. **Checkpoint sync.** On startup (and after prolonged disconnection), the miner loads the latest aggregated checkpoint from R2. This ensures your model state matches the network's current state.

2. **Inner training.** Runs 30 optimizer steps (Muon optimizer with AdamW for non-matrix parameters) on your data shard. Each step processes `batch_size` sequences of length 4096, accumulated over `micro_batch_size`-sized micro-batches. Training uses bf16 mixed precision.

3. **Gradient compression.** The DeMo (Decoupled Momentum) pipeline compresses your gradient update:
   - Momentum accumulation with decay (0.999) and subtraction (alpha=0.2)
   - DCT (Discrete Cosine Transform) of momentum tensors
   - Top-k sparsification (k=32, keeping 1/32 of coefficients)
   - 4-bin quantization over a range of 6 standard deviations
   - Error feedback carries residuals to the next window

4. **Upload.** Rank 0 uploads the compressed gradient to R2 storage. Typical upload size is well under 50 MB.

5. **Gather.** Downloads compressed gradients from peers via R2. The miner gathers from ~20 peers (`gather_peer_count`) plus up to 10 reserve peers.

6. **Outer step.** Applies Nesterov SGD (learning rate 0.4) on the aggregated gradient update from all gathered peers. This is the actual model weight update that advances the shared model.

7. **Repeat.** The miner waits for the next window and begins again.

## Monitoring

### Console logs

Use `--debug` for detailed logging or `--trace` for full verbosity. Default logging shows window boundaries, gather results, and outer step summaries.

### Weights & Biases

Set `WANDB_API_KEY` in `.env` and pass `--project my_project`. Tracked metrics include:

- Per-step training loss and gradient norms
- Compression statistics (momentum norms, compressed size)
- Gather success rates and peer counts
- Timing breakdown (training, compression, gather, upload, model update)
- GPU memory usage (allocated and cached)
- Effective batch size and tokens per second

### Dashboard

The live training dashboard at [hone.training](https://hone.training) shows loss curves, peer participation, and per-UID scores across the network.

## Troubleshooting

### OOM (Out of Memory)

- Reduce `micro_batch_size` in `hparams.json` (default 4)
- Reduce `batch_size` (default 192)
- `offload_optimizer_states` is enabled by default, which moves optimizer state to CPU between steps
- Use more GPUs with FSDP to shard the model

### Gradient upload failures

- Verify R2 write credentials in `.env` (read and write keys are separate)
- Check internet connectivity and upload bandwidth
- Look for timeout errors in logs during the gather/upload phase
- Confirm `R2_GRADIENTS_BUCKET_NAME` is correct

### Low scores

- Ensure your model is synced to the latest checkpoint. Miners training on stale checkpoints receive lower sync scores. The `sync_max_steps_behind` parameter (default 2) controls how far behind you can be.
- Check that inner training loss is decreasing each window.
- Verify internet speed -- slow uploads mean your gradient arrives late and may be missed by validators.
- Check gather success rate in logs. Consistently low rates indicate connectivity issues.

### Shard rotation

The dataset is split into shards that rotate based on `outer_steps_per_shard` (default 455 outer steps). Shard transitions happen automatically. If you see a log message about resetting shard schedule, this is normal behavior.

### Process crashes

The miner is designed to resume from the latest checkpoint on restart. Simply relaunch with the same command. The checkpoint system uses distributed checkpointing (DCP) and will catch up to the current network window.
