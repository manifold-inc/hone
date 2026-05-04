# Dense Validator Guide

## Overview

A validator evaluates miner contributions by measuring whether each miner's gradient actually improves the model. Validators gather gradients from miners, test them against evaluation data, compute scores, and set weights on-chain.

Validators require more compute than miners because they evaluate multiple miners per window and must hold the full model in memory for evaluation. Each evaluation involves saving model state, applying a miner's gradient, measuring loss change, and restoring -- repeated for up to 20 miners per window.

## Hardware Requirements

### Minimum

- 4x NVIDIA GPUs with 40GB+ VRAM each (A100 40GB)
- 128GB+ system RAM
- 500+ Mbps internet (validators download gradients from all miners each window)
- 200GB+ SSD

### Recommended

- 4-8x NVIDIA A100 80GB or H100/H200
- 256GB+ system RAM
- 1+ Gbps internet
- NVMe SSD for fast checkpoint I/O

## Prerequisites

Same as miner, plus:

- **Higher stake** -- validators need sufficient TAO stake to set weights effectively on the subnet
- **Reliable uptime** -- validators must stay online consistently; missed windows mean missed evaluations and stale weights

## Installation

```bash
cd hone
pip install -e .
```

## Configuration

### Environment variables

Same `.env` as the miner -- R2 credentials for gradients, aggregator, and dataset buckets, plus optional Wandb and HuggingFace tokens. See the [Dense Miner Guide](dense-miner-guide.md) for the full `.env` template.

Validators with `--store-gathers` enabled also need aggregator write credentials to upload checkpoints.

### Hyperparameters

Use the same `hparams/hparams.json` as the network. The `model_size` must match miners:

```json
"model_size": "1.4B"
```

Set `fsdp.dp_shard` to match your GPU count:

```json
"fsdp": {
    "dp_shard": 4,
    "compile": true,
    "mixed_precision": "bfloat16"
}
```

Key validator-specific hparams (all set in the shared `hparams.json`):

| Field | Default | Description |
|-------|---------|-------------|
| `gather_peer_count` | `20` | Number of miners to gather and evaluate per window |
| `reserve_peer_count` | `10` | Reserve peers used as fallback if primary gather fails |
| `uids_per_window` | `20` | Max UIDs evaluated per window |
| `windows_per_weights` | `3` | How often to set on-chain weights (every N windows) |
| `checkpoint_frequency` | `10` | Upload a checkpoint to R2 every N windows |
| `validator_sample_micro_bs` | `8` | Micro-batch multiplier for evaluation forward passes |
| `binary_score_ma_alpha` | `0.05` | Smoothing factor for binary moving average scores |
| `bma_threshold` | `0.10` | Minimum BMA score before penalty kicks in |
| `bma_warmup_windows` | `10` | Windows before BMA penalties apply to new miners |
| `time_window_delta_seconds` | `70` | How far back to look for miner gradient uploads |
| `idx_overlap_threshold` | `0.4` | Threshold for detecting gradient copying between miners |
| `openskill_beta` | `7` | OpenSkill rating volatility parameter |
| `openskill_tau` | `0.1` | OpenSkill rating dynamics parameter |

## Launching

### Multi-GPU (recommended)

```bash
torchrun --nproc_per_node=4 neurons/validator.py \
  --wallet.name my_wallet \
  --wallet.hotkey my_validator_hotkey \
  --netuid 268
```

### Single GPU (possible for 1.4B)

```bash
python neurons/validator.py \
  --wallet.name my_wallet \
  --wallet.hotkey my_validator_hotkey \
  --netuid 268
```

Single-GPU works for the 1.4B dense model because `offload_optimizer_states` is enabled by default, but evaluation throughput will be lower.

### With gradient storage

```bash
torchrun --nproc_per_node=4 neurons/validator.py \
  --wallet.name my_wallet \
  --wallet.hotkey my_validator_hotkey \
  --netuid 268 \
  --store-gathers
```

The `--store-gathers` flag saves gathered gradients to R2 for other nodes to sync from. This is important for network health -- it allows new miners and validators to bootstrap from stored checkpoints.

### Additional CLI flags

| Flag | Description |
|------|-------------|
| `--debug` | Enable debug-level logging |
| `--trace` | Enable trace-level logging (very verbose) |
| `--test` | Test mode -- use all peers without filtering |
| `--local` | Use a toy-size model for local development |
| `--project NAME` | Wandb project name (default: templar) |
| `--profile-iters N` | Enable PyTorch profiler for N iterations per trace (0 = disabled) |
| `--profile-dir PATH` | Directory to save profiler traces (default: `./log/profiler`) |

## What Happens During Validation

Each training window (26 blocks, ~5-6 minutes) the validator runs this cycle:

1. **Gather.** Downloads compressed gradients from ~20 miners via R2 storage. The validator looks at gradients uploaded within the `time_window_delta_seconds` (70s) window. Reserve peers (up to 10) are used as fallback if primary peers fail.

2. **Index overlap check.** Compares the top-k index sets across all gathered gradients to detect potential gradient copying. Miners with overlap above `idx_overlap_threshold` (0.4) are slashed. Severe overlap ("mega" level) results in a full score reset and temporary blacklisting.

3. **Per-miner evaluation.** For each of the gathered miners (up to `uids_per_window` = 20):
   - Save the current model state
   - Run a forward pass on evaluation data to compute loss **before** applying the gradient
   - Apply the miner's decompressed gradient to the model (scaled by `eval_lr_factor` = 0.5)
   - Run a forward pass on the same data to compute loss **after**
   - Restore the original model state
   - Score = loss_before - loss_after (positive means the gradient improved the model)

4. **Scoring.** Combines multiple signals into a final score per miner (details in the Scoring Mechanism section below).

5. **Outer step.** Applies the aggregated gradient from all gathered miners using Nesterov SGD (learning rate 0.4), advancing the validator's model to match the network state.

6. **Set weights.** Every `windows_per_weights` (3) windows, the validator submits updated on-chain weights to the Bittensor subnet. Weights are derived from final scores using power normalization (power=2.0).

7. **Checkpoint upload.** Every `checkpoint_frequency` (10) windows, the validator uploads its current model checkpoint to R2. This is the checkpoint that miners and other validators use to sync.

Gathered gradient data is offloaded to pinned CPU memory between evaluations to free GPU memory.

## Scoring Mechanism

### Gradient quality scoring

The primary signal. For each miner, the validator measures the change in evaluation loss when the miner's gradient is applied:

```
score = loss_before - loss_after
```

A positive score means the gradient reduced loss -- the miner contributed useful training signal. A negative score means the gradient made the model worse.

The evaluation uses the miner's own data shard indices combined with a holdout set, ensuring the gradient is tested on relevant data.

### Binary moving average (BMA)

A running average that tracks whether a miner consistently contributes positive gradients:

```
bma[uid] = (1 - alpha) * bma[uid] + alpha * (1 if score > 0 else 0)
```

With alpha=0.05, the BMA smooths over roughly 20 windows. Miners whose BMA drops below `bma_threshold` (0.10) after `bma_warmup_windows` (10) receive penalized scores. This prevents miners from gaming the system with occasional high scores interspersed with harmful gradients.

### OpenSkill ratings

An Elo-like rating system (PlackettLuce model) that ranks miners relative to each other within each evaluation window. Parameters:

- `beta=7` -- controls rating uncertainty
- `tau=0.1` -- controls rating dynamics speed

Miners are ranked by their gradient quality scores. The resulting ordinal rankings complement the raw magnitude-based scores.

### Sync scoring

Measures how up-to-date a miner's model state is. The validator tracks each miner's global step count and penalizes miners that are more than `sync_max_steps_behind` (2) outer steps behind the network. Miners training on stale checkpoints produce gradients that are less useful because they are computed against an outdated model.

### Missing gradient penalties

Miners who fail to upload a gradient for a window receive escalating penalties:

- 1 consecutive miss: score multiplied by 0.75
- 2 consecutive misses: score multiplied by 0.5
- 3+ consecutive misses: score set to 0

If a miner's gather success rate drops below `gather_peers_slash_threshold` (0.4), the score is immediately set to 0.

A special penalty score of -99.0 is assigned internally for completely missing gradients, ensuring they rank last in OpenSkill comparisons.

### Weight distribution

Final weights are computed from scores using power normalization:

```
weights[uid] = scores[uid]^power / sum(scores^power)
```

With `power=2.0`, this concentrates weight toward top-performing miners. Reserve miners receive weight that decays by `reserve_decay_ratio` (0.5). The `incentive_burn_rate` (1.0) directs a fraction of total weight to a burn address, reducing inflation.

## Monitoring

### Console logs

Use `--debug` for detailed per-evaluation logging or `--trace` for full verbosity including tensor shapes and timing for every operation.

### Weights & Biases

Set `WANDB_API_KEY` in `.env` and pass `--project my_project`. Validator-specific tracked metrics include:

- Per-UID evaluation scores (loss_before, loss_after, delta)
- Binary moving averages per UID
- OpenSkill ratings
- Gather success rates and timing
- Weight distribution across UIDs
- Checkpoint upload status and timing
- Slash events (overlap detection, missing gradients)
- GPU memory usage during evaluation

### Dashboard

The live dashboard at [hone.training](https://hone.training) provides a network-wide view including:

- Per-UID scores and weight history
- Gather participation rates
- Loss curves and training progress
- Slash event logs

## Troubleshooting

### Evaluation failures / OOM

- Evaluation loads each miner's gradient and runs forward passes, which can spike memory. The `offload_gather_results` mechanism (enabled by default) moves gathered gradient data to pinned CPU memory between evaluations.
- Reduce `validator_sample_micro_bs` (default 8) to lower evaluation batch size.
- Use more GPUs with FSDP to shard the model across devices.
- Check logs for which evaluation step triggered OOM -- a single corrupted gradient can cause outsized memory use.

### Slow gather

- Check internet download bandwidth. Validators download gradients from all `gather_peer_count` (20) peers each window.
- Increase `time_window_delta_seconds` (default 70) if miners are slow to upload, giving a wider collection window.
- Check R2 read credentials -- failed downloads show up as skipped UIDs in logs.
- A high skip rate may indicate many miners are offline or uploading late.

### Weight setting failures

- Verify your wallet has enough TAO staked on subnet 268.
- Check the subtensor connection -- the validator uses `bt.Subtensor` which connects to the Bittensor chain.
- Verify `--netuid 268` matches your registration.
- Look for weight-setting errors in the logs after each `windows_per_weights` cycle.

### Checkpoint upload failures

- Confirm aggregator write credentials in `.env` (`R2_AGGREGATOR_WRITE_ACCESS_KEY_ID` and `R2_AGGREGATOR_WRITE_SECRET_ACCESS_KEY`).
- Check available disk space -- checkpoints are staged locally before upload.
- Failed uploads are logged but do not stop the validator. Other validators will continue serving checkpoints.

### Stale scores

- If scores appear frozen, check that the evaluation loop is completing. Look for "gather task completed" in logs.
- Verify that `uids_per_window` is not set to 0.
- Check that the metagraph is refreshing -- a stale metagraph means the validator cannot discover new miners.

### Process recovery

The validator resumes from the latest checkpoint on restart. Relaunch with the same command. Score state (BMA, OpenSkill ratings, final scores) is maintained in memory and rebuilt from the evaluation history. After a restart, there is a brief warmup period while scores stabilize.
