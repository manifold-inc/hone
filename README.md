# Hone

Slim SparseLoCo distributed training for Bittensor subnet 5.

A minimal reimplementation of the [templar](https://github.com/one-covenant/templar) protocol with improved gradient verification, multi-model support, and FSDP multi-GPU training.

## Quick Start

```bash
# Install
pip install -e .

# Single-GPU miner
python scripts/run_miner.py --wallet.name mywallet --wallet.hotkey myhotkey --netuid 5

# Multi-GPU miner (FSDP)
torchrun --nproc_per_node=8 scripts/run_miner.py --wallet.name mywallet --netuid 5

# Validator
torchrun --nproc_per_node=8 scripts/run_validator.py --wallet.name mywallet --netuid 5
```

## Configuration

All settings via environment variables (prefixed `HONE_`) or CLI args. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--netuid` | 5 | Bittensor subnet ID |
| `--model_type` | llama | Model architecture (llama, mistral, qwen2, gemma, ouro) |
| `--inner_steps` | 30 | SparseLoCo inner optimization steps per window |
| `--batch_size` | 4 | Per-device batch size |

R2 bucket credentials are configured via environment variables:
```
HONE_R2_GRADIENTS_ACCOUNT_ID=...
HONE_R2_GRADIENTS_BUCKET_NAME=...
HONE_R2_GRADIENTS_READ_ACCESS_KEY_ID=...
HONE_R2_GRADIENTS_READ_SECRET_ACCESS_KEY=...
HONE_R2_GRADIENTS_WRITE_ACCESS_KEY_ID=...
HONE_R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY=...
```

## Architecture

```
src/hone/
├── config.py       # All configuration
├── chain.py        # Bittensor wrapper (wallet, subtensor, metagraph)
├── comms.py        # R2/S3 gradient storage
├── compress.py     # SparseLoCo: top-k, 2-bit quant, error feedback
├── verify.py       # Gradient verification (Freivalds + LossScore)
├── score.py        # OpenSkill scoring + weight allocation
├── report.py       # Dashboard telemetry client
├── model.py        # Multi-arch model registry + FSDP
├── distributed.py  # Process group helpers
├── data.py         # Sharded dataset loading
├── miner.py        # Miner neuron
└── validator.py    # Validator neuron
```

## Supported Models

The validator specifies which model architecture to train. Supported:

- **llama** - LLaMA/LLaMA-2/LLaMA-3 style
- **mistral** - Mistral
- **qwen2** - Qwen2
- **gemma** - Gemma
- **ouro** - ByteDance Ouro (Universal Transformer with early exit)
- Any HuggingFace `AutoModelForCausalLM`-compatible architecture

## Dashboard

Training metrics are reported to [hone.training](https://hone.training) via the hone-api. Set `HONE_HONE_API_URL` and `HONE_HONE_API_KEY` to enable.
