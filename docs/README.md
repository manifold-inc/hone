# Hone Documentation

## System

- [Architecture Overview](architecture.md) — How the system works end-to-end
- [Getting Started](getting-started.md) — Installation, configuration, and first run
- [Hyperparameters Reference](hparams-reference.md) — Full hparams.json field reference

## Operator Guides

- [Dense Miner Guide](dense-miner-guide.md) — Run a miner with the standard dense model
- [Dense Validator Guide](dense-validator-guide.md) — Run a validator with the dense model
- [MoE Miner Guide](moe-miner-guide.md) — Run a miner with Mixture-of-Experts
- [MoE Validator Guide](moe-validator-guide.md) — Run a validator with MoE
- [`blocks_per_window` Step-Down](blocks-per-window-stepdown.md) — When and how to ratchet the chain-window length down (P3b)

## Deep Dives

- [MoE Training](moe.md) — MoE architecture details and configuration reference
- [Compression](compression.md) — DeMo gradient compression tuning guide
- [TurboQuant Codec](turboquant.md) — Hadamard + Beta-Lloyd-Max scalar codec (P6, default-off, hard-gated)
