# Hyperparameters Reference

## Merge Order

Hone loads hyperparameters by merging four layers in order, where later layers override earlier ones:

1. **`DEFAULT_HPARAMS`** -- hardcoded defaults in `hone/src/hone/hparams.py` (lowest priority)
2. **`hparams/hparams.json`** -- base config (must define `model_size`)
3. **`hparams/{model_size}.json`** -- model-specific architecture config (e.g., `1.4B.json`, `35B-A3B.json`)
4. **`hparams/hparams-local-run.json`** -- optional local overrides, loaded only when the `--local` flag is passed (highest priority)

After merging, the system constructs a `SimpleNamespace` with all fields, attaches the tokenizer and `LoopLMConfig` model config, and returns it as the runtime hparams object.

---

## General

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `spec_version` | int | `1` | Schema version for hparams compatibility. Bump when making breaking changes to the hparams format. |
| `project` | str | `"hone"` | Project identifier. Used for R2 bucket paths and logging namespaces. |
| `model_size` | str | -- | Required. Selects the model architecture file (`hparams/{model_size}.json`). Examples: `"1.4B"`, `"35B-A3B"`, `"120M"`. |

## Training

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sequence_length` | int | `4096` | Token sequence length per sample. |
| `micro_batch_size` | int | -- | Number of sequences per micro-batch (per gradient accumulation step). |
| `target_batch_size` | int | -- | Target total batch size across all miners in the network. Used to compute gradient scaling. |
| `batch_size` | int | `8` | Local batch size per miner per window. Total tokens per window = `batch_size * sequence_length`. |
| `inner_steps` | int | -- | Number of local optimizer steps per training window before uploading gradients. |
| `max_inner_steps` | int | -- | Upper bound on inner steps. Caps how many local steps a miner can take if the window is long. |
| `outer_learning_rate` | float | -- | Learning rate for the outer (global) gradient aggregation step. Controls how aggressively aggregated gradients are applied to the global model. |
| `outer_momentum` | float | -- | Momentum coefficient for the outer SGD optimizer. Higher values let velocity accumulate compression noise across outer steps and amplify per-window param jumps; lower values (e.g. `0.5`) keep the outer step closer to the per-window gradient direction. |
| `outer_nesterov` | bool | `true` | Whether to use Nesterov look-ahead in the outer SGD optimizer. Nesterov adds an extra `+momentum*grad` on top of the regular momentum step (~1.5x larger updates). Disable when compression noise is high to keep outer steps from over-shooting. |
| `reset_inner_optimizer_per_window` | bool | `false` | When true, drop all inner-optimizer per-param state (Adam moments / Muon momentum buffer / step counter) and reset the manual LR-warmup counter at the start of every chain window. Removes the "stale momentum after outer-step discontinuity" overshoot that shows up as a one-step loss spike right after each outer step, at the cost of re-warming the inner optimizer (≈30 inner steps of (k+1)/N LR ramp). Useful when outer-step compression noise is high. |
| `weight_decay` | float | -- | Weight decay coefficient applied during outer optimization. |
| `max_grad_norm` | float | -- | Maximum gradient norm for clipping during outer optimization. |

## Window / Sync

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `blocks_per_window` | int | `2` | Number of Bittensor blockchain blocks per training window. Determines window duration (~12 seconds per block). |
| `windows_per_weights` | int | `10` | Number of windows between full weight checkpoint saves to R2. |
| `outer_steps_per_shard` | int | `455` | Number of outer optimization steps per data shard. Controls how the global step counter maps to data cycling. |
| `validator_offset` | int | -- | Block offset for the validator relative to miners. The validator evaluates gradients from `validator_offset` blocks behind the current block. |
| `checkpoint_frequency` | int | -- | How often (in windows) to save a full model checkpoint. |

## Compression (DeMo)

These parameters control DeMo (Decoupled Momentum) gradient compression, which reduces gradient upload sizes for bandwidth-efficient decentralized training.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `momentum_decay` | float | `0.999` | Decay factor for the momentum buffer used in DeMo compression. Higher values retain more history. |
| `momentum_subtraction_alpha` | float | -- | Alpha for momentum subtraction in DeMo. Controls how much of the previous momentum is subtracted before compression. |
| `topk_compression` | int | `32` | Top-k sparsification ratio. Only the top `1/topk_compression` fraction of gradient components are kept. A value of 32 means ~3.1% of components are transmitted. |
| `target_chunk` | int | `64` | Target chunk size for DCT-based compression. Gradients are split into chunks of this size before applying the DCT. |
| `use_dct` | bool | -- | Whether to use Discrete Cosine Transform for gradient compression. When true, DCT is applied to gradient chunks before top-k selection. |
| `quantization_bins` | int | -- | Number of quantization bins for gradient values after compression. Fewer bins = smaller uploads. |
| `quantization_range` | float | -- | Range of the quantization grid, in standard deviations. Values outside this range are clipped. |

## Peer Management

These parameters govern how miners discover, select, and manage peers for gradient exchange.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gather_peer_count` | int | -- | Number of peers to gather gradients from each window. |
| `gather_share` | float | -- | Fraction of gathered peers whose gradients are actually applied. The rest serve as backups. |
| `gather_top_ratio` | float | -- | Multiplier for selecting the top-performing peers. Peers are ranked and the top `gather_peer_count * gather_top_ratio` are considered. |
| `reserve_peer_count` | int | -- | Number of reserve peers maintained as fallbacks if primary peers go offline. |
| `reserve_decay_ratio` | float | -- | Decay factor for reserve peer scores. Lower values age out stale reserves faster. |
| `minimum_peers` | int | -- | Minimum number of peers required to proceed with gradient aggregation. If fewer peers are available, the miner waits. |
| `peer_replacement_frequency` | int | -- | How often (in windows) to refresh the peer list by replacing underperforming peers. |
| `peer_list_window_margin` | int | -- | Number of windows of margin when checking peer liveness. A peer is considered active if it uploaded within `recent_windows + peer_list_window_margin` windows. |
| `active_check_interval` | int | -- | Interval (in seconds) between active peer health checks. |
| `recent_windows` | int | -- | Number of recent windows to consider when evaluating peer activity. |
| `gather_peers_slash_threshold` | float | -- | Threshold below which a peer's score triggers slashing (score penalty). Peers consistently below this are penalized. |
| `uids_per_window` | int | -- | Maximum number of UIDs (miners) a validator evaluates per window. Caps validator compute per window. |
| `time_window_delta_seconds` | int | -- | Time tolerance (in seconds) for window synchronization. Peers whose window timestamps diverge by more than this are ignored. |
| `window_flush_headroom_seconds` | int | -- | Seconds of head-room reserved at the end of every chain window for the miner to compress, merge across PP/FSDP ranks, and PUT its gradient before the validator's `time_window_delta_seconds` deadline. The inner training loop exits early once the wall-clock distance to the next window's start drops below this. Set to 0 to disable (legacy behaviour: train until window flips). |
| `reset_inactivity_windows` | int | -- | Number of consecutive inactive windows before a peer's state is reset. |
| `sync_max_steps_behind` | int | -- | Maximum number of global steps a miner can fall behind before triggering a resync from checkpoint. |
| `exclude_negative_peers` | bool | -- | Whether to exclude peers with negative scores from gradient aggregation. |
| `consecutive_negative_threshold` | int | -- | Number of consecutive windows with negative score before a peer is dropped entirely. |

## Scoring

Parameters controlling how the validator scores miner gradient contributions.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `power_normalisation` | float | -- | Exponent for power normalization of scores. Higher values sharpen the score distribution, concentrating incentive on top miners. |
| `binary_score_ma_alpha` | float | -- | Exponential moving average alpha for the binary (positive/negative) score signal. Lower values smooth more aggressively. |
| `bma_threshold` | float | -- | Binary moving average threshold. Scores below this threshold are treated as negative contributions. |
| `bma_warmup_windows` | int | -- | Number of windows before the binary moving average is used for scoring. During warmup, raw scores are used. |
| `missing_gradient_penalty_score` | float | -- | Score assigned to miners who fail to upload gradients for a window. Should be strongly negative. |
| `openskill_beta` | float | -- | Beta parameter for the OpenSkill rating system. Controls the width of the performance uncertainty distribution. |
| `openskill_tau` | float | -- | Tau parameter for OpenSkill. Controls the dynamics (volatility) of rating changes. |
| `num_evaluation_bins` | int | -- | Number of bins used when evaluating gradient quality. Gradients are bucketed by magnitude and each bin is scored independently. |
| `idx_overlap_threshold` | float | -- | Minimum overlap ratio between a miner's compressed gradient indices and the validator's reference indices for the gradient to be considered valid. |
| `incentive_burn_rate` | float | -- | Rate at which incentive is burned (redistributed) from low-scoring miners. 1.0 means full redistribution. |
| `eval_lr_factor` | float | `0.5` | Scaling factor for the learning rate used during validator evaluation. The validator applies gradients at `outer_learning_rate * eval_lr_factor` to measure loss improvement. |
| `validator_sample_micro_bs` | int | -- | Micro-batch size used by the validator when sampling evaluation data. Independent of miner micro-batch size. |

## Optimizer

The optimizer config is a nested object with a `type` selector and per-optimizer settings.

### Top-Level

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `optimizer.type` | str | -- | Optimizer type. Either `"muon"` or `"adamw"`. |

### AdamW

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `optimizer.adamw.betas` | list[float] | `[0.9, 0.95]` | Beta coefficients for AdamW (first and second moment decay rates). |
| `optimizer.adamw.eps` | float | `1e-8` | Epsilon for numerical stability in AdamW. |
| `optimizer.adamw.weight_decay` | float | `0.1` | Weight decay for AdamW. Applied as decoupled weight decay. |
| `optimizer.adamw.learning_rate` | float | `3e-4` | Peak learning rate for AdamW. |
| `optimizer.adamw.scheduler.warmup_steps` | int | `1000` | Number of outer steps for linear warmup. |
| `optimizer.adamw.scheduler.warmup_inner_steps` | int | `30` | Number of inner steps during the warmup phase. May differ from `inner_steps` to allow slower warmup. |
| `optimizer.adamw.scheduler.t_max` | int | `140000` | Total steps for the cosine annealing schedule. |
| `optimizer.adamw.scheduler.eta_min_factor` | float | `0.1` | Minimum learning rate as a fraction of peak. At the end of cosine annealing, `lr = learning_rate * eta_min_factor`. |

### Muon

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `optimizer.muon.momentum` | float | `0.95` | Momentum coefficient for the Muon optimizer. |
| `optimizer.muon.weight_decay` | float | `0.01` | Weight decay for Muon. |
| `optimizer.muon.head_lr_scale` | float | `0.2` | Learning rate scale factor for the language model head. The head uses `learning_rate * head_lr_scale`. |
| `optimizer.muon.embed_lr_scale` | float | `0.2` | Learning rate scale factor for embedding parameters. |
| `optimizer.muon.scalar_lr_scale` | float | `0.2` | Learning rate scale factor for scalar parameters (norms, biases). |
| `optimizer.muon.learning_rate` | float | `2e-3` | Peak learning rate for Muon. |
| `optimizer.muon.scheduler.warmup_steps` | int | `1000` | Number of outer steps for linear warmup. |
| `optimizer.muon.scheduler.warmup_inner_steps` | int | `30` | Number of inner steps during the warmup phase. |
| `optimizer.muon.scheduler.t_max` | int | `140000` | Total steps for the cosine annealing schedule. |
| `optimizer.muon.scheduler.eta_min_factor` | float | `0.1` | Minimum learning rate as a fraction of peak. |

## Model Architecture

These fields are defined in `hparams/{model_size}.json` and describe the transformer architecture.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tokenizer_name` | str | `"HuggingFaceTB/SmolLM2-135M"` | HuggingFace tokenizer identifier. |
| `hidden_size` | int | `2048` | Hidden dimension of the transformer. |
| `num_hidden_layers` | int | `24` | Number of transformer decoder layers. |
| `num_attention_heads` | int | `16` | Number of attention heads. |
| `num_key_value_heads` | int | `16` | Number of key/value heads for grouped-query attention. When less than `num_attention_heads`, GQA is used. |
| `intermediate_size` | int | `5504` | FFN intermediate dimension. For SwiGLU, the actual gate projection is `2 * intermediate_size`. |
| `activation_function` | str | `"swiGLU"` | Activation function for the FFN. |
| `max_position_embeddings` | int | `4096` | Maximum sequence length supported by the positional encoding. |
| `rope_theta` | float | `10000.0` | Base frequency for Rotary Position Embeddings. Higher values extend effective context length. |
| `rms_norm_eps` | float | `1e-5` | Epsilon for RMSNorm layers. |
| `vocab_size` | int | `49152` | Vocabulary size. Overridden by model-size configs (e.g., 256000 for Gemma-based tokenizers). |
| `tie_embeddings` | bool | `true` | Whether to tie input embedding and output projection weights. Set to `false` for large MoE models. |

## MoE (Mixture of Experts)

MoE fields are set in the model-size config file when using a sparse MoE architecture.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_moe` | bool | `false` | Enable Mixture of Experts layers. |
| `num_experts` | int | `8` | Total number of experts per MoE layer. |
| `moe_top_k` | int | `2` | Number of experts activated per token (top-k routing). |
| `moe_intermediate_size` | int | `null` | FFN intermediate size for each expert. If null, defaults to the dense `intermediate_size`. |
| `shared_expert_intermediate_size` | int | `null` | FFN intermediate size for the shared expert (always active). If null, no shared expert is used. |
| `moe_layers` | list[int] | `null` | Explicit list of layer indices that use MoE. If null, all layers use MoE when `use_moe` is true. |
| `moe_aux_loss_coeff` | float | `0.01` | Coefficient for the auxiliary load-balancing loss that encourages even expert utilization. |

## FSDP

Controls PyTorch Fully Sharded Data Parallelism for intra-node GPU sharding.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `fsdp.dp_shard` | int | -- | Number of GPUs to shard across with FSDP. Typically set to the number of GPUs per node (e.g., 4 or 8). |
| `fsdp.compile` | bool | -- | Whether to use `torch.compile` for the model. Enables kernel fusion and graph optimization. |
| `fsdp.mixed_precision` | str | -- | Mixed precision policy. Typically `"bfloat16"`. Sets the dtype for computation while keeping master weights in fp32. |

## Pipeline Parallelism

Controls inter-node pipeline parallelism with ResBM activation compression. See the [Pipeline Parallelism guide](pipeline-parallelism.md) for full details.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pipeline.enabled` | bool | `false` | Enable pipeline parallelism. When true, the model's layers are split across PP stages. |
| `pipeline.num_stages` | int | `4` | Number of pipeline stages. Layers are distributed evenly across stages. |
| `pipeline.bottleneck_dim` | int | `16` | Bottleneck dimension for ResBM activation compression. Compression ratio = `hidden_dim / bottleneck_dim`. |

## Other

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `offload_optimizer_states` | bool | -- | Offload optimizer state tensors to CPU memory. Reduces GPU memory usage at the cost of slower optimizer steps. Useful for large models or nodes with limited GPU memory. |
