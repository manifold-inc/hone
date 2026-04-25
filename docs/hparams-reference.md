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
| `batch_size` | int | `8` | Local batch size per miner per inner step. Tokens per inner step = `batch_size * sequence_length`. With pipeline parallelism, microbatches per inner step `M = batch_size / (micro_batch_size * world_size_per_stage)` -- raising `batch_size` (or lowering `micro_batch_size`) grows `M` and shrinks the PP bubble fraction `P / (P + M - 1)`. Recommended setting for the 8B-A1B genesis_moe config: `128` (M=16 at world=2, bubble ~17% at PP=3). |
| `inner_steps` | int | -- | Number of local optimizer steps per training window before uploading gradients. Tokens per window = `batch_size * sequence_length * inner_steps`. When `batch_size` is increased to grow PP microbatches, drop `inner_steps` proportionally to keep the window's wall-clock similar; e.g. `batch_size: 32 + inner_steps: 30 -> batch_size: 128 + inner_steps: 8` keeps tokens/window roughly constant while cutting the PP bubble in half. |
| `max_inner_steps` | int | -- | Upper bound on inner steps. Caps how many local steps a miner can take if the window is long. Scale with `inner_steps`. |
| `outer_learning_rate` | float | -- | Learning rate for the outer (global) gradient aggregation step. Controls how aggressively aggregated gradients are applied to the global model. |
| `outer_momentum` | float | -- | Momentum coefficient for the outer SGD optimizer. Higher values let velocity accumulate compression noise across outer steps and amplify per-window param jumps; lower values (e.g. `0.5`) keep the outer step closer to the per-window gradient direction. |
| `outer_nesterov` | bool | `true` | Whether to use Nesterov look-ahead in the outer SGD optimizer. Nesterov adds an extra `+momentum*grad` on top of the regular momentum step (~1.5x larger updates). Disable when compression noise is high to keep outer steps from over-shooting. |
| `outer_max_grad_norm` | float \| null | `null` | Global L2 cap on the aggregated outer gradient before it's applied via the outer SGD step. The same idea as `max_grad_norm` for the inner optimizer, just applied per-window to the gather-result aggregate: `grad *= min(1, threshold / grad.norm())`. Bounds the per-window parameter jump and breaks the runaway feedback loop where each outer step grows the next one's L2 (we observed `Fingerprint global_l2`: 7 -> 32 -> 75 -> ... -> 727 over a single training run). Set to `null` to disable clipping (legacy behaviour). Recommended ~`5.0` for the 8B-A1B genesis_moe config. |
| `outer_lr_schedule` | str | `"constant"` | Schedule for the outer SGD learning rate. `"constant"` keeps `outer_learning_rate` fixed (legacy behaviour). `"cosine"` builds a `CosineAnnealingLR` ticked once per applied outer step that anneals from `outer_learning_rate` down to `outer_learning_rate * outer_lr_min_factor` over `outer_lr_t_max` outer steps. |
| `outer_lr_t_max` | int | `1000` | When `outer_lr_schedule="cosine"`, the number of outer steps over which the cosine completes one half-period (i.e. lands at `eta_min`). At ~5 minutes per window this is ~3.5 days of wall-clock. Ignored when schedule is constant. |
| `outer_lr_min_factor` | float | `0.1` | When `outer_lr_schedule="cosine"`, the floor multiplier for the cosine: `eta_min = outer_learning_rate * outer_lr_min_factor`. Ignored when schedule is constant. |
| `reset_inner_optimizer_per_window` | bool \| float | `false` | Per-window inner-optimizer state policy. `false`/`0.0`: preserve state via CPU offload+prefetch (legacy). `true`/`1.0`: hard clear (`optimizer.state.clear()`); inner Adam/Muon re-allocate zero buffers on the next step. Any `0.0 < x < 1.0`: soft decay -- multiply every state tensor by `x` in place. Soft decay (e.g. `0.5`) is the recommended setting for live training: removes the "stale momentum applied to discontinuous outer-stepped params" overshoot without throwing away the inner-loop's accumulated learning signal. Pair with `outer_max_grad_norm` for full per-window jump control. |
| `reset_warmup_per_window` | bool | `true` | When true, the per-window inner-optimizer reset (above) also resets the manual LR-warmup counter so the first 30 post-reset inner steps re-ramp the LR via the (k+1)/N branch. Set to `false` to keep the warmup counter sticky across windows -- recovers ~30 inner steps per window of effective compute, at the cost of the first inner step running with full LR + (decayed or zeroed) momentum. Safe with `outer_max_grad_norm` set, since clipping bounds the post-outer-step jump that warmup was protecting against. |
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
| `fsdp.compile` | bool | -- | Whether to use `torch.compile` for the model. In PP mode the compile is applied per-leaf submodule (`embed_tokens`, every `pp_stage.layers[i]`, ResBM `boundary.encoder`/`boundary.decoder`, `lm_head`, `norm`) since `_pp_run_1f1b` calls leaves directly and a root-level compile would be a no-op. In non-PP mode the whole model is compiled end-to-end. |
| `fsdp.mixed_precision` | str \| null | `null` | Mixed precision policy passed to `MixedPrecisionPolicy(param_dtype=..., reduce_dtype=fp32)` on every `fully_shard(...)` call. Accepts `"bfloat16"`/`"bf16"` (recommended), `"float16"`/`"fp16"`, `"float32"`/`"fp32"`, or `null` to disable the policy entirely (legacy behaviour, all-gathers in fp32). With `bfloat16` the per-layer all-gather wire bandwidth is halved and a redundant fp32->bf16 cast inside autocast is removed; `reduce_dtype` is pinned to fp32 for grad-reduce stability. |
| `fsdp.activation_checkpoint` | str \| null | `null` | Selective activation checkpointing on transformer blocks. `null` / `"none"` / `false` disables AC. `"selective"` (recommended) wraps each `Block` in `checkpoint_wrapper(..., NO_REENTRANT)`, trading ~25% extra backward compute for substantially lower activation memory -- in MoE this frees enough room to grow `batch_size`/`micro_batch_size` and shrink the PP bubble. `"full"` is reserved for a future per-submodule policy and currently behaves like `"selective"`. AC is applied before FSDP wrapping so it sees plain `nn.Module` children. |

## Pipeline Parallelism

Controls inter-node pipeline parallelism with ResBM activation compression. See the [Pipeline Parallelism guide](pipeline-parallelism.md) for full details.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pipeline.enabled` | bool | `false` | Enable pipeline parallelism. When true, the model's layers are split across PP stages. |
| `pipeline.num_stages` | int | `4` | Number of pipeline stages. Layers are distributed evenly across stages. Must match `--pp-num-stages` on the per-stage `torchrun` invocations *and* the validator's expectation -- bumping this requires consensus across the network. |
| `pipeline.bottleneck_dim` | int | `16` | Bottleneck dimension for ResBM activation compression. Compression ratio = `hidden_dim / bottleneck_dim`. |

## PP Transport

Controls the cross-stage activation/gradient transport used by `_pp_run_1f1b`. Two backends are supported under one class (`hone.PPTransport`): asynchronous TCP (default, works cross-node without RDMA) and an opt-in NCCL-P2P fast path for same-node setups.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pp_transport.async_send` | bool | `true` | Enable the background sender thread + bounded outbound queue. With `true`, `send_next` / `send_prev` enqueue the (already-staged-to-CPU) tensor and return immediately; a per-direction worker drains the queue to the socket while the next forward/backward microbatch runs on the GPU. With `false` the send is fully synchronous (legacy behaviour, useful for debugging). |
| `pp_transport.send_queue_depth` | int | `2` | Maximum number of in-flight queued tensors per direction. The bound exists to cap CPU memory growth under a slow peer; `2` lets the next forward overlap with the previous send, which is sufficient when stages are roughly balanced. Raise (e.g. `4`) only if the peer is consistently slower than compute. |
| `pp_transport.intra_node_nccl` | bool | `false` | Opt into the NCCL P2P fast path when both adjacent stages are on loopback. When enabled, builds a side `ProcessGroupNCCL` from a shared `TCPStore` rendezvous (`PP_NCCL_INIT_METHOD` env, default `tcp://127.0.0.1:29800`) and ships activations GPU-to-GPU over NVLink, skipping the D2H+serialize+H2D round-trip. Requires the side rendezvous to be reachable from every PP-stage process; falls back to async TCP transport on any bringup error. Default `false` so out-of-the-box runs use the (already-fast) async TCP path. |

## DataLoader

Controls how training tokens are fed into the inner loop.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataloader.num_workers` | int | `0` | Number of background worker processes that tokenize/collate microbatches. With `0` (legacy default) the main thread calls `next(loader_iter)` synchronously per microbatch, which becomes a noticeable share of the inner loop once the GPU compute is fast. Recommended: `4`. |
| `dataloader.pin_memory` | bool | `false` | Allocate output tensors in pinned CPU memory so the H2D copy can be zero-stage and overlap with compute. Recommended: `true` whenever `num_workers > 0`. |
| `dataloader.persistent_workers` | bool | `false` | Keep worker processes alive across epochs. Saves the per-epoch fork cost; requires `num_workers > 0`. Recommended: `true`. |
| `dataloader.prefetch_factor` | int | `2` | How many batches each worker pre-fetches; total queue depth is `num_workers * prefetch_factor`. Only honored when `num_workers > 0`. Recommended: `4`. |

## Other

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `offload_optimizer_states` | bool | -- | Offload optimizer state tensors to CPU memory. Reduces GPU memory usage at the cost of slower optimizer steps. Useful for large models or nodes with limited GPU memory. |
