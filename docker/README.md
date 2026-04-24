# Hone docker-compose stack

One-shot way to bring up a validator + a single PP-split miner on one
GPU box. Each pipeline stage of the miner runs in its own container
with one GPU pinned, sharing a single wallet hotkey -- the subnet sees
one UID per (cold, hot) pair regardless of how many stage containers
back it. Per-stage R2 gradient files are namespaced by the
``-stage{N}`` suffix that `Comms.put` adds, so they don't clobber
each other.

## What you get

The shipped `docker-compose.yml` defines three services:

| service | role | default GPU |
|---|---|---|
| `validator` | validator (single process, FSDP intra-process) | 0 |
| `miner-s0` | miner pipeline-stage 0 (embed + first half of layers) | 1 |
| `miner-s1` | miner pipeline-stage 1 (second half + norm + lm_head) | 2 |

Three GPUs total at the defaults; map your real GPU IDs in `.env`.

## Prerequisites

- Docker Engine ≥ 24 with `docker compose` v2.
- NVIDIA Container Toolkit (`nvidia-ctk runtime configure --runtime=docker`).
- An NVIDIA driver that supports CUDA 12.4+ (B200/H200/H100 ≥ 550).
- A bittensor wallet directory at `~/.bittensor/wallets` with two
  coldkey/hotkey trees (one validator, one miner). Both stage
  containers share the miner hotkey.
- R2 (or S3-compatible) credentials filled into `.env`.

## First-time setup

```bash
cd hone/docker
cp .env.example .env
$EDITOR .env                       # fill in wallets, GPUs, R2 keys

# Create the wallets if you don't have them yet:
btcli wallet new_coldkey --wallet.name hone-validator
btcli wallet new_hotkey  --wallet.name hone-validator --wallet.hotkey default
btcli wallet new_coldkey --wallet.name hone-miner
btcli wallet new_hotkey  --wallet.name hone-miner    --wallet.hotkey default

# Register each on the netuid (same NETUID as in .env):
btcli subnet register --wallet.name hone-validator --wallet.hotkey default --netuid 268
btcli subnet register --wallet.name hone-miner     --wallet.hotkey default --netuid 268

# Build the image once. Subsequent edits to source files are picked up
# automatically because /app is a bind-mount and the package is
# installed editable.
docker compose build
```

## Running

```bash
# Foreground (good for first run -- you'll see startup errors immediately):
docker compose up

# Detached:
docker compose up -d

# Logs:
docker compose logs -f validator
docker compose logs -f miner-s0 miner-s1
```

Healthy startup looks like this in `validator` logs:

```
[entrypoint] starting validator: netuid=268 wallet=hone-validator/default
[Init] code_version=...
[Init] Bittensor wallet loaded
...
Window -> 309xxx
```

And in each miner stage:

```
[entrypoint] starting miner: pp_stage=0/2 wallet=hone-miner/default ...
[PPTransport] stage=0/2 rank=0 connected to miner-s1:50000
...
---------- Window: 309xxx (Outer Steps Taken: 0) ----------
Inner Step 1, Batch 4, loss: ...
```

## Adding more pipeline stages

To split the miner into 3+ pipeline stages (matching `hparams.pipeline.num_stages`):

1. Bump `PP_NUM_STAGES` in `.env`.
2. Set `pipeline.num_stages` in `hparams/hparams.json` to the same value.
3. Copy the `miner-s1` block in `docker-compose.yml`, rename to
   `miner-s2`, set `PP_STAGE: "2"`, point its `PP_PEER_HOST_PREV` at
   `miner-s1`, and update `miner-s1`'s `PP_PEER_HOST_NEXT` to
   `miner-s2`. The first stage has only `PP_PEER_HOST_NEXT`, the last
   stage has only `PP_PEER_HOST_PREV`, middle stages have both.
4. Add a `MINER_S2_GPU_ID=N` line to `.env` and a matching
   `device_ids` entry in the new service block.

The PP listen-port can stay at `50000` for every container -- they're
each in their own network namespace, so ports never collide.

## Scaling stage width (intra-stage FSDP)

To run a stage across multiple GPUs (for larger configs like
`qwen-moe`), bump `NPROC_PER_NODE` and map more GPUs into the
container:

```yaml
miner-s0:
  environment:
    NPROC_PER_NODE: "4"
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ["1", "2", "3", "4"]
            capabilities: [gpu]
```

The entrypoint detects `NPROC_PER_NODE > 1` and switches from `python`
to `torchrun --nproc_per_node=...` automatically. Don't forget to set
`fsdp.dp_shard` in `hparams/hparams.json` to match.

## Single-stage (no PP) miner

For a non-PP miner (`pipeline.enabled: false` in hparams), drop the
`miner-s1` service entirely and set on `miner-s0`:

```yaml
miner-s0:
  environment:
    PP_NUM_STAGES: "1"
    NPROC_PER_NODE: "8"   # full FSDP across the box
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ["0", "1", "2", "3", "4", "5", "6", "7"]
            capabilities: [gpu]
```

## Stopping

```bash
docker compose down                 # stop and remove containers
docker compose down --volumes       # also drop the bind-mount logs (rare)
```

The training code installs SIGTERM handlers (see
`base_node._setup_signal_handlers`) and `tini` in the image forwards
signals correctly, so `docker stop` triggers a graceful shutdown that
flushes the current window's state before exiting.

## Troubleshooting

**`PPTransport ... could not connect to miner-s1:50000 within 120s`**
The peer container hasn't started yet. The compose file has
`depends_on: [validator]` only -- if `miner-s1` crashed at startup
(usually a wallet or R2 misconfiguration), you'll see this on
`miner-s0`. Inspect `docker compose logs miner-s1`.

**`OSError: [Errno 2] No such file or directory: '/root/.bittensor/wallets/...'`**
Wallet mount is wrong. Check `BITTENSOR_WALLETS_DIR` in `.env`. The
directory you point at must contain `<WALLET_NAME>/coldkeypub.txt` and
`<WALLET_NAME>/hotkeys/<WALLET_HOTKEY>`.

**`CUDA out of memory`**
Same triage as bare-metal -- start with smaller `micro_batch_size` in
`hparams.json`. The chunked cross-entropy in `hone/src/hone/loss.py`
caps fp32 logits at `chunk_tokens × V × 4` bytes per chunk so
`micro_batch_size` is no longer the limiting term, but expert
activation memory still scales with it.

**`Object gradient-... was uploaded too late`** in validator logs
The miner's window-flush headroom may be too tight under your hardware.
Bump `window_flush_headroom_seconds` in `hparams/hparams.json` (default
50) until uploads land before the validator's `time_window_delta_seconds`
deadline.
