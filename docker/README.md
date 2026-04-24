# Hone docker-compose stack

One-shot way to bring up a validator + N miners on a single GPU box.
Each miner pair runs a 2-stage pipeline-parallel split with
ResBM-compressed activations going over a Docker bridge network (TCP),
just like the production cross-node setup -- only the wire happens to
not leave the host.

## What you get

The shipped `docker-compose.yml` defines five services:

| service | role | default GPU |
|---|---|---|
| `validator` | validator (single process, FSDP intra-process) | 0 |
| `miner1-s0` | miner #1 pipeline-stage 0 | 1 |
| `miner1-s1` | miner #1 pipeline-stage 1 | 2 |
| `miner2-s0` | miner #2 pipeline-stage 0 | 3 |
| `miner2-s1` | miner #2 pipeline-stage 1 | 4 |

Two miners share one validator so you can exercise the multi-peer
gather path (the new `stage_id_filter` codepath in `Comms.gather`) on
a single node before going cross-node. Five GPUs total at the
defaults; map your real GPU IDs in `.env`.

## Prerequisites

- Docker Engine ≥ 24 with `docker compose` v2.
- NVIDIA Container Toolkit (`nvidia-ctk runtime configure --runtime=docker`).
- An NVIDIA driver that supports CUDA 12.4+ (B200/H200/H100 ≥ 550).
- A bittensor wallet directory at `~/.bittensor/wallets` containing
  three coldkey/hotkey trees:
  - one for the validator
  - one per miner (each miner needs its own hotkey -- gradient files
    in R2 are namespaced by hotkey, two miners on the same hotkey
    would clobber each other)
- R2 (or S3-compatible) credentials filled into `.env`.

## First-time setup

```bash
cd hone/docker
cp .env.example .env
$EDITOR .env                       # fill in wallets, GPUs, R2 keys

# Create the wallets if you don't have them yet:
btcli wallet new_coldkey --wallet.name hone-validator
btcli wallet new_hotkey  --wallet.name hone-validator --wallet.hotkey default
btcli wallet new_coldkey --wallet.name hone-miner1
btcli wallet new_hotkey  --wallet.name hone-miner1   --wallet.hotkey default
btcli wallet new_coldkey --wallet.name hone-miner2
btcli wallet new_hotkey  --wallet.name hone-miner2   --wallet.hotkey default
# Register each on the netuid (same netuid as NETUID in .env):
btcli subnet register --wallet.name hone-validator --wallet.hotkey default --netuid 268
btcli subnet register --wallet.name hone-miner1    --wallet.hotkey default --netuid 268
btcli subnet register --wallet.name hone-miner2    --wallet.hotkey default --netuid 268

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
docker compose logs -f miner1-s0 miner1-s1
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
[entrypoint] starting miner: pp_stage=0/2 wallet=hone-miner1/default ...
[PPTransport] stage=0/2 rank=0 connected to miner1-s1:50000
...
---------- Window: 309xxx (Outer Steps Taken: 0) ----------
Inner Step 1, Batch 4, loss: ...
```

## Adding a third (or fourth, ...) miner

Copy the `miner2-*` block in `docker-compose.yml`, rename to `miner3-s0`
/ `miner3-s1`, point `PP_PEER_HOST_NEXT` / `PP_PEER_HOST_PREV` at the
new sibling service name, and assign GPU IDs that aren't already used.
Add matching env vars to `.env`:

```env
MINER3_WALLET_NAME=hone-miner3
MINER3_WALLET_HOTKEY=default
MINER3_S0_GPU_ID=5
MINER3_S1_GPU_ID=6
```

The PP listen-port can stay at `50000` for every container -- they're
each in their own network namespace, so ports never collide.

## Scaling stage width (intra-stage FSDP)

To run a stage across multiple GPUs (for larger configs like
`qwen-moe`), bump `NPROC_PER_NODE` and map more GPUs into the
container:

```yaml
miner1-s0:
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

## Single-node-no-PP miner

For a non-PP miner (`pipeline.enabled: false` in hparams), use one
service per miner:

```yaml
solo-miner1:
  build: *build
  ...
  environment:
    <<: *common-env
    NODE_TYPE: miner
    PP_NUM_STAGES: "1"
    NPROC_PER_NODE: "8"   # full FSDP across the box
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

**`PPTransport ... could not connect to miner1-s1:50000 within 120s`**
The peer container hasn't started yet. The compose file has
`depends_on: [validator]` only -- if the peer crashed at startup
(usually a wallet or R2 misconfiguration), you'll see this on the
healthy stage. Inspect `docker compose logs miner1-s1`.

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
