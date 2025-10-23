# Hone Subnet — ARC‑AGI Problem Solving

A Bittensor subnet where **validators** continuously generate novel ARC‑AGI‑2 problems and score **miners** based on quality and latency
---

## Overview

- [**Validators**](#validator-setup)
  - Discover registered miners from chain, generate synthetic ARC‑AGI‑2 tasks, dispatch queries, score responses, and push **on‑chain weights** by performance.
  - Persists results + metrics to Postgres for analytics and stability.
- [**Miners**](#miner-setup)
  - Expose a minimal HTTP API (FastAPI) that receives ARC tasks and returns predicted grids.
  - You can plug your own solver (LLM, search, program synthesis, classical vision, HRMs, etc).

---

## Repository Layout (core)

```
hone/
├── common/                      # shared utilities
│   ├── chain.py                 # Bittensor chain interface
│   ├── constants.py
│   ├── epistula.py              # message signing/verification
│   ├── mock_chain.py            # mock chain for local tests
│   └── utils.py
├── validator/
│   ├── Dockerfile
│   ├── config.py
│   ├── cycle.py                 # main validation loop
│   ├── db.py                    # asyncpg DB client + queries
│   ├── discovery.py             # miner discovery from chain
│   ├── main.py                  # entrypoint
│   ├── query.py                 # miner query client
│   ├── scoring.py               # scoring & weight calc
│   ├── requirements.txt
│   ├── docker-compose.yml       # production (real chain)
│   ├── docker-compose.test.yml  # local test (mock chain + 3 miners)
│   └── sql/
│       └── init.sql             # DB schema (Postgres)
└── miner/
    ├── Dockerfile
    ├── __init__.py
    ├── config.py
    ├── handlers.py              # request handlers
    ├── main.py                  # FastAPI app
    └── arc/
      └── models.py              # request/response schemas
      └── solver.py              # plug your solver here
    ├── task_queue.py            # async job mgmt
    ├── keypair.py               # signing if needed
    ├── health.py                # liveness/readiness
    ├── query.py                 # (client helpers, if used)
    ├── check_task.py            # background check helpers
    └── requirements.txt
```

---

## Deployment (mainnet)

### Prerequisites

- **Python 3.10+**
- **Docker & Docker Compose**
- **Bittensor** installed (`btcli` available)
- A wallet with some **TAO** (Finney testnet or mainnet)

### Create/Import Wallets

```bash
# Coldkey
btcli wallet new_coldkey --wallet.name default

# Hotkeys
btcli wallet new_hotkey --wallet.name default --wallet.hotkey validator
btcli wallet new_hotkey --wallet.name default --wallet.hotkey miner
```

> [!NOTE] Wallet files location
> Store wallet files on the host under `~/.bittensor/wallets`.  
> The validator and miner containers **mount** this directory read‑only.


### Validator Setup

#### Requirements
Minimum: 
- 4 CPU
- 4gb RAM
- 10gb disk

#### Setup Environment Variables

Create **`validator/.env`** (or use your CI secrets).   
Defaults shown match `validator/docker-compose.yml`:

```ini
# ---- Chain ----
CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
NETUID=5

# ---- Wallets ----
WALLET_NAME=default
WALLET_HOTKEY=validator
WALLET_PATH=                          # optional custom path

# ---- DB ----
DB_URL=postgresql://postgres:postgres@db:5432/hone

# ---- Cycle config ----
CYCLE_DURATION=30                     # blocks (≈ 6 min on Finney)

# ---- Logging ----
LOG_LEVEL=INFO
```

> [!NOTE] 
> The compose file will also pass `DB_URL` explicitly.   
> If you keep both, `environment` usually wins over `.env`.

#### Register & Stake

```bash
# Register validator
btcli subnet register --netuid 5 --wallet.name default --wallet.hotkey validator

# Stake TAO
btcli stake add --wallet.name default --wallet.hotkey validator --amount 100
```

#### Start Validator (DB + App)

From `validator/`:

```bash
# Bring up Postgres + Adminer + Validator
docker-compose up -d

# Follow logs
docker-compose logs -f validator
```

This will:
- Start **Postgres 15** with schema applied from `sql/init.sql`
- Expose **Adminer** on `${ADMINER_PORT:-8080}` (optional UI)
- Start the **validator** on `${VALIDATOR_PORT:-8092}`

**Schema:** (from `sql/init.sql`)
- `miners(uid, hotkey, ip, port, stake, last_update_block, ...)`
- `query_results(...)` with **exact_match**, **partial_correctness**, **grid_similarity**, **efficiency_score**, **response_time**, **problem_id**
- `scores(...)` with aggregate metrics

### Miner Setup

#### Set Environment Variables

Create **`miner/.env`**:

```ini
# ---- Wallet ----
WALLET_NAME=default
WALLET_HOTKEY=miner

# ---- Server ----
HOST=0.0.0.0
MINER_PORT=8091
LOG_LEVEL=INFO

# ---- Testing toggles ----
SKIP_EPISTULA_VERIFY=false           # true for local dev only
```

If using LLMs, configure your provider:

**OpenAI:**
```ini
OPENAI_API_KEY=openaikei-...
OPENAI_MODEL=gpt-4o
```

**OpenRouter:**
```ini
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=mistralai/mistral-7b-instruct:free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1  # optional
```

#### Register Miner & Set IP on Chain

```bash
# Register the miner identity on the subnet
btcli subnet register --netuid 5 --wallet.name default --wallet.hotkey miner

# Set public IP + port on‑chain so the validator can discover you
python tools/post_ip_chain.py \
  --wallet-name default \
  --hotkey miner \
  --ip YOUR_PUBLIC_IP \
  --port 8091 # or any other port you're using
```

> [!TIP] Ensure ports are open
> Ensure your firewall allows inbound traffic to the miner port.

#### Start Miner (Docker)

From the repo root (or `miner/`):

```bash
docker build -t hone-miner miner/
docker run -d --name miner \
  -p 8091:8091 \
  -v ~/.bittensor/wallets:/root/.bittensor/wallets:ro \
  --env-file miner/.env \
  hone-miner

docker logs -f miner
```

> [!TIP]
> The miner serves **FastAPI** endpoints and should respond healthily at `/health`.

---

## How the Validator Works (High‑Level)

1. **Discovery** (`discovery.py`): Pull registered miners from chain (`common/chain.py`), ingest IP/port + stake.
2. **Cycle Loop** (`cycle.py`):
   - For `CYCLE_DURATION` blocks:
     - Generate **ARC‑AGI‑2** problem (`validator/synthetics/arc_agi2_generator.py`).
     - Query all miners concurrently (`query.py`, Epistula‑signed requests).
     - Score responses (`scoring.py`) using:
       - **Exact match** (≈40% weight)
       - **Partial correctness** (≈30%)
       - **Grid/pixel similarity** (≈20%)
       - **Efficiency** / latency (≈10%)
     - Persist results to Postgres (`db.py`).
   - Aggregate last N cycles into a **score** per miner.
3. **Weight Setting**: Push updated weights on‑chain (via chain interface).

> [!TIP] 
> Exact weights and confi constants can be tuned in `scoring.py`.  
> Responses are polled (for now 50 attempts, 10s intervals between them).

---

## Miner API (FastAPI)

- `GET /health` → basic liveness/readiness (from `health.py`)
- `POST /query` → solve an ARC task and return predicted grid
  - Typical request contains **problem definition** & **input grid(s)**
  - Response returns an **output grid**
- The **solver entrypoint** is in `solver.py`. Replace baseline logic with your own approach (LLM, search, programs, HRMs, etc ).
- Request handlers live in `handlers.py`. Background checks / queueing in `task_queue.py` and helpers in `check_task.py`.

> [!WARNING] 
> Validator <-> miner messages are signed/verified using **Epistula** (`common/epistula.py`).   
> In production, ***do not*** set `SKIP_EPISTULA_VERIFY=true`.

---

## Development / Local Testing (Mock Chain)

### One‑shot Local Test Script

```bash
chmod +x test_local.sh
./test_local.sh
```

This will:
- Start **Postgres**
- Launch **3 mock miners** (on different ports)
- Run **validator with mock chain** (`USE_MOCK_CHAIN=true`)
- Stream scoring, queries, and weight updates in real time

### Manual Local Test (compose)

From `validator/`:

```bash
docker-compose -f docker-compose.test.yml up
# validator on 8092, miners on 8091/8093/8094, Adminer on 8081

# Tail logs
docker logs validator-validator-1
docker logs validator-miner1-1

# Teardown
docker-compose -f docker-compose.test.yml down -v
```

> [!NOTE] 
> The test compose sets `USE_MOCK_CHAIN=true`, `SKIP_EPISTULA_VERIFY=true`, and points `DB_URL` to the test DB (`hone_test`).

---

## Configuration Reference

### Validator (env)

```ini
# Chain (testnet / mainnet)
CHAIN_ENDPOINT=wss://test.finney.opentensor.ai:443
NETUID=5

# Wallets
WALLET_NAME=default
WALLET_HOTKEY=validator
WALLET_PATH=

# DB
DB_URL=postgresql://postgres:postgres@db:5432/hone

# Cycles
CYCLE_DURATION=30

# Testing flags (for local only)
USE_MOCK_CHAIN=false
SKIP_EPISTULA_VERIFY=false

# Logging
LOG_LEVEL=INFO
```

### Miner (env)

```ini
WALLET_NAME=default
WALLET_HOTKEY=miner

HOST=0.0.0.0
MINER_PORT=8091
LOG_LEVEL=INFO

# Testing only
SKIP_EPISTULA_VERIFY=false
```

> [!TIP]
> The solver supports both OpenAI and OpenRouter. Configure the appropriate API key and model in your environment. The miner will automatically detect which provider to use based on the variables set.

---

## Database Schema (Postgres)

Applied from `validator/sql/init.sql` at container start:

- **miners**: on‑chain view of registered miners + metadata
- **query_results**: per‑task metrics and raw response JSON
- **scores**: aggregated performance features for weight setting

Indexes are created for `block`, `uid`, and `timestamp` fields for efficient querying.

---

## Troubleshooting

- **Validator cannot connect to DB**
  - Check `DB_URL` and that `db` service is healthy: `docker ps`, `docker logs validator-db-1`
- **Miner not discovered**
  - Ensure you registered miner hotkey on correct `NETUID`.
  - Post correct **public IP/port** on‑chain and open firewall/NAT for the port.
- **Signature/verification failures**
  - Set `SKIP_EPISTULA_VERIFY=false` in production on both sides.
  - Verify the miner and validator use the intended wallets.
- **No scores updating**
  - Check `CYCLE_DURATION` and validator logs. Ensure miners respond within the timeout (~30s).

---

## Security Notes

- Wallets are mounted read‑only from host: `~/.bittensor/wallets:/root/.bittensor/wallets:ro`
- Keep `SKIP_EPISTULA_VERIFY=false` in production.
- Never expose Adminer to the public internet without access control.

---

## License
