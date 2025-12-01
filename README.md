# Hone Subnet — ARC-AGI-2 Benchmarking on Bittensor

A Bittensor subnet where **validators** evaluate **miners** on their ability to solve novel ARC-AGI-2 reasoning problems. Miners don't run solvers directly—they point to a git repository containing their solution, which is executed in a secure GPU sandbox.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Validator Setup](#validator-setup)
- [Miner Setup](#miner-setup)
- [Building Your Solver](#building-your-solver)
- [Local Testing with Sandbox Runner](#local-testing-with-sandbox-runner)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Security Notes](#security-notes)

---

## Overview

### How It Works

1. **Miners** expose an HTTP endpoint (`/info`) that returns a pointer to their solution repository
2. **Validators** fetch miner info, submit jobs to a **Sandbox Runner** (secure GPU execution service)
3. The Sandbox Runner clones the miner's repo, builds a Docker image, runs prep (with internet) and inference (isolated), then calculates metrics
4. Validators aggregate `exact_match_rate` scores and set on-chain weights using exponential distribution

### Scoring Mechanism

- **Metric**: `exact_match_rate` — percentage of ARC problems solved correctly
- **Minimum floor**: 20% accuracy required to qualify
- **Top 5** miners above floor receive rewards
- **No qualifiers**: If no miners meet the floor, 100% is burned

### Key Features

- **Submission caching**: Identical repo+branch+commit combinations use cached scores (no redundant evaluation)
- **Daily limits**: Configurable submissions per miner per day (default: 1)
- **GPU isolation**: Inference runs without network access
- **vLLM support**: Optional LLM sidecar for transformer-based solvers

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BITTENSOR CHAIN                                │
│                    (miner registration, weights, stake)                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                 ┌────────────────────┼────────────────────┐
                 │                    │                    │
                 ▼                    ▼                    ▼
          ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
          │  VALIDATOR  │      │  VALIDATOR  │      │  VALIDATOR  │
          │             │      │             │      │             │
          │ • Discover  │      │             │      │             │
          │ • Query     │      │             │      │             │
          │ • Score     │      │             │      │             │
          │ • Set wts   │      │             │      │             │
          └──────┬──────┘      └─────────────┘      └─────────────┘
                 │
                 │ Submit jobs via API
                 ▼
          ┌─────────────────────────────────────────────────────┐
          │                  SANDBOX RUNNER                      │
          │                                                      │
          │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
          │  │ H200 #0 │  │ H200 #1 │  │ H200 #2 │  │ H200 #3 │ │
          │  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │
          │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
          │  │ H200 #4 │  │ H200 #5 │  │ H200 #6 │  │ H200 #7 │ │
          │  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │
          │                                                      │
          │  • Clone repo → Build image → Run prep → Run infer  │
          │  • Calculate exact_match_rate against held-out data │
          └─────────────────────────────────────────────────────┘
                 │
                 │ Fetch /info
                 ▼
          ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
          │   MINER 1   │      │   MINER 2   │      │   MINER N   │
          │             │      │             │      │             │
          │  /info →    │      │  /info →    │      │  /info →    │
          │  repo_url   │      │  repo_url   │      │  repo_url   │
          │  weight_cls │      │  weight_cls │      │  weight_cls │
          └─────────────┘      └─────────────┘      └─────────────┘
```

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Docker & Docker Compose**
- **Bittensor CLI** (`btcli`)
- **NVIDIA GPU + drivers** (for sandbox runner / local testing)
- TAO for registration and staking

### Create Wallets

```bash
# create coldkey
btcli wallet new_coldkey --wallet.name default

# create hotkeys
btcli wallet new_hotkey --wallet.name default --wallet.hotkey validator
btcli wallet new_hotkey --wallet.name default --wallet.hotkey miner
```

---

## Validator Setup

### Requirements

- 4+ CPU cores
- 8GB+ RAM
- 20GB disk
- Reliable network connection

### 1. Clone Repository

```bash
git clone https://github.com/manifold-inc/hone.git
cd hone/validator
```

### 2. Configure Environment

Create `validator/.env`:

```ini
# chain
NETUID=5
CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443

# wallet
WALLET_NAME=default
WALLET_HOTKEY=validator
WALLET_PATH=/root/.bittensor/wallets

# database
DB_URL=postgresql://postgres:postgres@db:5432/hone
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=hone

# sandbox runner connection
SANDBOX_RUNNER_ENDPOINT=http://your-sandbox-runner:8000
SANDBOX_RUNNER_API_KEY=your_api_key_here
SANDBOX_RUNNER_TIMEOUT_HOURS=3

# scoring parameters
MAX_SUBMISSIONS_PER_DAY=1
MIN_ACCURACY_FLOOR=0.20
TOP_MINERS_COUNT=5

# cycle timing
CYCLE_DURATION=30
```

### 3. Register and Stake

```bash
# register validator on subnet
btcli subnet register --netuid 5 --wallet.name default --wallet.hotkey validator

# stake TAO
btcli stake add --wallet.name default --wallet.hotkey validator --amount 100
```

### 4. Start Validator

```bash
cd validator
make up
```

This starts:
- PostgreSQL database
- Adminer (DB UI on port 8080)
- Validator service

### 5. Monitor

```bash
# view logs
make logs

# check status
make status

# auto-updater logs
make logs-update
```

The validator includes auto-update functionality that pulls and restarts on new commits.

---

## Miner Setup

### Requirements

- Public IP address
- Open port (default: 8091)
- Minimal compute (the heavy lifting happens in sandbox)

### 1. Clone Repository

```bash
git clone https://github.com/manifold-inc/hone.git
cd hone
```

### 2. Configure Environment

Create `miner/.env`:

```ini
WALLET_NAME=default
WALLET_HOTKEY=miner
MINER_PORT=8091

# your solution repository
MINER_REPO_URL=https://github.com/your-username/your-arc-solver
MINER_REPO_BRANCH=main
MINER_REPO_PATH=              # subdirectory if needed
MINER_WEIGHT_CLASS=1xH200     # 1xH200, 2xH200, 4xH200, or 8xH200

# vLLM settings (optional)
MINER_USE_VLLM=true
VLLM_MODEL=unsloth/Meta-Llama-3.1-8B-Instruct
VLLM_DTYPE=half
VLLM_GPU_MEMORY_UTIL=0.8
VLLM_MAX_MODEL_LEN=12000
```

### 3. Register Miner

```bash
# register on subnet
btcli subnet register --netuid 5 --wallet.name default --wallet.hotkey miner

# set your public IP on-chain so validators can discover you
python tools/post_ip_chain.py \
  --wallet-name default \
  --hotkey miner \
  --ip YOUR_PUBLIC_IP \
  --port 8091
```

### 4. Start Miner

```bash
# build and run
docker build -t hone-miner -f miner/Dockerfile .
docker run -d --name miner \
  -p 8091:8091 \
  -v ~/.bittensor/wallets:/root/.bittensor/wallets:ro \
  --env-file miner/.env \
  hone-miner
```

### 5. Verify

```bash
# check health
curl http://localhost:8091/health

# check info endpoint (what validators see)
curl http://localhost:8091/info
```

Expected `/info` response:
```json
{
  "repo_url": "https://github.com/your-username/your-arc-solver",
  "repo_branch": "main",
  "weight_class": "1xH200",
  "use_vllm": true,
  "vllm_config": {
    "model": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "dtype": "half",
    "gpu_memory_utilization": 0.8,
    "max_model_len": 12000
  },
  "version": "1.0.0",
  "hotkey": "5Abc...xyz"
}
```

---

## Building Your Solver

Your solution lives in a git repository. The sandbox runner clones it, builds a Docker image, and runs two phases:

### Required Files

```
your-solver-repo/
├── Dockerfile           # builds your execution environment
├── requirements.txt     # python dependencies
├── arc_main.py          # entry point (CLI wrapper)
├── arc_prep_phase.py    # downloads models, data (internet ON)
├── arc_inference_phase.py  # solves problems (internet OFF)
├── arc_solver_llm.py    # your solver implementation (or any name)
└── arc_utils.py         # I/O utilities
```

### Execution Flow

```
┌────────────────────────────────────────────────────────────────┐
│                      PREP PHASE                                │
│                   (internet enabled)                           │
│                                                                │
│  • Download model weights from HuggingFace                     │
│  • Download any auxiliary data                                 │
│  • Models saved to /app/models                                 │
│                                                                │
│  Command: python arc_main.py --phase prep --input /input       │
│                              --output /output                  │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                    INFERENCE PHASE                             │
│                   (internet disabled)                          │
│                                                                │
│  • Load models from /app/models                                │
│  • Read problems from /input/miner_current_dataset.json        │
│  • Solve each problem                                          │
│  • Write predictions to /output/results.json                   │
│                                                                │
│  Command: python arc_main.py --phase inference --input /input  │
│                              --output /output                  │
└────────────────────────────────────────────────────────────────┘
```

### Input Format

`/input/miner_current_dataset.json`:
```json
{
  "tasks": [
    {
      "task_hash": "abc123...",
      "train_examples": [
        {"input": [[0,1,2],[3,4,5]], "output": [[5,4,3],[2,1,0]]}
      ],
      "test_input": [[1,2,3],[4,5,6]],
      "metadata": {}
    }
  ]
}
```

### Output Format

`/output/results.json`:
```json
{
  "phase": "inference",
  "status": "success",
  "predictions": [
    {
      "problem_index": 0,
      "task_hash": "abc123...",
      "predicted_output": [[6,5,4],[3,2,1]]
    }
  ]
}
```

### Solver Interface

Your solver must implement:

```python
class ARCSolver:
    def __init__(self, use_vllm: bool = True):
        # initialize your model/algorithm
        pass
    
    def solve(
        self,
        train_examples: List[Dict],  # [{"input": grid, "output": grid}, ...]
        test_input: List[List[int]]  # 2D grid of ints 0-9
    ) -> List[List[int]]:            # 2D grid prediction
        # your solving logic here
        pass
```

### Using vLLM

If `use_vllm=true` in your miner config, a vLLM server runs alongside your container on a shared network. Connect via:

```python
from openai import OpenAI

vllm_api_base = os.environ.get("VLLM_API_BASE", "http://vllm-container:8000")
client = OpenAI(base_url=f"{vllm_api_base}/v1", api_key="dummy")

response = client.chat.completions.create(
    model="your-model-name",  # discovered via client.models.list()
    messages=[...],
    temperature=0.1,
    max_tokens=2000
)
```

### Example Solver

See `miner-solution-example/` for a complete reference implementation with:
- HuggingFace model download in prep phase
- vLLM-based inference with fallback heuristics
- Proper error handling and logging

---

## Local Testing with Sandbox Runner

Test your solver locally before submitting to mainnet.

### 1. Set Up Sandbox Runner

```bash
cd sandbox_runner

# configure
cp config.yaml.example config.yaml
# edit config.yaml with your settings

# create .env
cat > .env << EOF
API_KEYS=test-key-123
GPU_COUNT=1
LOG_LEVEL=INFO
EOF

# start
make up
```

### 2. Generate Test Dataset

The sandbox runner generates daily datasets automatically, but you can trigger manually:

```bash
# inside sandbox_runner container or locally
python -c "
from synthetics.dataset_manager import DatasetManager
from pathlib import Path
import asyncio

dm = DatasetManager(Path('/app/data/datasets'))
asyncio.run(dm.generate_daily_dataset())
"
```

### 3. Submit Test Job

```bash
curl -X POST http://localhost:8000/v1/jobs/submit \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-key-123" \
  -d '{
    "repo_url": "https://github.com/your-username/your-arc-solver",
    "repo_branch": "main",
    "repo_path": "",
    "weight_class": "1xH200",
    "miner_hotkey": "test-miner",
    "validator_hotkey": "test-validator",
    "priority": 5,
    "use_vllm": true,
    "vllm_config": {
      "model": "unsloth/Meta-Llama-3.1-8B-Instruct",
      "dtype": "half",
      "gpu_memory_utilization": 0.8,
      "max_model_len": 12000
    }
  }'
```

Response:
```json
{
  "job_id": "job_abc123def456",
  "status": "pending",
  "queue_position": 0
}
```

### 4. Monitor Job

```bash
# check status
curl http://localhost:8000/v1/jobs/job_abc123def456 \
  -H "X-API-Key: test-key-123"

# get metrics (after completion)
curl http://localhost:8000/v1/jobs/job_abc123def456/metrics \
  -H "X-API-Key: test-key-123"

# stream logs
curl http://localhost:8000/v1/logs/job_abc123def456/tail?lines=100 \
  -H "X-API-Key: test-key-123"
```

### 5. Check Results

```json
{
  "job_id": "job_abc123def456",
  "status": "completed",
  "metrics": {
    "aggregate": {
      "total_problems": 100,
      "num_solved": 85,
      "num_exact_matches": 23,
      "exact_match_rate": 0.2706,
      "avg_partial_correctness": 0.4521,
      "avg_grid_similarity": 0.6234
    }
  }
}
```

### Local Development Workflow

```bash
# 1. make changes to your solver
vim your-solver/arc_solver_llm.py

# 2. commit and push
git add -A && git commit -m "improve pattern matching" && git push

# 3. submit new job to local sandbox
curl -X POST http://localhost:8000/v1/jobs/submit ...

# 4. check results
curl http://localhost:8000/v1/jobs/{job_id}/metrics ...

# 5. iterate until satisfied with exact_match_rate
```

---

## Configuration Reference

### Validator Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NETUID` | `5` | Subnet UID |
| `CHAIN_ENDPOINT` | mainnet | Bittensor chain endpoint |
| `WALLET_NAME` | `default` | Wallet name |
| `WALLET_HOTKEY` | `validator` | Hotkey name |
| `DB_URL` | - | PostgreSQL connection string |
| `SANDBOX_RUNNER_ENDPOINT` | - | Sandbox runner API URL |
| `SANDBOX_RUNNER_API_KEY` | - | API key for sandbox |
| `SANDBOX_RUNNER_TIMEOUT_HOURS` | `3` | Max job execution time |
| `SANDBOX_POLL_INTERVAL` | `30` | Seconds between status polls |
| `MAX_SUBMISSIONS_PER_DAY` | `1` | Submissions per miner per day |
| `MIN_ACCURACY_FLOOR` | `0.20` | Minimum exact_match_rate |
| `TOP_MINERS_COUNT` | `5` | Number of miners to reward |
| `CYCLE_DURATION` | `30` | Blocks per query cycle |

### Miner Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WALLET_NAME` | `default` | Wallet name |
| `WALLET_HOTKEY` | `miner` | Hotkey name |
| `MINER_PORT` | `8091` | HTTP server port |
| `MINER_REPO_URL` | - | Your solver repository URL |
| `MINER_REPO_BRANCH` | `main` | Git branch |
| `MINER_REPO_COMMIT` | - | Specific commit (optional) |
| `MINER_REPO_PATH` | - | Subdirectory in repo |
| `MINER_WEIGHT_CLASS` | `1xH200` | GPU requirement |
| `MINER_USE_VLLM` | `false` | Enable vLLM sidecar |

### Weight Classes

| Class | GPUs | Use Case |
|-------|------|----------|
| `1xH200` | 1 | Small models, heuristics |
| `2xH200` | 2 | Medium models |
| `4xH200` | 4 | Large models with tensor parallelism |
| `8xH200` | 8 | Very large models |

---

## Troubleshooting

### Validator Issues

**Cannot connect to database**
```bash
# check postgres is running
docker ps | grep db

# check logs
docker logs validator-db-1

# verify connection string
echo $DB_URL
```

**Sandbox runner unreachable**
```bash
# test connectivity
curl $SANDBOX_RUNNER_ENDPOINT/health

# check API key
curl -H "X-API-Key: $SANDBOX_RUNNER_API_KEY" $SANDBOX_RUNNER_ENDPOINT/v1/status
```

**Weights not setting**
- Ensure sufficient stake
- Check rate limiting (must wait between weight updates)
- Verify validator UID is registered

### Miner Issues

**Not discovered by validators**
```bash
# verify on-chain registration
btcli subnet list --netuid 5

# check IP is set correctly
btcli subnet metagraph --netuid 5 | grep YOUR_HOTKEY

# re-post IP if needed
python tools/post_ip_chain.py --wallet-name default --hotkey miner --ip YOUR_IP --port 8091
```

**/info endpoint not working**
```bash
# test locally
curl http://localhost:8091/info

# check logs
docker logs miner
```

**Jobs failing in sandbox**
- Check Dockerfile builds successfully locally
- Verify all required files exist
- Check prep phase has internet access
- Ensure inference phase doesn't require network

### Sandbox Runner Issues

**GPU allocation failures**
```bash
# check GPU status
curl http://localhost:8000/v1/status -H "X-API-Key: ..."

# verify nvidia-smi works
nvidia-smi
```

**Docker network conflicts**
```bash
# cleanup stale networks
docker network prune -f

# remove specific network
docker network rm sandbox-job-xyz
```

**vLLM not starting**
- Check GPU memory is sufficient
- Verify model exists in /app/models after prep phase
- Check vLLM logs: `curl .../v1/logs/{job_id}/tail?lines=100`

---

## Security Notes

- **Wallet files** are mounted read-only
- **Inference phase** runs with network disabled (`network_mode: none`)
- **Capabilities dropped**: `CAP_SYS_ADMIN`, `CAP_NET_ADMIN`, `CAP_SYS_MODULE`, `CAP_SYS_PTRACE`, `CAP_SYS_RAWIO`
- **No new privileges** flag enabled on containers
- **Validation dataset** (with expected outputs) never exposed to miners
- Never expose Adminer or sandbox runner APIs publicly without authentication

---

## Repository Structure

```
hone/
├── common/                 # shared utilities (chain, epistula, etc.)
├── miner/                  # miner HTTP server
├── miner-solution-example/ # reference solver implementation
├── sandbox_runner/         # GPU execution service
│   ├── api/                # REST API routes
│   ├── core/               # job queue, GPU pool, scheduler, executor
│   ├── execution/          # Docker execution logic
│   ├── synthetics/         # ARC problem generation
│   └── utils/              # metrics, validation, S3
├── validator/              # validator service
│   ├── autoupdate/         # auto-update scripts
│   └── sql/                # database schema
├── telemetry/              # optional telemetry service
└── tools/                  # CLI utilities
```

---

## License

See [LICENSE](LICENSE) file.

---

## Links

- [ARC-AGI-2 Dataset](https://arcprize.org/)
- [Bittensor Documentation](https://docs.bittensor.com/)
- [Subnet Registration Guide](https://docs.bittensor.com/subnets/)
