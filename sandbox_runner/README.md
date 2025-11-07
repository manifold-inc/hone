# Hone Subnet Sandbox Runner

Secure GPU execution service for Bittensor subnet miners with multi-layer security isolation.

## Features

### Core Capabilities
- **Multi-Mode Execution**: Docker + gVisor (most secure) → Docker only → Direct execution (fallback chain)
- **GPU Management**: Smart allocation and scheduling across 8x H200 GPUs
- **Two-Phase Execution**: 
  - Prep phase: Internet enabled for model downloads (2hr timeout)
  - Inference phase: Network blocked for security (1hr timeout)
- **Intelligent Scheduling**: Optimize parallel GPU utilization (e.g., run 4x 1-GPU jobs simultaneously)

### Security Layers
1. **Container Isolation**: Docker with optional gVisor user-space kernel
2. **Resource Limits**: cgroups v2 for CPU, memory, PID control
3. **Network Control**: Phase-based internet access control
4. **Syscall Filtering**: Seccomp profiles to limit attack surface
5. **Process Isolation**: Linux namespaces (PID, network, mount)

### API & Authentication
- **HTTPS-only** RESTful API on port 8443
- **Dual Authentication**: Epistula signatures + API keys
- **Rate Limiting**: Per-validator request throttling
- **Prometheus Metrics**: Real-time monitoring on port 9090

## Quick Start

### Prerequisites
- Ubuntu 24.04 LTS (or compatible)
- Python 3.11+
- NVIDIA GPUs with drivers installed
- Docker (optional but recommended)
- gVisor (optional for maximum security)

### Installation

```bash
git clone <repository-url>
cd sandbox_runner

bash setup.sh

```

### Configuration

Edit `config.yaml` to customize:

```yaml
runner:
  id: "runner-1"
  name: "My Sandbox Runner"
  location: "us-east-1"

hardware:
  gpu_count: 8
  gpu_type: "H200"
  cpu_cores: 96
  memory_gb: 1024

execution:
  mode: "docker+gvisor"  # or "docker" or "direct"
  fallback_on_error: true
  cpu_limit: 32
  memory_limit_gb: 256
  prep_timeout_seconds: 7200
  inference_timeout_seconds: 3600

security:
  gvisor:
    enabled: true
    platform: "ptrace"  # or "kvm"
  
  network_policy:
    prep_allow_internet: true
    inference_block_internet: true
    allowed_prep_domains:
      - "huggingface.co"
      - "github.com"

storage:
  s3_bucket: "hone-subnet-data"
  s3_region: "us-east-1"
```

### Running

#### Local Development
```bash
source venv/bin/activate
python3 main.py
python3 main.py --config config.local.yaml --log-level DEBUG
```


## Architecture

### Components

```
sandbox_runner/
├── api/                    # FastAPI gateway and routes
│   ├── gateway.py         # Main FastAPI application
│   ├── routes.py          # API endpoints
│   └── auth.py            # Authentication (Epistula + API keys)
│
├── core/                   # Core orchestration
│   ├── meta_manager.py    # Central coordinator
│   ├── gpu_pool.py        # GPU allocation manager
│   ├── job_queue.py       # Priority-based queue
│   ├── scheduler.py       # Intelligent job scheduler
│   └── executor.py        # Job execution orchestrator
│
├── execution/             # Execution modes
│   ├── docker_gvisor.py  # Docker + gVisor (most secure)
│   ├── docker_only.py    # Docker only (fallback)
│   └── direct.py         # Direct execution (last resort)
│
├── security/              # Security layers
│   ├── cgroups.py        # cgroups v2 resource management
│   ├── isolation.py      # Linux namespace isolation
│   ├── network.py        # Network policy enforcement
│   └── seccomp.py        # Syscall filtering
│
├── monitoring/            # Observability
│   ├── metrics.py        # Prometheus metrics
│   └── logging.py        # Structured logging
│
├── utils/                 # Utilities
│   ├── s3.py             # S3 storage operations
│   └── validation.py     # Input validation
│
├── config.py              # Configuration management
└── main.py               # Application entry point
```

### Job Lifecycle

```
PENDING → CLONING → BUILDING → PREP → INFERENCE → COMPLETED
                                ↓         ↓
                              FAILED   TIMEOUT
```

1. **PENDING**: Job queued, waiting for GPU allocation
2. **CLONING**: Git repository clone
3. **BUILDING**: Docker image build (or dependency installation)
4. **PREP**: Execute with internet access for downloads
5. **INFERENCE**: Execute without internet for security
6. **COMPLETED**: Results uploaded to S3

### Execution Modes

#### 1. Docker + gVisor (Most Secure)
- User-space kernel interception
- System call filtering
- Better isolation than standard containers

#### 2. Docker Only (Fallback)
- Standard Docker isolation
- Resource limits via cgroups
- Network namespace control

#### 3. Direct Execution (Last Resort)
- OS-level isolation only
- cgroups for resource limits
- Linux namespaces for isolation

## API Usage

### Submit Job

```bash
curl -X POST https://localhost:8443/v1/jobs/submit \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -H "X-Epistula-Signature: signature" \
  -H "X-Epistula-Request-ID: uuid" \
  -H "X-Epistula-Timestamp: 2025-01-01T00:00:00Z" \
  -d '{
    "repo_url": "https://github.com/user/miner-repo",
    "repo_branch": "main",
    "weight_class": "1xH200",
    "input_data_s3_path": "s3://bucket/inputs/data.json",
    "output_data_s3_path": "s3://bucket/outputs/results.json",
    "miner_hotkey": "5Abc...xyz",
    "priority": 5
  }'
```

### Get Job Status

```bash
curl https://localhost:8443/v1/jobs/{job_id} \
  -H "X-API-Key: your-api-key"
```

### Get Runner Status

```bash
curl https://localhost:8443/v1/status \
  -H "X-API-Key: your-api-key"
```

## Monitoring

### Prometheus Metrics

Available at `http://localhost:9090/metrics`:

- `sandbox_jobs_submitted_total` - Total jobs submitted
- `sandbox_jobs_completed_total` - Total jobs completed (by status)
- `sandbox_jobs_active` - Currently active jobs
- `sandbox_gpu_utilization_percent` - GPU utilization per device
- `sandbox_gpu_memory_used_bytes` - GPU memory usage
- `sandbox_queue_depth` - Jobs in queue by weight class
- `sandbox_api_request_duration_seconds` - API latency

### Health Check

```bash
curl https://localhost:8443/health
```