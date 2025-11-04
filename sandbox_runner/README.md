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
# Clone repository
git clone <repository-url>
cd sandbox_runner

# Run setup script
bash setup.sh

# Or manual setup:
make setup-all
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
# Activate virtual environment
source venv/bin/activate

# Run with default config
python3 main.py

# Run with custom config
python3 main.py --config config.local.yaml --log-level DEBUG
```

#### Docker
```bash
# Build image
make docker-build

# Run container
make docker-run

# Or use docker-compose
docker-compose up -d
```

#### System Service
```bash
# Install as systemd service
sudo make install-service

# Start service
sudo systemctl start sandbox-runner

# Check status
sudo systemctl status sandbox-runner

# View logs
sudo journalctl -u sandbox-runner -f
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

## Development

### Running Tests

```bash
# All tests
make test

# Integration tests only
make test-integration

# With coverage
pytest --cov=. --cov-report=html
```

### Code Quality

```bash
# Linting
make lint

# Formatting
make format

# Type checking
mypy .
```

### Debug Mode

```bash
# Run with debug logging
python3 main.py --log-level DEBUG

# Interactive Docker shell
make docker-run-dev
```

## Security Considerations

### Production Deployment

1. **SSL Certificates**: Replace self-signed certs with proper CA-signed certificates
2. **API Keys**: Use secure random keys, rotate regularly
3. **Epistula**: Configure validator public keys properly
4. **Network**: Use firewall rules to restrict access
5. **Monitoring**: Enable Falco for runtime security monitoring

### Attack Surface Reduction

- Containers run as `nobody` user (UID 65534)
- Capabilities dropped (CAP_SYS_ADMIN, CAP_NET_ADMIN, etc.)
- Read-only root filesystem where possible
- Seccomp profiles limit available syscalls
- Network completely blocked during inference

## Troubleshooting

### Docker Not Available

```bash
# Check Docker status
sudo systemctl status docker

# Add user to docker group
sudo usermod -aG docker $USER

# Restart Docker
sudo systemctl restart docker
```

### gVisor Not Working

```bash
# Check runsc
runsc --version

# Verify Docker runtime
docker info | grep -i runtime

# Test gVisor container
docker run --runtime=runsc hello-world
```

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --gpus all nvidia/cuda:12.0-base nvidia-smi

# Install nvidia-docker2
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### cgroups Issues

```bash
# Check cgroups version
mount | grep cgroup

# Enable cgroups v2 (if needed)
sudo grub2-editconfig
# Add: systemd.unified_cgroup_hierarchy=1

# Reboot
sudo reboot
```

## Performance Tuning

### GPU Scheduling
- Adjust `starvation_threshold_seconds` in scheduler for large job fairness
- Use priority levels (0-10) to prioritize important jobs
- Monitor queue depth per weight class

### Resource Limits
- Tune `cpu_limit` and `memory_limit_gb` per job requirements
- Adjust `max_processes` to prevent fork bombs
- Set appropriate timeouts for prep and inference phases

### Network Performance
- For prep phase, ensure high bandwidth for model downloads
- Consider caching commonly used models
- Use S3 transfer acceleration for faster uploads/downloads
