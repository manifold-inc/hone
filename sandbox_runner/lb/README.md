# Sandbox Runner Load Balancer

Routes validator requests to multiple sandbox runners with intelligent caching and sticky routing.

## Features

- **Solution Caching**: Caches completed job metrics by `(repo_url, branch, commit, path, weight_class)`
  - Automatically resolves latest commit from GitHub if not provided
  - Dual-layer storage: in-memory for speed, SQLite for persistence
  - Configurable TTL (default: 7 days)

- **Sticky Routing**: Routes all requests for a job_id to the same sandbox runner

- **Load Balancing**: Routes new jobs to the healthiest runner with lowest queue depth

- **Health Checks**: Periodic health checks with automatic failover

- **API Key Authentication**: Simple API key validation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python load_balancer.py \
    --runners https://runner1.example.com https://runner2.example.com \
    --port 8080
```

### With PM2

```bash
pm2 start load_balancer.py --interpreter python3 --name sandbox-lb -- \
    --runners https://runner1.example.com https://runner2.example.com \
    --port 8080
```

### Full Options

```bash
python load_balancer.py \
    --runners https://runner1.com https://runner2.com \
    --port 8080 \
    --cache-dir ./lb_cache \
    --cache-ttl-days 7 \
    --health-check-interval 30 \
    --log-level INFO
```

## API Endpoints

### Health (No Auth)

```
GET /health
GET /v1/health
```

### Jobs (Requires X-API-Key header)

```
POST /v1/jobs/submit     - Submit a new job
GET  /v1/jobs/{job_id}   - Get job status
GET  /v1/jobs/{job_id}/metrics - Get job metrics
DELETE /v1/jobs/{job_id} - Cancel a job
```

### Status (Requires X-API-Key header)

```
GET /v1/status           - Full load balancer status
GET /v1/stats            - Detailed statistics
GET /v1/cache/entries    - List cached entries
```

## API Key Management

API keys are stored in `{cache-dir}/api_keys.json`.

On first run, a default key `lb-dev-key-12345` is created.

To add/modify keys, edit the JSON file:

```json
{
  "keys": [
    "lb-dev-key-12345",
    "your-production-key"
  ]
}
```

## Cache Behavior

1. **On job submit**: 
   - Resolve commit hash if not provided (via `git ls-remote`)
   - Check cache for `(repo_url, branch, commit, path, weight_class)`
   - If cache hit: return cached metrics immediately with `from_cache: true`
   - If cache miss: forward to healthiest runner

2. **On metrics request**:
   - If job completed successfully: cache metrics for future requests
   - Return metrics to validator

3. **Cache cleanup**:
   - Expired entries cleaned up hourly
   - Entries persist across restarts via SQLite

## Architecture

```
Validators                Load Balancer              Sandbox Runners
    │                          │                          │
    │ POST /v1/jobs/submit     │                          │
    ├─────────────────────────►│                          │
    │                          │ 1. Resolve commit        │
    │                          │ 2. Check cache           │
    │                          │ 3. Route to runner ─────►│
    │                          │                          │
    │ GET /v1/jobs/{id}        │ (sticky routing)         │
    ├─────────────────────────►├─────────────────────────►│
    │                          │                          │
    │ GET /v1/jobs/{id}/metrics│                          │
    ├─────────────────────────►├─────────────────────────►│
    │                          │ 4. Cache result          │
    │◄─────────────────────────┤◄─────────────────────────│
```

## Validator Integration

Update your validator's sandbox client to point to the load balancer:

```python
# Before
SANDBOX_RUNNER_ENDPOINT=https://runner1.example.com

# After
SANDBOX_RUNNER_ENDPOINT=http://localhost:8080
SANDBOX_RUNNER_API_KEY=lb-dev-key-12345
```

The load balancer API is fully compatible with the sandbox runner API.

## Monitoring

### Check Status

```bash
curl -H "X-API-Key: lb-dev-key-12345" http://localhost:8080/v1/status
```

### View Cache Stats

```bash
curl -H "X-API-Key: lb-dev-key-12345" http://localhost:8080/v1/stats
```

### List Cache Entries

```bash
curl -H "X-API-Key: lb-dev-key-12345" http://localhost:8080/v1/cache/entries
```

## Files

```
lb_cache/
├── cache.db       # SQLite cache database
└── api_keys.json  # API keys configuration
```

## Environment Variables

None required - all configuration via CLI arguments.

## Troubleshooting

### All runners unavailable (503)

- Check that runner URLs are correct
- Check runner health: `curl https://runner1.example.com/v1/health`
- Check load balancer health endpoint for runner status

### Cache not working

- Ensure `cache-dir` is writable
- Check `cache.db` exists and has entries
- Verify commit resolution is working (check logs for `git ls-remote` output)

### Job not found (404)

- Job may have been cleaned up after completion
- Check if job_id starts with `cache_` (indicates cache hit)