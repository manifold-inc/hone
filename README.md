# Hone Subnet - ARC AGI Problem Solving

A Bittensor subnet where miners compete to solve ARC AGI 2 (3 coming soon) problems and validators score them based novel synthetic ARC AGI 2 data.

## Architecture
- **Validators**: Generate synthetic ARC AGI 2 problems, query miners continuously, and set weights based on solving performance
- **Miners**: Provide API endpoints to solve ARC problems, currently uses basic strategies (replace with your custom solution)

## Prerequisites
- Python 3.10+
- Docker & Docker Compose  
- Bittensor wallet with TAO (for mainnet/testnet)

## Local Testing (No Bittensor Required)

### Quick Test Setup
```bash
# Run complete local test environment
chmod +x test_local.sh
./test_local.sh

# This will:
# - Start PostgreSQL database
# - Launch 3 mock miners
# - Run validator with mock chain
# - Show real-time scoring and weight setting
```

### Manual Test Setup
```bash
# Start test environment
cd validator
docker-compose -f docker-compose.test.yml up

# Check logs
docker logs validator-validator-1  # See validator activity
docker logs validator-miner1-1      # See miner solving problems

# Stop test environment
docker-compose -f docker-compose.test.yml down -v
```

## Production Setup

### Initial Setup
```bash
# Create wallets
btcli wallet new_coldkey --wallet.name default
btcli wallet new_hotkey --wallet.name default --wallet.hotkey miner
btcli wallet new_hotkey --wallet.name default --wallet.hotkey validator

# Setup environment files
make setup
# OR manually:
cp validator/.env.example validator/.env
cp miner/.env.example miner/.env

# Edit the .env files with your wallet names
```

### Miner Deployment

#### 1. Register and Set IP
```bash
# Register on subnet
btcli subnet register --netuid 5 --wallet.name default --wallet.hotkey miner

# Set IP on chain
python tools/post_ip_chain.py --wallet-name default --hotkey miner --ip YOUR_PUBLIC_IP --port 8091
```

#### 2. Run Miner
```bash
# Docker (recommended)
docker build -t hone-miner miner/
docker run -d --name miner -p 8091:8091 \
  -v ~/.bittensor/wallets:/root/.bittensor/wallets:ro \
  --env-file miner/.env \
  hone-miner

# OR Python
pip install -r miner/requirements.txt
source miner/.env && python -m miner.main
```

#### 3. Implement Your Solver
Replace the basic `ARCSolver` in `miner/handlers.py` with your neural network or advanced solving approach.

### Validator Deployment

#### 1. Register and Stake
```bash
# Register
btcli subnet register --netuid 5 --wallet.name default --wallet.hotkey validator

# Stake
btcli stake add --wallet.name default --wallet.hotkey validator --amount 100
```

#### 2. Run Validator
```bash
# Start validator with database
make val

# OR manually:
cd validator && docker-compose up -d

# Monitor logs
docker-compose logs -f validator

# Stop
make down
```

## Configuration

### Validator Environment Variables
```env
# Network
NETUID=5
CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443

# Wallet
WALLET_NAME=default
WALLET_HOTKEY=validator

# Cycles (in blocks, 1 block ≈ 12 seconds)
CYCLE_DURATION=30  # Query miners for 30 blocks (~6 minutes)

# Database
DB_URL=postgresql://postgres:postgres@db:5432/hone

# Testing
USE_MOCK_CHAIN=false  # Set to true for local testing
```

### Miner Environment Variables
```env
WALLET_NAME=default
WALLET_HOTKEY=miner
MINER_PORT=8091

# Testing
SKIP_EPISTULA_VERIFY=false  # Set to true for local testing
```

## How It Works

### Validator Cycle
1. **Discovery Phase**: Find all registered miners on the subnet
2. **Query Phase**: For `CYCLE_DURATION` blocks:
   - Generate synthetic ARC AGI 2 problems
   - Send problems to all miners
   - Evaluate responses with metrics:
     - Exact match (40% weight)
     - Partial correctness (30% weight) 
     - Grid similarity (20% weight)
     - Response time efficiency (10% weight)
3. **Scoring Phase**: Calculate aggregate scores over last 4 cycles
4. **Weight Setting**: Update on-chain weights based on performance
5. **Repeat**: Start next cycle

### Miner Response Flow
1. Receive ARC problem (input grid)
2. Apply solving strategies to find output
3. Return predicted output grid
4. Get scored by validator

## Scoring Metrics

- **Exact Match**: Output perfectly matches expected solution
- **Partial Correctness**: Considers shape, color distribution, and pattern similarity
- **Grid Similarity**: Pixel-wise comparison of output vs expected
- **Efficiency**: Response time relative to timeout (30 seconds max)

## Database Schema

The validator maintains a PostgreSQL database tracking:
- Miner registrations and IPs
- Query results with all metrics
- Historical scores for weight calculation

## API Endpoints

### Miner Endpoints
- `POST /query`: Receive and solve ARC problems
- `GET /health`: Health check

### Message Protocol
Uses Epistula for secure validator-miner communication with signature verification.

## Development

### Project Structure
```
hone/
├── common/           # Shared utilities
│   ├── chain.py     # Bittensor chain interface
│   ├── epistula.py  # Message signing/verification
│   └── mock_chain.py # Testing without Bittensor
├── miner/
│   ├── handlers.py  # ARC solver implementation
│   ├── query.py     # Query endpoint
│   └── main.py      # FastAPI app
├── validator/
│   ├── synthetics/  # ARC problem generation
│   ├── cycle.py     # Main validation loop
│   ├── query.py     # Miner querying
│   ├── scoring.py   # Performance scoring
│   └── db.py        # Database operations
└── test_local.sh    # Local testing script
```

## Troubleshooting

### Logs
```bash
# Validator logs
docker-compose logs -f validator

# Miner logs
docker logs -f miner

# Database queries
docker exec -it validator-db-1 psql -U postgres -d hone
```


## License