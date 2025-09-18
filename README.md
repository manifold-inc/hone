# Hone Subnet Quick Setup

## Prerequisites
- Python 3.10+
- Docker & Docker Compose  
- Bittensor wallet with TAO

## Initial Setup

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

# Edit the .env files with your wallet names:
# validator/.env - Set WALLET_NAME and WALLET_HOTKEY
# miner/.env - Set WALLET_NAME and WALLET_HOTKEY
```

## Miner

### 1. Register and Set IP

```bash
# Register on subnet
btcli subnet register --netuid 5 --wallet.name default --wallet.hotkey miner

# Set IP on chain
pip install substrate-interface netaddr
python set_miner_ip.py --wallet-name default --hotkey miner --ip YOUR_PUBLIC_IP --port 8091
```

### 2. Run

```bash
# Docker
docker build -t hone-miner miner/
docker run -d --name miner -p 8091:8091 \
  -v ~/.bittensor/wallets:/root/.bittensor/wallets:ro \
  --env-file miner/.env \
  hone-miner

# OR Python
pip install -r miner/requirements.txt
source miner/.env && python -m miner.main
```

## Validator

### 1. Register and Stake

```bash
# Register
btcli subnet register --netuid 5 --wallet.name default --wallet.hotkey validator

# Stake (optional but recommended)
btcli stake add --wallet.name default --wallet.hotkey validator --amount 100
```

### 2. Run

```bash
# Start validator with database
make val

# OR manually:
cd validator && docker-compose up -d

# Stop everything
make down
```

## Environment Files

**validator/.env.example**
```env
NETUID=5
CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
WALLET_NAME=default
WALLET_HOTKEY=validator
CYCLE_DURATION=30
DB_URL=postgresql://postgres:postgres@db:5432/hone
```

**miner/.env.example**
```env
WALLET_NAME=default
WALLET_HOTKEY=miner
MINER_PORT=8091
```

## Verify

```bash
# Check miner logs
docker logs miner

# Check validator logs  
cd validator && docker-compose logs -f validator

# Check chain registration
btcli subnet metagraph --netuid 5
```