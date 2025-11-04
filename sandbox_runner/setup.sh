#!/bin/bash
# Hone Subnet Sandbox Runner - Setup Script
# This script helps you quickly set up the sandbox runner environment

set -e  # Exit on error

echo "=========================================="
echo "Hone Subnet Sandbox Runner Setup"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.11"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION detected"
else
    echo -e "${RED}✗${NC} Python 3.11+ required, found $PYTHON_VERSION"
    exit 1
fi

# Check for NVIDIA GPUs
echo ""
echo "Checking for NVIDIA GPUs..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo -e "${GREEN}✓${NC} Found $GPU_COUNT NVIDIA GPU(s)"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}⚠${NC} nvidia-smi not found. GPU support may not be available."
fi

# Check for Docker
echo ""
echo "Checking for Docker..."
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
    echo -e "${GREEN}✓${NC} Docker $DOCKER_VERSION detected"
else
    echo -e "${YELLOW}⚠${NC} Docker not found. Docker execution mode will not be available."
    echo "Install Docker: https://docs.docker.com/engine/install/"
fi

# Check for gVisor
echo ""
echo "Checking for gVisor..."
if command -v runsc &> /dev/null; then
    echo -e "${GREEN}✓${NC} gVisor (runsc) detected"
else
    echo -e "${YELLOW}⚠${NC} gVisor not found. Docker+gVisor mode will not be available."
    echo "Install gVisor: https://gvisor.dev/docs/user_guide/install/"
fi

# Create virtual environment
echo ""
echo "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
else
    echo -e "${YELLOW}⚠${NC} Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}✓${NC} Dependencies installed"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p logs
mkdir -p data/inputs
mkdir -p data/outputs
echo -e "${GREEN}✓${NC} Directories created"

# Generate SSL certificates for testing
echo ""
echo "Generating SSL certificates for testing..."
SSL_DIR="ssl"
mkdir -p $SSL_DIR

if [ ! -f "$SSL_DIR/runner.crt" ]; then
    openssl req -x509 -newkey rsa:4096 -nodes \
        -keyout $SSL_DIR/runner.key \
        -out $SSL_DIR/runner.crt \
        -days 365 \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost" \
        2>/dev/null
    echo -e "${GREEN}✓${NC} SSL certificates generated in $SSL_DIR/"
    echo -e "${YELLOW}⚠${NC} These are self-signed certificates for testing only!"
else
    echo -e "${YELLOW}⚠${NC} SSL certificates already exist"
fi

# Create local config if it doesn't exist
echo ""
echo "Setting up configuration..."
if [ ! -f "config.local.yaml" ]; then
    cp config.yaml config.local.yaml
    
    # Update SSL paths in local config
    sed -i "s|/etc/ssl/certs/runner.crt|$PWD/$SSL_DIR/runner.crt|g" config.local.yaml
    sed -i "s|/etc/ssl/private/runner.key|$PWD/$SSL_DIR/runner.key|g" config.local.yaml
    
    echo -e "${GREEN}✓${NC} Configuration file created: config.local.yaml"
    echo "Please edit config.local.yaml to customize your settings"
else
    echo -e "${YELLOW}⚠${NC} config.local.yaml already exists"
fi

# Run tests
echo ""
echo "Running Phase 1 tests..."
if python test_phase1.py -v; then
    echo -e "${GREEN}✓${NC} All tests passed!"
else
    echo -e "${RED}✗${NC} Some tests failed. Please check the output above."
fi

# Print summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit config.local.yaml with your settings"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Start the server: python main.py --config config.local.yaml"
echo ""
echo "For testing without authentication:"
echo "  python main.py --config config.local.yaml"
echo ""
echo "The API will be available at: https://localhost:8443"
echo "Metrics endpoint: http://localhost:9090/metrics"
echo ""
echo -e "${YELLOW}Note:${NC} Self-signed certificates will trigger browser warnings."
echo "For production, obtain proper SSL certificates."
echo ""