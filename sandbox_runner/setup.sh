#!/bin/bash

set -e  # Exit on error

echo "=========================================="
echo "Hone Subnet Sandbox Runner - Complete Setup"
echo "=========================================="
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' 

# Detect OS
OS="unknown"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
fi

echo -e "${BLUE}Detected OS: $OS${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Function to install packages based on OS
install_package() {
    local package=$1
    echo -e "${BLUE}Installing $package...${NC}"
    
    if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        sudo apt-get update -qq
        sudo apt-get install -y "$package"
    elif [ "$OS" = "centos" ] || [ "$OS" = "rhel" ] || [ "$OS" = "fedora" ]; then
        sudo yum install -y "$package"
    else
        echo -e "${RED}Unknown OS. Please install $package manually.${NC}"
        return 1
    fi
}

# =============================================================================
# 1. Check and Install Python 3.11+
# =============================================================================
echo "=========================================="
echo "1. Checking Python 3.11+"
echo "=========================================="

if command_exists python3; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION detected"
    else
        echo -e "${YELLOW}⚠${NC} Python $PYTHON_VERSION is too old, need 3.11+"
        echo "Installing Python 3.11..."
        
        if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
            sudo apt-get update
            sudo apt-get install -y software-properties-common
            sudo add-apt-repository -y ppa:deadsnakes/ppa
            sudo apt-get update
            sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
            # Create symlink
            sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
        else
            echo -e "${RED}Please install Python 3.11+ manually${NC}"
            exit 1
        fi
    fi
else
    echo -e "${RED}✗${NC} Python3 not found. Installing..."
    install_package python3
    install_package python3-venv
    install_package python3-pip
fi

# Install pip if missing
if ! command_exists pip3; then
    echo "Installing pip..."
    install_package python3-pip
fi

echo ""

# =============================================================================
# 2. Check NVIDIA GPUs and Drivers
# =============================================================================
echo "=========================================="
echo "2. Checking NVIDIA GPUs"
echo "=========================================="

if command_exists nvidia-smi; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    echo -e "${GREEN}✓${NC} Found $GPU_COUNT NVIDIA GPU(s)"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}⚠${NC} nvidia-smi not found."
    echo "GPU support will not be available without NVIDIA drivers."
    echo "To install NVIDIA drivers, visit: https://www.nvidia.com/Download/index.aspx"
fi

echo ""

# =============================================================================
# 3. Check and Install Docker
# =============================================================================
echo "=========================================="
echo "3. Checking Docker"
echo "=========================================="

if command_exists docker; then
    DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
    echo -e "${GREEN}✓${NC} Docker $DOCKER_VERSION detected"
else
    echo -e "${YELLOW}⚠${NC} Docker not found. Installing..."
    
    if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        # Install Docker
        sudo apt-get update
        sudo apt-get install -y ca-certificates curl gnupg lsb-release
        
        # Add Docker's official GPG key
        sudo mkdir -p /etc/apt/keyrings
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        
        # Set up repository
        echo \
          "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
          $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        
        # Install Docker Engine
        sudo apt-get update
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
        
        # Start Docker
        sudo systemctl start docker
        sudo systemctl enable docker
        
        echo -e "${GREEN}✓${NC} Docker installed"
    else
        echo -e "${YELLOW}Please install Docker manually: https://docs.docker.com/engine/install/${NC}"
    fi
fi

# Add current user to docker group
if command_exists docker; then
    if ! groups | grep -q docker; then
        echo "Adding user to docker group..."
        sudo usermod -aG docker $USER || true
        echo -e "${YELLOW}Note: You may need to log out and back in for docker group changes to take effect${NC}"
    fi
fi

echo ""

# =============================================================================
# 4. Check and Install nvidia-docker2
# =============================================================================
echo "=========================================="
echo "4. Checking nvidia-docker2"
echo "=========================================="

if command_exists docker && command_exists nvidia-smi; then
    if docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi &>/dev/null; then
        echo -e "${GREEN}✓${NC} nvidia-docker2 is working"
    else
        echo -e "${YELLOW}⚠${NC} nvidia-docker2 not configured. Installing..."
        
        if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
            distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
            
            sudo apt-get update
            sudo apt-get install -y nvidia-docker2
            sudo systemctl restart docker
            
            echo -e "${GREEN}✓${NC} nvidia-docker2 installed"
        fi
    fi
else
    echo -e "${YELLOW}⚠${NC} Skipping nvidia-docker2 (Docker or NVIDIA drivers not available)"
fi

echo ""

# =============================================================================
# 5. Check and Install gVisor
# =============================================================================
echo "=========================================="
echo "5. Checking gVisor"
echo "=========================================="

if command_exists runsc; then
    echo -e "${GREEN}✓${NC} gVisor (runsc) detected"
    runsc --version
else
    echo -e "${YELLOW}⚠${NC} gVisor not found. Installing..."
    
    # Install gVisor
    (
        set -e
        ARCH=$(uname -m)
        URL=https://storage.googleapis.com/gvisor/releases/release/latest/${ARCH}
        wget ${URL}/runsc ${URL}/runsc.sha512 ${URL}/containerd-shim-runsc-v1 ${URL}/containerd-shim-runsc-v1.sha512 -P /tmp
        sha512sum -c /tmp/runsc.sha512
        sha512sum -c /tmp/containerd-shim-runsc-v1.sha512
        sudo mv /tmp/runsc /tmp/containerd-shim-runsc-v1 /usr/local/bin
        sudo chmod a+rx /usr/local/bin/runsc /usr/local/bin/containerd-shim-runsc-v1
        rm -f /tmp/runsc.sha512 /tmp/containerd-shim-runsc-v1.sha512
        
        echo -e "${GREEN}✓${NC} gVisor installed"
    ) || echo -e "${YELLOW}gVisor installation failed. Continuing...${NC}"
    
    # Configure Docker to use gVisor
    if command_exists docker && command_exists runsc; then
        echo "Configuring Docker to use gVisor..."
        sudo mkdir -p /etc/docker
        cat <<EOF | sudo tee /etc/docker/daemon.json > /dev/null
{
    "runtimes": {
        "runsc": {
            "path": "/usr/local/bin/runsc"
        }
    }
}
EOF
        sudo systemctl restart docker || true
        echo -e "${GREEN}✓${NC} Docker configured for gVisor"
    fi
fi

echo ""

# =============================================================================
# 6. Install System Dependencies
# =============================================================================
echo "=========================================="
echo "6. Installing System Dependencies"
echo "=========================================="

SYSTEM_DEPS="git curl wget build-essential libssl-dev openssl"

if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
    echo "Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y $SYSTEM_DEPS
    echo -e "${GREEN}✓${NC} System dependencies installed"
fi

echo ""

# =============================================================================
# 7. Create Virtual Environment
# =============================================================================
echo "=========================================="
echo "7. Setting up Python Virtual Environment"
echo "=========================================="

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
else
    echo -e "${YELLOW}⚠${NC} Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo ""

# =============================================================================
# 8. Install Python Dependencies
# =============================================================================
echo "=========================================="
echo "8. Installing Python Dependencies"
echo "=========================================="

if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip install -r requirements.txt
    echo -e "${GREEN}✓${NC} Python dependencies installed"
else
    echo -e "${RED}✗${NC} requirements.txt not found"
    exit 1
fi

echo ""

# =============================================================================
# 9. Create Directories
# =============================================================================
echo "=========================================="
echo "9. Creating Directories"
echo "=========================================="

mkdir -p logs
mkdir -p data/inputs
mkdir -p data/outputs
mkdir -p ssl

echo -e "${GREEN}✓${NC} Directories created"
echo ""

# =============================================================================
# 10. Generate SSL Certificates
# =============================================================================
echo "=========================================="
echo "10. Generating SSL Certificates"
echo "=========================================="

SSL_DIR="ssl"

if [ ! -f "$SSL_DIR/runner.crt" ]; then
    echo "Generating self-signed SSL certificates..."
    openssl req -x509 -newkey rsa:4096 -nodes \
        -keyout $SSL_DIR/runner.key \
        -out $SSL_DIR/runner.crt \
        -days 365 \
        -subj "/C=US/ST=State/L=City/O=HoneSubnet/CN=localhost" \
        2>/dev/null
    
    chmod 600 $SSL_DIR/runner.key
    chmod 644 $SSL_DIR/runner.crt
    
    echo -e "${GREEN}✓${NC} SSL certificates generated in $SSL_DIR/"
    echo -e "${YELLOW}⚠${NC} These are self-signed certificates for testing only!"
else
    echo -e "${YELLOW}⚠${NC} SSL certificates already exist"
fi

echo ""

# =============================================================================
# 11. Create Configuration
# =============================================================================
echo "=========================================="
echo "11. Setting up Configuration"
echo "=========================================="

if [ ! -f "config.local.yaml" ]; then
    echo "Creating local configuration..."
    cp config.yaml config.local.yaml
    
    # Update SSL paths in local config
    FULL_PATH=$(pwd)
    sed -i "s|/etc/ssl/certs/runner.crt|$FULL_PATH/$SSL_DIR/runner.crt|g" config.local.yaml
    sed -i "s|/etc/ssl/private/runner.key|$FULL_PATH/$SSL_DIR/runner.key|g" config.local.yaml
    
    # Update GPU count if we can detect it
    if command_exists nvidia-smi; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        sed -i "s|gpu_count: 8|gpu_count: $GPU_COUNT|g" config.local.yaml
    fi
    
    echo -e "${GREEN}✓${NC} Configuration file created: config.local.yaml"
else
    echo -e "${YELLOW}⚠${NC} config.local.yaml already exists"
fi

echo ""

# =============================================================================
# 12. Check cgroups v2
# =============================================================================
echo "=========================================="
echo "12. Checking cgroups v2"
echo "=========================================="

if [ -f "/sys/fs/cgroup/cgroup.controllers" ]; then
    echo -e "${GREEN}✓${NC} cgroups v2 is available"
else
    echo -e "${YELLOW}⚠${NC} cgroups v2 not available"
    echo "Direct execution mode may not work properly"
    echo "To enable cgroups v2, add 'systemd.unified_cgroup_hierarchy=1' to kernel parameters"
fi

echo ""

# =============================================================================
# Summary
# =============================================================================
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""

# Print status summary
echo "Component Status:"
echo "=================="

# Python
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Python 3.11+"
else
    echo -e "${RED}✗${NC} Python 3.11+"
fi

# NVIDIA GPUs
if command_exists nvidia-smi; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    echo -e "${GREEN}✓${NC} NVIDIA GPUs ($GPU_COUNT detected)"
else
    echo -e "${YELLOW}⚠${NC} NVIDIA GPUs (not available)"
fi

# Docker
if command_exists docker; then
    echo -e "${GREEN}✓${NC} Docker"
else
    echo -e "${YELLOW}⚠${NC} Docker"
fi

# nvidia-docker2
if command_exists docker && docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi &>/dev/null; then
    echo -e "${GREEN}✓${NC} nvidia-docker2"
else
    echo -e "${YELLOW}⚠${NC} nvidia-docker2"
fi

# gVisor
if command_exists runsc; then
    echo -e "${GREEN}✓${NC} gVisor"
else
    echo -e "${YELLOW}⚠${NC} gVisor"
fi

# cgroups v2
if [ -f "/sys/fs/cgroup/cgroup.controllers" ]; then
    echo -e "${GREEN}✓${NC} cgroups v2"
else
    echo -e "${YELLOW}⚠${NC} cgroups v2"
fi