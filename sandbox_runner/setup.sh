set -e

export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
export NEEDRESTART_SUSPEND=1

echo "=========================================="
echo "Hone Subnet Sandbox Runner"
echo "COMPLETE AUTO-INSTALL"
echo "=========================================="
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}ERROR: Must run as root${NC}"
    echo "Run: sudo ./setup.sh"
    exit 1
fi

OS="unknown"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
fi

echo -e "${BLUE}OS: $OS${NC}"
echo ""

command_exists() {
    command -v "$1" &> /dev/null
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Project directory: $SCRIPT_DIR"
echo ""

# =============================================================================
# 1. Python 3.11+
# =============================================================================
echo "=========================================="
echo "1. Python 3.11+"
echo "=========================================="

PYTHON_OK=false
if command_exists python3; then
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION"
        PYTHON_OK=true
    fi
fi

if [ "$PYTHON_OK" = false ]; then
    echo "Installing Python 3.11..."
    apt-get update -qq
    apt-get install -y -qq software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa >/dev/null 2>&1
    apt-get update -qq
    apt-get install -y -qq python3.11 python3.11-venv python3.11-dev python3-pip
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
    echo -e "${GREEN}✓${NC} Python 3.11 installed"
fi

echo ""

# =============================================================================
# 2. NVIDIA GPUs
# =============================================================================
echo "=========================================="
echo "2. NVIDIA GPUs"
echo "=========================================="

if command_exists nvidia-smi; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    echo -e "${GREEN}✓${NC} $GPU_COUNT GPU(s) detected"
else
    echo -e "${YELLOW}⚠${NC} No NVIDIA drivers"
fi

echo ""

# =============================================================================
# 3. Docker
# =============================================================================
echo "=========================================="
echo "3. Docker"
echo "=========================================="

if command_exists docker; then
    echo -e "${GREEN}✓${NC} Docker installed"
else
    echo "Installing Docker..."
    apt-get update -qq
    apt-get install -y -qq ca-certificates curl gnupg lsb-release
    
    mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    
    systemctl start docker
    systemctl enable docker
    
    echo -e "${GREEN}✓${NC} Docker installed"
fi

echo ""

# =============================================================================
# 4. NVIDIA Container Toolkit
# =============================================================================
echo "=========================================="
echo "4. NVIDIA Container Toolkit"
echo "=========================================="

if command_exists docker && command_exists nvidia-smi; then
    echo "Configuring NVIDIA Container Toolkit..."
    
    if ! dpkg -l | grep -q nvidia-container-toolkit; then
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null
        
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
        
        apt-get update -qq 2>/dev/null
        apt-get install -y -qq nvidia-container-toolkit 2>/dev/null
    fi
    
    nvidia-ctk runtime configure --runtime=docker >/dev/null 2>&1
    systemctl restart docker
    
    echo -e "${GREEN}✓${NC} NVIDIA Container Toolkit configured"
fi

echo ""

# =============================================================================
# 5. System Dependencies
# =============================================================================
echo "=========================================="
echo "5. System Dependencies"
echo "=========================================="

apt-get install -y -qq \
    git curl wget \
    build-essential \
    libssl-dev openssl \
    unzip \
    net-tools \
    htop \
    nano >/dev/null 2>&1

echo -e "${GREEN}✓${NC} System dependencies installed"
echo ""

# =============================================================================
# 6. Node.js & npm
# =============================================================================
echo "=========================================="
echo "6. Node.js & npm"
echo "=========================================="

if command_exists npm; then
    NPM_VERSION=$(npm --version 2>/dev/null || echo "unknown")
    echo -e "${GREEN}✓${NC} npm $NPM_VERSION already installed"
else
    echo "Installing Node.js and npm..."
    curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
    apt-get install -y -qq nodejs
    echo -e "${GREEN}✓${NC} Node.js and npm installed"
fi

echo ""

# =============================================================================
# 7. PM2
# =============================================================================
echo "=========================================="
echo "7. PM2"
echo "=========================================="

if command_exists pm2; then
    PM2_VERSION=$(pm2 --version 2>/dev/null || echo "unknown")
    echo -e "${GREEN}✓${NC} PM2 $PM2_VERSION already installed"
else
    echo "Installing PM2..."
    npm install -g pm2
    
    if command_exists pm2; then
        PM2_VERSION=$(pm2 --version 2>/dev/null || echo "unknown")
        echo -e "${GREEN}✓${NC} PM2 $PM2_VERSION installed"
    else
        echo -e "${RED}✗${NC} PM2 installation failed"
    fi
fi

echo ""

# =============================================================================
# 8. gVisor
# =============================================================================
echo "=========================================="
echo "8. gVisor (Optional)"
echo "=========================================="

if ! command_exists runsc; then
    echo "Installing gVisor..."
    
    (
        ARCH=$(uname -m)
        URL=https://storage.googleapis.com/gvisor/releases/release/latest/${ARCH}
        
        cd /tmp
        wget -q ${URL}/runsc ${URL}/runsc.sha512 \
             ${URL}/containerd-shim-runsc-v1 ${URL}/containerd-shim-runsc-v1.sha512
        
        sha512sum -c runsc.sha512 >/dev/null 2>&1
        sha512sum -c containerd-shim-runsc-v1.sha512 >/dev/null 2>&1
        
        chmod a+rx runsc containerd-shim-runsc-v1
        mv runsc containerd-shim-runsc-v1 /usr/local/bin/
        rm -f runsc.sha512 containerd-shim-runsc-v1.sha512
        
        echo -e "${GREEN}✓${NC} gVisor installed"
    ) || echo -e "${YELLOW}⚠${NC} gVisor install failed (optional)"
    
    if command_exists docker && command_exists runsc; then
        cat > /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "runsc": {
            "path": "/usr/local/bin/runsc"
        }
    }
}
EOF
        systemctl restart docker
    fi
else
    echo -e "${GREEN}✓${NC} gVisor already installed"
fi

echo ""

# =============================================================================
# 9. Python Environment
# =============================================================================
echo "=========================================="
echo "9. Python Environment"
echo "=========================================="

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
else
    echo -e "${GREEN}✓${NC} Virtual environment exists"
fi

echo "Installing Python packages..."
source venv/bin/activate
pip install --upgrade pip setuptools wheel -q
pip install -r requirements.txt -q
deactivate

echo -e "${GREEN}✓${NC} Python packages installed"
echo ""

# =============================================================================
# 10. Directories
# =============================================================================
echo "=========================================="
echo "10. Directories"
echo "=========================================="

mkdir -p logs data/inputs data/outputs ssl

echo -e "${GREEN}✓${NC} Directories created"
echo ""

# =============================================================================
# 11. SSL Certificates
# =============================================================================
echo "=========================================="
echo "11. SSL Certificates"
echo "=========================================="

if [ ! -f "ssl/runner.crt" ]; then
    openssl req -x509 -newkey rsa:4096 -nodes \
        -keyout ssl/runner.key \
        -out ssl/runner.crt \
        -days 365 \
        -subj "/C=US/ST=State/L=City/O=HoneSubnet/CN=localhost" \
        2>/dev/null
    
    chmod 600 ssl/runner.key
    chmod 644 ssl/runner.crt
    
    echo -e "${GREEN}✓${NC} SSL certificates generated"
else
    echo -e "${GREEN}✓${NC} SSL certificates exist"
fi

echo ""

# =============================================================================
# 12. Configuration
# =============================================================================
echo "=========================================="
echo "12. Configuration"
echo "=========================================="

if [ ! -f "config.local.yaml" ]; then
    cp config.yaml config.local.yaml
    
    sed -i "s|/etc/ssl/certs/runner.crt|$SCRIPT_DIR/ssl/runner.crt|g" config.local.yaml
    sed -i "s|/etc/ssl/private/runner.key|$SCRIPT_DIR/ssl/runner.key|g" config.local.yaml
    
    if command_exists nvidia-smi; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        sed -i "s|gpu_count: 8|gpu_count: $GPU_COUNT|g" config.local.yaml
    fi
    
    if command_exists runsc && command_exists docker; then
        EXEC_MODE="docker+gvisor"
    elif command_exists docker; then
        EXEC_MODE="docker"
        sed -i "s|mode: \"docker+gvisor\"|mode: \"docker\"|g" config.local.yaml
    else
        EXEC_MODE="direct"
        sed -i "s|mode: \"docker+gvisor\"|mode: \"direct\"|g" config.local.yaml
    fi
    
    echo -e "${GREEN}✓${NC} Configuration created (mode: $EXEC_MODE)"
else
    echo -e "${GREEN}✓${NC} Configuration exists"
fi

echo ""

# =============================================================================
# Summary
# =============================================================================
echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""

echo "Status:"
python3 --version && echo -e "${GREEN}✓${NC} Python $(python3 --version | awk '{print $2}')"
command_exists nvidia-smi && echo -e "${GREEN}✓${NC} NVIDIA GPUs ($(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l))" || echo -e "${YELLOW}⚠${NC} NVIDIA GPUs"
command_exists docker && echo -e "${GREEN}✓${NC} Docker" || echo -e "${RED}✗${NC} Docker"
command_exists runsc && echo -e "${GREEN}✓${NC} gVisor" || echo -e "${YELLOW}⚠${NC} gVisor"
command_exists npm && echo -e "${GREEN}✓${NC} npm $(npm --version 2>/dev/null)" || echo -e "${RED}✗${NC} npm"
command_exists pm2 && echo -e "${GREEN}✓${NC} PM2 $(pm2 --version 2>/dev/null)" || echo -e "${RED}✗${NC} PM2"

echo ""
echo "To run:"
echo "  cd $SCRIPT_DIR"
echo "  source venv/bin/activate"
echo "  python -m main --config config.local.yaml"
echo ""
echo "API: https://localhost:8443"
echo "Metrics: http://localhost:9090/metrics"
echo ""