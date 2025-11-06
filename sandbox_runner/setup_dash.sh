#!/bin/bash
# Quick setup script for Sandbox Runner Dashboard

set -e

echo "=========================================="
echo "Sandbox Runner Dashboard Setup"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Python 3 is required but not installed.${NC}"
    exit 1
fi

echo -e "${BLUE}1. Installing dashboard dependencies...${NC}"
pip install streamlit==1.29.0 plotly==5.18.0 pandas==2.1.4 httpx==0.25.2 -q

echo -e "${GREEN}✓${NC} Dependencies installed"
echo ""

# Create .streamlit directory
echo -e "${BLUE}2. Creating Streamlit configuration...${NC}"
mkdir -p .streamlit

# Create config.toml if it doesn't exist
if [ ! -f .streamlit/config.toml ]; then
    cat > .streamlit/config.toml << 'EOF'
[theme]
primaryColor = "#4169E1"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
EOF
    echo -e "${GREEN}✓${NC} Created .streamlit/config.toml"
else
    echo -e "${YELLOW}ℹ${NC} .streamlit/config.toml already exists"
fi

# Create secrets.toml if it doesn't exist
if [ ! -f .streamlit/secrets.toml ]; then
    cat > .streamlit/secrets.toml << 'EOF'
# API Configuration
API_BASE_URL = "http://localhost:8080"
API_KEY = "dev-key-12345"

# Optional settings
REFRESH_INTERVAL = 5
MAX_LOG_LINES = 1000
EOF
    echo -e "${GREEN}✓${NC} Created .streamlit/secrets.toml"
    echo -e "${YELLOW}⚠${NC}  Remember to update API_BASE_URL and API_KEY in .streamlit/secrets.toml"
else
    echo -e "${YELLOW}ℹ${NC} .streamlit/secrets.toml already exists"
fi