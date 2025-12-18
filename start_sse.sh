#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Embedding Agent Startup (SSE Mode)${NC}"
echo "================================"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $PYTHON_VERSION"

# Check minimum Python version (3.8)
MIN_VERSION="3.8"
if [ "$(printf '%s\n' "$MIN_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$MIN_VERSION" ]; then
    echo -e "${RED}Error: Python $MIN_VERSION or higher is required${NC}"
    exit 1
fi

# Check required environment variables
if [ -z "$RS_HTTP_FACADE_TOKEN" ]; then
    echo -e "${RED}Error: Authentication token is required${NC}"
    echo "Please set the RS_HTTP_FACADE_TOKEN environment variable"
    exit 1
fi

# Configuration values are now hardcoded in agent_sse.py
echo "RS Facade URL: https://nsq.fred.org.ru (hardcoded)"
echo "Task Stream: embedding_tasks (hardcoded)"
echo "Task Group: embedding-agent (hardcoded)"
echo "Result Stream: embedding_results (hardcoded)"
echo "Web Port: 8080 (hardcoded)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Detect platform and install appropriate PyTorch
echo -e "${YELLOW}Detecting platform...${NC}"
OS=$(uname -s)
ARCH=$(uname -m)

if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    echo "Platform: macOS Apple Silicon (M1/M2/M3)"
    echo "Installing PyTorch with MPS support..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio
elif [ "$OS" = "Linux" ]; then
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "Platform: Linux with NVIDIA GPU"
        echo "Installing PyTorch with CUDA support..."
        pip install --upgrade pip
        # Install PyTorch with CUDA 11.8 (widely compatible)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "Platform: Linux (CPU only)"
        echo "Installing PyTorch (CPU version)..."
        pip install --upgrade pip
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
else
    echo "Platform: Generic (CPU only)"
    echo "Installing PyTorch (CPU version)..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio
fi

# Install remaining dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Run the SSE agent
echo -e "${GREEN}Starting agent (SSE mode)...${NC}"
echo "================================"
python3 agent_sse.py
