#!/bin/bash
# Creates a venv with PyTorch for this platform and runs the agent.
#   Linux + NVIDIA: CUDA build;  macOS Apple Silicon: default build (MPS);  otherwise CPU build.
# Usage: AGENT_TOKEN=... ./start.sh [--cpu] [--benchmark N]
set -e
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
if ! command -v "$PYTHON" >/dev/null; then
  echo "python3 not found (need Python 3.10+)"; exit 1
fi

if [ ! -d venv ]; then
  echo "Creating venv..."
  "$PYTHON" -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

if ! python -c "import torch" 2>/dev/null; then
  pip install --upgrade pip
  OS=$(uname -s)
  if [ "$OS" = "Darwin" ]; then
    echo "macOS: installing PyTorch with MPS (Apple GPU) support"
    pip install torch
  elif command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU found: installing PyTorch with CUDA"
    pip install torch --index-url https://download.pytorch.org/whl/cu124
  else
    echo "No GPU: installing CPU-only PyTorch"
    pip install torch --index-url https://download.pytorch.org/whl/cpu
  fi
fi
pip install -q -r requirements.txt

exec python -u agent.py "$@"
