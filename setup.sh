#!/bin/bash
# SatQuery AI - Local Setup Script
# Run this to set up the development environment

set -e

echo "============================================"
echo "  SatQuery AI - Local Development Setup"
echo "============================================"
echo ""

# Check for required tools
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "ERROR: $1 is not installed."
        echo "  $2"
        exit 1
    fi
}

check_command "node" "Install Node.js 18+ from https://nodejs.org/"
check_command "bun" "Install Bun from https://bun.sh/"

echo "[1/4] Installing frontend dependencies..."
bun install

echo ""
echo "[2/4] Setting up Python environment..."
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "  WARNING: Python not found. Training pipeline requires Python 3.10+"
    echo "  Install from https://python.org/"
    PYTHON=""
fi

if [ -n "$PYTHON" ]; then
    echo "  Using $($PYTHON --version 2>&1)"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
        echo "  Creating virtual environment..."
        $PYTHON -m venv .venv
    fi
    
    echo "  Activating virtual environment..."
    source .venv/bin/activate
    
    echo "  Installing Python dependencies..."
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
    
    echo "  Python setup complete!"
else
    echo "  Skipping Python setup (Python not found)"
fi

echo ""
echo "[3/4] Fetching training data (synthetic fallback)..."
if [ -n "$PYTHON" ] && [ -f "training/fetch_training_data.py" ]; then
    source .venv/bin/activate 2>/dev/null || true
    $PYTHON training/fetch_training_data.py --dataset bigearthnet --output ./data --max-samples 100 2>/dev/null || echo "  (Data fetch skipped - can be done manually later)"
fi

echo ""
echo "[4/4] Running typecheck..."
bun tsc -b --noEmit 2>&1 && echo "  TypeScript: All checks passed!" || echo "  WARNING: Type errors found"

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Quick start:"
echo "  bun run dev          # Start the web app"
echo ""
echo "Training:"
echo "  Open training/01_domain_adaptation.ipynb in Google Colab"
echo ""
echo "Data fetching:"
echo "  python training/fetch_training_data.py --dataset all --output ./data"
echo ""
