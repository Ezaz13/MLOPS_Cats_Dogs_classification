#!/bin/bash
# ============================================================
# DVC Setup Script for Linux/Mac
# Cats & Dogs Classification MLOps Project
# ============================================================

set -e  # Exit on error

# Change to project root directory (parent of scripts folder)
cd "$(dirname "$0")/.."

echo "============================================================"
echo "DVC SETUP SCRIPT"
echo "============================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python is not installed"
    echo "Please install Python and try again."
    exit 1
fi

echo "[1/5] Checking DVC installation..."
if ! command -v dvc &> /dev/null; then
    echo "DVC is not installed. Installing DVC..."
    pip install dvc
else
    echo "DVC is already installed."
fi

echo ""
echo "[2/5] Initializing DVC repository..."
python3 src/dvc_scripts/init_dvc.py
if [ $? -ne 0 ]; then
    echo "ERROR: DVC initialization failed"
    exit 1
fi

echo ""
echo "[3/5] Verifying DVC setup..."
dvc status || echo "WARNING: DVC status check failed"

echo ""
echo "[4/5] Displaying pipeline DAG..."
dvc dag || echo "WARNING: Could not display pipeline DAG"

echo ""
echo "[5/5] Setup complete!"
echo ""
echo "============================================================"
echo "DVC SETUP COMPLETED SUCCESSFULLY"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Run data ingestion: dvc repro data_ingestion"
echo "  2. Run full pipeline: dvc repro"
echo "  3. Or use: python src/dvc_scripts/run_pipeline.py --help"
echo ""
