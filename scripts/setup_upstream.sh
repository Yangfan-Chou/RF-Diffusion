#!/bin/bash
# Setup script to download upstream RF-Diffusion code, dataset, and model weights.

set -e

UPSTREAM_DIR="$(dirname "$0")/upstream/RF-Diffusion"

echo "=== RF-Diffusion Setup Script ==="
echo "Upstream directory: $UPSTREAM_DIR"
echo ""

# Check if upstream already exists
if [ -d "$UPSTREAM_DIR" ]; then
    echo "Upstream RF-Diffusion already exists at $UPSTREAM_DIR"
    echo "Skipping clone..."
else
    echo "Cloning upstream RF-Diffusion..."
    mkdir -p "$(dirname "$0")/upstream"
    git clone https://github.com/mobicom24/RF-Diffusion.git "$UPSTREAM_DIR"
fi

cd "$UPSTREAM_DIR"

# Download dataset and model
echo ""
echo "Downloading dataset and model weights..."

if [ ! -d "dataset/wifi" ] || [ ! -d "model/wifi" ]; then
    wget -q --show-progress https://github.com/mobicom24/RF-Diffusion/releases/download/dataset_model/dataset.zip
    wget -q --show-progress https://github.com/mobicom24/RF-Diffusion/releases/download/dataset_model/model.zip

    echo "Extracting archives..."
    unzip -q dataset.zip
    unzip -q model.zip

    echo "Restructuring directories..."
    mkdir -p dataset
    mv wifi fmcw mimo dataset/ 2>/dev/null || true

    # Cleanup
    rm -f dataset.zip model.zip
else
    echo "Dataset and model already exist, skipping download."
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To verify the installation, run:"
echo "  python upstream/RF-Diffusion/inference.py --task_id 0"
