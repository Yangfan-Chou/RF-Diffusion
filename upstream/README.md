# Upstream RF-Diffusion

This directory is intentionally left empty in the repository. The official
[RF-Diffusion repository](https://github.com/mobicom24/RF-Diffusion) (GPL-3.0)
is fetched on demand via the setup script.

## Why Not a Git Submodule?

A git submodule would tightly couple this repository to a specific commit of the
upstream, complicating maintenance. Instead, we fetch the upstream at a known
good state using the setup script (see below).

## Setup

Run the setup script from the project root:

```bash
# Linux / macOS
bash scripts/setup_upstream.sh

# Windows
powershell scripts/setup_upstream.ps1
```

This script will:

1. Clone the official RF-Diffusion repository into `upstream/RF-Diffusion/`
2. Download the dataset and model weights from the official GitHub releases
3. Extract and restructure the directories

## What Gets Fetched

- **Code**: The full upstream RF-Diffusion inference pipeline
  (`upstream/RF-Diffusion/`)
- **Dataset**: Wi-Fi, FMCW, and MIMO signal data
  (`upstream/RF-Diffusion/dataset/`)
- **Models**: Pretrained model checkpoints
  (`upstream/RF-Diffusion/model/`)

## Running After Setup

Once the setup script has completed, you can run experiments directly:

```bash
# Wi-Fi evaluation
python scripts/run_wifi_sampling.py --num-samples 5 --strategy native

# 5G MIMO evaluation
python scripts/run_5g_mimo.py --num-samples 5

# Efficiency sweep
python scripts/run_efficiency.py --num-samples 3
```
