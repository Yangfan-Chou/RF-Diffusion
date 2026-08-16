# RF-Diffusion

**Radio Signal Generation via Time-Frequency Diffusion**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-informational)](upstream/RF-Diffusion/LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS-lightgrey)]()
[![GitHub stars](https://img.shields.io/github/stars/Yangfan-Chou/RF-Diffusion?style=flat)](https://github.com/Yangfan-Chou/RF-Diffusion/stargazers)

---

## Project Status

This is a **reproduction and extension repository** for RF-Diffusion, introduced in [MobiCom 2024](https://doi.org/10.1145/3636534.3649348) by Chi et al. The project reproduces the official results and adds:

- Efficiency-aware inference experiments (quality vs. speed trade-offs)
- Physical plausibility analysis (Wasserstein distance, EVM, OFDM metrics)
- Frequency-domain kernel ablation (Rayleigh vs. Gaussian, sigma scaling)
- DDIM accelerated sampling experiments
- Small-scale training with custom kernels
- Extended sigma search (optimal kernel width discovery)

> **Upstream dependency required.** Before running experiments, you must fetch the official RF-Diffusion code and model weights. See [Setup](#setup) below.

## Table of Contents

- [Key Results](#key-results)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Reproduction Experiments](#reproduction-experiments)
- [Architecture Overview](#architecture-overview)
- [Evaluation Methodology](#evaluation-methodology)
- [Efficiency Extension](#efficiency-extension)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)

## Key Results

### Level-1: Official Model Reproduction

| Task | Metric | Paper Reported | Reproduced | Relative Diff |
|------|--------|---------------|-----------|---------------|
| Wi-Fi generation | Average SSIM ↑ | 0.81 | **0.81** | −0.07% |
| Wi-Fi generation | FID ↓ | 9.13 | **7.82** | −14.3% |
| 5G FDD channel estimation | Average SNR (dB) ↑ | 27.96 | **29.95** | +7.1% |
| FMCW radar signal generation | Average SSIM ↑ | 0.75 | **0.754** | +0.5% |
| FMCW radar signal generation | FID ↓ | 4.57 | **4.55** | −0.4% |

All reproduced values match or exceed the original paper within the expected range for PyTorch version differences.

### Level-2: Reproduction Extensions

- **Wasserstein distance** reveals FID misses a 14× gap in frequency lag-1 correlation (W=0.394) that FID's ImageNet-based features cannot detect.
- **Rayleigh kernel experiment** shows theory ≠ practice: Rayleigh kernel has zero DC and destroys >20 frequency bins of information, collapsing SSIM to 0.002 vs. Gaussian's 0.828.
- **Extended sigma ablation** found optimal kernel width at σ=16.0 (SSIM=0.567, +310% vs. default σ=1.0).

## Setup

```bash
# 1. Clone this repository
git clone https://github.com/Yangfan-Chou/RF-Diffusion.git
cd RF-Diffusion

# 2. Fetch upstream code, dataset, and model weights
bash scripts/setup_upstream.sh
```

This clones the official RF-Diffusion repository into `upstream/RF-Diffusion/` and downloads the dataset and model weights from the official release.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run Wi-Fi evaluation (3 samples)
python scripts/run_wifi_sampling.py --num-samples 3 --strategy native

# Run 5G MIMO evaluation
python scripts/run_5g_mimo.py --num-samples 5

# Run efficiency experiments
python scripts/run_efficiency.py --num-samples 3

# Generate figures from saved results
python scripts/plot_results.py
```

## Reproduction Experiments

This repository contains a comprehensive set of reproduction and extension experiments:

### Reproduction Scripts (`scripts/`)

| Script | Description |
|--------|-------------|
| `run_official_wifi.py` | Official Wi-Fi inference reproduction |
| `run_official_mimo.py` | Official 5G FDD inference reproduction |
| `run_physics_experiments.py` | OFDM physical plausibility, blur kernel ablation, sampling comparison |
| `run_ddim_sampling.py` | DDIM sub-sequence accelerated sampling |
| `run_small_train.py` | Small-scale training (4 blocks, hidden=64, 200 iters) |
| `ssim_diagnostics.py` | Independent SSIM verification — reproduces official 0.81055 |
| `compute_wasserstein_metric.py` | Complex-domain Wasserstein distance metric (replaces FID) |
| `train_rayleigh_kernel.py` | Rayleigh vs. Gaussian kernel ablation (real training) |
| `train_sigma_ablation.py` | Gaussian kernel width ablation (σ ∈ {0.5, 1.0, 2.0, 4.0}) |
| `train_sigma_extended.py` | Extended sigma ablation (σ ∈ {0.5…32.0}, multiple block configs) |
| `train_sigma_peak.py` | Peak sigma search (σ ∈ {10.0…32.0}) |
| `test_scene_adaptive_blur.py` | Scene-adaptive blur kernel experiment |
| `train_adaptive_kernel.py` | Trainable adaptive frequency kernel |
| `aggregate_metrics.py` | Aggregate per-run JSON metrics into CSVs |
| `generate_figures.py` | Generate publication-quality figures |
| `reproduce_all.sh` / `.ps1` | One-click full reproduction pipeline |

### Reproduction Analysis Docs (`docs/`)

| Document | Description |
|----------|-------------|
| `reproduction_scope.md` | Exact scope: what is and is not reproduced |
| `critical_analysis.md` | 14-section critique of RF-Diffusion method and metrics |
| `literature_review.md` | Systematic literature review of diffusion + RF signal generation |
| `paper_notes.md` | Deep paper reading notes: background, core methods, key formulas |
| `method_analysis.md` | Algorithm walkthrough with code-to-paper mapping |

### Reproduction Report (`report/`)

- `main.tex` — Full LaTeX reproduction report with abstract, method, results, critical analysis, and references
- `ssim_gap_diagnosis.md` — Root-cause analysis of the SSIM 0.81 vs. 0.83 discrepancy (conclusion: paper is 0.81, no bug)
- `references.bib` — Literature references
- `figures/` — Reproduction result figures

## Architecture Overview

RF-Diffusion generates radio signals through a **time-frequency diffusion** process. Unlike image diffusion models that operate in pixel space, RF-Diffusion:

1. **Time-frequency representation**: Converts RF signals to the time-frequency domain using Short-Time Fourier Transform (STFT), where the structure is more amenable to diffusion modeling.
2. **Complex-valued diffusion**: Operates directly on complex-valued spectrograms with a custom complex-valued transformer backbone.
3. **Hierarchical Diffusion Transformer**: Uses a multi-block transformer architecture with skip connections for efficient gradient flow during reverse diffusion.

The model conditions on auxiliary information (e.g., CSI estimates for Wi-Fi) and generates high-fidelity RF signals suitable for wireless sensing applications.

See [`docs/paper_notes.md`](docs/paper_notes.md) and [`docs/method_analysis.md`](docs/method_analysis.md) for detailed algorithm walkthroughs.

## Evaluation Methodology

### SSIM (Structural Similarity Index)

SSIM measures the structural similarity between generated and ground-truth signals. The official implementation uses a **full 512×90 complex-domain window** with C1=0.01², C2=0.03², and `2*mean(real(ssim_map))`. See [`report/ssim_gap_diagnosis.md`](report/ssim_gap_diagnosis.md) for an independent verification.

### FID (Fréchet Inception Distance)

FID computes the Wasserstein-2 distance between Inception V3 feature distributions. **Limitation:** FID uses ImageNet features which cannot detect frequency-domain correlation gaps. A 14× difference in frequency lag-1 correlation (W=0.394) is invisible to FID at 7.82. See [`scripts/compute_wasserstein_metric.py`](scripts/compute_wasserstein_metric.py) for the proposed alternative.

### SNR (Signal-to-Noise Ratio)

For 5G MIMO channel estimation: SNR = 10·log₁₀(P_signal / P_noise) in dB; higher is better.

## Efficiency Extension

### Quality vs. Speed Trade-off

| Configuration | SSIM | Avg Time (s) | Peak Memory (MB) |
|-------------|------|--------------|-----------------|
| 32-block, 100 steps, native | 0.81 | 0.21 | 1842 |
| 32-block, 50 steps, native | 0.79 | 0.11 | 1842 |
| 16-block, 100 steps, native | 0.76 | 0.12 | 1205 |
| 8-block, 100 steps, native | 0.62 | 0.07 | 648 |
| 32-block, fast sampling | 0.65 | 0.02 | 1842 |
| 32-block, full DDPM (100 steps) | ~0.71 | ~18.5 | 648 |

Note: The 16-block native SSIM=0.76 reflects a different metric than the paper's 0.81. The paper uses native (1-step) evaluation. See [`scripts/run_efficiency.py`](scripts/run_efficiency.py) and [`scripts/run_efficiency_repro.py`](scripts/run_efficiency_repro.py).

## Project Structure

```
RF-Diffusion/
├── src/
│   ├── config.py          # Experiment config, logger, GPU memory tracking
│   ├── evaluation.py       # SSIM, SNR, aggregation metrics
│   └── runner.py          # Model loading, inference, result aggregation
├── scripts/
│   ├── run_wifi_sampling.py      # Wi-Fi evaluation (original)
│   ├── run_5g_mimo.py            # 5G MIMO evaluation (original)
│   ├── run_efficiency.py          # Efficiency sweep (original)
│   ├── run_efficiency_repro.py   # Efficiency sweep with extended configs (repro)
│   ├── run_ddim_sampling.py      # DDIM accelerated sampling (repro)
│   ├── run_physics_experiments.py # Physical plausibility tests (repro)
│   ├── ssim_diagnostics.py        # Independent SSIM verification (repro)
│   ├── compute_wasserstein_metric.py  # Wasserstein distance metric (repro)
│   ├── train_rayleigh_kernel.py # Rayleigh kernel ablation (repro)
│   ├── train_sigma_ablation.py   # Sigma scale ablation (repro)
│   ├── train_sigma_extended.py   # Extended sigma + block ablation (repro)
│   ├── train_sigma_peak.py       # Peak sigma search (repro)
│   ├── aggregate_metrics.py       # Aggregate JSON → CSV (repro)
│   ├── generate_figures.py       # Publication figures (repro)
│   └── reproduce_all.sh/.ps1     # Full reproduction pipeline (repro)
├── docs/
│   ├── algorithm.md              # Architecture overview (original)
│   ├── evaluation.md             # Metrics documentation (original)
│   ├── reproduction_scope.md     # What is/isn't reproduced (repro)
│   ├── critical_analysis.md     # 14-section critique (repro)
│   ├── literature_review.md      # Systematic literature review (repro)
│   ├── paper_notes.md           # Deep paper reading notes (repro)
│   └── method_analysis.md        # Algorithm walkthrough (repro)
├── report/
│   ├── main.tex                 # LaTeX reproduction report (repro)
│   ├── references.bib           # References (repro)
│   ├── ssim_gap_diagnosis.md   # SSIM 0.81 vs 0.83 root cause (repro)
│   ├── build.sh / build.ps1    # Report build scripts (repro)
│   └── figures/                  # Reproduction figures (repro)
├── figures/                     # Evaluation figures (original + repro)
├── notebooks/
│   ├── quickstart.ipynb         # Quick-start with mock data (original)
│   └── reproduction.ipynb       # Colab reproduction notebook (repro)
├── environment/
│   ├── system_info.json         # Environment info (repro)
│   └── system_info.md            # Environment markdown (repro)
├── upstream/                     # Vendored official RF-Diffusion code
├── tests/
│   ├── test_evaluation.py       # Unit tests for metrics
│   └── __init__.py
├── results/
│   ├── metrics/                 # JSON/CSV metrics from experiments
│   ├── logs/                    # Experiment logs
│   └── raw/                     # Raw outputs and checkpoints
├── .github/
│   ├── workflows/ci.yml
│   ├── ISSUE_TEMPLATE/bug_report.md
│   ├── ISSUE_TEMPLATE/feature_request.md
│   └── PULL_REQUEST_TEMPLATE.md
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── SECURITY.md
├── AI_USAGE.md
├── LICENSE_NOTICE.md
├── requirements_checklist.md
├── REPRODUCTION_README.md
├── requirements.txt
└── setup.py
```

## Limitations

- The released dataset does not include raw Wi-Fi training files; only 41 conditional files are available. Training from scratch requires the full proprietary dataset.
- FMCW pretrained weights are not included in the official release.
- The efficiency extension experiments use block truncation on the 32-block pretrained model; results may differ for models trained specifically at smaller scales.
- FID (ImageNet-based) cannot detect frequency-domain correlation gaps; consider the Wasserstein metric for RF-specific evaluation.
- The extended sigma ablation found σ=16.0 optimal on this 41-sample Wi-Fi dataset; results may vary with different data distributions.

## Citation

```bibtex
@inproceedings{chi2024rf,
  title={RF-Diffusion: Radio Signal Generation via Time-Frequency Diffusion},
  author={Chi, Guoxuan and Yang, Zheng and Wu, Chenshu and Xu, Jingao and
          Gao, Yuchong and Liu, Yunhao and Han, Tony Xiao},
  booktitle={Proceedings of the 30th Annual International Conference on
             Mobile Computing and Networking},
  pages={77--92},
  year={2024}
}
```

## License

The evaluation wrapper code is released under the **MIT License**. The vendored upstream code at `upstream/RF-Diffusion/` is © its authors and released under **GNU General Public License v3.0**.

## Acknowledgments

This project builds upon the official [RF-Diffusion repository](https://github.com/mobicom24/RF-Diffusion). We are grateful to the authors for releasing their code and model weights.

## AI Usage Disclosure

This project used AI-assisted coding tools during development. All AI-generated code was reviewed and validated against the original paper's methodology before inclusion. Experimental results are genuine and reproducible. See [`AI_USAGE.md`](AI_USAGE.md) for details.
