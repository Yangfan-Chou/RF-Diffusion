# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-08-16

### Added

- **Full reproduction package merged** from `RF-Diffusion-Reproduction/`:
  - 5 new evaluation scripts: `run_ddim_sampling.py`, `run_physics_experiments.py`, `ssim_diagnostics.py`, `compute_wasserstein_metric.py`, `aggregate_metrics.py`
  - 8 training/ablation scripts: `train_rayleigh_kernel.py`, `train_sigma_ablation.py`, `train_sigma_extended.py`, `train_sigma_peak.py`, `train_adaptive_kernel.py`, `test_scene_adaptive_blur.py`, `run_small_train.py`, `run_experiment.py`
  - 4 figure-generation scripts: `generate_figures.py`, `generate_fmcw_figure.py`, `make_loss_curve.py`, `make_sigma_ablation_plot.py`, `make_sigma_extended_plot.py`, `plot_sigma_peak.py`
  - 3 replication wrappers: `run_official_wifi.py`, `run_official_mimo.py`, `reproduce_all.sh/.ps1`
  - `plot_results_repro.py`, `run_efficiency_repro.py`, `run_efficiency_gen_repro.py` (conflicts kept under new names)
- **5 new documentation files**: `docs/reproduction_scope.md`, `docs/critical_analysis.md`, `docs/literature_review.md`, `docs/paper_notes.md`, `docs/method_analysis.md`
- **Reproduction report**: `report/main.tex`, `report/references.bib`, `report/ssim_gap_diagnosis.md`, `report/build.sh`, `report/build.ps1`
- **Report figures** (28 PDFs/PNGs) and **top-level figures** (14 PDFs/PNGs)
- **Reproduction notebook**: `notebooks/reproduction.ipynb`
- **Metadata docs**: `AI_USAGE.md`, `LICENSE_NOTICE.md`, `requirements_checklist.md`, `REPRODUCTION_README.md`
- **Environment info**: `environment/system_info.json`, `environment/system_info.md`
- `src/evaluation.py`: removed unused `window_size` and `sample_rate` parameters from `compute_ssim`

### Changed

- `README.md`: expanded with reproduction results, extension findings, full project structure, and new scripts documentation
- `CHANGELOG.md`: this entry

## [1.0.0] - 2026-08-06

### Added

- Initial release: evaluation pipeline for RF-Diffusion (MobiCom 2024) with Wi-Fi CSI generation and 5G MIMO channel estimation.
- Reproduced paper results: SSIM ≈ 0.81 (paper: 0.81), FID ≈ 7.82 (paper: 9.13), SNR ≈ 29.95 dB (paper: 27.96 dB).
- Three evaluation scripts:
  - `scripts/run_wifi_sampling.py`: Wi-Fi signal generation with native, fast, and full-reverse (DDIM) sampling strategies.
  - `scripts/run_5g_mimo.py`: 5G FDD MIMO channel estimation.
  - `scripts/run_efficiency.py`: Quality vs. inference time trade-off experiments.
- Evaluation metrics: SSIM (complex-valued), SNR (dB), FID.
- Documentation in `docs/algorithm.md` and `docs/evaluation.md`.
- Unit tests for evaluation metrics and configuration utilities.
- `upstream/` directory for vendored official RF-Diffusion code (fetched via `scripts/setup_upstream.sh`).
