# Reproduction Scope

This file describes exactly what this project reproduces and what it does not,
to avoid misrepresenting any partial reproduction as a full one.

## What is reproduced

### Level-1: official pretrained model reproduction
- Wi-Fi data generation (`python inference.py --task_id 0`).
  - Uses the official released checkpoint.
  - Average SSIM, FID computed identically to the paper's evaluation.
- 5G FDD channel estimation (`python inference.py --task_id 2`).
  - Uses the official released checkpoint.
  - Average SNR (dB) computed identically to the paper's evaluation.

### Level-2: standardized wrapper framework
- `scripts/run_experiment.py` argparse-driven runner with task/mode/seed.
- `scripts/run_official_wifi.py`, `scripts/run_official_mimo.py` directly
  invoke the official inference scripts (Level-1 raw reproduction).
- `scripts/run_efficiency.py`, `scripts/run_efficiency_gen.py` measure
  SSIM/time/memory trade-offs for various block counts and step counts.
- Each run produces a unique `run_id` and stores metrics to
  `results/metrics/<task>_<mode>_<run_id>.json`.

### Level-3: small-scale training
- `scripts/run_small_train.py` runs 200 iterations with a 4-block, hidden-64
  variant on 41 Wi-Fi training samples (released cond files).
- Loss curve saved to JSON.

## What is NOT reproduced

- **Full official training**. The official training requires the full
  Widar3.0 dataset (10+ GB), 32 blocks, hidden 128/256, and many GPU-days.
  We were only given the cond/ folder with 41 .mat files; raw training
  data was not released.
- **FMCW task**. The released model.zip does not include a pre-trained
  FMCW checkpoint, only Wi-Fi and MIMO. Skipped.
- **EEG denoising task**. Not in the official paper's main scope.
- **Downstream gesture recognition** (Section 7.1 of the paper). Would
  require a separate classifier training pipeline and is outside the
  reproduction scope.

## Distinguishing results

| Result type   | Where stored                              | How labeled          |
|---------------|-------------------------------------------|----------------------|
| Paper reported| `results/metrics/paper_comparison.csv`    | "Paper Reported"     |
| Reproduced    | `results/metrics/main_results.csv`        | "Reproduced"         |
| Small-train   | `results/metrics/small_train_*.json`      | "small-train" mode   |
| Extension     | `results/metrics/efficiency_*.json`       | "efficiency" configs |

The CSV `paper_comparison.csv` makes the comparison explicit.