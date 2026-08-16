# AI Usage Disclosure

This document records how generative AI was used during the project.

## Where AI was used

1. **Repository understanding**: AI read the official RF-Diffusion code
   (inference.py, train.py, tfdiff/diffusion.py, tfdiff/wifi_model.py,
   complex/complex_module.py) and summarised the structure for me.
2. **Bug hunting**: AI traced an indexing bug in the official
   `SignalDiffusion.degrade_fn` shape (size 512 vs 90) and pointed me to
   the right data reshape convention used by the official learner.
3. **Scripting**: AI drafted the wrapper scripts (`src/runner.py`,
   `scripts/run_experiment.py`, `scripts/run_efficiency*.py`,
   `scripts/run_small_train.py`, `scripts/plot_results.py`,
   `scripts/aggregate_metrics.py`). All these scripts were executed and
   their outputs were inspected.
4. **Documentation**: AI drafted `docs/paper_notes.md`,
   `docs/literature_review.md`, `docs/method_analysis.md`,
   `docs/critical_analysis.md`, `docs/reproduction_scope.md` and the
   LaTeX report. All factual claims about the paper were verified
   against the paper text and the official code.

## What AI did NOT do

- AI did not invent or fabricate any experimental result, SSIM/FID/SNR
  number, run time, or memory usage.
- AI did not modify the official core algorithm
  (`upstream/RF-Diffusion/tfdiff/*.py` and
  `upstream/RF-Diffusion/complex/*.py` remain untouched).
- AI did not skip running the experiments.

## Verification I performed personally

- I ran every `python scripts/*.py` invocation myself and inspected every
  output (log files, JSON metrics, CSV, PDF/PNG figures).
- I cross-checked the reproduced numbers against the paper:
  - Wi-Fi SSIM 0.81 (paper 0.81, matches within ±0.001).
  - Wi-Fi FID 7.82 (paper 9.13, lower is better).
  - 5G SNR 29.95 dB (paper 27.96 dB, 7.1% higher).
- I read the official inference log to extract the SSIM/FID/SNR lines
  that appear in main_results.csv.

## Known limits

- The official training data (raw/) was not fully released; only 41
  condition files in cond/. Small-scale training therefore uses these
  files as a subset. This is documented in
  `docs/reproduction_scope.md` and `scripts/run_small_train.py`.

## AI model used

- Cursor built-in assistant (Claude-based) for code drafting and doc
  drafting. Reasoning model = composer-2.5-fast (per Cursor settings).
- DeepSeek API was offered (`sk-b009941df1374ddd94f0cb4876e60b40`) but
  not needed for this project; Cursor assistant covered all needs.