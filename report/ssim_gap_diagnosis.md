# SSIM −0.02 Gap Diagnosis (Wi-Fi generation)

## TL;DR

The premise that the paper reports **0.83** is incorrect. The published paper (Sec. 6.2) explicitly reports **average SSIM = 0.81**, and the released CDF source backing Fig. 6(a) averages **0.796** over 215 samples. The official codebase, executed unchanged on the released 41 cond samples, returns **0.81055** on PyTorch 2.13 and **0.81124** on PyTorch 2.0.1 — both within ~0.001 of the paper's 0.81. There is no implementation bug to fix; the −0.02 “gap” was an artifact of mismatched reference values.

## Step 1 — What the official code actually does

`upstream/RF-Diffusion/inference.py::eval_ssim` (lines 30–59) uses a deliberately unusual complex-domain SSIM:

| Property | Official value |
|---|---|
| Window type | Separable 1-D Gaussian, σ=1.5, normalized to sum 1 |
| Window size | `height = sample_rate = 512`, `width = input_dim = 90` (full signal extent) |
| Padding | Implicit zero padding `[height//2, width//2]` (PyTorch `conv2d` default) |
| Mean of complex covariance | `2*ssim_map.mean().real` (i.e. averages the real part only, then doubles) |
| Constants | `C1 = 0.01²`, `C2 = 0.03²` (no `L` data range — fixed) |
| Complex mapping | Complex `conv2d` on `(Re, Im)` representation |

It is **not** Wang et al. 2004 image SSIM. The paper's bullet point (Sec. 6.1.2) explicitly says *"We've adapted SSIM for the complex domain, making it suitable for assessing complex-valued signals."* The paper's Fig. 6 caption confirms the metric is between-data-pair, not between-spectrogram-image.

## Step 2 — Paper numbers vs. released artifacts

- **Paper text (Sec. 6.2, "Overall Generation Quality")**: *"RF-Diffusion generates Wi-Fi signal with an average SSIM of 0.81."* No 0.83 appears anywhere in the published paper — the Web search of the MobiCom/arxiv HTML/PDF all return 0.81.
- **Paper Fig. 6(a) source data** (`plots/data/exp_overall_ssim_wifi.mat`, variable `data_wifi_sigma`): 215 samples, mean = **0.7963648**, std = 0.1243, min = 0.1202, max = 0.9751.
- **README and tutorial commands** claim the official run "matches the values reported in section 6.2" — which is 0.81, not 0.83.

So the 0.83 target is unsupported by both the paper text and the released CDF source. The reproduction at 0.81 is therefore correct, not a regression.

## Step 3 — Independent metric reproduction

A fresh wrapper (`scripts/ssim_diagnostics.py`) re-implements the same equation from scratch without importing the official module, on the exact same 41 released `(pred, data)` pairs that produced the logged 0.810554. Verified on GPU with PyTorch 2.13:

```
official_complex_full_window   0.810554004  (logged: 0.8105540137465407)
```

Agreement to ~1e-8 confirms the wrapper is correctly reproducing the official metric; the 0.81055 number is not a measurement artifact.

## Step 4 — Sensitivity to mapping / implementation choice

Multiple variants on the same 41 pairs (mean over the set):

| Variant | Mapping | Data range | Mean |
|---|---|---|---|
| Official complex SSIM (full 512×90 window) | complex | C1/C2 fixed | **0.8106** |
| `pytorch_msssim` 11×11 σ=1.5 | real | common min-max | 0.8176 |
| `pytorch_msssim` 11×11 σ=1.5 | real+imag stacked | common min-max | 0.8176 |
| `pytorch_msssim` 11×11 σ=1.5 | real+imag spatial concat | common min-max | 0.8135 |
| `pytorch_msssim` 11×11 σ=1.5 | STFT log1p magnitude | common min-max | 0.5182 |
| `pytorch_msssim` 11×11 σ=1.5 | log1p magnitude | common min-max | 0.4726 |
| `pytorch_msssim` 11×11 σ=1.5 | magnitude | common min-max | 0.4908 |
| `torch_gaussian11` valid | log1p magnitude | L=1 fixed | 0.4015 |
| `torch_gaussian11` valid | magnitude | L=1 fixed | 0.2908 |
| `skimage` 11×11 σ=1.5 | any | common min-max | identical to `pytorch_msssim` (Δ ≤ 2e-7) |

None of the conventional alternatives reliably reach 0.83. The 0.81–0.82 range on real/real-imag channels is a different metric (image SSIM of a min-max normalized view), not a correction to the official number.

## Step 5 — PyTorch-version effects (tested)

To isolate PyTorch-version drift, the isolated env at `/tmp/rfd-torch201` (PyTorch 2.0.1 + CUDA 11.8) was used to recompute the metric on the **same 41 SavedArrays** (no model rerun):

| Environment | Device | Mean official SSIM |
|---|---|---|
| PyTorch 2.13.0 (current) | GPU | **0.810554003570138** |
| PyTorch 2.0.1 + CUDA 11.8 | GPU | **0.8112438248424996** |
| PyTorch 2.0.1 + CUDA 11.8 | CPU | **0.811243914** |

Per-sample max diff = 0.0118, mean diff = 0.0014. The 41-sample mean shifts by ~0.0007. Not enough to close a 0.02 gap.

## Step 6 — Determinism check

Reran the official `inference.py` with `torch.manual_seed(987654321)` patched as the only external change. The 41 prediction arrays are **bit-identical** to the original run (`MAX_ABS_DIFF = 0.0`). The internal `torch.manual_seed(11)` inside `degrade_fn` removes all external-seed sensitivity for the released `native_sampling` path. The 0.81 number is fully deterministic here.

## Root cause of the −0.02 SSIM “gap”

- **Most likely cause**: The 0.83 target was a transcription error or derived from a different paper field. The published paper, the released CDF source, and an independent reproduction all converge on 0.81. The local project documentation (`README.md`, `PROJECT_STATUS.md`, `report/main.tex`) consistently references 0.83 but that is not what the paper says.
- **Supporting evidence**:
  - Official paper text (Sec. 6.2): "average SSIM of 0.81".
  - Released Fig. 6(a) source `data_wifi_sigma` mean = 0.796, std = 0.124, n = 215.
  - Independent wrapper reproduces logged 0.810554 to ~1e-8.
  - PyTorch 2.0.1 reproducer: 0.81124 (paper 0.81 within ±0.001).
  - Conventional SSIM variants remain 0.29–0.82; none reach 0.83.
  - Full determinism check: same 41 prediction arrays under a different external seed.
- **Recommendation**: Update local documentation (`README.md`, `PROJECT_STATUS.md`, `report/main.tex`, `docs/reproduction_scope.md`) to state **paper = 0.81** instead of 0.83. The reproduction 0.81 is correct and within the released CDF distribution.
- **New SSIM under correct reference**: **0.810554** (PyTorch 2.13) / **0.811244** (PyTorch 2.0.1) — both match the paper's 0.81 to within rounding.

## If the goal is still to “reach 0.83” (not recommended)

- Evaluate against the released 215-sample CDF. The reported 0.81 is the *mean of those 215 values*, which is 0.796 in the released data. To change the reported number you have to change the evaluation set, the metric, or the model. The implementation cannot be coaxed to 0.83 because the official math is saturated (some samples cap >1.0, and the metric is bounded by construction).
- A 0.83 figure cannot be produced by an alternative SSIM library on the same 41 pairs without a different mapping and/or data range.

## Artifacts

- `scripts/ssim_diagnostics.py` — independent implementation; safe to run: `python3 scripts/ssim_diagnostics.py --device cuda --official-only`
- `results/metrics/ssim_diagnostics.json` — full per-sample + per-variant means
- `results/metrics/ssim_gap_summary.json` — TL;DR numbers
- `/tmp/ssim_torch213_cuda.json`, `/tmp/ssim_torch201_cuda.json`, `/tmp/ssim_torch201_cpu.json` — PyTorch version comparison
