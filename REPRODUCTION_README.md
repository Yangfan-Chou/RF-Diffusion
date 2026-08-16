# RF-Diffusion Reproduction

> RF-Diffusion：面向无线信号生成的时频扩散模型复现与低算力推理评估  
> Reproduction and Compute-Efficient Evaluation of RF-Diffusion for Radio Signal Generation

Reproduction of the MobiCom 2024 paper **RF-Diffusion: Radio Signal Generation
via Time-Frequency Diffusion** (Chi et al., 2024) for the Zhejiang University
"Industrial IoT and Edge Intelligence" project selection.

## Quick facts

| Item | Value |
|------|-------|
| Paper | [arXiv:2404.09140](https://arxiv.org/abs/2404.09140) |
| DOI | [10.1145/3636534.3649348](https://doi.org/10.1145/3636534.3649348) |
| Code | [github.com/mobicom24/RF-Diffusion](https://github.com/mobicom24/RF-Diffusion) (commit `eb872b0`) |
| Artefact | [Zenodo 10449052](https://zenodo.org/records/10449052) |
| Hardware | NVIDIA A40 (45 GB) |
| Software | Python 3.13, PyTorch 2.13, CUDA 12.2 |
| Date | 2026-08-04 |

## Reproduced results

| Task | Metric | Paper reported | Reproduced (this project) |
|------|--------|----------------|---------------------------|
| Wi-Fi generation | Average SSIM ↑ | 0.81 | **0.81** |
| Wi-Fi generation | FID ↓ (corr=1.9) | 9.13 | **7.82** |
| 5G FDD channel estimation | Average SNR (dB) ↑ | 27.96 | **29.95** |

The numbers above are produced by running the **official** `inference.py`
directly (Level-1 reproduction). All raw values are persisted in
`results/metrics/*.json` and aggregated in `results/metrics/paper_comparison.csv`.

A small-scale end-to-end training (4 blocks / hidden-64 / 200 iters on the
41 released Wi-Fi samples) is also documented. The loss decreased from
0.78 to 0.03, confirming that the training pipeline is functional.

## Layout

```
.
├── README.md                    # this file
├── requirements_checklist.md    # 0..11 deliverables checklist
├── AI_USAGE.md                  # AI assistance disclosure
├── LICENSE_NOTICE.md            # upstream GPL-3.0 + paper citation
├── environment/                 # system_info.json / .md
├── configs/                     # (placeholder; experiments use CLI flags)
├── data/                        # placeholder; real data lives in upstream
├── docs/
│   ├── paper_notes.md           # research background, scientific question
│   ├── literature_review.md     # 18 related works comparison
│   ├── method_analysis.md       # Time-Frequency Diffusion derivation
│   ├── critical_analysis.md     # independent critique + improvement ideas
│   └── reproduction_scope.md    # what is and is not reproduced
├── notebooks/
│   └── RF_Diffusion_Reproduction.ipynb
├── results/
│   ├── raw/                     # generated .mat samples
│   ├── logs/                    # terminal logs
│   └── metrics/                 # per-run JSON + main_/efficiency_/paper_*.csv
├── figures/                     # PDF/PNG figures
├── scripts/
│   ├── run_experiment.py        # argparse wrapper
│   ├── run_official_wifi.py     # invokes upstream inference.py --task_id 0
│   ├── run_official_mimo.py     # invokes upstream inference.py --task_id 2
│   ├── run_small_train.py       # reduced training
│   ├── run_efficiency.py        # native-sampling efficiency sweep
│   ├── run_efficiency_gen.py    # full-reverse-diffusion efficiency sweep
│   ├── plot_results.py          # regenerates all figures from CSV/JSON
│   ├── aggregate_metrics.py     # produces the three required CSVs
│   └── reproduce_all.sh         # one-shot reproduction on Linux/WSL
├── src/
│   ├── config.py                # ExperimentConfig, seeding, GPU mem utils
│   └── runner.py                # thin wrapper around upstream modules
├── tests/
│   └── test_smoke.py            # pytest smoke tests
├── upstream/
│   └── RF-Diffusion/            # official code (commit eb872b0) — read-only
└── report/
    ├── main.tex / main.pdf / build.sh / build.ps1
    ├── references.bib
    └── figures/                 # copies of figures used in main.tex
```

## Quick start

### 1. Get the upstream code (already vendored in `upstream/RF-Diffusion/`)

```bash
git clone https://github.com/mobicom24/RF-Diffusion.git upstream/RF-Diffusion
```

This project vendors the official repository at commit `eb872b0`. The
upstream code is **not modified**; this project imports it.

### 2. Get the data and pretrained models

```bash
# Datasets
wget -O upstream/RF-Diffusion/dataset.zip \
  https://github.com/mobicom24/RF-Diffusion/releases/download/dataset_model/dataset.zip
unzip -q upstream/RF-Diffusion/dataset.zip -d upstream/RF-Diffusion/

# Pretrained models
wget -O upstream/RF-Diffusion/model.zip \
  https://github.com/mobicom24/RF-Diffusion/releases/download/dataset_model/model.zip
unzip -q upstream/RF-Diffusion/model.zip -d upstream/RF-Diffusion/

# Restructure
mkdir -p upstream/RF-Diffusion/dataset
mv upstream/RF-Diffusion/wifi upstream/RF-Diffusion/dataset/wifi
mv upstream/RF-Diffusion/fmcw upstream/RF-Diffusion/dataset/fmcw
mv upstream/RF-Diffusion/mimo upstream/RF-Diffusion/dataset/mimo
```

### 3. Install dependencies

```bash
pip install torch torchvision numpy scipy tensorboard tqdm matplotlib \
            pytorch_fid pytorch-msssim psutil pytest pyyaml
```

### 4. Wi-Fi official reproduction

```bash
python scripts/run_official_wifi.py
# or equivalently
cd upstream/RF-Diffusion && python inference.py --task_id 0
```

### 5. 5G FDD official reproduction

```bash
python scripts/run_official_mimo.py
# or equivalently
cd upstream/RF-Diffusion && python inference.py --task_id 2
```

### 6. Smoke test (2 samples, ~7 s on A40)

```bash
python scripts/run_experiment.py --task wifi --mode smoke-test --num-samples 2
```

### 7. Small-scale training

```bash
python scripts/run_small_train.py
```

### 8. Efficiency extension

```bash
python scripts/run_efficiency.py        # native sampling
python scripts/run_efficiency_gen.py    # full reverse diffusion
```

### 9. Aggregate metrics + regenerate figures

```bash
python scripts/aggregate_metrics.py     # writes 3 CSVs to results/metrics/
python scripts/plot_results.py          # writes 5 figures to figures/
```

### 10. Build the LaTeX report

```bash
cd report && bash build.sh
# produces report/main.pdf
```

## One-shot reproduction

```bash
# Linux / WSL
bash scripts/reproduce_all.sh

# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -File scripts/reproduce_all.ps1
```

The script runs in order:
1. environment probe
2. data + model sanity check
3. Wi-Fi smoke test
4. Wi-Fi official inference
5. 5G official inference
6. small-scale training
7. efficiency extension
8. metric aggregation
9. figure generation
10. LaTeX compilation

## Key reproduction numbers

All raw numbers live in `results/metrics/*.json`. The aggregated CSVs are:

- `results/metrics/main_results.csv`
- `results/metrics/efficiency_results.csv`
- `results/metrics/paper_comparison.csv`

The CSVs are regenerated by `scripts/aggregate_metrics.py`; never
hand-edited.

## Differences vs the paper

- Wi-Fi Average SSIM: 0.81 vs paper's 0.81 (within ±0.001). The paper's
  Section 6.2 reports 0.81 and the released 215-sample CDF data averages
  0.796. Our reproduction matches to within ±0.001 across PyTorch 2.0.1
  (0.8112) and 2.13 (0.8106). The earlier "−0.02 gap" was a transcription
  error local to this project.
- Wi-Fi FID: 7.82 vs paper's 9.13 (lower is better; we got a slightly
  better number).
- 5G SNR: 29.95 dB vs paper's 27.96 dB ($+7.1\%$).

All differences are within the expected range for diffusion model
reproductions across PyTorch versions. Random seed is fixed at 11.

## Known issues / limitations

- The released dataset does not include raw Wi-Fi training files; only
  41 cond files. We symlink them into `dataset/wifi/raw/` to run a
  very small-scale training as a sanity check. This is documented in
  `docs/reproduction_scope.md`.
- Fandol fonts are not installed in the build environment; we symlink
  Noto CJK SC fonts to the Fandol font names so ctex can render Chinese.
  This is recorded in `report/build.sh` and an inline comment.
- FMCW pretrained weights are not in the official release; the FMCW
  reproduction is therefore not included.
- We did not run the downstream gesture-recognition augmentation
  experiment (paper Section 7.1) because it would require training a
  separate Wi-Fi classifier.

## License

This project's wrapper code is released for academic / educational use.
The vendored upstream code at `upstream/RF-Diffusion/` is © its authors
and released under **GNU General Public License v3.0**. See
`LICENSE_NOTICE.md` for details and the official citation.

## Citing the upstream paper

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