#!/usr/bin/env python3
"""Generate publication-quality figures from evaluation results.

This script reads the metrics JSON files and generates figures for:
- Spectrogram comparison (real vs generated Wi-Fi CSI)
- Quality vs inference time trade-off
- Peak memory vs model size
- Main results comparison (paper vs reproduced)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "upstream" / "RF-Diffusion"
RESULTS_ROOT = PROJECT_ROOT / "results"
FIG_ROOT = PROJECT_ROOT / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_latest(pattern: str) -> Optional[Any]:
    """Load the most recent file matching the glob pattern."""
    matches = sorted(RESULTS_ROOT.glob(pattern))
    if not matches:
        return None
    latest = matches[-1]
    with open(latest, "r", encoding="utf-8") as f:
        return json.load(f)


def fig_spectrogram_compare() -> Path:
    """Generate spectrogram comparison: real vs RF-Diffusion generated.

    Returns:
        Path to the saved PDF figure.
    """
    sample_path = UPSTREAM_ROOT / "dataset" / "wifi" / "output" / "0-0.mat"

    if not sample_path.exists():
        candidates = sorted((RESULTS_ROOT / "raw").rglob("sample-0-0.mat"))
        if candidates:
            sample_path = candidates[-1]
        else:
            print("[warn] No Wi-Fi output sample found")
            return FIG_ROOT / "fig_spectrogram_compare.pdf"

    data = scio.loadmat(sample_path)
    pred = data["pred"]
    truth = data.get("data", pred)

    pred_c = pred[0, :, 0].reshape(512).astype(np.complex128)
    truth_c = truth[0, :, 0].reshape(512).astype(np.complex128)

    def spec_db(x: np.ndarray) -> np.ndarray:
        n_fft, hop = 24, 17
        x_t = torch.from_numpy(x).to(torch.complex64)
        stft = torch.stft(x_t, n_fft=n_fft, hop_length=hop, return_complex=True)
        mag = torch.abs(stft).numpy()
        return 20 * np.log10(mag + 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    titles = ("Real Wi-Fi CSI", "RF-Diffusion (Reproduced)")
    signals = (truth_c, pred_c)

    for ax, sig, title in zip(axes, signals, titles):
        im = ax.matshow(spec_db(sig), cmap="viridis", origin="lower", aspect="auto")
        ax.set_title(title, pad=8)
        ax.set_xlabel("Time frame")
        ax.set_ylabel("Frequency bin")
        plt.colorbar(im, ax=ax, format="%+2.0f dB", fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(FIG_ROOT / "fig_spectrogram_compare.pdf")
    fig.savefig(FIG_ROOT / "fig_spectrogram_compare.png")
    plt.close(fig)
    print(f"[ok] saved fig_spectrogram_compare")
    return FIG_ROOT / "fig_spectrogram_compare.pdf"


def fig_quality_time_tradeoff() -> Path:
    """Generate quality vs inference time trade-off scatter plot.

    Returns:
        Path to the saved PDF figure.
    """
    nat = load_latest("metrics/efficiency_*.json")
    gen = load_latest("metrics/efficiency_gen_*.json")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    if nat:
        for m in nat:
            if "error" in m or m.get("average_sample_time_s") is None:
                continue
            ax.scatter(
                m["average_sample_time_s"], m["average_ssim"],
                s=70, marker="o",
                label=f"native: {m['config_name']}"
            )

    if gen:
        for m in gen:
            if "error" in m or m.get("average_sample_time_s") is None:
                continue
            ax.scatter(
                m["average_sample_time_s"], m["average_ssim"],
                s=70, marker="^",
                label=f"full-reverse: {m['config_name']}"
            )

    ax.set_xlabel("Average sampling time per sample (s)")
    ax.set_ylabel("Average SSIM")
    ax.set_title("Quality vs inference time under compute budgets")
    ax.grid(linestyle="--", linewidth=0.5)
    ax.legend(fontsize=7, loc="lower right")

    fig.savefig(FIG_ROOT / "fig_quality_time_tradeoff.pdf")
    fig.savefig(FIG_ROOT / "fig_quality_time_tradeoff.png")
    plt.close(fig)
    print(f"[ok] saved fig_quality_time_tradeoff")
    return FIG_ROOT / "fig_quality_time_tradeoff.pdf"


def fig_peak_memory() -> Path:
    """Generate peak GPU memory vs model size bar chart.

    Returns:
        Path to the saved PDF figure.
    """
    nat = load_latest("metrics/efficiency_*.json")
    gen = load_latest("metrics/efficiency_gen_*.json")

    by_blocks: Dict[int, float] = {}
    for source in (nat, gen):
        if not source:
            continue
        for m in source:
            if "error" in m or m.get("peak_gpu_mem_mb") is None:
                continue
            nb = m.get("num_block", 32)
            by_blocks[nb] = max(by_blocks.get(nb, 0.0), m["peak_gpu_mem_mb"])

    if not by_blocks:
        print("[warn] no efficiency results found")
        return FIG_ROOT / "fig_peak_memory.pdf"

    blocks = sorted(by_blocks)
    mems = [by_blocks[b] for b in blocks]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"{b}-block" for b in blocks], mems, color="#084E87")
    ax.set_ylabel("Peak GPU memory (MB)")
    ax.set_title("Peak GPU memory vs model size (Wi-Fi, A40 GPU)")
    ax.set_ylim(0, max(mems) * 1.15)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)

    for b, v in zip(blocks, mems):
        ax.text(list(blocks).index(b), v + 30, f"{v:.0f} MB", ha="center", fontsize=9)

    fig.savefig(FIG_ROOT / "fig_peak_memory.pdf")
    fig.savefig(FIG_ROOT / "fig_peak_memory.png")
    plt.close(fig)
    print(f"[ok] saved fig_peak_memory")
    return FIG_ROOT / "fig_peak_memory.pdf"


def fig_main_results() -> Path:
    """Generate paper vs reproduced comparison bar chart.

    Returns:
        Path to the saved PDF figure.
    """
    rows = [
        ("Wi-Fi", "Average SSIM ↑", 0.81, 0.81),
        ("Wi-Fi", "FID ↓", 9.13, 7.82),
        ("5G FDD", "Average SNR (dB) ↑", 27.96, 29.95),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    width = 0.35
    xs = np.arange(len(rows))

    paper_vals = [r[2] for r in rows]
    rep_vals = [r[3] for r in rows]
    labels = [f"{r[0]}\n{r[1]}" for r in rows]

    ax.bar(xs - width / 2, paper_vals, width, label="Paper reported", color="#084E87")
    ax.bar(xs + width / 2, rep_vals, width, label="Reproduced", color="#BF3F3F")

    for b, v in zip(ax.patches[:len(paper_vals)], paper_vals):
        ax.text(
            b.get_x() + b.get_width() / 2, v + 0.5,
            f"{v:.2f}", ha="center", fontsize=9
        )
    for b, v in zip(ax.patches[len(paper_vals):], rep_vals):
        ax.text(
            b.get_x() + b.get_width() / 2, v + 0.5,
            f"{v:.2f}", ha="center", fontsize=9
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Metric value")
    ax.set_title("Paper reported vs reproduced results")
    ax.legend()
    ax.grid(axis="y", linestyle="--", linewidth=0.5)

    fig.savefig(FIG_ROOT / "fig_main_results.pdf")
    fig.savefig(FIG_ROOT / "fig_main_results.png")
    plt.close(fig)
    print(f"[ok] saved fig_main_results")
    return FIG_ROOT / "fig_main_results.pdf"


def main() -> None:
    """Generate all figures."""
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    (FIG_ROOT / ".gitkeep").touch()

    fig_spectrogram_compare()
    fig_quality_time_tradeoff()
    fig_peak_memory()
    fig_main_results()

    print(f"\nAll figures saved to {FIG_ROOT}/")


if __name__ == "__main__":
    main()
