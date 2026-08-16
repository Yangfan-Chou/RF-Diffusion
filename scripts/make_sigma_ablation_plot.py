#!/usr/bin/env python3
"""
Sigma-scale ablation figure: SSIM bar chart + loss curves.
Reads the latest results/metrics/sigma_ablation_*.csv
"""
from __future__ import annotations

import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(BASE)
METRICS_DIR = os.path.join(REPO_ROOT, "results", "metrics")
OUT_DIR    = os.path.join(REPO_ROOT, "report", "figures")

os.makedirs(OUT_DIR, exist_ok=True)

# ── locate latest run ────────────────────────────────────────────────────────
json_files = sorted(glob.glob(os.path.join(METRICS_DIR, "sigma_ablation_*.json")))
csv_files  = sorted(glob.glob(os.path.join(METRICS_DIR, "sigma_ablation_*.csv")))

if not csv_files:
    raise FileNotFoundError("No sigma_ablation_*.csv found in results/metrics/")

LATEST_JSON = json_files[-1]
LATEST_CSV  = csv_files[-1]

print(f"Using JSON: {LATEST_JSON}")
print(f"Using CSV:  {LATEST_CSV}")

# ── load ──────────────────────────────────────────────────────────────────────
with open(LATEST_JSON) as f:
    metrics = json.load(f)

with open(LATEST_CSV) as f:
    rows = list(csv.DictReader(f))

results = metrics["results"]          # list of dicts with loss_curve, ssim_mean, etc.

# Sort by sigma_scale
results_sorted = sorted(results, key=lambda r: r["sigma_scale"])

sigma_scales  = [r["sigma_scale"]  for r in results_sorted]
sigma_labels  = [r["sigma_label"]  for r in results_sorted]
ssim_means    = [r["ssim_mean"]    for r in results_sorted]
ssim_stds     = [r["ssim_std"]     for r in results_sorted]
final_losses  = [r["final_loss"]   for r in results_sorted]
loss_curves   = [r["loss_curve"]   for r in results_sorted]

best_idx      = int(np.argmax(ssim_means))
best_sigma    = sigma_scales[best_idx]
best_ssim     = ssim_means[best_idx]
default_ssim  = ssim_means[1]   # sigma_scale=1.0 is at index 1
delta_pct     = (best_ssim - default_ssim) / default_ssim * 100

print(f"\nBest: sigma_scale={best_sigma}  SSIM={best_ssim:.4f}")
print(f"Delta vs default (sigma=1.0): {delta_pct:+.2f}%")

# ── figure ────────────────────────────────────────────────────────────────────
COLORS = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63"]
SIGMA_LABELS_SHORT = ["0.5\n(narrow)", "1.0\n(default)", "2.0\n(wide)", "4.0\n(very-wide)"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# ── (a) SSIM bar chart ────────────────────────────────────────────────────────
ax = axes[0]
x = np.arange(len(sigma_scales))
bars = ax.bar(x, ssim_means, yerr=ssim_stds,
              color=COLORS, edgecolor="white", linewidth=0.8,
              capsize=5, error_kw={"linewidth": 1.2}, alpha=0.88)

# Highlight best bar
bars[best_idx].set_edgecolor("#FFD700")
bars[best_idx].set_linewidth(2.5)

# Annotate bars
for i, (bar, ssim, sigma) in enumerate(zip(bars, ssim_means, sigma_scales)):
    ax.annotate(f"σ={sigma}\n{ssim:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 8), textcoords="offset points",
                ha="center", va="bottom", fontsize=9,
                fontweight="bold" if i == best_idx else "normal",
                color=COLORS[i])

ax.set_xticks(x)
ax.set_xticklabels(SIGMA_LABELS_SHORT, fontsize=9)
ax.set_xlabel("Gaussian kernel width scale (σ)", fontsize=11)
ax.set_ylabel("SSIM (amplitude, 41 samples)", fontsize=11)
ax.set_title("(a) SSIM vs Gaussian Kernel Width", fontsize=12, fontweight="bold")
ax.set_ylim(0, 0.38)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax.grid(axis="y", alpha=0.3)
ax.tick_params(axis="y", labelsize=9)

# ── (b) Loss curves ───────────────────────────────────────────────────────────
ax = axes[1]
for i, (curve, color, sigma, label) in enumerate(zip(
        loss_curves, COLORS, sigma_scales, sigma_labels)):
    linestyle = "--" if sigma == 1.0 else "-"
    lw = 2.0 if sigma == best_sigma else 1.3
    alpha = 0.9 if sigma == best_sigma else 0.7
    ax.plot(curve, color=color, lw=lw, linestyle=linestyle,
            alpha=alpha, label=f"σ={sigma} ({label}), final={final_losses[i]:.3f}")

ax.set_xlabel("Iteration", fontsize=11)
ax.set_ylabel("Training Loss (MSE)", fontsize=11)
ax.set_title("(b) Loss Curves (4-block, 100 iters)", fontsize=12, fontweight="bold")
ax.set_xlim(0, 100)
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="upper right")

fig.suptitle(
    "Gaussian Kernel Width Ablation (Wi-Fi, 4-block model, 100 iters)\n"
    f"Best: σ={best_sigma} (SSIM={best_ssim:.4f}, {delta_pct:+.1f}% vs default σ=1.0)",
    fontsize=12, y=1.02,
)
plt.tight_layout()

# ── save ──────────────────────────────────────────────────────────────────────
pdf_path = os.path.join(OUT_DIR, "fig_sigma_ablation.pdf")
png_path = os.path.join(OUT_DIR, "fig_sigma_ablation.png")
fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
fig.savefig(png_path, bbox_inches="tight", dpi=150)
print(f"\nSaved: {pdf_path}")
print(f"Saved: {png_path}")
