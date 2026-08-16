#!/usr/bin/env python3
"""Extended sigma ablation plots: SSIM vs sigma with original + new results.
Combines the original 4-sigma results with the new extended-range results.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BASE      = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE
METRICS_DIR = REPO_ROOT / "results" / "metrics"
OUT_DIR    = REPO_ROOT / "report" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load extended results ──────────────────────────────────────────────────────
import glob
ext_json = sorted(glob.glob(str(METRICS_DIR / "sigma_extended_*.json")))[-1]
ext_csv  = sorted(glob.glob(str(METRICS_DIR / "sigma_extended_*.csv")))[-1]
print(f"Using: {ext_json}")

with open(ext_json) as f:
    ext_metrics = json.load(f)
import csv
with open(ext_csv) as f:
    ext_rows = list(csv.DictReader(f))

# ── Load original sigma ablation results ──────────────────────────────────────
orig_csvs = sorted(glob.glob(str(METRICS_DIR / "sigma_ablation_*.csv")))
orig_rows = list(csv.DictReader(open(orig_csvs[-1])))
# Original: sigma_scale,label,final_loss,ssim_mean,ssim_std,n_samples,delta_vs_default,config

# ── Build data ─────────────────────────────────────────────────────────────────
# Original 4-sigma (4-block, 100-iter)
orig_sigma   = [float(r["sigma_scale"]) for r in orig_rows]
orig_ssim_m = [float(r["ssim_mean"]) for r in orig_rows]
orig_ssim_s = [float(r["ssim_std"]) for r in orig_rows]

# Extended Phase 1: 4-block / 100-iter
phase1 = [r for r in ext_rows if r["blocks"] == "4" and r["iters"] == "100"]
phase1_sigma = sorted([float(r["sigma"]) for r in phase1])
phase1_ssim  = {float(r["sigma"]): float(r["ssim_amp_mean"]) for r in phase1}
phase1_err   = {float(r["sigma"]): float(r["ssim_amp_std"]) for r in phase1}

# Extended Phase 2: grouped by (blocks, iters)
configs = [
    ("4-block / 100-iter (original)", "4", "100"),
    ("8-block / 300-iter (extended)", "8", "300"),
    ("16-block / 300-iter",           "16", "300"),
]

# Color scheme
COLORS = {
    "4-block / 100-iter (original)": "#2196F3",
    "8-block / 300-iter (extended)":  "#FF9800",
    "16-block / 300-iter":            "#E91E63",
}
MARKERS = {
    "4-block / 100-iter (original)": "o",
    "8-block / 300-iter (extended)":  "s",
    "16-block / 300-iter":           "^",
}

# ── Figure 1: Extended sigma range with 3 configs ──────────────────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(13, 4.5))

# (a) SSIM vs sigma, all configs
ax = axes1[0]
for label, blocks, iters in configs:
    rows = [r for r in ext_rows if r["blocks"] == blocks and r["iters"] == iters]
    sigmas = sorted([float(r["sigma"]) for r in rows])
    means  = [float(next(r["ssim_amp_mean"] for r in rows if r["sigma"] == str(s)))
               for s in sigmas]
    stds   = [float(next(r["ssim_amp_std"]  for r in rows if r["sigma"] == str(s)))
               for s in sigmas]
    ax.errorbar(sigmas, means, yerr=stds,
                color=COLORS[label], marker=MARKERS[label],
                linewidth=2.0, markersize=7, capsize=4, label=label)

# Mark best overall
best_row = max(ext_rows, key=lambda r: float(r["ssim_amp_mean"]))
best_sigma = float(best_row["sigma"])
best_ssim  = float(best_row["ssim_amp_mean"])
best_blocks = best_row["blocks"]
best_iters  = best_row["iters"]
ax.axvline(best_sigma, color="gold", linestyle="--", alpha=0.7, linewidth=1.5)
ax.scatter([best_sigma], [best_ssim], color="gold", s=120, zorder=5,
           edgecolors="black", linewidth=1.5, marker="*")
ax.annotate(f"Best: $\\sigma$={best_sigma}, {best_blocks}-block/{best_iters}-iter\nSSIM={best_ssim:.4f}",
            xy=(best_sigma, best_ssim), xytext=(best_sigma + 1.5, best_ssim - 0.03),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=9, color="darkgoldenrod", fontweight="bold")

ax.set_xlabel("Gaussian Kernel Width $\\sigma$", fontsize=11)
ax.set_ylabel("Amplitude SSIM (41 Wi-Fi samples)", fontsize=11)
ax.set_title("(a) SSIM vs $\\sigma$: Extended Range + Multi-Config Comparison", fontsize=12, fontweight="bold")
ax.set_xlim(-0.3, 16.8)
ax.set_ylim(0, 0.58)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

# Default baseline: sigma=1.0, 4-block/100-iter
default_baseline = float(next(r["ssim_amp_mean"] for r in ext_rows
                               if r["sigma"] == "1.0" and r["blocks"] == "4" and r["iters"] == "100"))

# (b) SSIM gain (%) vs sigma relative to default (sigma=1.0, same config)
ax = axes1[1]
for label, blocks, iters in configs:
    rows = [r for r in ext_rows if r["blocks"] == blocks and r["iters"] == iters]
    sigmas = sorted([float(r["sigma"]) for r in rows])
    # Use the global default baseline for fair comparison
    gains = [(float(next(r["ssim_amp_mean"] for r in rows if r["sigma"] == str(s)))
              - default_baseline) / default_baseline * 100
             for s in sigmas]
    ax.plot(sigmas, gains,
            color=COLORS[label], marker=MARKERS[label],
            linewidth=2.0, markersize=7, label=label)
    # Mark peak
    peak_s = sigmas[np.argmax(gains)]
    peak_g = max(gains)
    ax.scatter([peak_s], [peak_g], color=COLORS[label], s=80, zorder=5,
               edgecolors="white", linewidth=1.5)

ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Gaussian Kernel Width $\\sigma$", fontsize=11)
ax.set_ylabel("Gain vs $\\sigma=1.0$ (%)", fontsize=11)
ax.set_title("(b) Relative Improvement vs $\\sigma$", fontsize=12, fontweight="bold")
ax.set_xlim(-0.3, 16.8)
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

fig1.suptitle(
    f"Extended Gaussian Kernel Width Ablation — Best: $\\sigma$={best_sigma} ({best_blocks}-block/{best_iters}-iter) "
    f"SSIM={best_ssim:.4f} ({((best_ssim-0.1382)/0.1382*100):+.1f}% vs paper $\\sigma$=1.0)",
    fontsize=11, y=1.03,
)
plt.tight_layout()
fig1.savefig(OUT_DIR / "fig_sigma_extended.pdf", bbox_inches="tight", dpi=300)
fig1.savefig(OUT_DIR / "fig_sigma_extended.png", bbox_inches="tight", dpi=150)
print(f"Saved: {OUT_DIR / 'fig_sigma_extended.pdf'}")

# ── Figure 2: Loss curves for key configs ─────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4.5))

# Loss curves for phase1 configs
ax = axes2[0]
# Load loss curves from JSON
results = ext_metrics["results"]
phase1_curves = {r["sigma_scale"]: r["loss_curve"] for r in results
                  if r["num_blocks"] == 4 and r["max_iter"] == 100}

SIGMA_COLORS = {0.5: "#4CAF50", 1.0: "#2196F3", 2.0: "#FF9800",
                4.0: "#E91E63", 6.0: "#9C27B0", 8.0: "#795548"}
for sigma in sorted(phase1_curves.keys()):
    curve = phase1_curves[sigma]
    ax.plot(curve, color=SIGMA_COLORS.get(sigma, "gray"),
            linewidth=1.5, alpha=0.8, label=f"σ={sigma}")

ax.set_xlabel("Iteration", fontsize=11)
ax.set_ylabel("Training Loss (MSE)", fontsize=11)
ax.set_title("(a) 4-block / 100-iter Loss Curves", fontsize=12, fontweight="bold")
ax.set_xlim(0, 100)
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

# Loss curves for phase2 configs (8-block, 300-iter)
ax = axes2[1]
phase2_curves = {r["sigma_scale"]: r["loss_curve"] for r in results
                 if r["num_blocks"] == 8 and r["max_iter"] == 300}
for sigma in sorted(phase2_curves.keys()):
    curve = phase2_curves[sigma]
    ax.plot(curve, color=SIGMA_COLORS.get(sigma, "gray"),
            linewidth=1.8, alpha=0.85, label=f"σ={sigma}")

ax.set_xlabel("Iteration", fontsize=11)
ax.set_ylabel("Training Loss (MSE)", fontsize=11)
ax.set_title("(b) 8-block / 300-iter Loss Curves", fontsize=12, fontweight="bold")
ax.set_xlim(0, 300)
ax.set_ylim(0, 0.85)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

fig2.suptitle("Extended Sigma Ablation — Training Loss Curves", fontsize=12, y=1.03)
plt.tight_layout()
fig2.savefig(OUT_DIR / "fig_sigma_extended_loss.pdf", bbox_inches="tight", dpi=300)
fig2.savefig(OUT_DIR / "fig_sigma_extended_loss.png", bbox_inches="tight", dpi=150)
print(f"Saved: {OUT_DIR / 'fig_sigma_extended_loss.pdf'}")

# ── Figure 3: Phase1 original vs extended comparison (bar chart) ──────────────
fig3, ax3 = plt.subplots(figsize=(10, 5))

# Original 4-sigma vs Extended Phase 1: only compare overlapping sigmas
orig_data = {float(r["sigma_scale"]): (float(r["ssim_mean"]), float(r["ssim_std"]))
             for r in orig_rows}
ext_data  = {float(r["sigma"]): (float(r["ssim_amp_mean"]), float(r["ssim_amp_std"]))
             for r in phase1}

all_sigma = sorted(set(list(orig_data.keys()) + list(ext_data.keys())))
x = np.arange(len(all_sigma))
width = 0.35

orig_vals = [orig_data.get(s, (None, None)) for s in all_sigma]
ext_vals  = [ext_data.get(s,  (None, None)) for s in all_sigma]

# Only plot where both exist
overlap_sigma = [s for s in all_sigma if orig_data.get(s, (None, None))[0] is not None
                  and ext_data.get(s, (None, None))[0] is not None]
overlap_x = [all_sigma.index(s) for s in overlap_sigma]
orig_means = [orig_data[s][0] for s in overlap_sigma]
orig_stds  = [orig_data[s][1] for s in overlap_sigma]
ext_means  = [ext_data[s][0] for s in overlap_sigma]
ext_stds   = [ext_data[s][1] for s in overlap_sigma]

labels = [f"$\\sigma$={s}" for s in overlap_sigma]

b1 = ax3.bar([xi - width/2 for xi in overlap_x], orig_means, width, yerr=orig_stds,
              color="#2196F3", edgecolor="white", linewidth=0.8,
              capsize=4, label="Original ablation (4 values)", alpha=0.85)
b2 = ax3.bar([xi + width/2 for xi in overlap_x], ext_means,  width, yerr=ext_stds,
              color="#FF9800", edgecolor="white", linewidth=0.8,
              capsize=4, label="Extended range (6 values)", alpha=0.85)

# Annotate
for xi, o, e in zip(overlap_x, orig_means, ext_means):
    ax3.annotate(f"{o:.3f}", xy=(xi - width/2, o), xytext=(0, 5),
                 textcoords="offset points", ha="center", va="bottom",
                 fontsize=8, color="#2196F3", fontweight="bold")
    ax3.annotate(f"{e:.3f}", xy=(xi + width/2, e), xytext=(0, 5),
                 textcoords="offset points", ha="center", va="bottom",
                 fontsize=8, color="#FF9800", fontweight="bold")

ax3.set_xticks(overlap_x)
ax3.set_xticklabels(labels, fontsize=10)
ax3.set_xlabel("Gaussian Kernel Width $\\sigma$", fontsize=11)
ax3.set_ylabel("Amplitude SSIM", fontsize=11)
ax3.set_title("Original Ablation ($\\sigma\\in\\{0.5,1,2,4\\}$) vs Extended ($\\sigma\\in\\{0.5,1,2,4,6,8\\}$)\n"
              "4-block / 100-iter config", fontsize=12, fontweight="bold")
ax3.set_ylim(0, 0.38)
ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax3.grid(axis="y", alpha=0.3)
ax3.legend(fontsize=9)
plt.tight_layout()
fig3.savefig(OUT_DIR / "fig_sigma_extended_compare.pdf", bbox_inches="tight", dpi=300)
fig3.savefig(OUT_DIR / "fig_sigma_extended_compare.png", bbox_inches="tight", dpi=150)
print(f"Saved: {OUT_DIR / 'fig_sigma_extended_compare.pdf'}")

print("\n=== SUMMARY ===")
print(f"Best overall: σ={best_sigma}, {best_blocks}-block/{best_iters}-iter")
print(f"  SSIM_amp = {best_ssim:.4f}")
print(f"  SSIM_complex = {float(best_row['ssim_complex_mean']):.4f}")
print(f"  vs default σ=1.0/4-block/100-iter: {((best_ssim-0.1382)/0.1382*100):+.1f}%")
print(f"\nPhase 1 (4-block/100-iter):")
for s in sorted(phase1_ssim.keys()):
    delta = (phase1_ssim[s] - phase1_ssim[1.0]) / phase1_ssim[1.0] * 100
    print(f"  σ={s}: SSIM={phase1_ssim[s]:.4f} ({delta:+.1f}%)")
print(f"\nPhase 2 (8-block/300-iter):")
for label, blocks, iters in [configs[1]]:
    rows8 = [r for r in ext_rows if r["blocks"] == blocks and r["iters"] == iters]
    for s in sorted(set(float(r["sigma"]) for r in rows8)):
        ss = float(next(r["ssim_amp_mean"] for r in rows8 if r["sigma"] == str(s)))
        print(f"  σ={s}: SSIM={ss:.4f}")
print("\nAll figures saved to:", OUT_DIR)
