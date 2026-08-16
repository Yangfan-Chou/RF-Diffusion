#!/usr/bin/env python3
"""
Plot training loss curves for Gaussian vs Scene-Adaptive (Rayleigh) kernel experiments.
Reads from:
  - Gaussian: results/metrics/small_train_20260804_234505_3a2c85.json  (200 iters, Gaussian kernel)
  - Adaptive: results/metrics/adaptive_kernel_*.json  (not yet available; falls back gracefully)
  - Rayleigh:  results/metrics/rayleigh_training_20260807_000721_745736.json (Rayleigh kernel, 100 iters)
Outputs:
  - report/figures/fig_loss_curve.pdf
  - report/figures/fig_loss_curve.png
"""

import json
import os
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE)
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "metrics")
OUT_DIR = os.path.join(REPO_ROOT, "report", "figures")

GAUSSIAN_JSON = os.path.join(RESULTS_DIR, "small_train_20260804_234505_3a2c85.json")
RAYLEIGH_JSON  = os.path.join(RESULTS_DIR, "rayleigh_training_20260807_000721_745736.json")
ADAPTIVE_GLOB  = os.path.join(RESULTS_DIR, "adaptive_kernel_*.json")

os.makedirs(OUT_DIR, exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def try_load(path):
    try:
        return load_json(path)
    except FileNotFoundError:
        return None


# ── load data ────────────────────────────────────────────────────────────────
gaussian_data = try_load(GAUSSIAN_JSON)
rayleigh_data = try_load(RAYLEIGH_JSON)

adaptive_files = sorted(glob.glob(ADAPTIVE_GLOB))
adaptive_data  = try_load(adaptive_files[0]) if adaptive_files else None

# ── extract loss curves ────────────────────────────────────────────────────────
def extract(data, label):
    if data is None:
        return None, None, None
    curve = data.get("training", {}).get("loss_curve", [])
    final = data.get("training", {}).get("final_loss", None)
    if final is None and curve:
        final = curve[-1]
    return curve, final, label


g_curve, g_final, _ = extract(gaussian_data, "Gaussian")
r_curve, r_final, _ = extract(rayleigh_data,  "Rayleigh")
a_curve, a_final, _ = extract(adaptive_data,   "Adaptive")

# Decide which dataset to show on the right subplot:
# Priority: adaptive > rayleigh (rayleigh demonstrates the Rayleigh-kernel failure case)
right_label  = "Adaptive" if a_curve else ("Rayleigh" if r_curve else None)
right_curve  = a_curve if a_curve else r_curve
right_final  = a_final if a_final else r_final
right_color  = "#2196F3" if right_label == "Adaptive" else "#E91E63"
right_legend = f"{right_label}, final={right_final:.4f}" if right_final is not None else right_label

# ── plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# Left: Gaussian
ax = axes[0]
if g_curve:
    iterations = list(range(len(g_curve)))
    ax.plot(iterations, g_curve, color="#4CAF50", lw=1.5, alpha=0.85)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Training Loss", fontsize=11)
    ax.set_title("(a) Gaussian Kernel", fontsize=12)
    ax.set_xlim(0, len(g_curve))
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(True, alpha=0.3)
    legend_str = f"Gaussian, final={g_final:.4f}" if g_final is not None else "Gaussian"
    ax.plot([], [], color="#4CAF50", lw=1.5, label=legend_str)
    ax.legend(fontsize=10)
else:
    ax.text(0.5, 0.5, "Gaussian data\nnot found", ha="center", va="center",
            transform=ax.transAxes, fontsize=11, color="gray")
    ax.set_title("(a) Gaussian Kernel", fontsize=12)

# Right: Adaptive or Rayleigh
ax = axes[1]
if right_curve:
    iterations = list(range(len(right_curve)))
    ax.plot(iterations, right_curve, color=right_color, lw=1.5, alpha=0.85)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_title(f"(b) {right_label} Kernel", fontsize=12)
    ax.set_xlim(0, len(right_curve))
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.plot([], [], color=right_color, lw=1.5, label=right_legend)
    ax.legend(fontsize=10)
else:
    ax.text(0.5, 0.5, f"{right_label} data\nnot found", ha="center", va="center",
            transform=ax.transAxes, fontsize=11, color="gray")
    ax.set_title(f"(b) {right_label} Kernel", fontsize=12)

fig.suptitle(
    "Training Loss: Gaussian vs Scene-Adaptive Kernel\n(4 blocks, hidden=64, 100–200 iters)",
    fontsize=13, y=1.02,
)

note = (
    "Note: adaptive_kernel_*.json not yet available. "
    "Right panel shows Rayleigh kernel result as reference."
    if not adaptive_data else ""
)
if note:
    fig.text(0.5, -0.04, note, ha="center", fontsize=8, color="gray",
              style="italic")

plt.tight_layout()

# ── save ──────────────────────────────────────────────────────────────────────
pdf_path = os.path.join(OUT_DIR, "fig_loss_curve.pdf")
png_path = os.path.join(OUT_DIR, "fig_loss_curve.png")
fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
fig.savefig(png_path, bbox_inches="tight", dpi=150)
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
