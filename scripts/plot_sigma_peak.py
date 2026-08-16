"""Plot SSIM vs sigma curve with peak annotation."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# All data combined: sigma, blocks, iters, ssim_amp_mean, ssim_amp_std
# New peak search data (8-block/100-iter, mean of seeds 42,43)
peak_data = {
    "sigma": [10.0, 12.0, 16.0, 20.0, 32.0],
    "ssim_amp_mean": [],
    "ssim_amp_std": [],
}
# Manual aggregation from CSV
raw = {
    10.0: [(0.4653, 0.0211), (0.5193, 0.0300)],
    12.0: [(0.4714, 0.0232), (0.5366, 0.0242)],
    16.0: [(0.5244, 0.0267), (0.6087, 0.0233)],
    20.0: [(0.5488, 0.0213), (0.5354, 0.0238)],
    32.0: [(0.5314, 0.0233), (0.5046, 0.0267)],
}
for s in peak_data["sigma"]:
    means = [x[0] for x in raw[s]]
    stds  = [x[1] for x in raw[s]]
    peak_data["ssim_amp_mean"].append(np.mean(means))
    peak_data["ssim_amp_std"].append(np.std(means))  # std across seeds

# Previous extended data (4-block/100-iter, seed=11)
prev_data = {
    "sigma": [0.5, 1.0, 2.0, 4.0, 6.0, 8.0],
    "ssim_amp_mean": [0.1157, 0.1382, 0.1733, 0.2361, 0.2892, 0.3121],
    "ssim_amp_std":  [0.0099, 0.0115, 0.0131, 0.0116, 0.0138, 0.0168],
}

# Previous extended data (8-block/300-iter, seed=11)
prev_b8i300 = {
    "sigma": [4.0, 6.0, 8.0],
    "ssim_amp_mean": [0.4264, 0.4583, 0.4809],
    "ssim_amp_std":  [0.0190, 0.0194, 0.0187],
}

# Setup figure
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(8, 5))

# Plot 4-block/100-iter (seed=11)
ax.errorbar(prev_data["sigma"], prev_data["ssim_amp_mean"],
            yerr=prev_data["ssim_amp_std"],
            marker="o", markersize=6, linewidth=1.8,
            color="#2196F3", label="4-block / 100-iter (seed=11)",
            capsize=4, alpha=0.8)

# Plot 8-block/100-iter new peak data (mean±std over seeds)
ax.errorbar(peak_data["sigma"], peak_data["ssim_amp_mean"],
            yerr=peak_data["ssim_amp_std"],
            marker="s", markersize=7, linewidth=2.0,
            color="#FF5722", label="8-block / 100-iter (mean±std, seeds 42,43)",
            capsize=4, alpha=0.9)

# Plot 8-block/300-iter previous data
ax.errorbar(prev_b8i300["sigma"], prev_b8i300["ssim_amp_mean"],
            yerr=prev_b8i300["ssim_amp_std"],
            marker="^", markersize=6, linewidth=1.8,
            color="#4CAF50", label="8-block / 300-iter (seed=11)",
            capsize=4, alpha=0.8)

# Mark the peak point σ=16
peak_idx = peak_data["sigma"].index(16.0)
peak_x = 16.0
peak_y = peak_data["ssim_amp_mean"][peak_idx]
ax.axvline(x=peak_x, color="#FF5722", linestyle="--", alpha=0.5, linewidth=1.2)
ax.annotate(f"Peak\nσ=16.0\nSSIM={peak_y:.3f}",
            xy=(peak_x, peak_y),
            xytext=(peak_x + 4, peak_y + 0.01),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#FF5722", lw=1.2),
            color="#FF5722",
            ha="left")

# Mark the inflection region
ax.axvspan(8.0, 20.0, alpha=0.08, color="orange", label="Inflection region [8, 20]")

# Mark σ=8 baseline
ax.axvline(x=8.0, color="gray", linestyle=":", alpha=0.6, linewidth=1.2)
ax.text(8.2, 0.08, "σ=8 (prev. best)", fontsize=8, color="gray", rotation=90, va="bottom")

ax.set_xlabel("Gaussian kernel width σ")
ax.set_ylabel("Amplitude SSIM")
ax.set_title("SSIM vs σ: Peak Found at σ=16.0 (8-block / 100-iter)")
ax.set_xlim(-1, 35)
ax.set_ylim(0, 0.72)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", fontsize=9)

# Add text box with key finding
textstr = "Optimal σ = 16.0\nSSIM = 0.567 ± 0.042\n(8-block/100-iter, 2 seeds)"
props = dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8, edgecolor="orange")
ax.text(0.97, 0.35, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment="right", bbox=props)

plt.tight_layout()
out_path = "/data/zhaoshiqian/talents/talent-16/CCSE/RF-Diffusion-Reproduction/report/figures/fig_sigma_peak_verification.pdf"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

# Also save as PNG for preview
plt.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print("Saved PNG version.")
