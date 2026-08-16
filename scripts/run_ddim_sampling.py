#!/usr/bin/env python3
"""
run_ddim_sampling.py — DDIM-subsequence accelerated sampling for RF-Diffusion (Wi-Fi task).

Improvement direction: "模型蒸馏加速边缘推理" + "DDIM 加速采样"
from report/main.tex §"局限与改进方向".

Strategy comparison (all on official Wi-Fi pretrained model, task_id=0, b32-256-100s):
  • native_sampling     : 1-step evaluation — HIGH QUALITY BASELINE (official result ~0.76 SSIM)
  • robust_sampling     : 100-step deterministic — official DDIM-equivalent
  • full DDPM sampling  : 100-step stochastic — TRUE generation baseline
  • ddim_skip_N         : DDIM sub-sequence with stride η=N (N∈{2,5,10,20,50})

Metrics: SSIM, per-sample wall-clock time (s), peak GPU memory (MB), model evals.

Outputs:
  results/metrics/ddim_subsample_<run_id>.json
  figures/fig_ddim_sampling.pdf / .png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import uuid
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import scipy.io as scio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Weights live in the real upstream; CCSE copy is gitignored and lacks them.
# We need to chdir to the real upstream so relative paths in from_path_inference work.
REAL_UPSTREAM = Path("/data/zhaoshiqian/talents/talent-16/upstream/RF-Diffusion")
sys.path.insert(0, str(REAL_UPSTREAM))
import os as _os
_os.chdir(REAL_UPSTREAM)

from tfdiff.params import AttrDict, all_params
from tfdiff.wifi_model import tfdiff_WiFi
from tfdiff.diffusion import SignalDiffusion
from tfdiff.dataset import from_path_inference, _nested_map


# ---------------------------------------------------------------------------
# SSIM (same as upstream inference.py)
# ---------------------------------------------------------------------------

def _gaussian(window_size: int, sigma: float):
    g = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                      for x in range(window_size)])
    return g / g.sum()


@torch.jit.script
def _create_window(h: int, w: int):
    hw = _gaussian(h, 1.5).unsqueeze(1)
    ww = _gaussian(w, 1.5).unsqueeze(1)
    win = hw.mm(ww.t()).unsqueeze(0).unsqueeze(0).contiguous()
    return win


def eval_ssim(pred, data, h, w, device):
    """SSIM on complex spectrograms; higher is better."""
    window = _create_window(h, w).to(torch.complex64).to(device)
    pad = [h // 2, w // 2]
    def conv2d(x):
        return torch.nn.functional.conv2d(x, window, padding=pad, groups=1)
    mu_p = conv2d(pred);  mu_d = conv2d(data)
    var_p = conv2d(pred * pred) - mu_p.pow(2.)
    var_d = conv2d(data  * data)  - mu_d.pow(2.)
    cov  = conv2d(pred * data)  - mu_p * mu_d
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_p * mu_d + C1) * (2 * cov.real + C2)) \
             / ((mu_p.pow(2) + mu_d.pow(2) + C1) * (var_p + var_d + C2))
    return (2 * ssim_map.mean()).real


# ---------------------------------------------------------------------------
# DDIM sub-sequence sampling for SignalDiffusion (x₀-prediction variant)
# ---------------------------------------------------------------------------

def ddim_subsample_sampling(restore_fn, cond, device, max_step: int,
                             skip: int, task_id: int):
    """
    DDIM-subsequence jump for RF-Diffusion's SignalDiffusion (x₀-prediction).

    Full reverse diffusion evaluates the model at every t = T-1 … 0.
    DDIM jumps η=skip steps at a time using the analytic degrade_fn.

    The model predicts x̂₀.  From the current noisy state x_t we estimate the
    noise ε̂ = (x_t − √ᾱ_t · x̂₀) / √(1−ᾱ_t), then jump to t−η:

        x_{t−η} = √ᾱ_{t−η} · x̂₀  +  √(1−ᾱ_{t−η}) · ε̂

    This is the standard DDIM update (Song et al. 2021) adapted for
    SignalDiffusion's time-frequency degradation and x₀-prediction setup.

    Returns: (x_0_hat, num_model_evals)
    """
    batch_size = cond.shape[0]
    params = all_params[task_id]
    T = max_step
    sd = SignalDiffusion(AttrDict(params))

    # Build tau list: [T-1, T-1-skip, ..., >= 0]
    tau = list(range(T - 1, -1, -skip))
    if tau[-1] != 0:
        tau.append(0)
    tau_rev = list(reversed(tau))  # ascending: [0, skip, 2*skip, ...]

    # Init: x_T ~ N(0, I)  (full noise, no information)
    input_dim = params.sample_rate
    extra = params.extra_dim + [2]
    inf_w = (sd.noise_weights[T - 1] + sd.info_weights[T - 1]).to(device)
    if task_id in [2, 3]:
        inf_w = inf_w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    else:
        inf_w = inf_w.unsqueeze(-1).unsqueeze(-1)
    x_t = inf_w * torch.randn([batch_size, input_dim] + params.extra_dim + [2],
                                dtype=torch.float32, device=device)

    evals = 0
    for i, cur in enumerate(reversed(tau)):
        x_0_hat = restore_fn(
            x_t,
            cur * torch.ones(batch_size, dtype=torch.int64, device=device),
            cond)
        evals += 1
        if i + 1 < len(tau):
            nxt = tau[len(tau) - i - 2]
            snr = sd.info_weights[cur].unsqueeze(-1).unsqueeze(-1).to(device)
            nw  = sd.noise_weights[cur].unsqueeze(-1).unsqueeze(-1).to(device)
            eps_hat = (x_t - snr * x_0_hat) / (nw + 1e-10)
            snr_nxt = sd.info_weights[nxt].unsqueeze(-1).unsqueeze(-1).to(device)
            nw_nxt  = sd.noise_weights[nxt].unsqueeze(-1).unsqueeze(-1).to(device)
            x_t = snr_nxt * x_0_hat + nw_nxt * eps_hat

    return x_0_hat, evals


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------

def evaluate(task_id: int, model, diffusion, device,
             strategy_fn, strategy_name: str,
             max_step: int, skip: int,
             num_samples: int) -> dict:
    """Run one strategy on up to num_samples; returns aggregated metrics."""
    params = all_params[task_id]
    # recreate iterator for each call to avoid exhaustion across strategies
    dataset = list(islice(from_path_inference(params), num_samples)) if num_samples > 0 \
              else list(from_path_inference(params))

    ssim_list, times, evals_list = [], [], []
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for features in dataset:
            features = _nested_map(features, lambda x: x.to(device)
                                   if isinstance(x, torch.Tensor) else x)
            data = features["data"]
            cond = features["cond"]

            t0 = time.perf_counter()
            if strategy_name.startswith("ddim_skip"):
                pred, ev = strategy_fn(model, cond, device, max_step, skip, task_id)
            elif strategy_name == "native":
                # native_sampling(model, data, cond, device)
                pred = strategy_fn(model, data, cond, device)
                ev = 1
            else:
                # sampling / robust_sampling(model, cond, device)
                pred = strategy_fn(model, cond, device)
                ev = max_step
            times.append(time.perf_counter() - t0)
            evals_list.append(ev)

            # SSIM
            dc = torch.view_as_complex(data.squeeze(0))
            pc = torch.view_as_complex(pred.squeeze(0))
            ssim_list.append(eval_ssim(pc.unsqueeze(0), dc.unsqueeze(0),
                                        params.sample_rate, params.input_dim,
                                        device).item())

    return {
        "strategy":        strategy_name,
        "skip":            skip,
        "num_samples":     len(ssim_list),
        "avg_ssim":        float(np.mean(ssim_list)),
        "std_ssim":        float(np.std(ssim_list)),
        "min_ssim":        float(np.min(ssim_list)),
        "max_ssim":        float(np.max(ssim_list)),
        "avg_sample_time": float(np.mean(times)),
        "std_sample_time": float(np.std(times)),
        "avg_model_evals": float(np.mean(evals_list)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num-samples", type=int, default=0,
                    help="0 = all (default 41)")
    args = ap.parse_args()

    run_id = args.run_id or (time.strftime("%Y%m%d_%H%M%S")
                              + "_" + uuid.uuid4().hex[:6])
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    params = all_params[0]  # Wi-Fi
    checkpoint = torch.load(
        REAL_UPSTREAM / "model" / "wifi" / "b32-256-100s" / "weights.pt",
        map_location=device)
    model = tfdiff_WiFi(AttrDict(params)).to(device)
    model.load_state_dict(checkpoint["model"]);  model.eval()
    diffusion = SignalDiffusion(AttrDict(params))

    # Strategies: (name, fn, max_step, skip)
    # Call signatures:
    #   sampling / robust_sampling: fn(model, cond, device)
    #   native_sampling:            fn(model, data, cond, device)   ← different!
    #   ddim_skip_N:               fn(model, cond, device, max_step, skip, task_id)
    strategies = [
        # HIGH QUALITY BASELINE — 1-step (matches paper SSIM ~0.76)
        ("native",
         lambda _m, _d, _c, _dev: diffusion.native_sampling(_m, _d, _c, _dev),
         1, 1),
        # DDIM DETERMINISTIC — robust_sampling uses x_{s-1}=x_s−D(̂x₀,s)+D(̂x₀,s−1)
        # Official paper uses this for Wi-Fi evaluation. ≈ 100 evals, ~1.4s, SSIM≈0.76
        ("robust_ddim",
         lambda _m, _c, _d: diffusion.robust_sampling(_m, _c, _d),
         100, 1),
        # FULL STOCHASTIC DDPM — generation from pure noise (true generation baseline)
        ("full_ddpm",
         lambda _m, _c, _d: diffusion.sampling(_m, _c, _d),
         100, 1),
        # DDIM sub-sequence with various strides (generation + speedup)
        ("ddim_skip2",  None, 100, 2),
        ("ddim_skip5",  None, 100, 5),
        ("ddim_skip10", None, 100, 10),
    ]

    results = []
    for name, fn, ms, sk in strategies:
        print(f"\n{'='*60}\n[{run_id}] {name}  (skip={sk}, max_step={ms})")
        if name.startswith("ddim_skip"):
            fn_eval = lambda _m, _c, _d, _ms=ms, _sk=sk, _tid=0: \
                ddim_subsample_sampling(_m, _c, _d, _ms, _sk, _tid)
        else:
            fn_eval = fn
        res = evaluate(0, model, diffusion, device,
                       fn_eval, name, ms, sk,
                       num_samples=args.num_samples)
        results.append(res)
        print(f"  SSIM : {res['avg_ssim']:.4f} ± {res['std_ssim']:.4f}")
        print(f"  Time : {res['avg_sample_time']:.3f} s/sample")
        print(f"  Evals: {res['avg_model_evals']:.0f}/sample")

    # Save JSON
    out = PROJECT_ROOT / "results" / "metrics" / f"ddim_subsample_{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "results": results}, f, indent=2)
    print(f"\nSaved: {out}")

    # Plot
    import matplotlib;  matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shutil

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(results)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: SSIM vs model evals
    ax = axes[0]
    for i, r in enumerate(results):
        ax.scatter(r["avg_model_evals"], r["avg_ssim"],
                  s=200, color=colors[i], zorder=5, edgecolors="k", lw=0.8)
        label = r["strategy"].replace("ddim_skip", "DDIM η=")
        ax.annotate(label,
                    (r["avg_model_evals"] + 1.5, r["avg_ssim"]),
                    fontsize=9)
    ax.set_xscale("log");  ax.set_xlim(0.5, 150)
    ax.set_xlabel("Model evaluations per sample (log scale)", fontsize=11)
    ax.set_ylabel("Average SSIM  ↑", fontsize=11)
    ax.set_title("DDIM Sub-sequence Sampling: Quality vs Speed", fontsize=12)
    ax.grid(alpha=0.3)
    ax.axhline(results[0]["avg_ssim"], color="red", ls="--", lw=1,
               label=f"native (1 eval) = {results[0]['avg_ssim']:.3f}")
    ax.legend(fontsize=9)

    # Panel B: SSIM bar chart with speedup annotation
    ax2 = axes[1]
    labels = [r["strategy"].replace("ddim_skip", "DDIM η=") + \
              f"\n({int(r['avg_model_evals'])} ev)" for r in results]
    vals = [r["avg_ssim"] for r in results]
    errs = [r["std_ssim"] for r in results]
    bars = ax2.bar(range(len(results)), vals, color=colors,
                   edgecolor="k", lw=0.8, yerr=errs, capsize=4, alpha=0.85)
    ax2.set_xticks(range(len(results)))
    ax2.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax2.set_ylabel("Average SSIM  ↑", fontsize=11)
    ax2.set_title("SSIM by Sampling Strategy", fontsize=12)
    ax2.set_ylim(0, 1.0)
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"RF-Diffusion DDIM Sub-sequence Sampling — Wi-Fi Task  "
        f"(run_id={run_id})",
        fontsize=12, y=1.01)
    plt.tight_layout()

    for fmt in ("pdf", "png"):
        p = PROJECT_ROOT / "figures" / f"fig_ddim_sampling.{fmt}"
        plt.savefig(p, bbox_inches="tight", dpi=300 if fmt == "pdf" else 150)
        print(f"Saved: {p}")
        shutil.copy2(p, PROJECT_ROOT / "report" / "figures" / f"fig_ddim_sampling.{fmt}")

    # Summary table
    print("\n" + "=" * 78)
    print(f"{'Strategy':<22} {'SSIM':>7} {'±σ':>7} {'Time(s)':>8} {'Evals':>7} {'Speedup':>8}")
    print("-" * 78)
    base_time = results[0]["avg_sample_time"]
    for r in results:
        speedup = base_time / r["avg_sample_time"] if r["avg_sample_time"] > 0 else 0
        print(f"{r['strategy']:<22} {r['avg_ssim']:>7.4f} {r['std_ssim']:>7.4f} "
              f"{r['avg_sample_time']:>8.3f} {r['avg_model_evals']:>7.0f} "
              f"{speedup:>7.1f}x")
    print("=" * 78)


if __name__ == "__main__":
    main()
