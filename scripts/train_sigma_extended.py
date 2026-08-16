"""Extended sigma ablation: wider sigma range, more blocks, more iterations.

Extends the original ablation from:
  sigma ∈ {0.5, 1.0, 2.0, 4.0}  (4-block, 100-iter)
to:
  sigma ∈ {0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0}
  blocks ∈ {4, 8, 16}
  iters ∈ {100, 300, 500}

Each config runs 1–3 seeds, outputs amplitude SSIM + complex SSIM.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "upstream" / "RF-Diffusion"
RESULTS_ROOT  = PROJECT_ROOT / "results"
sys.path.insert(0, str(UPSTREAM_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy.io as scio
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from src.config import get_logger, peak_gpu_memory_mb, reset_peak_memory, set_seed
from tfdiff.dataset import from_path
from tfdiff.params import AttrDict, all_params
from tfdiff.wifi_model import tfdiff_WiFi
from tfdiff.diffusion import SignalDiffusion

LOGGER = get_logger("rfdiff.sigma_extended")

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _resolve_wifi_dirs():
    candidates = [
        UPSTREAM_ROOT / "dataset" / "wifi",
        Path("/data/zhaoshiqian/talents/talent-16/upstream/RF-Diffusion/dataset/wifi"),
    ]
    for root in candidates:
        cond = root / "cond"
        raw  = root / "raw"
        if cond.exists() and list(cond.glob("user*.mat")):
            return cond, raw
    raise FileNotFoundError("Wi-Fi data not found")


# ---------------------------------------------------------------------------
# SignalDiffusion subclass with scaled Gaussian kernel variance
# ---------------------------------------------------------------------------
class ScaledGaussianDiffusion(SignalDiffusion):
    """SignalDiffusion with the Gaussian kernel variance scaled by `sigma_scale`.

    sigma_scale > 1 → wider kernel (more frequency-domain spreading)
    sigma_scale < 1 → narrower kernel (less spreading, closer to multipath)
    """

    def __init__(self, params, sigma_scale: float = 1.0):
        self._sigma_scale = float(sigma_scale)
        super().__init__(params)
        self._rescale_kernels()

    def _rescale_kernels(self):
        n_freq = self.input_dim  # = 512
        blur_std = 5.0 * self._sigma_scale  # sigma_scale=1.0 → blur_std=5 (paper-default)

        def _make_kernel() -> torch.Tensor:
            samples = torch.arange(n_freq, dtype=torch.float32)
            center = n_freq // 2
            g = torch.exp(-((samples - center) ** 2) / (2 * blur_std ** 2))
            base = (n_freq * g / g.sum()).float()
            return base.unsqueeze(0).expand(self.params.max_step, -1)

        self.gaussian_kernel     = _make_kernel()
        self.gaussian_kernel_bar = _make_kernel()
        self.info_weights = (
            self.gaussian_kernel_bar
            * torch.sqrt(self.alpha_bar).unsqueeze(-1)
        )
        self.noise_weights = self.get_noise_weights()


# ---------------------------------------------------------------------------
# Official SSIM (matches upstream inference.py)
# ---------------------------------------------------------------------------
@torch.jit.script
def _gaussian_window(ws: int, sigma: float):
    g = torch.tensor([math.exp(-(x - ws // 2) ** 2 / float(2 * sigma ** 2))
                      for x in range(ws)])
    return g / g.sum()


def _make_window(h: int, w: int):
    hw = _gaussian_window(h, 1.5).unsqueeze(1)
    ww = _gaussian_window(w, 1.5).unsqueeze(1)
    return (hw.mm(ww.t()).unsqueeze(0).unsqueeze(0)
             .expand(1, 1, h, w).contiguous())


def eval_ssim(pred, data, H, W, device):
    window = _make_window(H, W).to(device)
    pad = [H // 2, W // 2]
    mu_p  = F.conv2d(pred,  window, padding=pad, groups=1)
    mu_d  = F.conv2d(data,  window, padding=pad, groups=1)
    mu_p2 = mu_p.pow(2.0)
    mu_d2 = mu_d.pow(2.0)
    mu_pd = mu_p * mu_d
    var_p = F.conv2d(pred * pred, window, padding=pad, groups=1) - mu_p2
    var_d = F.conv2d(data * data, window, padding=pad, groups=1) - mu_d2
    cov   = F.conv2d(pred * data,  window, padding=pad, groups=1) - mu_pd
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_p * mu_d + C1) * (2 * cov.real + C2)) / \
               ((mu_p2 + mu_d2 + C1) * (var_p + var_d + C2))
    return float((2 * ssim_map.mean().real).item())


# ---------------------------------------------------------------------------
# Train one model with given sigma, blocks, hidden, iters
# ---------------------------------------------------------------------------
def train_and_eval(sigma_scale: float,
                   num_blocks: int,
                   hidden_dim: int,
                   max_iter: int,
                   seed: int,
                   params_base: AttrDict,
                   cond_files: list,
                   train_raw: Path,
                   cond_dir: Path,
                   device: torch.device) -> dict:
    """Train one model with scaled Gaussian kernel, evaluate SSIM."""

    set_seed(seed)
    run_tag = f"sigma{sigma_scale}_b{num_blocks}_h{hidden_dim}_i{max_iter}_s{seed}"
    ckpt_dir = RESULTS_ROOT / "raw" / f"sigma_extended_{run_tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("─── Training sigma=%.1f blocks=%d hidden=%d iters=%d seed=%d ───",
                sigma_scale, num_blocks, hidden_dim, max_iter, seed)

    params_i = AttrDict(dict(params_base))
    params_i.data_dir = [str(train_raw)]
    params_i.cond_dir = [str(cond_dir)]
    params_i.batch_size = 4
    params_i.learning_rate = 1e-3
    params_i.max_iter = max_iter
    params_i.num_block = num_blocks
    params_i.hidden_dim = hidden_dim
    params_i.embed_dim = hidden_dim
    params_i.model_dir = str(ckpt_dir)

    dataset   = from_path(AttrDict(params_i))
    model     = tfdiff_WiFi(AttrDict(params_i)).to(device)
    optim     = AdamW(model.parameters(), lr=params_i.learning_rate)
    diffusion = ScaledGaussianDiffusion(AttrDict(params_i), sigma_scale=sigma_scale)

    losses: list[float] = []
    reset_peak_memory(device)
    t0 = time.perf_counter()
    iter_idx = 0

    for epoch in range(999):
        for batch in dataset:
            data = batch["data"].to(device)
            cond = batch["cond"].to(device)
            B = data.shape[0]
            t  = torch.randint(0, params_i.max_step, (B,),
                               dtype=torch.int64, device=device)
            x_t   = diffusion.degrade_fn(data, t.cpu(), params_i.task_id)
            x_hat = model(x_t, t, cond)
            loss  = torch.mean((x_hat - data) ** 2)
            optim.zero_grad(); loss.backward(); optim.step()
            losses.append(loss.item())
            iter_idx += 1
            if iter_idx % 50 == 0:
                LOGGER.info("  iter=%d loss=%.6f", iter_idx, loss.item())
            if iter_idx >= params_i.max_iter:
                break
        if iter_idx >= params_i.max_iter:
            break

    train_time = time.perf_counter() - t0
    LOGGER.info("  Training done: %d iters in %.1fs, final loss=%.6f",
                iter_idx, train_time, losses[-1])

    # Save checkpoint
    ckpt_path = ckpt_dir / "weights.pt"
    torch.save({"model": model.state_dict(), "params": dict(params_i),
                "sigma_scale": sigma_scale}, ckpt_path)

    # Inference on all 41 test samples
    ssim_amp_list: list[float] = []
    ssim_complex_list: list[float] = []
    model.eval()
    with torch.no_grad():
        for p in cond_files:
            mat  = scio.loadmat(str(p), verify_compressed_data_integrity=False)
            feat = mat["feature"]
            cond = torch.from_numpy(mat["cond"]).to(torch.complex64).squeeze(0)

            data_view = torch.view_as_real(torch.from_numpy(feat)).permute(1, 2, 0)
            down  = F.interpolate(data_view, 512, mode="nearest-exact")
            norm  = (down - down.mean()) / down.std()
            d_norm = norm.permute(2, 0, 1).contiguous()

            d_in = d_norm.unsqueeze(0).to(device)
            c_in = torch.view_as_real(cond).unsqueeze(0).to(device)

            t_max = params_i.max_step - 1
            x_s   = diffusion.degrade_fn(
                d_in,
                t_max * torch.ones(1, dtype=torch.int64),
                params_i.task_id,
            ).float()
            x_hat = model(
                x_s,
                t_max * torch.ones(1, dtype=torch.int64, device=device),
                c_in,
            )
            d_c = torch.view_as_complex(d_in.squeeze(0).contiguous())
            p_c = torch.view_as_complex(x_hat.squeeze(0).contiguous())
            d_m = d_c.abs().float(); p_m = p_c.abs().float()

            # Amplitude SSIM (real-valued magnitude)
            cur_amp = eval_ssim(p_m.unsqueeze(0).unsqueeze(0),
                                d_m.unsqueeze(0).unsqueeze(0),
                                512, 90, device)

            # Complex SSIM: stack real and imaginary along height dim
            # view_as_real returns float64 by default; cast to float32
            d_rv = torch.view_as_real(d_c).float()   # [512, 90, 2]
            p_rv = torch.view_as_real(p_c).float()
            # Stack along height: [2, 512, 90] -> unsqueeze -> [1, 1, 1024, 90]
            d_stacked = torch.cat([d_rv[:, :, 0], d_rv[:, :, 1]], dim=0).unsqueeze(0).unsqueeze(0)
            p_stacked = torch.cat([p_rv[:, :, 0], p_rv[:, :, 1]], dim=0).unsqueeze(0).unsqueeze(0)
            cur_complex = eval_ssim(p_stacked, d_stacked, 1024, 90, device)
            ssim_amp_list.append(cur_amp)
            ssim_complex_list.append(cur_complex)

    amp_mean = float(np.mean(ssim_amp_list))
    amp_std  = float(np.std(ssim_amp_list))
    complex_mean = float(np.mean(ssim_complex_list))
    complex_std  = float(np.std(ssim_complex_list))
    LOGGER.info("  SSIM amp:  %.4f ± %.4f", amp_mean, amp_std)
    LOGGER.info("  SSIM comp: %.4f ± %.4f", complex_mean, complex_std)

    return {
        "sigma_scale": sigma_scale,
        "num_blocks": num_blocks,
        "hidden_dim": hidden_dim,
        "max_iter": max_iter,
        "seed": seed,
        "final_loss": float(losses[-1]),
        "loss_curve": losses,
        "train_time_s": float(train_time),
        "peak_gpu_mem_mb": peak_gpu_memory_mb(device),
        "ssim_amp_mean": amp_mean,
        "ssim_amp_std":  amp_std,
        "ssim_complex_mean": complex_mean,
        "ssim_complex_std":  complex_std,
        "n_samples": len(ssim_amp_list),
        "ckpt_dir": str(ckpt_dir),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    run_id  = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    log_dir = RESULTS_ROOT / "logs" / f"sigma_extended_{run_id}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Attach file handler to logger
    fh = logging.FileHandler(log_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger("rfdiff.sigma_extended").addHandler(fh)

    set_seed(11)
    cond_dir, raw_dir = _resolve_wifi_dirs()
    cond_files = sorted(cond_dir.glob("user*.mat"))
    LOGGER.info("Found %d Wi-Fi samples", len(cond_files))

    # Symlink raw -> cond for training
    train_raw = raw_dir
    train_raw.mkdir(exist_ok=True)
    for src in cond_files:
        dst = train_raw / src.name
        if not dst.exists():
            try:
                dst.symlink_to(src)
            except OSError:
                shutil.copy(src, dst)

    # Build base params
    params_base = AttrDict(dict(all_params[0]))
    params_base.task_id     = 0
    params_base.sample_rate = 512
    params_base.input_dim   = 90
    params_base.extra_dim  = [90]
    params_base.cond_dim  = 6
    params_base.num_heads  = 4
    params_base.signal_diffusion = True
    params_base.max_step   = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device: %s", device)

    # ---- Experimental grid ----
    # Phase 1: replicate original 4-block/100-iter with extended sigma
    SIGMA_PHASE1 = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
    SEEDS_PHASE1 = [11]

    # Phase 2: larger blocks/iters for best sigma from phase 1
    # We'll fill in after phase 1 completes (or predefine if we want)
    PHASE2_SIGMAS = [4.0, 6.0, 8.0]   # top candidates
    PHASE2_BLOCKS = [8, 16]
    PHASE2_ITERS  = [300]
    SEEDS_PHASE2  = [11]

    all_results: list[dict] = []

    # ---- Phase 1: 4-block / 100-iter extended sigma ----
    LOGGER.info("=" * 60)
    LOGGER.info("PHASE 1: 4-block / 100-iter with extended sigma range")
    LOGGER.info("=" * 60)

    for sigma_scale in SIGMA_PHASE1:
        for seed in SEEDS_PHASE1:
            result = train_and_eval(
                sigma_scale, 4, 64, 100, seed,
                params_base, cond_files, train_raw, cond_dir, device
            )
            all_results.append(result)
            import time as _t; _t.sleep(3)

    # Print phase 1 summary
    LOGGER.info("=" * 60)
    LOGGER.info("PHASE 1 SUMMARY:")
    LOGGER.info("%-10s %8s %10s %8s", "sigma", "final_loss", "SSIM_amp", "SSIM_comp")
    for r in all_results:
        LOGGER.info("%-10.1f %.6f   %.4f     %.4f",
                    r["sigma_scale"], r["final_loss"],
                    r["ssim_amp_mean"], r["ssim_complex_mean"])

    # Find best sigma from phase 1 for phase 2 focus
    phase1_by_sigma = {}
    for r in all_results:
        s = r["sigma_scale"]
        if s not in phase1_by_sigma:
            phase1_by_sigma[s] = []
        phase1_by_sigma[s].append(r)

    best_sigma_p1 = max(phase1_by_sigma.keys(),
                         key=lambda s: np.mean([x["ssim_amp_mean"] for x in phase1_by_sigma[s]]))
    LOGGER.info("Best sigma from Phase 1: %.1f (SSIM=%.4f)",
                best_sigma_p1,
                np.mean([x["ssim_amp_mean"] for x in phase1_by_sigma[best_sigma_p1]]))

    # ---- Phase 2: more blocks / iters ----
    LOGGER.info("=" * 60)
    LOGGER.info("PHASE 2: More blocks/iters for top sigma values")
    LOGGER.info("=" * 60)

    # Predefined phase 2 configs based on phase 1 trend (sigma=4,6,8 with 8/16 blocks)
    for sigma_scale in PHASE2_SIGMAS:
        for num_blocks in PHASE2_BLOCKS:
            for max_iter in PHASE2_ITERS:
                for seed in SEEDS_PHASE2:
                    result = train_and_eval(
                        sigma_scale, num_blocks, 64, max_iter, seed,
                        params_base, cond_files, train_raw, cond_dir, device
                    )
                    all_results.append(result)
                    import time as _t; _t.sleep(5)

    # ---- Final summary ----
    LOGGER.info("=" * 60)
    LOGGER.info("FINAL SUMMARY:")
    LOGGER.info("%-8s %6s %6s %8s %10s %8s %10s %8s",
               "sigma", "blocks", "iters", "loss", "SSIM_amp", "std", "SSIM_comp", "time_s")
    for r in all_results:
        LOGGER.info("%-8.1f %6d %6d %.6f  %.4f    %.4f  %.4f    %.1f",
                   r["sigma_scale"], r["num_blocks"], r["max_iter"],
                   r["final_loss"], r["ssim_amp_mean"], r["ssim_amp_std"],
                   r["ssim_complex_mean"], r["train_time_s"])

    # Identify best overall
    best = max(all_results, key=lambda r: r["ssim_amp_mean"])
    default_result = next((r for r in all_results
                           if r["sigma_scale"] == 1.0 and r["num_blocks"] == 4
                           and r["max_iter"] == 100), None)
    default_ssim = default_result["ssim_amp_mean"] if default_result else None

    LOGGER.info("BEST OVERALL: sigma=%.1f blocks=%d iters=%d SSIM=%.4f",
                best["sigma_scale"], best["num_blocks"], best["max_iter"],
                best["ssim_amp_mean"])
    if default_ssim:
        delta_pct = (best["ssim_amp_mean"] - default_ssim) / default_ssim * 100
        LOGGER.info("  vs default (sigma=1.0, 4-block, 100-iter): %+.2f%%", delta_pct)

    # ---- Save results ----
    metrics = {
        "task": "wifi",
        "mode": "sigma-extended-ablation",
        "run_id": run_id,
        "phase1_sigmas": SIGMA_PHASE1,
        "phase2_sigmas": PHASE2_SIGMAS,
        "phase2_blocks": PHASE2_BLOCKS,
        "phase2_iters": PHASE2_ITERS,
        "results": all_results,
        "best": {
            "sigma_scale": best["sigma_scale"],
            "num_blocks": best["num_blocks"],
            "max_iter": best["max_iter"],
            "ssim_amp_mean": best["ssim_amp_mean"],
            "ssim_amp_std": best["ssim_amp_std"],
        },
        "device": str(device),
    }

    metrics_path = RESULTS_ROOT / "metrics" / f"sigma_extended_{run_id}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, np.floating) else x)
    LOGGER.info("Metrics: %s", metrics_path)

    # CSV
    csv_path = RESULTS_ROOT / "metrics" / f"sigma_extended_{run_id}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma", "blocks", "hidden", "iters", "seed",
                    "final_loss", "ssim_amp_mean", "ssim_amp_std",
                    "ssim_complex_mean", "ssim_complex_std",
                    "train_time_s", "peak_gpu_mem_mb", "n_samples"])
        for r in all_results:
            w.writerow([r["sigma_scale"], r["num_blocks"], r["hidden_dim"], r["max_iter"],
                        r["seed"], r["final_loss"],
                        r["ssim_amp_mean"], r["ssim_amp_std"],
                        r["ssim_complex_mean"], r["ssim_complex_std"],
                        r["train_time_s"], r["peak_gpu_mem_mb"], r["n_samples"]])
    LOGGER.info("CSV: %s", csv_path)

    LOGGER.info("DONE. Run ID: %s", run_id)


if __name__ == "__main__":
    main()
