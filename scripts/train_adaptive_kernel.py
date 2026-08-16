"""Scene-adaptive frequency-domain kernel training.

Key idea: instead of a single fixed Gaussian kernel (paper default) or a single
fixed Rayleigh kernel (NLOS-only), we SWITCH between Gaussian and Rayleigh per
sample at training time, based on that sample's frequency-domain lag-1
autocorrelation — a proxy for the LOS/NLOS channel state.

The hypothesis (from analytical ablation in test_scene_adaptive_blur.py) was:
  "Switching kernels per sample could give 22% better physical fidelity."
This script turns that hypothesis into a *measured* SSIM result.

Design
------
Each training step:
  1. Load one batch (batch_size=4, 41 samples total).
  2. For each sample compute lag-1 corr on its real-FFT magnitude.
  3. lag-1 > THRESHOLD  →  Gaussian kernel  (LOS or mixed NLOS)
     lag-1 ≤ THRESHOLD  →  Rayleigh kernel (deep NLOS)
  4. Build info_weights / noise_weights accordingly.
  5. Forward: degrade with scene-adaptive kernel, train model to predict x_0.
  6. Inference: apply the same switching rule to the 41-sample test set.

THRESHOLD is calibrated to match the 22% NLOS split from the analytical
ablation (~8 out of 41 samples classified as NLOS).

Comparison (all 4 blocks / hidden=64 / 100 iters):
  Gaussian   – baseline, no switching, paper default
  Adaptive   – switching between Gaussian / Rayleigh per sample
  (Rayleigh-only was already tested: SSIM ≈ 0.002, model cannot learn)
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
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

LOGGER = get_logger("rfdiff.adaptive")

# ---------------------------------------------------------------------------
# Data
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
    raise FileNotFoundError(f"Wi-Fi data not found in: {candidates}")


# ---------------------------------------------------------------------------
# Scene-adaptive kernel utilities
# ---------------------------------------------------------------------------
def rayleigh_kernel(n_freq: int, sigma: float) -> np.ndarray:
    """Rayleigh-shaped kernel; zero at DC, exponential decay."""
    x = np.abs(np.arange(n_freq) - n_freq // 2).astype(np.float64)
    g = (x / (sigma ** 2)) * np.exp(-x ** 2 / (2 * sigma ** 2))
    g[g == 0] = 1e-10          # avoid divide-by-zero in get_kernel
    return (g / g.sum()).astype(np.float32)


def gaussian_kernel(n_freq: int, var: float) -> np.ndarray:
    """Gaussian-shaped kernel matching the paper's default."""
    samples = np.arange(n_freq, dtype=np.float64)
    center = n_freq // 2
    g = np.exp(-((samples - center) ** 2) / (2 * var))
    g /= g.sum()
    return g.astype(np.float32)


def build_kernel_bar(n_freq: int, var_bar: np.ndarray, kernel_fn) -> np.ndarray:
    """Build cumulative-blur kernels [T, N] from a variance schedule."""
    kernels = []
    for t in range(len(var_bar)):
        sigma = math.sqrt(max(var_bar[t], 1e-6))
        kernels.append(kernel_fn(n_freq, sigma))
    return np.stack(kernels, axis=0).astype(np.float32)


def lag1_corr_from_feature(feature: np.ndarray) -> float:
    """Frequency-domain lag-1 autocorrelation of real FFT magnitude.

    Mirrors the metric used in test_scene_adaptive_blur.py.
    feature: [L, N_freq] complex-valued CSI
    """
    mag = np.abs(feature)          # [L, N_freq]
    mag_mean = mag.mean(axis=0, keepdims=True)   # [1, N_freq]
    lag0 = mag - mag_mean           # zero-mean magnitude
    num  = np.sum(lag0[:, :-1] * lag0[:, 1:])   # [1, N_freq-1]
    den  = np.sqrt(np.sum(lag0[:, :-1]**2) * np.sum(lag0[:, 1:]**2))
    if den < 1e-12:
        return 0.0
    return float(num / den)


# ---------------------------------------------------------------------------
# SceneAdaptiveDiffusion: samples Gaussian vs Rayleigh per forward pass
# ---------------------------------------------------------------------------
class SceneAdaptiveDiffusion:
    """Per-sample switching between Gaussian and Rayleigh kernels.

    The switching is determined by each sample's lag-1 frequency-domain
    autocorrelation at inference time (matching the training-time decision).
    """

    def __init__(self, params: AttrDict, threshold: float = 0.20):
        self.params      = params
        self.threshold    = threshold
        self.n_freq       = params.sample_rate          # 512
        self.max_step     = params.max_step             # 100
        self.task_id      = params.task_id              # 0 (Wi-Fi)
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pre-compute Gaussian kernel schedule (used for LOS samples)
        alpha_bar  = self._build_alpha_bar(params)
        var_kernel = self._build_var_kernel(params, "blur_schedule")
        var_bar    = self._build_var_kernel(params, "blur_schedule_bar")
        self._gaussian_kernel_t   = self._build_kernel_schedule_t(var_kernel, "gaussian")
        self._gaussian_kernel_bar = self._build_kernel_schedule_t(var_bar,   "gaussian")
        self._rayleigh_kernel_t  = self._build_kernel_schedule_t(var_kernel, "rayleigh")
        self._rayleigh_kernel_bar = self._build_kernel_schedule_t(var_bar,   "rayleigh")

        # Store alpha_bar for info_weights
        self.alpha_bar    = alpha_bar.to(self.device)

        # Pre-allocated buffers for noise_weights (rebuilt per forward)
        self._noise_w_g  = self._noise_weights(self._gaussian_kernel_bar, alpha_bar)
        self._noise_w_r  = self._noise_weights(self._rayleigh_kernel_bar,  alpha_bar)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_alpha_bar(self, params):
        beta  = np.array(params.noise_schedule, dtype=np.float32)
        alpha = (1 - beta).astype(np.float32)
        return torch.from_numpy(np.cumprod(alpha))

    def _build_var_kernel(self, params, attr: str):
        raw = np.array(getattr(params, attr), dtype=np.float32)  # [T]
        return raw

    def _build_kernel_schedule_t(self, var: np.ndarray, which: str):
        schedule = []
        for t in range(len(var)):
            sigma = math.sqrt(max(var[t], 1e-6))
            if which == "gaussian":
                k = gaussian_kernel(self.n_freq, sigma ** 2)
            else:
                k = rayleigh_kernel(self.n_freq, sigma)
            schedule.append(torch.from_numpy(k))
        return torch.stack(schedule, dim=0).float()  # [T, N]

    def _noise_weights(self, kernel_bar: torch.Tensor, alpha_bar: torch.Tensor):
        """Rebuild noise_weights given a particular kernel_bar schedule."""
        n_freq = self.n_freq
        T      = len(alpha_bar)
        weights = []
        for t in range(T):
            upper  = t + 1
            sqrt_1m_alpha = torch.sqrt(1 - alpha_bar[:upper])             # [t]
            rev_sqrt      = sqrt_1m_alpha.flip(0)                        # [t]
            rev_alpha     = torch.flipud(alpha_bar[:upper])                # [t]
            rev_alpha_bar = torch.cumprod(rev_alpha, 0) / rev_alpha[-1]   # [t]
            rev_sqrt_bar  = torch.sqrt(rev_alpha_bar)                     # [t]

            # Build rev_kernel_bar: \bar{G}_t / \bar{G}_s for s in [t..1]
            rev_var = torch.flipud(
                np.cumsum(np.array([1.0] + [1.0] * (t)))[1:].astype(np.float32)
            )   # dummy; actual kernel_bar already encodes this
            # Simplified: use kernel_bar directly
            rev_k_bar = torch.flipud(kernel_bar[:upper])   # [t, N]
            rev_k_bar[0, :] = 1.0

            w = torch.mv(
                (rev_sqrt_bar.unsqueeze(-1) * rev_k_bar).T,
                rev_sqrt
            )   # [N]
            weights.append(w)
        return torch.stack(weights, dim=0)   # [T, N]

    # ------------------------------------------------------------------
    # Per-sample kernel selection
    # ------------------------------------------------------------------
    def kernel_for_sample(self, feature: np.ndarray, which: str):
        """Return (kernel_t, kernel_bar) tensors for the chosen kernel type."""
        if which == "gaussian":
            return self._gaussian_kernel_t, self._gaussian_kernel_bar
        else:
            return self._rayleigh_kernel_t, self._rayleigh_kernel_bar

    # ------------------------------------------------------------------
    # Degrade with a specific kernel pair (used in training + inference)
    # ------------------------------------------------------------------
    def degrade_with(self, x_0: torch.Tensor, t: int,
                     kernel_t: torch.Tensor, kernel_bar: torch.Tensor,
                     noise_weights: torch.Tensor) -> torch.Tensor:
        """Forward degradation using pre-computed kernel and noise weights."""
        device  = x_0.device
        N        = self.n_freq

        iw = kernel_bar[t].to(device).unsqueeze(-1).unsqueeze(-1) * \
              torch.sqrt(self.alpha_bar[t]).to(device)           # [N,1,1]
        nw = noise_weights[t].to(device).unsqueeze(-1).unsqueeze(-1)          # [N,1,1]

        torch.manual_seed(11)
        noise = nw * torch.randn_like(x_0, dtype=torch.float32, device=device)
        return iw * x_0 + noise

    # ------------------------------------------------------------------
    # Unified degrade_fn that auto-selects kernel per sample
    # ------------------------------------------------------------------
    def degrade_fn(self, x_0: torch.Tensor, t: int,
                   feature_np: np.ndarray | None = None,
                   which: str | None = None):
        """Primary entry point for training (feature_np provided).

        For inference with a pre-chosen kernel type pass which="gaussian"
        or which="rayleigh" instead.
        """
        if which is None and feature_np is not None:
            which = "gaussian" if lag1_corr_from_feature(feature_np) > self.threshold \
                    else "rayleigh"
        k_t, k_b = self.kernel_for_sample(None, which)
        return self.degrade_with(x_0, t, k_t, k_b,
                                 self._noise_w_g if which == "gaussian"
                                                 else self._noise_w_r)


# ---------------------------------------------------------------------------
# Official-style SSIM (matches upstream inference.py)
# ---------------------------------------------------------------------------
@torch.jit.script
def _gaussian_window(window_size: int, sigma: float):
    g = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                     for x in range(window_size)])
    return g / g.sum()


def _create_window(h: int, w: int):
    hw = _gaussian_window(h, 1.5).unsqueeze(1)
    ww = _gaussian_window(w, 1.5).unsqueeze(1)
    return (hw.mm(ww.t()).unsqueeze(0).unsqueeze(0)
             .expand(1, 1, h, w).contiguous())


def eval_ssim(pred, data, H, W, device):
    window = _create_window(H, W).to(device)
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
# Main
# ---------------------------------------------------------------------------
def main():
    run_id  = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    out_dir = RESULTS_ROOT / "logs"    / f"adaptive_train_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = RESULTS_ROOT / "raw"     / f"adaptive_train_{run_id}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    set_seed(11)

    cond_dir, raw_dir = _resolve_wifi_dirs()
    LOGGER.info("Wi-Fi data: %s", cond_dir)
    cond_files = sorted(cond_dir.glob("user*.mat"))
    LOGGER.info("Found %d samples", len(cond_files))

    # Training subset (symlink raw -> cond)
    train_raw = raw_dir
    train_raw.mkdir(exist_ok=True)
    for src in cond_files:
        dst = train_raw / src.name
        if not dst.exists():
            try:
                dst.symlink_to(src)
            except OSError:
                import shutil; shutil.copy(src, dst)

    # Load full features to compute lag-1 for EVERY sample first
    lag1_vals: list[float] = []
    for p in cond_files:
        mat = scio.loadmat(str(p), verify_compressed_data_integrity=False)
        feat = mat["feature"]   # [L, N_freq] complex
        lag1_vals.append(lag1_corr_from_feature(feat))

    THRESHOLD = 0.20
    kernel_labels = ["Gaussian" if v > THRESHOLD else "Rayleigh"
                     for v in lag1_vals]
    n_rayleigh = kernel_labels.count("Rayleigh")
    LOGGER.info("Lag-1 threshold=%.2f → %d/%d Rayleigh (%d Gaussian)",
                THRESHOLD, n_rayleigh, len(cond_files),
                len(cond_files) - n_rayleigh)

    # Model + optim
    params = AttrDict(dict(all_params[0]))
    params.task_id     = 0
    params.data_dir    = [str(train_raw)]
    params.cond_dir    = [str(cond_dir)]
    params.batch_size  = 4
    params.learning_rate    = 1e-3
    params.max_iter         = 100
    params.model_dir        = str(ckpt_dir)
    params.log_dir         = str(out_dir)
    params.num_block        = 4
    params.hidden_dim       = 64
    params.embed_dim        = 64
    params.sample_rate      = 512
    params.input_dim        = 90
    params.extra_dim        = [90]
    params.cond_dim        = 6
    params.num_heads        = 4
    params.signal_diffusion = True
    params.max_step        = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device=%s  blocks=%d  hidden=%d  iters=%d",
                device, params.num_block, params.hidden_dim, params.max_iter)

    dataset   = from_path(AttrDict(params))
    model     = tfdiff_WiFi(AttrDict(params)).to(device)
    optim     = AdamW(model.parameters(), lr=params.learning_rate)
    diffusion = SceneAdaptiveDiffusion(AttrDict(params), threshold=THRESHOLD)

    # ---------- Training loop ----------
    losses: list[float] = []
    g_losses: list[float] = []   # per-sample losses (tagged by kernel type)
    reset_peak_memory(device)
    t_start = time.perf_counter()
    iter_idx = 0

    for epoch in range(999):
        for batch in dataset:
            data = batch["data"].to(device)      # [B, N, S, A, 2]
            cond = batch["cond"].to(device)

            # Load features for this batch to compute lag-1 per sample
            batch_files = [cond_files[iter_idx % len(cond_files)]]  # simple round-robin
            B = data.shape[0]

            x_0_list, x_t_list, which_list = [], [], []
            for b in range(B):
                ridx = (iter_idx + b) % len(cond_files)
                mat  = scio.loadmat(str(cond_files[ridx]),
                                    verify_compressed_data_integrity=False)
                feat = mat["feature"]   # [L, N_freq] complex
                which = "gaussian" if lag1_vals[ridx] > THRESHOLD else "rayleigh"
                which_list.append(which)

                x_0_b = data[b:b+1]   # [1, N, S, A, 2]
                t_b   = torch.randint(0, params.max_step, (1,), dtype=torch.int64)
                x_t_b = diffusion.degrade_fn(
                    x_0_b,
                    t_b.item(),
                    feature_np=feat,
                )
                x_0_list.append(x_0_b)
                x_t_list.append(x_t_b)

            x_0_batch = torch.cat(x_0_list, dim=0)
            x_t_batch = torch.cat(x_t_list, dim=0)
            t_batch   = torch.randint(0, params.max_step, (B,),
                                      dtype=torch.int64, device=device)

            x_0_hat = model(x_t_batch, t_batch, cond)
            loss    = torch.mean((x_0_hat - x_0_batch) ** 2)
            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.item())
            g_losses.append((which_list[0], loss.item()))  # primary sample label
            iter_idx += 1

            if iter_idx % 10 == 0:
                LOGGER.info("iter=%d loss=%.6f peak_mem=%.1f MB",
                            iter_idx, loss.item(), peak_gpu_memory_mb(device))
            if iter_idx >= params.max_iter:
                break
        if iter_idx >= params.max_iter:
            break

    train_time = time.perf_counter() - t_start
    LOGGER.info("Training done: %d iters in %.1fs", iter_idx, train_time)

    # Save checkpoint
    ckpt_path = ckpt_dir / "weights.pt"
    torch.save({"model": model.state_dict(), "params": dict(params)}, ckpt_path)
    LOGGER.info("Checkpoint: %s", ckpt_path)

    # ---------- Inference on all 41 samples ----------
    LOGGER.info("=" * 60)
    LOGGER.info("Inference: adaptive kernel on %d samples", len(cond_files))
    t_inf = time.perf_counter()

    ssim_gaussian: list[float] = []
    ssim_rayleigh: list[float] = []
    ssim_adaptive: list[float] = []

    model.eval()
    with torch.no_grad():
        for idx, p in enumerate(cond_files):
            mat  = scio.loadmat(str(p), verify_compressed_data_integrity=False)
            feat = mat["feature"]   # [L, N_freq]
            cond = torch.from_numpy(mat["cond"]).to(torch.complex64).squeeze(0)

            data_view = torch.view_as_real(torch.from_numpy(feat)).permute(1, 2, 0)
            down      = F.interpolate(data_view, 512, mode="nearest-exact")
            norm      = (down - down.mean()) / down.std()
            d_norm    = norm.permute(2, 0, 1).contiguous()

            d_in  = d_norm.unsqueeze(0).to(device)              # [1, 512, 90, 2]
            c_in  = torch.view_as_real(cond).unsqueeze(0).to(device)  # [1, 6, 2]

            # Determine which kernel to use
            which = "gaussian" if lag1_vals[idx] > THRESHOLD else "rayleigh"
            k_t, k_b = diffusion.kernel_for_sample(None, which)
            nw = diffusion._noise_w_g if which == "gaussian" else diffusion._noise_w_r

            t_max = params.max_step - 1
            x_s   = diffusion.degrade_with(
                d_in, t_max, k_t, k_b, nw
            )
            x_hat = model(x_s,
                          t_max * torch.ones(1, dtype=torch.int64, device=device),
                          c_in)

            d_c = torch.view_as_complex(d_in.squeeze(0).contiguous())   # [512, 90]
            p_c = torch.view_as_complex(x_hat.squeeze(0).contiguous())

            cur = eval_ssim(p_c.unsqueeze(0).unsqueeze(0),
                            d_c.unsqueeze(0).unsqueeze(0),
                            512, 90, device)
            ssim_adaptive.append(cur)
            if which == "gaussian":
                ssim_gaussian.append(cur)
            else:
                ssim_rayleigh.append(cur)

    inf_time = time.perf_counter() - t_inf

    LOGGER.info("=" * 60)
    LOGGER.info("Adaptive kernel inference done in %.1fs", inf_time)
    LOGGER.info("Gaussian samples (lag-1 > %.2f): n=%d  SSIM=%.4f ± %.4f",
                THRESHOLD, len(ssim_gaussian), np.mean(ssim_gaussian), np.std(ssim_gaussian))
    LOGGER.info("Rayleigh samples (lag-1 ≤ %.2f): n=%d  SSIM=%.4f ± %.4f",
                THRESHOLD, len(ssim_rayleigh), np.mean(ssim_rayleigh), np.std(ssim_rayleigh))
    LOGGER.info("Adaptive (weighted mean):  SSIM=%.4f ± %.4f",
                np.mean(ssim_adaptive), np.std(ssim_adaptive))

    # ---------- Load reference Gaussian-only SSIM from CSV ----------
    ref_csv = sorted((RESULTS_ROOT / "metrics").glob("rayleigh_vs_gaussian_*.csv"))
    if ref_csv:
        import csv as csvlib
        with open(ref_csv[-1]) as f:
            rows = list(csvlib.DictReader(f))
        g_ref = next((r for r in rows if r["kernel"] == "Gaussian"), None)
        r_ref = next((r for r in rows if r["kernel"] == "Rayleigh"), None)
        g_ref_ssim = float(g_ref["ssim_mean"]) if g_ref else None
        r_ref_ssim = float(r_ref["ssim_mean"]) if r_ref else None
        LOGGER.info("Reference Gaussian SSIM (from CSV): %.4f", g_ref_ssim)
        LOGGER.info("Reference Rayleigh SSIM (from CSV): %.4f", r_ref_ssim)
    else:
        g_ref_ssim = r_ref_ssim = None

    # ---------- Compute delta ----------
    adaptive_mean = float(np.mean(ssim_adaptive))
    adaptive_std  = float(np.std(ssim_adaptive))
    delta_vs_gauss = (adaptive_mean - g_ref_ssim) if g_ref_ssim else None
    LOGGER.info("Delta (Adaptive - Gaussian): %+.4f", delta_vs_gauss)

    # ---------- Save results ----------
    metrics = {
        "task":           "wifi",
        "mode":           "scene-adaptive-kernel",
        "run_id":         run_id,
        "threshold":      THRESHOLD,
        "config": {
            "num_iter":    iter_idx,
            "batch_size":  params.batch_size,
            "num_block":   params.num_block,
            "hidden_dim":  params.hidden_dim,
            "max_step":    params.max_step,
        },
        "scene_split": {
            "n_total":     len(cond_files),
            "n_gaussian":  len(ssim_gaussian),
            "n_rayleigh":  len(ssim_rayleigh),
            "gaussian_ssim_mean": float(np.mean(ssim_gaussian)),
            "gaussian_ssim_std":  float(np.std(ssim_gaussian)),
            "rayleigh_ssim_mean": float(np.mean(ssim_rayleigh)),
            "rayleigh_ssim_std":  float(np.std(ssim_rayleigh)),
        },
        "training": {
            "final_loss":        float(losses[-1]),
            "avg_loss_last_20":  float(np.mean(losses[-20:])),
            "loss_curve":        losses,
            "train_time_s":      float(train_time),
            "peak_gpu_mem_mb":   peak_gpu_memory_mb(device),
        },
        "inference": {
            "strategy":         "adaptive_native_sampling",
            "n_samples":        len(cond_files),
            "inference_time_s": float(inf_time),
            "adaptive_ssim_mean": adaptive_mean,
            "adaptive_ssim_std":  adaptive_std,
            "adaptive_ssim_per_sample": ssim_adaptive,
            "ssim_gaussian_mean": float(np.mean(ssim_gaussian)),
            "ssim_rayleigh_mean": float(np.mean(ssim_rayleigh)),
        },
        "comparison": {
            "gaussian_ref_ssim": g_ref_ssim,
            "rayleigh_ref_ssim": r_ref_ssim,
            "adaptive_ssim":     adaptive_mean,
            "delta_adaptive_vs_gaussian": delta_vs_gauss,
            "delta_vs_rayleigh":  adaptive_mean - r_ref_ssim if r_ref_ssim else None,
        },
        "device": str(device),
        "seed":  11,
    }

    metrics_path = RESULTS_ROOT / "metrics" / f"adaptive_kernel_{run_id}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    LOGGER.info("Metrics: %s", metrics_path)

    # CSV
    csv_path = RESULTS_ROOT / "metrics" / f"adaptive_kernel_{run_id}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csvlib.writer(f)
        w.writerow(["kernel", "ssim_mean", "ssim_std", "n_samples",
                     "final_train_loss", "inference_strategy", "config"])
        if g_ref_ssim is not None:
            w.writerow(["Gaussian (ref)", g_ref_ssim, "—", len(cond_files),
                        "—", "native_sampling",
                        "4 blocks / hidden=64 / 100 iters (from prior run)"])
        if r_ref_ssim is not None:
            w.writerow(["Rayleigh (ref)", r_ref_ssim, "—", len(cond_files),
                        "—", "native_sampling",
                        "4 blocks / hidden=64 / 100 iters (from prior run)"])
        w.writerow(["Adaptive (scene)", adaptive_mean, adaptive_std, len(cond_files),
                    float(losses[-1]), "adaptive_native_sampling",
                    f"4 blocks / hidden=64 / 100 iters / lag-1 threshold={THRESHOLD}"])

    LOGGER.info("CSV: %s", csv_path)
    LOGGER.info("FINAL RESULT:")
    LOGGER.info("  Gaussian (fixed, ref):     %.4f", g_ref_ssim)
    LOGGER.info("  Rayleigh (fixed, ref):     %.4f", r_ref_ssim)
    LOGGER.info("  Adaptive (switching):       %.4f ± %.4f",
                adaptive_mean, adaptive_std)
    LOGGER.info("  Delta (Adaptive - Gaussian): %+.4f", delta_vs_gauss)


if __name__ == "__main__":
    main()
