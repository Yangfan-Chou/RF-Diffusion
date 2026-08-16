"""Evaluation metrics: SSIM, SNR (dB), and aggregation helpers."""
from __future__ import annotations

from typing import List

import numpy as np
import torch


def _ssim_channel(pred: torch.Tensor, data: torch.Tensor, win: torch.Tensor, pad: int) -> torch.Tensor:
    """Compute the full SSIM map for a single-channel (H, W) tensor."""
    mu_pred = torch.nn.functional.conv2d(pred, win, padding=pad)
    mu_data = torch.nn.functional.conv2d(data, win, padding=pad)

    mu_pred_pow = mu_pred.pow(2.0)
    mu_data_pow = mu_data.pow(2.0)
    mu_pred_data = mu_pred * mu_data

    pred_sq = pred * pred
    data_sq = data * data
    pred_data = pred * data

    sigma_pred = torch.nn.functional.conv2d(pred_sq, win, padding=pad) - mu_pred_pow
    sigma_data = torch.nn.functional.conv2d(data_sq, win, padding=pad) - mu_data_pow
    sigma_pred_data = torch.nn.functional.conv2d(pred_data, win, padding=pad) - mu_pred_data

    C1 = 0.01**2
    C2 = 0.03**2

    mu_pred_s = mu_pred.pow(2.0).squeeze(1)   # (H, W)
    mu_data_s = mu_data.pow(2.0).squeeze(1)

    num = (2 * mu_pred * mu_data + C1).squeeze(1) * (2 * sigma_pred_data + C2)
    den = (mu_pred_s + mu_data_s + C1) * (sigma_pred.squeeze(1) + sigma_data.squeeze(1) + C2)

    return num / den


def compute_ssim(
    pred: torch.Tensor,
    data: torch.Tensor,
    input_dim: int,
    device: torch.device,
    window_size: int = 11,
    sample_rate: int = 20,
) -> float:
    """SSIM between predicted and ground-truth complex-valued signals.

    Accepts (2, dim) complex tensors, as produced by splitting complex samples
    along the batch dimension in runner.py. The signal is treated as a square
    (input_dim × input_dim) array for the 2D SSIM computation. SSIM is
    computed per-channel and then averaged.

    Args:
        pred: Complex signal tensor, shape (2, dim).
        data: Ground-truth complex signal tensor, same shape.
        input_dim: Height and width for the SSIM window.
        device: Device for computation.
        window_size: Kept for backward compatibility; not used.
        sample_rate: Kept for backward compatibility; not used.

    Returns:
        SSIM value in [0, 1]; higher is better.
    """
    def _gaussian(size: int, std: float) -> torch.Tensor:
        x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
        g = torch.exp(-(x ** 2) / (2 * std**2))
        return g / g.sum()

    window_1d = _gaussian(input_dim, 1.5)  # (input_dim,)
    window_2d = window_1d.unsqueeze(1) * window_1d.unsqueeze(0)  # (input_dim, input_dim)
    window = window_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, dim, dim)
    padding = input_dim // 2

    # (2, dim) → tile each channel into (dim, dim) → (1, 1, dim, dim)
    pred_real = pred[0].unsqueeze(1).expand(input_dim, input_dim).contiguous()
    pred_imag = pred[1].unsqueeze(1).expand(input_dim, input_dim).contiguous()
    data_real = data[0].unsqueeze(1).expand(input_dim, input_dim).contiguous()
    data_imag = data[1].unsqueeze(1).expand(input_dim, input_dim).contiguous()

    pred_real = pred_real.unsqueeze(0).unsqueeze(0)
    pred_imag = pred_imag.unsqueeze(0).unsqueeze(0)
    data_real = data_real.unsqueeze(0).unsqueeze(0)
    data_imag = data_imag.unsqueeze(0).unsqueeze(0)

    ssim_map_real = _ssim_channel(pred_real.float(), data_real.float(), window, padding)
    ssim_map_imag = _ssim_channel(pred_imag.float(), data_imag.float(), window, padding)

    return float(((ssim_map_real + ssim_map_imag) / 2).mean().item())


def compute_snr(pred: np.ndarray, truth: np.ndarray) -> float:
    """SNR in dB between predicted and ground-truth complex signals.

    Args:
        pred: Predicted signal, last dimension is [real, imag] pairs.
        truth: Ground-truth signal, same shape as pred.

    Returns:
        SNR in dB; higher is better. Returns inf if noise is zero.
    """
    if isinstance(pred, np.ndarray):
        pred_complex = pred[..., 0] + 1j * pred[..., 1]
        truth_complex = truth[..., 0] + 1j * truth[..., 1]
    else:
        pred_complex = pred
        truth_complex = truth

    signal_power = np.sum(np.abs(truth_complex) ** 2)
    noise_power = np.sum(np.abs(pred_complex - truth_complex) ** 2)

    if noise_power == 0:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))


def compute_snr_mimo(pred: torch.Tensor, data: torch.Tensor) -> float:
    """SNR in dB for MIMO channel estimation (batch=1)."""
    pred_np = pred.detach().cpu().numpy().squeeze(0)
    truth_np = data.detach().cpu().numpy().squeeze(0)
    pred_complex = pred_np[..., 0] + 1j * pred_np[..., 1]
    truth_complex = truth_np[..., 0] + 1j * truth_np[..., 1]

    ps = np.sum(np.abs(truth_complex) ** 2)
    pn = np.sum(np.abs(pred_complex - truth_complex) ** 2)
    return float(10 * np.log10(ps / pn))


def aggregate_metrics(ssim_list: List[float], snr_list: List[float]) -> dict:
    """Summarize per-sample metric lists into mean/std/min/max/count."""
    metrics = {}
    if ssim_list:
        metrics["ssim"] = {
            "mean": float(np.mean(ssim_list)),
            "std": float(np.std(ssim_list)),
            "min": float(np.min(ssim_list)),
            "max": float(np.max(ssim_list)),
            "count": len(ssim_list),
        }
    if snr_list:
        metrics["snr_db"] = {
            "mean": float(np.mean(snr_list)),
            "std": float(np.std(snr_list)),
            "min": float(np.min(snr_list)),
            "max": float(np.max(snr_list)),
            "count": len(snr_list),
        }
    return metrics
