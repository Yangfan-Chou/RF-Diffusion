#!/usr/bin/env python3
"""Independent SSIM diagnostics for released RF-Diffusion Wi-Fi outputs.

This script does not import or modify the official inference implementation. It
reconstructs its input preprocessing, reproduces its complex SSIM equation, and
compares several conventional real-image SSIM mappings on the same 41 pairs.
"""

from __future__ import annotations

import argparse
import json
import math
from glob import glob
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.io as scio
import torch
import torch.nn.functional as F


def gaussian_1d(size: int, sigma: float, *, device: torch.device) -> torch.Tensor:
    values = torch.tensor(
        [math.exp(-((x - size // 2) ** 2) / (2 * sigma**2)) for x in range(size)],
        dtype=torch.float32,
        device=device,
    )
    return values / values.sum()


def gaussian_window(height: int, width: int, *, device: torch.device) -> torch.Tensor:
    h = gaussian_1d(height, 1.5, device=device).unsqueeze(1)
    w = gaussian_1d(width, 1.5, device=device).unsqueeze(1)
    return h.mm(w.t()).unsqueeze(0).unsqueeze(0)


def official_complex_ssim(pred: torch.Tensor, data: torch.Tensor) -> float:
    """Exact equation from upstream/RF-Diffusion/inference.py::eval_ssim."""
    height, width = data.shape
    window = gaussian_window(height, width, device=data.device).to(torch.complex64)
    pred_4d = pred[None, None]
    data_4d = data[None, None]
    padding = [height // 2, width // 2]
    mu_pred = F.conv2d(pred_4d, window, padding=padding)
    mu_data = F.conv2d(data_4d, window, padding=padding)
    mu_pred_pow = mu_pred.pow(2.0)
    mu_data_pow = mu_data.pow(2.0)
    mu_pred_data = mu_pred * mu_data
    var_pred = F.conv2d(pred_4d * pred_4d, window, padding=padding) - mu_pred_pow
    var_data = F.conv2d(data_4d * data_4d, window, padding=padding) - mu_data_pow
    cov = F.conv2d(pred_4d * data_4d, window, padding=padding) - mu_pred_data
    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu_pred * mu_data + c1) * (2 * cov.real + c2)) / (
        (mu_pred_pow + mu_data_pow + c1) * (var_pred + var_data + c2)
    )
    return float((2 * ssim_map.mean().real).item())


def standard_torch_ssim(
    pred: torch.Tensor,
    data: torch.Tensor,
    *,
    padding: str,
    data_range: float = 1.0,
) -> float:
    """Wang-style Gaussian SSIM for real tensors shaped [C, H, W]."""
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
        data = data.unsqueeze(0)
    channels = pred.shape[0]
    base_window = gaussian_window(11, 11, device=pred.device)
    window = base_window.expand(channels, 1, 11, 11).contiguous()
    pad = 5 if padding == "same" else 0
    pred_4d = pred.unsqueeze(0).float()
    data_4d = data.unsqueeze(0).float()
    mu_pred = F.conv2d(pred_4d, window, padding=pad, groups=channels)
    mu_data = F.conv2d(data_4d, window, padding=pad, groups=channels)
    mu_pred_sq = mu_pred.square()
    mu_data_sq = mu_data.square()
    mu_cross = mu_pred * mu_data
    var_pred = F.conv2d(pred_4d.square(), window, padding=pad, groups=channels) - mu_pred_sq
    var_data = F.conv2d(data_4d.square(), window, padding=pad, groups=channels) - mu_data_sq
    covariance = F.conv2d(pred_4d * data_4d, window, padding=pad, groups=channels) - mu_cross
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    value = ((2 * mu_cross + c1) * (2 * covariance + c2)) / (
        (mu_pred_sq + mu_data_sq + c1) * (var_pred + var_data + c2)
    )
    return float(value.mean().item())


def common_minmax(pred: torch.Tensor, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    low = torch.minimum(pred.min(), data.min())
    high = torch.maximum(pred.max(), data.max())
    scale = (high - low).clamp_min(torch.finfo(torch.float32).eps)
    return (pred - low) / scale, (data - low) / scale


def map_real(x: torch.Tensor) -> torch.Tensor:
    return x.real.unsqueeze(0)


def map_magnitude(x: torch.Tensor) -> torch.Tensor:
    return x.abs().unsqueeze(0)


def map_log_magnitude(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(x.abs()).unsqueeze(0)


def map_complex_channels(x: torch.Tensor) -> torch.Tensor:
    return torch.stack((x.real, x.imag), dim=0)


def map_complex_spatial(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((x.real, x.imag), dim=1).unsqueeze(0)


def map_stft_log_magnitude(x: torch.Tensor) -> torch.Tensor:
    # Treat the 90 CSI streams as channels and transform each 512-sample series.
    spectrogram = torch.stft(
        x.transpose(0, 1),
        n_fft=24,
        hop_length=17,
        window=torch.ones(24, device=x.device),
        return_complex=True,
    )
    return torch.log1p(spectrogram.abs())


def pytorch_msssim_value(pred: torch.Tensor, data: torch.Tensor) -> float:
    from pytorch_msssim import ssim as msssim_ssim

    return float(
        msssim_ssim(
            pred.unsqueeze(0).float(),
            data.unsqueeze(0).float(),
            data_range=1.0,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            nonnegative_ssim=False,
        ).item()
    )


def skimage_value(pred: torch.Tensor, data: torch.Tensor) -> float:
    from skimage.metrics import structural_similarity as skimage_ssim

    pred_np = pred.detach().cpu().numpy().transpose(1, 2, 0)
    data_np = data.detach().cpu().numpy().transpose(1, 2, 0)
    return float(
        skimage_ssim(
            data_np,
            pred_np,
            data_range=1.0,
            channel_axis=-1,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            win_size=11,
        )
    )


def load_pairs(cond_dir: Path, output_dir: Path, device: torch.device):
    # This intentionally preserves the official unsorted recursive-glob order.
    input_paths = glob(str(cond_dir / "**" / "user*.mat"), recursive=True)
    if not input_paths:
        raise FileNotFoundError(f"No user*.mat inputs under {cond_dir}")
    for index, input_path in enumerate(input_paths):
        output_path = output_dir / f"{index}-0.mat"
        if not output_path.exists():
            raise FileNotFoundError(output_path)
        sample = scio.loadmat(input_path, verify_compressed_data_integrity=False)
        source = torch.from_numpy(sample["feature"]).to(torch.complex64)
        real_view = torch.view_as_real(source).permute(1, 2, 0)
        resized = F.interpolate(real_view, 512, mode="nearest-exact")
        normalized = (resized - resized.mean()) / resized.std()
        data = torch.view_as_complex(normalized.permute(2, 0, 1).contiguous()).to(device)
        pred_np = scio.loadmat(output_path)["pred"]
        pred = torch.from_numpy(pred_np).to(torch.complex64).squeeze(0).to(device)
        yield index, Path(input_path), pred, data


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    stderr = float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std_population": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "standard_error": stderr,
        "ci95_normal_low": float(array.mean() - 1.96 * stderr),
        "ci95_normal_high": float(array.mean() + 1.96 * stderr),
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cond-dir",
        type=Path,
        default=root / "upstream/RF-Diffusion/dataset/wifi/cond",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "upstream/RF-Diffusion/dataset/wifi/output",
    )
    parser.add_argument(
        "--paper-data",
        type=Path,
        default=root / "upstream/RF-Diffusion/plots/data/exp_overall_ssim_wifi.mat",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--official-only",
        action="store_true",
        help="Skip third-party and standard-image variants (useful in legacy environments)",
    )
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    metric_values: dict[str, list[float]] = {
        "official_complex_full_window": [],
    }
    mappings: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "real": map_real,
        "magnitude": map_magnitude,
        "log1p_magnitude": map_log_magnitude,
        "real_imag_2channel": map_complex_channels,
        "real_imag_spatial_concat": map_complex_spatial,
        "stft_log1p_magnitude": map_stft_log_magnitude,
    }
    sample_records = []

    for index, input_path, pred, data in load_pairs(args.cond_dir, args.output_dir, device):
        official = official_complex_ssim(pred, data)
        metric_values["official_complex_full_window"].append(official)
        record = {"index": index, "input": str(input_path), "official": official}
        if not args.official_only:
            for mapping_name, mapping in mappings.items():
                mapped_pred = mapping(pred).float()
                mapped_data = mapping(data).float()
                norm_pred, norm_data = common_minmax(mapped_pred, mapped_data)
                variants = {
                    f"torch_gaussian11_valid_{mapping_name}_fixed_L1": standard_torch_ssim(
                        mapped_pred, mapped_data, padding="valid"
                    ),
                    f"torch_gaussian11_same_{mapping_name}_fixed_L1": standard_torch_ssim(
                        mapped_pred, mapped_data, padding="same"
                    ),
                    f"torch_gaussian11_valid_{mapping_name}_common_minmax": standard_torch_ssim(
                        norm_pred, norm_data, padding="valid"
                    ),
                    f"pytorch_msssim_{mapping_name}_common_minmax": pytorch_msssim_value(
                        norm_pred, norm_data
                    ),
                    f"skimage_gaussian11_{mapping_name}_common_minmax": skimage_value(
                        norm_pred, norm_data
                    ),
                }
                for name, value in variants.items():
                    metric_values.setdefault(name, []).append(value)
                if mapping_name in {"magnitude", "real", "log1p_magnitude"}:
                    record.update({name: value for name, value in variants.items()})
        sample_records.append(record)
        print(f"processed {index + 1:2d}: official={official:.9f}")

    summaries = {name: summarize(values) for name, values in metric_values.items()}
    paper_summary = None
    if args.paper_data.exists():
        released = scio.loadmat(args.paper_data)["data_wifi_sigma"].reshape(-1)
        paper_summary = summarize(released.tolist())

    result = {
        "runtime": {
            "python_torch": torch.__version__,
            "device": str(device),
            "cuda": torch.version.cuda,
        },
        "definitions": {
            "official": "full 512x90 separable Gaussian sigma=1.5; complex x*x moments; zero padding [256,45]; 2*mean(real(ssim_map))",
            "standard": "11x11 Gaussian sigma=1.5; real-valued Wang SSIM; C1=(0.01L)^2, C2=(0.03L)^2",
            "common_minmax": "one shared min/max across prediction and target for each pair, then L=1",
            "stft": "rectangular n_fft=24, hop_length=17, log1p magnitude, 90 streams as channels",
        },
        "metrics": summaries,
        "released_plot_data_wifi_sigma": paper_summary,
        "samples": sample_records,
    }
    print("\nMetric means:")
    for name, stats in sorted(summaries.items(), key=lambda item: item[0]):
        print(f"{name:75s} {stats['mean']:.9f}  std={stats['std_population']:.9f}")
    if paper_summary:
        print(
            "released_plot_data_wifi_sigma"
            f"{'':45s} {paper_summary['mean']:.9f}  n={paper_summary['n']}"
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
