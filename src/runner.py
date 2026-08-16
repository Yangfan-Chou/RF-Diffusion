"""Adapter for running RF-Diffusion inference without modifying upstream code."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.io as scio
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = PROJECT_ROOT / "upstream" / "RF-Diffusion"
RESULTS_ROOT = PROJECT_ROOT / "results"

sys.path.insert(0, str(UPSTREAM_ROOT))

from tfdiff.dataset import _nested_map, from_path_inference
from tfdiff.diffusion import GaussianDiffusion, SignalDiffusion
from tfdiff.eeg_model import tfdiff_eeg
from tfdiff.fmcw_model import tfdiff_fmcw
from tfdiff.mimo_model import tfdiff_mimo
from tfdiff.params import AttrDict, all_params
from tfdiff.wifi_model import tfdiff_WiFi

from src.config import (
    ExperimentConfig,
    RESULTS_ROOT,
    UPSTREAM_ROOT,
    get_logger,
    peak_gpu_memory_mb,
    reset_peak_memory,
    set_seed,
)
from src.evaluation import compute_ssim, compute_snr_mimo

LOGGER = get_logger("rfdiff.runner")


def build_task_params(task: str) -> AttrDict:
    """Return task parameters pointing to upstream data directories."""
    task_id = {"wifi": 0, "fmcw": 1, "mimo": 2}[task]
    params = all_params[task_id]
    new_params = AttrDict(dict(params))
    new_params.task_id = task_id
    suffix = "200s" if task == "mimo" else "100s"
    new_params.model_dir = str(UPSTREAM_ROOT / "model" / task / f"b32-256-{suffix}")
    new_params.cond_dir = [str(UPSTREAM_ROOT / "dataset" / task / "cond")]
    new_params.out_dir = str(RESULTS_ROOT / "raw" / task / "samples")
    new_params.data_dir = [str(UPSTREAM_ROOT / "dataset" / task / "raw")]
    new_params.fid_data_dir = str(UPSTREAM_ROOT / "dataset" / task / "img_matric" / "data")
    new_params.fid_pred_dir = str(UPSTREAM_ROOT / "dataset" / task / "img_matric" / "pred")
    return new_params


def build_model(params: AttrDict, device: torch.device) -> torch.nn.Module:
    """Load and return the pretrained RF-Diffusion model for the task."""
    if params.task_id == 0:
        model = tfdiff_WiFi(AttrDict(params)).to(device)
    elif params.task_id == 1:
        model = tfdiff_fmcw(AttrDict(params)).to(device)
    elif params.task_id == 2:
        model = tfdiff_mimo(AttrDict(params)).to(device)
    elif params.task_id == 3:
        model = tfdiff_eeg(AttrDict(params)).to(device)
    else:
        raise ValueError(f"Unknown task_id: {params.task_id}")

    checkpoint = torch.load(f"{params.model_dir}/weights.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def evaluate_pretrained(
    experiment_cfg: ExperimentConfig,
    sampling_strategy: str = "native",
    max_step: Optional[int] = None,
) -> Dict[str, Any]:
    """Run inference on the pretrained model and collect metrics."""
    set_seed(experiment_cfg.seed)
    params = build_task_params(experiment_cfg.task)
    os.makedirs(params.out_dir, exist_ok=True)

    if max_step is not None:
        params.max_step = max_step

    device = torch.device(
        "cuda"
        if experiment_cfg.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    LOGGER.info(
        "Running task=%s device=%s model_dir=%s strategy=%s",
        experiment_cfg.task, device, params.model_dir, sampling_strategy
    )

    model = build_model(params, device)

    if params.signal_diffusion:
        diffusion = SignalDiffusion(AttrDict(params))
    else:
        diffusion = GaussianDiffusion(AttrDict(params))

    params.override({"cond_dir": params.cond_dir})
    dataset = from_path_inference(AttrDict(params))

    ssim_list: List[float] = []
    snr_list: List[float] = []
    sample_times: List[float] = []

    reset_peak_memory(device)
    start_total = time.perf_counter()

    with torch.no_grad():
        for sample_idx, features in enumerate(tqdm(dataset, desc=f"RF-Diffusion/{experiment_cfg.task}")):
            if experiment_cfg.num_samples is not None and sample_idx >= experiment_cfg.num_samples:
                break

            features = _nested_map(
                features,
                lambda x: x.to(device) if isinstance(x, torch.Tensor) else x
            )
            data = features["data"]
            cond = features["cond"]

            sample_start = time.perf_counter()

            if experiment_cfg.task in ("wifi", "fmcw"):
                if sampling_strategy == "fast":
                    pred = diffusion.fast_sampling(model, cond, device)
                elif sampling_strategy == "full_reverse":
                    pred = diffusion.sampling(model, cond, device)
                else:
                    pred = diffusion.native_sampling(model, data, cond, device)

                data_samples = [torch.view_as_complex(s) for s in torch.split(data, 1, dim=0)]
                pred_samples = [torch.view_as_complex(s) for s in torch.split(pred, 1, dim=0)]

                for batch_idx, p_sample in enumerate(pred_samples):
                    d_sample = data_samples[batch_idx]
                    cur_ssim = compute_ssim(
                        p_sample, d_sample,
                        params.input_dim, device
                    )
                    ssim_list.append(cur_ssim)

                    scio.savemat(
                        os.path.join(params.out_dir, f"sample-{sample_idx}-{batch_idx}.mat"),
                        {"pred": p_sample.cpu().numpy(), "data": d_sample.cpu().numpy()}
                    )

            elif experiment_cfg.task == "mimo":
                pred = diffusion.fast_sampling(model, cond, device)
                snr = compute_snr_mimo(pred, data)
                snr_list.append(snr)

                scio.savemat(
                    os.path.join(params.out_dir, f"sample-{sample_idx}.mat"),
                    {"pred": pred.cpu().numpy(), "data": data.cpu().numpy()}
                )
            else:
                raise NotImplementedError(f"Task {experiment_cfg.task} not implemented")

            if device.type == "cuda":
                torch.cuda.synchronize()
            sample_times.append(time.perf_counter() - sample_start)

    total_time = time.perf_counter() - start_total

    metrics: Dict[str, Any] = {
        "task": experiment_cfg.task,
        "mode": experiment_cfg.mode,
        "sampling_strategy": sampling_strategy,
        "num_samples": len(sample_times),
        "average_sample_time_s": float(np.mean(sample_times)) if sample_times else 0.0,
        "total_time_s": total_time,
        "peak_gpu_mem_mb": peak_gpu_memory_mb(device),
        "device": str(device),
        "model_dir": params.model_dir,
        "config_seed": experiment_cfg.seed,
    }

    if ssim_list:
        metrics["average_ssim"] = float(np.mean(ssim_list))
        metrics["ssim_std"] = float(np.std(ssim_list))
        metrics["ssim_min"] = float(np.min(ssim_list))
        metrics["ssim_max"] = float(np.max(ssim_list))

    if snr_list:
        metrics["average_snr_db"] = float(np.mean(snr_list))
        metrics["snr_std"] = float(np.std(snr_list))
        metrics["snr_min"] = float(np.min(snr_list))
        metrics["snr_max"] = float(np.max(snr_list))

    LOGGER.info("Results: %s", {
        k: v for k, v in metrics.items()
        if not isinstance(v, (list, np.ndarray)) and k != "model_dir"
    })

    return metrics


def truncate_model_blocks(model: torch.nn.Module, n_blocks: int) -> torch.nn.Module:
    """Keep only the first n_blocks of the model (for efficiency experiments)."""
    if hasattr(model, "module"):
        model = model.module
    model.blocks = torch.nn.ModuleList(list(model.blocks)[:n_blocks])
    return model
