"""Efficiency experiments for RF-Diffusion.

Measures quality (SSIM), latency, peak GPU memory for several
configurations of the official pretrained Wi-Fi model. Configurations:
- num_block: 16, 32 (we load the pretrained 32-block model and the
  32-block model is the only pretrained; for 16-block we modify the
  trained model in-memory by keeping only the first 16 blocks of state).
- max_step: 10, 50, 100  (number of diffusion sampling steps).
- Sampling strategy: native_sampling vs fast_sampling (single-step).
- Inference device: cuda vs cpu.
"""
from __future__ import annotations

import copy
import gc
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "upstream" / "RF-Diffusion"
RESULTS_ROOT = PROJECT_ROOT / "results"

sys.path.insert(0, str(UPSTREAM_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import scipy.io as scio  # noqa: E402
import torch  # noqa: E402

from src.config import get_logger, peak_gpu_memory_mb, reset_peak_memory, set_seed  # noqa: E402
from src.runner import build_model, build_task_params, compute_ssim  # noqa: E402
from tfdiff.dataset import _nested_map, from_path_inference  # noqa: E402
from tfdiff.params import AttrDict  # noqa: E402

LOGGER = get_logger("rfdiff.efficiency")


def truncate_model_blocks(model: torch.nn.Module, n_blocks: int) -> torch.nn.Module:
    """Keep only the first n_blocks blocks of the model (in-place)."""
    if hasattr(model, "module"):
        model = model.module
    blocks = model.blocks
    new_blocks = torch.nn.ModuleList(list(blocks)[:n_blocks])
    model.blocks = new_blocks
    return model


def measure(config_name: str, cfg: Dict[str, Any], samples: int = 5,
            device: torch.device = torch.device("cuda")) -> Dict[str, Any]:
    set_seed(11)
    params = build_task_params(cfg["task"])
    if cfg.get("num_block") is not None:
        params.num_block = cfg["num_block"]
    if cfg.get("max_step") is not None:
        params.max_step = cfg["max_step"]
    model = build_model(AttrDict(params), device)
    if cfg.get("truncate_blocks") is not None:
        truncate_model_blocks(model, cfg["truncate_blocks"])
    if device.type == "cpu":
        model = model.to(device)

    from tfdiff.diffusion import SignalDiffusion
    diffusion = SignalDiffusion(AttrDict(params))
    params.cond_dir = params.cond_dir
    params.batch_size = 1
    params.inference_batch_size = 1
    dataset = from_path_inference(AttrDict(params))

    ssim_list: List[float] = []
    sample_times: List[float] = []
    n_used = 0
    reset_peak_memory(device)
    total_start = time.perf_counter()
    with torch.no_grad():
        for idx, features in enumerate(dataset):
            if n_used >= samples:
                break
            features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
            data = features["data"]
            cond = features["cond"]
            sample_start = time.perf_counter()
            if cfg.get("strategy") == "fast":
                pred = diffusion.fast_sampling(model, cond, device)
            else:
                pred = diffusion.native_sampling(model, data, cond, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            sample_times.append(time.perf_counter() - sample_start)
            data_samples = [torch.view_as_complex(s) for s in torch.split(data, 1, dim=0)]
            pred_samples = [torch.view_as_complex(s) for s in torch.split(pred, 1, dim=0)]
            for b, p_sample in enumerate(pred_samples):
                cur_ssim = compute_ssim(p_sample, data_samples[b],
                                        params.sample_rate, params.input_dim, device)
                ssim_list.append(cur_ssim)
            n_used += 1
    total_time = time.perf_counter() - total_start

    metrics = {
        "config_name": config_name,
        "task": cfg["task"],
        "num_block": cfg.get("truncate_blocks") or cfg.get("num_block") or 32,
        "max_step": params.max_step,
        "strategy": cfg.get("strategy", "native"),
        "device": str(device),
        "samples": n_used,
        "average_ssim": float(np.mean(ssim_list)) if ssim_list else None,
        "ssim_std": float(np.std(ssim_list)) if ssim_list else None,
        "average_sample_time_s": float(np.mean(sample_times)) if sample_times else None,
        "total_time_s": total_time,
        "peak_gpu_mem_mb": peak_gpu_memory_mb(device),
    }
    LOGGER.info("[%s] avg_ssim=%.4f avg_t=%.3fs peak_mem=%.1fMB",
                config_name, metrics["average_ssim"] or 0.0,
                metrics["average_sample_time_s"] or 0.0,
                metrics["peak_gpu_mem_mb"])
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def main() -> None:
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    out_path = RESULTS_ROOT / "metrics" / f"efficiency_{run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    samples = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = [
        ("b32_step100_native", {"task": "wifi", "truncate_blocks": 32, "max_step": 100, "strategy": "native"}),
        ("b32_step50_native", {"task": "wifi", "truncate_blocks": 32, "max_step": 50, "strategy": "native"}),
        ("b32_step10_native", {"task": "wifi", "truncate_blocks": 32, "max_step": 10, "strategy": "native"}),
        ("b16_step100_native", {"task": "wifi", "truncate_blocks": 16, "max_step": 100, "strategy": "native"}),
        ("b8_step100_native", {"task": "wifi", "truncate_blocks": 8, "max_step": 100, "strategy": "native"}),
        ("b32_step100_fast", {"task": "wifi", "truncate_blocks": 32, "max_step": 100, "strategy": "fast"}),
    ]

    for name, cfg in configs:
        try:
            m = measure(name, cfg, samples=samples, device=device)
            results.append(m)
        except Exception as e:
            LOGGER.exception("Config %s failed: %s", name, e)
            results.append({"config_name": name, "error": str(e)})

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    LOGGER.info("Saved efficiency metrics: %s", out_path)


if __name__ == "__main__":
    main()