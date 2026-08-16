"""Quality-vs-efficiency comparison using the genuine reverse-diffusion
sampling (not native_sampling which uses ground-truth degradation).
"""
from __future__ import annotations

import gc
import json
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
import torch  # noqa: E402

from src.config import get_logger, peak_gpu_memory_mb, reset_peak_memory, set_seed  # noqa: E402
from src.runner import build_model, build_task_params, compute_ssim  # noqa: E402
from tfdiff.dataset import _nested_map, from_path_inference  # noqa: E402
from tfdiff.params import AttrDict  # noqa: E402

LOGGER = get_logger("rfdiff.effgen")


def truncate_blocks(model, n):
    if hasattr(model, "module"):
        model = model.module
    model.blocks = torch.nn.ModuleList(list(model.blocks)[:n])
    return model


def measure_gen(name: str, cfg: Dict[str, Any], samples: int = 3,
                device: torch.device = torch.device("cuda")) -> Dict[str, Any]:
    set_seed(11)
    params = build_task_params(cfg["task"])
    if cfg.get("max_step") is not None:
        params.max_step = cfg["max_step"]
    # Build model with the original 32-block config to load checkpoint, then
    # truncate to truncate_blocks if requested.
    model = build_model(AttrDict(params), device)
    if cfg.get("truncate_blocks") is not None and cfg["truncate_blocks"] < model.blocks.__len__():
        truncate_blocks(model, cfg["truncate_blocks"])
    params.num_block = model.blocks.__len__()
    from tfdiff.diffusion import SignalDiffusion
    diffusion = SignalDiffusion(AttrDict(params))
    params.cond_dir = params.cond_dir
    params.batch_size = 1
    params.inference_batch_size = 1
    dataset = from_path_inference(AttrDict(params))

    ssim_list: List[float] = []
    sample_times: List[float] = []
    reset_peak_memory(device)
    n_used = 0
    with torch.no_grad():
        for idx, features in enumerate(dataset):
            if n_used >= samples:
                break
            features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
            data = features["data"]
            cond = features["cond"]
            sample_start = time.perf_counter()
            # Use full sampling (from noise, not data)
            pred = diffusion.sampling(model, cond, device)
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
    metrics = {
        "config_name": name,
        "task": cfg["task"],
        "num_block": cfg.get("truncate_blocks") or 32,
        "max_step": params.max_step,
        "samples": n_used,
        "average_ssim": float(np.mean(ssim_list)) if ssim_list else None,
        "ssim_std": float(np.std(ssim_list)) if ssim_list else None,
        "average_sample_time_s": float(np.mean(sample_times)) if sample_times else None,
        "total_time_s": sum(sample_times),
        "peak_gpu_mem_mb": peak_gpu_memory_mb(device),
        "sampling_mode": "full_reverse",
    }
    LOGGER.info("[%s] ssim=%.4f time=%.3fs mem=%.1fMB",
                name, metrics["average_ssim"] or 0.0,
                metrics["average_sample_time_s"] or 0.0,
                metrics["peak_gpu_mem_mb"])
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def main() -> None:
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    out_path = RESULTS_ROOT / "metrics" / f"efficiency_gen_{run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    samples = 2  # full reverse is slower, keep small
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = [
        ("full_b32_step100", {"task": "wifi", "truncate_blocks": 32, "max_step": 100}),
        ("full_b32_step50", {"task": "wifi", "truncate_blocks": 32, "max_step": 50}),
        ("full_b32_step20", {"task": "wifi", "truncate_blocks": 32, "max_step": 20}),
        ("full_b16_step100", {"task": "wifi", "truncate_blocks": 16, "max_step": 100}),
        ("full_b16_step20", {"task": "wifi", "truncate_blocks": 16, "max_step": 20}),
    ]
    results = []
    for name, cfg in configs:
        try:
            m = measure_gen(name, cfg, samples=samples, device=device)
            results.append(m)
        except Exception as e:
            LOGGER.exception("Config %s failed: %s", name, e)
            results.append({"config_name": name, "error": str(e)})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    LOGGER.info("Saved %s", out_path)


if __name__ == "__main__":
    main()