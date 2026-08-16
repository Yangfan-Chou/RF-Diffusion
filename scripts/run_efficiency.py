#!/usr/bin/env python3
"""Efficiency experiments: quality vs inference time trade-off analysis.

This script sweeps different model configurations and sampling strategies
to analyze the quality-efficiency trade-off of RF-Diffusion.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"

sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.config import (
    RESULTS_ROOT,
    get_logger,
    peak_gpu_memory_mb,
    reset_peak_memory,
    set_seed,
)
from src.runner import build_model, build_task_params, truncate_model_blocks
from src.evaluation import compute_ssim
from tfdiff.dataset import _nested_map, from_path_inference
from tfdiff.diffusion import SignalDiffusion
from tfdiff.params import AttrDict

LOGGER = get_logger("rfdiff.efficiency")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RF-Diffusion efficiency experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all efficiency experiments (default)
  python scripts/run_efficiency.py --gpu 0 --num-samples 3

  # Run only native sampling experiments
  python scripts/run_efficiency.py --mode native --gpu 0

  # Run only full reverse diffusion experiments
  python scripts/run_efficiency.py --mode full_reverse --gpu 0
        """
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument(
        "--mode",
        choices=["all", "native", "full_reverse"],
        default="all",
        help="Experiment mode"
    )
    parser.add_argument(
        "--num-samples", type=int, default=3,
        help="Number of samples per configuration"
    )
    parser.add_argument(
        "--max-step", type=int, default=None,
        help="Override max diffusion steps"
    )
    return parser.parse_args()


def measure_config(
    name: str,
    cfg: Dict[str, Any],
    num_samples: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Measure quality and efficiency for a single configuration.

    Args:
        name: Configuration name for logging.
        cfg: Configuration dictionary with task, truncate_blocks, max_step, strategy.
        num_samples: Number of samples to evaluate.
        device: PyTorch device.

    Returns:
        Dictionary of measured metrics.
    """
    set_seed(11)
    params = build_task_params(cfg["task"])

    if cfg.get("max_step") is not None:
        params.max_step = cfg["max_step"]

    model = build_model(AttrDict(params), device)

    if cfg.get("truncate_blocks") is not None:
        truncate_model_blocks(model, cfg["truncate_blocks"])

    num_blocks = model.blocks.__len__() if hasattr(model, "blocks") else params.num_block

    diffusion = SignalDiffusion(AttrDict(params))

    dataset = from_path_inference(AttrDict(params))

    ssim_list: List[float] = []
    sample_times: List[float] = []
    reset_peak_memory(device)

    n_used = 0
    with torch.no_grad():
        for idx, features in enumerate(dataset):
            if n_used >= num_samples:
                break

            features = _nested_map(
                features,
                lambda x: x.to(device) if isinstance(x, torch.Tensor) else x
            )
            data = features["data"]
            cond = features["cond"]

            sample_start = time.perf_counter()

            strategy = cfg.get("strategy", "native")
            if strategy == "fast":
                pred = diffusion.fast_sampling(model, cond, device)
            elif strategy == "full_reverse":
                pred = diffusion.sampling(model, cond, device)
            else:
                pred = diffusion.native_sampling(model, data, cond, device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            sample_times.append(time.perf_counter() - sample_start)

            data_samples = [torch.view_as_complex(s) for s in torch.split(data, 1, dim=0)]
            pred_samples = [torch.view_as_complex(s) for s in torch.split(pred, 1, dim=0)]

            for b, p_sample in enumerate(pred_samples):
                cur_ssim = compute_ssim(
                    p_sample, data_samples[b],
                    params.input_dim, device
                )
                ssim_list.append(cur_ssim)

            n_used += 1

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    metrics = {
        "config_name": name,
        "task": cfg["task"],
        "num_block": num_blocks,
        "max_step": params.max_step,
        "strategy": cfg.get("strategy", "native"),
        "device": str(device),
        "samples": n_used,
        "average_ssim": float(np.mean(ssim_list)) if ssim_list else None,
        "ssim_std": float(np.std(ssim_list)) if ssim_list else None,
        "average_sample_time_s": float(np.mean(sample_times)) if sample_times else None,
        "total_time_s": sum(sample_times),
        "peak_gpu_mem_mb": peak_gpu_memory_mb(device),
    }

    LOGGER.info(
        "[%s] SSIM=%.4f Time=%.3fs Mem=%.1fMB",
        name, metrics["average_ssim"] or 0, metrics["average_sample_time_s"] or 0,
        metrics["peak_gpu_mem_mb"]
    )

    return metrics


def main() -> None:
    """Main entry point for efficiency experiments."""
    args = parse_args()

    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    LOGGER.info("Run ID: %s", run_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)
        LOGGER.info("Using GPU %d: %s", args.gpu, torch.cuda.get_device_name(args.gpu))

    num_samples = args.num_samples
    max_step_override = args.max_step

    configs: List[tuple] = []

    if args.mode in ("all", "native"):
        configs.extend([
            ("b32_step100_native", {"task": "wifi", "truncate_blocks": 32,
                                     "max_step": max_step_override or 100, "strategy": "native"}),
            ("b32_step50_native", {"task": "wifi", "truncate_blocks": 32,
                                    "max_step": max_step_override or 50, "strategy": "native"}),
            ("b32_step10_native", {"task": "wifi", "truncate_blocks": 32,
                                    "max_step": max_step_override or 10, "strategy": "native"}),
            ("b16_step100_native", {"task": "wifi", "truncate_blocks": 16,
                                     "max_step": max_step_override or 100, "strategy": "native"}),
            ("b8_step100_native", {"task": "wifi", "truncate_blocks": 8,
                                    "max_step": max_step_override or 100, "strategy": "native"}),
        ])

    if args.mode in ("all", "native"):
        configs.append(
            ("b32_step100_fast", {"task": "wifi", "truncate_blocks": 32,
                                   "max_step": max_step_override or 100, "strategy": "fast"})
        )

    if args.mode in ("all", "full_reverse"):
        configs.extend([
            ("full_b32_step100", {"task": "wifi", "truncate_blocks": 32,
                                   "max_step": max_step_override or 100, "strategy": "full_reverse"}),
            ("full_b32_step50", {"task": "wifi", "truncate_blocks": 32,
                                  "max_step": max_step_override or 50, "strategy": "full_reverse"}),
            ("full_b32_step20", {"task": "wifi", "truncate_blocks": 32,
                                  "max_step": max_step_override or 20, "strategy": "full_reverse"}),
            ("full_b16_step100", {"task": "wifi", "truncate_blocks": 16,
                                   "max_step": max_step_override or 100, "strategy": "full_reverse"}),
        ])

    results: List[Dict[str, Any]] = []

    for name, cfg in configs:
        try:
            metrics = measure_config(name, cfg, num_samples, device)
            results.append(metrics)
        except Exception as e:
            LOGGER.exception("Config %s failed: %s", name, e)
            results.append({"config_name": name, "error": str(e)})

    output_path = RESULTS_ROOT / "metrics" / f"efficiency_{run_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    LOGGER.info("=" * 60)
    LOGGER.info("Results saved to: %s", output_path)
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()
