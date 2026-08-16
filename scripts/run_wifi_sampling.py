#!/usr/bin/env python3
"""Wi-Fi signal generation using RF-Diffusion.

This script runs the full RF-Diffusion evaluation pipeline on Wi-Fi CSI data,
supporting multiple sampling strategies for quality-efficiency trade-off analysis.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"

sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.config import (
    ExperimentConfig,
    RESULTS_ROOT,
    get_logger,
    set_seed,
)
from src.runner import evaluate_pretrained

LOGGER = get_logger("rfdiff.wifi")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RF-Diffusion Wi-Fi signal generation and evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with 5 samples using default native sampling
  python scripts/run_wifi_sampling.py --num-samples 5 --gpu 0

  # Run fast sampling (single-step) with 10 samples
  python scripts/run_wifi_sampling.py --num-samples 10 --strategy fast --gpu 0

  # Run full reverse diffusion with custom max steps
  python scripts/run_wifi_sampling.py --num-samples 3 --strategy full_reverse --max-step 50 --gpu 0

  # Smoke test (2 samples) on CPU
  python scripts/run_wifi_sampling.py --num-samples 2 --device cpu
        """
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device index (default: 0)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Number of samples to generate. None means all available samples."
    )
    parser.add_argument(
        "--seed", type=int, default=11,
        help="Random seed for reproducibility (default: 11)"
    )
    parser.add_argument(
        "--strategy",
        choices=["native", "fast", "full_reverse", "all"],
        default="native",
        help="Sampling strategy: native (DDPM), fast (single-step), "
             "full_reverse (DDIM from noise), or 'all' to run all strategies"
    )
    parser.add_argument(
        "--max-step", type=int, default=None,
        help="Maximum diffusion steps (overrides model default)"
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu"], default="cuda",
        help="Device to run on (default: cuda)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Custom output directory for results"
    )
    parser.add_argument(
        "--notes", type=str, default="",
        help="Additional notes for this run"
    )
    return parser.parse_args()


def run_single_strategy(
    cfg: ExperimentConfig,
    strategy: str,
    run_id: str,
    max_step: int | None = None
) -> dict:
    """Run evaluation for a single sampling strategy.

    Args:
        cfg: Base experiment configuration.
        strategy: Sampling strategy name.
        run_id: Unique run identifier.
        max_step: Optional max diffusion steps.

    Returns:
        Dictionary of results for this strategy.
    """
    LOGGER.info("=" * 60)
    LOGGER.info("Running strategy: %s", strategy)
    LOGGER.info("=" * 60)

    cfg.mode = f"wifi_{strategy}"
    if max_step is not None:
        cfg.mode += f"_step{max_step}"

    output_dir = RESULTS_ROOT / "raw" / f"wifi_{strategy}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = output_dir

    try:
        metrics = evaluate_pretrained(
            cfg,
            sampling_strategy=strategy,
            max_step=max_step
        )
        metrics["strategy"] = strategy
        return metrics
    except Exception as e:
        LOGGER.exception("Strategy %s failed: %s", strategy, e)
        return {
            "strategy": strategy,
            "error": str(e),
            "task": cfg.task,
            "mode": cfg.mode,
        }


def main() -> None:
    """Main entry point for Wi-Fi sampling evaluation."""
    args = parse_args()

    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    LOGGER.info("Run ID: %s", run_id)
    LOGGER.info("Arguments: %s", vars(args))

    set_seed(args.seed)

    cfg = ExperimentConfig(
        task="wifi",
        mode="wifi",
        seed=args.seed,
        num_samples=args.num_samples,
        device=args.device,
    )
    cfg.notes = args.notes

    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        LOGGER.info("Using GPU %d: %s", args.gpu, torch.cuda.get_device_name(args.gpu))
    elif args.device == "cuda":
        LOGGER.warning("CUDA requested but not available, falling back to CPU")
        cfg.device = "cpu"

    strategies = ["all"] if args.strategy == "all" else [args.strategy]

    all_results = {}
    for strategy in strategies:
        max_step = args.max_step if args.strategy != "all" else None
        if strategy == "full_reverse":
            max_step = args.max_step or 100
        result = run_single_strategy(cfg, strategy, run_id, max_step)
        all_results[strategy] = result

    summary_path = RESULTS_ROOT / "metrics" / f"wifi_summary_{run_id}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    LOGGER.info("=" * 60)
    LOGGER.info("Summary saved to: %s", summary_path)
    LOGGER.info("=" * 60)

    for strategy, result in all_results.items():
        if "error" in result:
            LOGGER.error("Strategy %s FAILED: %s", strategy, result["error"])
        else:
            LOGGER.info(
                "Strategy %s: SSIM=%.4f, Time=%.3fs, Memory=%.1fMB",
                strategy,
                result.get("average_ssim", 0),
                result.get("average_sample_time_s", 0),
                result.get("peak_gpu_mem_mb", 0)
            )


if __name__ == "__main__":
    main()
