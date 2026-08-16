#!/usr/bin/env python3
"""5G MIMO channel estimation using RF-Diffusion.

This script evaluates RF-Diffusion on 5G FDD MIMO channel estimation,
measuring SNR (Signal-to-Noise Ratio) as the primary quality metric.
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

LOGGER = get_logger("rfdiff.mimo")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RF-Diffusion 5G MIMO channel estimation evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 5G MIMO evaluation with 5 samples on GPU 0
  python scripts/run_5g_mimo.py --num-samples 5 --gpu 0

  # Run on CPU (slower)
  python scripts/run_5g_mimo.py --num-samples 3 --device cpu
        """
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device index (default: 0)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Number of samples to evaluate. None means all available."
    )
    parser.add_argument(
        "--seed", type=int, default=11,
        help="Random seed for reproducibility (default: 11)"
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu"], default="cuda",
        help="Device to run on (default: cuda)"
    )
    parser.add_argument(
        "--notes", type=str, default="",
        help="Additional notes for this run"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for 5G MIMO evaluation."""
    args = parse_args()

    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    LOGGER.info("Run ID: %s", run_id)
    LOGGER.info("Arguments: %s", vars(args))

    set_seed(args.seed)

    cfg = ExperimentConfig(
        task="mimo",
        mode="mimo",
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

    output_dir = RESULTS_ROOT / "raw" / f"mimo_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = output_dir

    try:
        metrics = evaluate_pretrained(cfg, sampling_strategy="fast")
        metrics["run_id"] = run_id

        summary_path = RESULTS_ROOT / "metrics" / f"mimo_summary_{run_id}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)

        LOGGER.info("=" * 60)
        LOGGER.info("Results saved to: %s", summary_path)
        LOGGER.info("=" * 60)
        LOGGER.info(
            "5G MIMO Results: SNR=%.2fdB (std=%.2f), Time=%.3fs, Memory=%.1fMB",
            metrics.get("average_snr_db", 0),
            metrics.get("snr_std", 0),
            metrics.get("average_sample_time_s", 0),
            metrics.get("peak_gpu_mem_mb", 0)
        )

    except Exception as e:
        LOGGER.exception("5G MIMO evaluation failed: %s", e)
        raise


if __name__ == "__main__":
    main()
