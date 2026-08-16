#!/usr/bin/env python3
"""Evaluate generated .mat files and produce aggregate metrics as JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import scipy.io as scio
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"

sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_logger
from src.evaluation import aggregate_metrics, compute_snr, compute_ssim

LOGGER = get_logger("rfdiff.evaluate")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate generated RF-Diffusion .mat files"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory containing generated .mat files"
    )
    parser.add_argument(
        "--task", choices=["wifi", "fmcw", "mimo"], default="wifi",
        help="Task type for metric selection"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: input_dir/evaluation.json)"
    )
    return parser.parse_args()


def evaluate_directory(input_dir: Path, task: str) -> Dict[str, Any]:
    """Evaluate all .mat files in a directory."""
    mat_files = sorted(input_dir.glob("*.mat"))
    if not mat_files:
        LOGGER.warning("No .mat files found in %s", input_dir)
        return {"error": "No .mat files found", "task": task}

    LOGGER.info("Found %d sample files", len(mat_files))

    ssim_list: List[float] = []
    snr_list: List[float] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for mat_file in tqdm(mat_files, desc="Evaluating"):
        try:
            data = scio.loadmat(mat_file)
            pred = data["pred"]
            truth = data.get("data", data["pred"])

            if task == "mimo":
                snr = compute_snr(pred, truth)
                snr_list.append(snr)
            else:
                pred_complex = pred[0, :, 0].reshape(512).astype(np.complex128)
                truth_complex = truth[0, :, 0].reshape(512).astype(np.complex128)
                pred_tensor = torch.from_numpy(
                    np.stack([pred_complex.real, pred_complex.imag])
                ).squeeze(1)
                truth_tensor = torch.from_numpy(
                    np.stack([truth_complex.real, truth_complex.imag])
                ).squeeze(1)

                ssim = compute_ssim(
                    pred_tensor.to(torch.complex64),
                    truth_tensor.to(torch.complex64),
                    input_dim=512, device=device
                )
                ssim_list.append(ssim)

        except Exception as e:
            LOGGER.warning("Failed to evaluate %s: %s", mat_file, e)

    metrics = aggregate_metrics(ssim_list, snr_list)
    metrics["task"] = task
    metrics["num_files"] = len(mat_files)
    metrics["evaluated"] = len(ssim_list) + len(snr_list)

    return metrics


def main() -> None:
    """Main entry point."""
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output) if args.output else input_dir / "evaluation.json"

    results = evaluate_directory(input_dir, args.task)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    LOGGER.info("Results saved to: %s", output_path)
    LOGGER.info("Summary: %s", {
        k: v for k, v in results.items()
        if k not in ("task", "num_files", "evaluated")
    })


if __name__ == "__main__":
    main()
