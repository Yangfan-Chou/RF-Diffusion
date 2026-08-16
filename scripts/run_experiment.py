"""Top-level experiment runner.

Supports three commands:
- pretrained: run the official pretrained model on a downstream task.
- smoke-test: run a small number of samples for verification.
- small-train: train a reduced-size model on a Wi-Fi subset.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from src.config import (  # noqa: E402
    ExperimentConfig,
    RESULTS_ROOT,
    get_logger,
    set_seed,
)
from src.runner import evaluate_pretrained  # noqa: E402

LOGGER = get_logger("rfdiff.experiment")


def run_pretrained(cfg: ExperimentConfig) -> Dict[str, Any]:
    return evaluate_pretrained(cfg)


def run_smoke(cfg: ExperimentConfig) -> Dict[str, Any]:
    cfg.num_samples = cfg.num_samples or 2
    return evaluate_pretrained(cfg)


def run_small_train(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Reduce the official Wi-Fi config and run a short training session."""
    raise NotImplementedError(
        "Small-scale training is implemented in scripts/run_small_train.py"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RF-Diffusion reproduction runner")
    parser.add_argument("--task", choices=["wifi", "fmcw", "mimo"], default="wifi")
    parser.add_argument("--mode", choices=["pretrained", "smoke-test", "small-train"],
                        default="pretrained")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    if args.config:
        cfg = ExperimentConfig.from_yaml(Path(args.config))
    else:
        cfg = ExperimentConfig(
            task=args.task,
            mode=args.mode,
            seed=args.seed,
            num_samples=args.num_samples,
            device=args.device,
            batch_size=args.batch_size,
        )
    cfg.notes = f"cli: task={cfg.task} mode={cfg.mode} num_samples={cfg.num_samples}"
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    cfg.output_dir = RESULTS_ROOT / "raw" / f"{cfg.task}_{cfg.mode}_{run_id}"
    cfg.log_dir = RESULTS_ROOT / "logs" / f"{cfg.task}_{cfg.mode}_{run_id}"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Run id: %s", run_id)

    set_seed(cfg.seed)

    if cfg.mode == "pretrained":
        metrics = run_pretrained(cfg)
    elif cfg.mode == "smoke-test":
        metrics = run_smoke(cfg)
    elif cfg.mode == "small-train":
        metrics = run_small_train(cfg)
    else:
        raise ValueError(cfg.mode)

    metrics_path = RESULTS_ROOT / "metrics" / f"{cfg.task}_{cfg.mode}_{run_id}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    LOGGER.info("Saved metrics to %s", metrics_path)
    LOGGER.info("Summary: %s", json.dumps(
        {k: v for k, v in metrics.items() if not isinstance(v, list)}, indent=2))


if __name__ == "__main__":
    main()