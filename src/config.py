"""Project-level configuration: experiment setup, logging, GPU memory tracking, and random seeds."""
from __future__ import annotations

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "upstream" / "RF-Diffusion"
RESULTS_ROOT = PROJECT_ROOT / "results"

sys.path.insert(0, str(UPSTREAM_ROOT))


@dataclass
class ExperimentConfig:
    """Holds parameters for a single evaluation run."""

    task: str = "wifi"
    mode: str = "pretrained"
    seed: int = 11
    num_samples: Optional[int] = None
    device: str = "cuda"
    batch_size: int = 1
    output_dir: Path = field(default_factory=lambda: RESULTS_ROOT / "raw" / "default")
    log_dir: Path = field(default_factory=lambda: RESULTS_ROOT / "logs" / "default")
    notes: str = ""

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        """Load config from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        data.setdefault("task", "wifi")
        data.setdefault("mode", "pretrained")
        data.setdefault("seed", 11)
        data.setdefault("device", "cuda")
        data.setdefault("batch_size", 1)
        return cls(**data)


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with standard formatting, creating it once per name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def peak_gpu_memory_mb(device: torch.device) -> float:
    """Return peak GPU memory allocation in MB."""
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def reset_peak_memory(device: torch.device) -> None:
    """Reset peak memory stats for the device."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
