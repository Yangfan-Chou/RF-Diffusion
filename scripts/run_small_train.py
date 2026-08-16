"""Small-scale training of RF-Diffusion on a Wi-Fi subset.

This script does NOT aim to reproduce the full training. It runs a *very*
reduced-size Wi-Fi training session to demonstrate that the framework can
be trained end-to-end on this machine.

Key reductions vs official:
- num_block: 32 -> 4
- hidden_dim: 128 -> 64
- max_iter:   unlimited -> 200
- batch_size: 32 -> 4
- subset:     full dataset -> first 64 samples in dataset/wifi/cond
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "upstream" / "RF-Diffusion"
RESULTS_ROOT = PROJECT_ROOT / "results"

sys.path.insert(0, str(UPSTREAM_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.optim import AdamW  # noqa: E402

from src.config import RESULTS_ROOT as _RR, get_logger, peak_gpu_memory_mb, reset_peak_memory, set_seed  # noqa: E402
from tfdiff.dataset import from_path  # noqa: E402
from tfdiff.params import AttrDict, all_params  # noqa: E402
from tfdiff.wifi_model import tfdiff_WiFi  # noqa: E402

LOGGER = get_logger("rfdiff.smalltrain")


def main() -> None:
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    out_dir = RESULTS_ROOT / "logs" / f"small_train_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = RESULTS_ROOT / "raw" / f"small_train_{run_id}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    set_seed(11)

    # Take a small subset of cond files for training (max 64 files).
    cond_dir = UPSTREAM_ROOT / "dataset" / "wifi" / "cond"
    raw_dir = UPSTREAM_ROOT / "dataset" / "wifi" / "raw"
    train_cond = cond_dir
    # The official release only ships cond/ files which contain both 'feature'
    # (the target signal) and 'cond'. We create a small raw/ subset by symlinking.
    train_raw = raw_dir
    train_raw.mkdir(exist_ok=True)
    cond_files = sorted(cond_dir.glob("user*.mat"))[:64]
    if not list(train_raw.glob("user*.mat")):
        for src in cond_files:
            dst = train_raw / src.name
            if not dst.exists():
                try:
                    dst.symlink_to(src)
                except OSError:
                    import shutil
                    shutil.copy(src, dst)
    LOGGER.info("Created training subset with %d files in %s", len(cond_files), train_raw)

    params = AttrDict(dict(all_params[0]))
    params.task_id = 0
    params.data_dir = [str(train_raw)]
    params.cond_dir = [str(train_cond)]
    params.batch_size = 4
    params.learning_rate = 1e-3
    params.max_iter = 200
    params.model_dir = str(ckpt_dir)
    params.log_dir = str(out_dir)
    params.num_block = 4   # down from 32
    params.hidden_dim = 64  # down from 128
    params.embed_dim = 64
    params.sample_rate = 512
    params.input_dim = 90
    params.extra_dim = [90]
    params.cond_dim = 6
    params.num_heads = 4
    params.signal_diffusion = True
    params.max_step = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device: %s", device)
    LOGGER.info("Params: blocks=%d hidden=%d", params.num_block, params.hidden_dim)

    dataset = from_path(AttrDict(params))
    LOGGER.info("Dataset size: %d batches", len(dataset))

    model = tfdiff_WiFi(AttrDict(params)).to(device)
    optim = AdamW(model.parameters(), lr=params.learning_rate)

    # Build diffusion object manually (we only need degrade_fn and reverse-time kernels).
    from tfdiff.diffusion import SignalDiffusion
    diffusion = SignalDiffusion(AttrDict(params))

    losses: list[float] = []
    reset_peak_memory(device)
    start = time.perf_counter()
    iter_idx = 0
    for epoch in range(999):
        for batch in dataset:
            data = batch["data"].to(device)
            cond = batch["cond"].to(device)
            # data is [B, N, S, 2]; keep it as-is to match official learner
            B = data.shape[0]
            t = torch.randint(0, params.max_step, (B,), dtype=torch.int64, device=device)
            # Use official degrade_fn shape expectations [B, N, S*A, 2]
            # For Wi-Fi: S*A = 90 (treating A=1 as collapsed into S), data is [B, N, S*A, 2]
            x_t = diffusion.degrade_fn(data, t.cpu(), params.task_id)
            # network predicts x_0_hat conditioned on cond
            x_0_hat = model(x_t, t, cond)
            loss = torch.mean((x_0_hat - data) ** 2)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
            iter_idx += 1
            if iter_idx % 10 == 0:
                LOGGER.info("iter=%d loss=%.6f peak_mem=%.1f MB",
                            iter_idx, loss.item(), peak_gpu_memory_mb(device))
            if iter_idx >= params.max_iter:
                break
        if iter_idx >= params.max_iter:
            break
    total_time = time.perf_counter() - start

    ckpt_path = ckpt_dir / "weights.pt"
    torch.save({"model": model.state_dict(), "params": dict(params)}, ckpt_path)

    metrics = {
        "task": "wifi",
        "mode": "small-train",
        "run_id": run_id,
        "num_iter": iter_idx,
        "batch_size": params.batch_size,
        "num_block": params.num_block,
        "hidden_dim": params.hidden_dim,
        "final_loss": float(losses[-1]) if losses else None,
        "avg_loss_last_20": float(np.mean(losses[-20:])) if len(losses) >= 20 else float(np.mean(losses)) if losses else None,
        "loss_curve": losses,
        "total_time_s": total_time,
        "peak_gpu_mem_mb": peak_gpu_memory_mb(device),
        "device": str(device),
        "config_seed": 11,
        "notes": "Small-scale training with 4 blocks, 64 hidden dim, 200 iters. "
                 "Not comparable to full RF-Diffusion training.",
    }
    metrics_path = RESULTS_ROOT / "metrics" / f"small_train_{run_id}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    LOGGER.info("Saved metrics: %s", metrics_path)
    LOGGER.info("Final loss: %.6f | Total time: %.1fs | Peak GPU: %.1f MB",
                metrics["final_loss"], total_time, metrics["peak_gpu_mem_mb"])


if __name__ == "__main__":
    main()