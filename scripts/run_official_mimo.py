"""Run the official 5G/MIMO inference (Level-1 reproduction)."""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "upstream" / "RF-Diffusion"
RESULTS_ROOT = PROJECT_ROOT / "results"


def main() -> None:
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    log_path = RESULTS_ROOT / "logs" / f"official_mimo_{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["PYTHONPATH"] = f"{UPSTREAM_ROOT}:{env.get('PYTHONPATH', '')}"
    start = time.perf_counter()
    cmd = ["python3", "inference.py", "--task_id", "2"]
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"CMD: {' '.join(cmd)}\n\n")
        proc = subprocess.run(cmd, cwd=str(UPSTREAM_ROOT), env=env,
                              stdout=logf, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - start
    metrics = {
        "task": "mimo_5g",
        "mode": "official_full",
        "run_id": run_id,
        "elapsed_s": elapsed,
        "log_path": str(log_path),
        "return_code": proc.returncode,
    }
    metrics_path = RESULTS_ROOT / "metrics" / f"official_mimo_{run_id}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[official_mimo] done in {elapsed:.1f}s, return code {proc.returncode}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()