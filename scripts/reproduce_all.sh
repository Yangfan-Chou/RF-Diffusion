#!/usr/bin/env bash
# One-shot reproduction script.
# Usage: bash scripts/reproduce_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p results/{raw,logs,metrics}

log() { printf "[reproduce_all %s] %s\n" "$(date '+%H:%M:%S')" "$*"; }

log "1/10 Environment probe"
python - <<'PY'
import json, platform
from pathlib import Path
info = {
    "python": platform.python_version(),
    "torch": __import__("torch").__version__,
    "cuda_available": __import__("torch").cuda.is_available(),
    "device_name": __import__("torch").cuda.get_device_name(0)
              if __import__("torch").cuda.is_available() else "cpu",
}
print(json.dumps(info, indent=2))
PY

log "2/10 Data + model sanity check"
UPSTREAM="$ROOT/upstream/RF-Diffusion"
[[ -d "$UPSTREAM/dataset/wifi/cond" ]] || { log "FATAL: dataset/wifi/cond missing"; exit 1; }
[[ -d "$UPSTREAM/model/wifi" ]] || { log "FATAL: model/wifi missing"; exit 1; }
[[ -d "$UPSTREAM/model/mimo" ]] || { log "FATAL: model/mimo missing"; exit 1; }
log "  cond files: $(ls $UPSTREAM/dataset/wifi/cond | wc -l)"
log "  Wi-Fi weights: $(ls $UPSTREAM/model/wifi/*/weights.pt 2>/dev/null | wc -l)"

log "3/10 Wi-Fi smoke test"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  python scripts/run_experiment.py --task wifi --mode smoke-test --num-samples 2 \
  | tee results/logs/repro_smoke.log | tail -10

log "4/10 Wi-Fi official inference"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  python scripts/run_official_wifi.py \
  | tee results/logs/repro_wifi.log | tail -5

log "5/10 5G FDD official inference"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} \
  python scripts/run_official_mimo.py \
  | tee results/logs/repro_mimo.log | tail -5

log "6/10 Small-scale training"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} \
  python scripts/run_small_train.py \
  | tee results/logs/repro_small_train.log | tail -5

log "7/10 Efficiency extension"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} \
  python scripts/run_efficiency.py \
  | tee results/logs/repro_eff.log | tail -10

log "8/10 Aggregate metrics"
python scripts/aggregate_metrics.py

log "9/10 Generate figures"
python scripts/plot_results.py

log "10/10 Build LaTeX report"
(cd report && bash build.sh) 2>&1 | tail -5

log "DONE. See results/metrics/ and figures/ for outputs."