#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CANDIDATE=${1:?existing baseline candidate directory}
PARENT=${2:-$ROOT/model_design/runs/stage2-dbaasp-expanded-300/checkpoints/final.pt}
MAIN=${3:-$ROOT/model_design/runs/s5-continuous-four-domain-v1/main}
RUN_ROOT=$ROOT/model_design/runs/s5-continuous-four-domain-v1
STATE=$RUN_ROOT/evaluation/fixed/best_fixed_state.json
CONFIG=$ROOT/model_design/configs/s5_continuous.yaml
PYTHON_BIN=${PYTHON_BIN:-/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python}
export REPO_ROOT=$ROOT
export PYTHONPATH=$ROOT/model_design:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=7
if "$PYTHON_BIN" -m muse_encoder.s5.evaluate_candidate \
  --config "$CONFIG" --checkpoint "$PARENT" \
  --evaluation-root "$(dirname "$CANDIDATE")" --resume-from "$CANDIDATE" \
  --mode full --device cuda:0 --encoding-devices 0 --cpu-workers 14 \
  --state "$STATE" --bootstrap; then
  "$PYTHON_BIN" - "$STATE" "$PARENT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
state = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
best = state["best_fixed"]
assert best["complete"] and not best["smoke"] and best["is_full_a_board"]
assert (best["scored_tasks"], best["num_tasks"]) == (90, 90)
assert best["shadow_status"] == "ok" and best["shadow_ok"] == 8
assert best["fixed_probe_protocol"] == "muse_encoder.s5_fixed_robust_probe.v1"
assert best["encoder_missing_fraction"] <= 0.001
assert best["frozen_checkpoint"]["sha256"] == hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
print(json.dumps({"baseline_ready": True, "J_encoder_fixed": best["J_encoder_fixed"],
                  "shadow_ok": best["shadow_ok"]}), flush=True)
PY
  TRAIN_PID=$(cat "$MAIN/orchestrator.pid")
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "baseline 已通过，但正式训练驱动不存在，拒绝孤立启动 watcher" >&2
    exit 1
  fi
  WATCH_LOG=$RUN_ROOT/logs/main_watcher.local7.log
  if [[ -e "$MAIN/watcher.pid" || -e "$WATCH_LOG" ]]; then
    echo "watcher 已存在，拒绝覆盖" >&2
    exit 1
  fi
  nohup setsid bash "$ROOT/model_design/launches/s5-continuous-20261104/start_evaluation.sh" \
    "$MAIN" "$RUN_ROOT/evaluation/fixed/candidates" "$STATE" > "$WATCH_LOG" 2>&1 < /dev/null &
  WATCH_PID=$!
  echo "$WATCH_PID" > "$MAIN/watcher.pid"
  echo "baseline 已通过；watcher=$WATCH_PID；未重复启动训练"
else
  RC=$?
  echo "baseline 未通过，未启动 watcher；训练进程不受影响" >&2
  exit "$RC"
fi
