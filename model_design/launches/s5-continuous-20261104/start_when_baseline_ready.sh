#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
RUN_DIR=${1:?formal run_dir}
PARENT=${2:?parent checkpoint}
BASELINE_PID_FILE=${3:?baseline pid file}
STATE=${4:-$ROOT/model_design/runs/s5-continuous-four-domain-v1/evaluation/fixed/best_fixed_state.json}
TARGET=${5:-1000000}
REMOTE=${REMOTE:-172.17.0.178}
PYTHON_BIN=${PYTHON_BIN:-/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python}
LOG_ROOT=$(dirname "$RUN_DIR")
POLL_SECONDS=${POLL_SECONDS:-60}
while [[ ! -f "$STATE" ]]; do
  BASELINE_PID=$(cat "$BASELINE_PID_FILE")
  if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" "kill -0 '$BASELINE_PID' 2>/dev/null"; then
    echo "baseline 进程已退出且 best_fixed_state.json 未生成，拒绝启动连续训练" >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done
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
print(json.dumps({"baseline_ready": True, "J_encoder_fixed": best["J_encoder_fixed"], "shadow_ok": best["shadow_ok"]}))
PY
mkdir -p "$RUN_DIR"
TRAIN_LOG="$LOG_ROOT/logs/main.orchestrator.log"
WATCH_LOG="$LOG_ROOT/logs/main_watcher.log"
nohup setsid env MASTER_PORT=29828 bash "$ROOT/model_design/launches/s5-continuous-20261104/start_two_node_training.sh" \
  "$RUN_DIR" "$PARENT" "$TARGET" > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$RUN_DIR/orchestrator.pid"
nohup setsid bash "$ROOT/model_design/launches/s5-continuous-20261104/start_evaluation.sh" \
  "$RUN_DIR" "$LOG_ROOT/evaluation/fixed/candidates" "$STATE" > "$WATCH_LOG" 2>&1 &
WATCH_PID=$!
echo "$WATCH_PID" > "$RUN_DIR/watcher.pid"
echo "连续训练已启动：orchestrator=$TRAIN_PID watcher=$WATCH_PID"
