#!/usr/bin/env bash
set -euo pipefail
RUN_DIR=${1:?run_dir}; PARENT=${2:?自有父 checkpoint}; TARGET=${3:-1000000}
export REPO_ROOT=${REPO_ROOT:?先 export REPO_ROOT=/项目根目录}
export PYTHONPATH="$REPO_ROOT/code/model_design:${PYTHONPATH:-}"
CONFIG="$REPO_ROOT/code/model_design/configs/s5_continuous.yaml"
GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6}
NPROC=${NPROC:-7}
export CUDA_VISIBLE_DEVICES="$GPUS"
mkdir -p "$RUN_DIR"
failures=0
while true; do
  if [[ -f "$RUN_DIR/latest.json" ]]; then
    STEP=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "$RUN_DIR/latest.json")
    [[ "$STEP" -lt "$TARGET" ]] || break
    MODE=(--resume "$RUN_DIR/latest.json")
  else
    MODE=(--init-weights "$PARENT")
  fi
  set +e
  torchrun --standalone --nproc_per_node="$NPROC" -m muse_encoder.s5.train \
    --config "$CONFIG" --run-dir "$RUN_DIR" --stop-after "$TARGET" "${MODE[@]}" \
    >> "$RUN_DIR/driver.log" 2>&1
  RC=$?
  set -e
  [[ $RC -eq 0 ]] && break
  failures=$((failures + 1))
  [[ $failures -le 20 ]] || { echo '连续失败超过 20 次，停止自动重启并保留现场' >&2; exit $RC; }
  sleep $((failures * 30))
done
