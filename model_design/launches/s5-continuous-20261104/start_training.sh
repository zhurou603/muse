#!/usr/bin/env bash
set -euo pipefail
RUN_DIR=${1:?run_dir}; PARENT=${2:?自有父 checkpoint}; TARGET=${3:-1000000}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export REPO_ROOT=${REPO_ROOT:-$ROOT}
export PYTHONPATH="$REPO_ROOT/model_design:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
CONFIG=${CONFIG:-$REPO_ROOT/model_design/configs/s5_continuous.yaml}
TORCHRUN_BIN=${TORCHRUN_BIN:-/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/torchrun}
PYTHON_BIN=${PYTHON_BIN:-/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python}
GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6}
NPROC=${NPROC:-7}
export CUDA_VISIBLE_DEVICES="$GPUS"
mkdir -p "$RUN_DIR"
failures=0
while true; do
  if [[ -f "$RUN_DIR/latest.json" ]]; then
    STEP=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "$RUN_DIR/latest.json")
    [[ "$STEP" -lt "$TARGET" ]] || break
    MODE=(--resume "$RUN_DIR/latest.json")
  else
    MODE=(--init-weights "$PARENT")
  fi
  set +e
  "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC" -m muse_encoder.s5.train \
    --config "$CONFIG" --run-dir "$RUN_DIR" --stop-after "$TARGET" "${MODE[@]}" \
    >> "$RUN_DIR/driver.log" 2>&1
  RC=$?
  set -e
  [[ $RC -eq 0 ]] && break
  failures=$((failures + 1))
  [[ $failures -le 20 ]] || { echo '连续失败超过 20 次，停止自动重启并保留现场' >&2; exit "$RC"; }
  sleep $((failures * 30))
done
