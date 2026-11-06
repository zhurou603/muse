#!/usr/bin/env bash
set -euo pipefail
RUN_DIR=${1:?训练 run_dir}; EVAL_ROOT=${2:?独立 evaluation_root}; STATE=${3:?best_fixed_state.json}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export REPO_ROOT=${REPO_ROOT:-$ROOT}
export PYTHONPATH="$REPO_ROOT/model_design:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export CUDA_VISIBLE_DEVICES=${EVAL_GPU:-7}
PYTHON_BIN=${PYTHON_BIN:-/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python}
CONFIG=${CONFIG:-$REPO_ROOT/model_design/configs/s5_continuous.yaml}
mkdir -p "$EVAL_ROOT"
"$PYTHON_BIN" -m muse_encoder.s5.watch_evaluation \
  --config "$CONFIG" \
  --train-run "$RUN_DIR" --evaluation-root "$EVAL_ROOT" --state "$STATE" \
  --device cuda:0 --poll-seconds 300
