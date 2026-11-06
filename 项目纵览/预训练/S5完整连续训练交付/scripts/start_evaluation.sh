#!/usr/bin/env bash
set -euo pipefail
RUN_DIR=${1:?训练 run_dir}; EVAL_ROOT=${2:?独立 evaluation_root}; STATE=${3:?best_fixed_state.json}
export REPO_ROOT=${REPO_ROOT:?先 export REPO_ROOT=/项目根目录}
export PYTHONPATH="$REPO_ROOT/code/model_design:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${EVAL_GPU:-7}
mkdir -p "$EVAL_ROOT"
python3 -m muse_encoder.s5.watch_evaluation \
  --config "$REPO_ROOT/code/model_design/configs/s5_continuous.yaml" \
  --train-run "$RUN_DIR" --evaluation-root "$EVAL_ROOT" --state "$STATE" \
  --device cuda:0 --poll-seconds 300
