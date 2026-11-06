#!/usr/bin/env bash
set -euo pipefail
RUN_DIR=${1:?run_dir}; PARENT=${2:?自有父 checkpoint}; TARGET=${3:-1000000}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export REPO_ROOT=${REPO_ROOT:-$ROOT}
export PYTHONPATH="$REPO_ROOT/model_design:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYTHON_BIN=${PYTHON_BIN:-/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python}
CONFIG=${CONFIG:-$REPO_ROOT/model_design/configs/s5_continuous.yaml}
MASTER_ADDR=${MASTER_ADDR:-172.17.0.171}
MASTER_PORT=${MASTER_PORT:-29815}
REMOTE=${REMOTE:-172.17.0.178}
MAX_FAILURES=${MAX_FAILURES:-20}
mkdir -p "$RUN_DIR"
export NCCL_SOCKET_IFNAME==eth0 GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA==mlx5_bond_0:1,mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1
export NCCL_IB_GID_INDEX=3 NCCL_IB_DISABLE=0 NCCL_NET=IB
export S5_PROCESS_PREFETCH=${S5_PROCESS_PREFETCH:-1}
export S5_NCCL_TIMEOUT_SECONDS=${S5_NCCL_TIMEOUT_SECONDS:-900}
LOCAL_PID=""
REMOTE_PID=""

stop_local() {
  if [[ -n "$LOCAL_PID" ]] && kill -0 "$LOCAL_PID" 2>/dev/null; then
    pkill -TERM -P "$LOCAL_PID" 2>/dev/null || true
    kill -TERM "$LOCAL_PID" 2>/dev/null || true
    wait "$LOCAL_PID" 2>/dev/null || true
  fi
}

stop_remote() {
  if [[ -n "$REMOTE_PID" ]]; then
    ssh -o BatchMode=yes "$REMOTE" "pkill -TERM -P '$REMOTE_PID' 2>/dev/null || true; kill -TERM '$REMOTE_PID' 2>/dev/null || true" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  stop_local
  stop_remote
}
trap cleanup EXIT INT TERM

failures=0
attempt=1
while compgen -G "$RUN_DIR/driver.*.attempt${attempt}.*" >/dev/null; do
  attempt=$((attempt + 1))
done
while true; do
  if [[ -f "$RUN_DIR/latest.json" ]]; then
    STEP=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "$RUN_DIR/latest.json")
    [[ "$STEP" -lt "$TARGET" ]] || break
    MODE=(--resume "$RUN_DIR/latest.json")
  else
    MODE=(--init-weights "$PARENT")
  fi
  printf -v MODE_TEXT '%q ' "${MODE[@]}"
  REMOTE_PID_FILE="$RUN_DIR/driver.node1.attempt${attempt}.pid"
  REMOTE_RC_FILE="$RUN_DIR/driver.node1.attempt${attempt}.rc"
  LOCAL_LOG="$RUN_DIR/driver.node0.attempt${attempt}.log"
  REMOTE_LOG="$RUN_DIR/driver.node1.attempt${attempt}.log"
  for path in "$REMOTE_PID_FILE" "$REMOTE_RC_FILE" "$LOCAL_LOG" "$REMOTE_LOG"; do
    [[ ! -e "$path" ]] || { echo "拒绝覆盖 $path" >&2; exit 2; }
  done
  ssh -f -o BatchMode=yes "$REMOTE" "cd '$ROOT' && echo \$\$ > '$REMOTE_PID_FILE' && export REPO_ROOT='$REPO_ROOT' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='$ROOT/model_design' OMP_NUM_THREADS='$OMP_NUM_THREADS' PYTORCH_ALLOC_CONF='$PYTORCH_ALLOC_CONF' CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_SOCKET_IFNAME==eth0 GLOO_SOCKET_IFNAME=eth0 NCCL_IB_HCA==mlx5_bond_0:1,mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1 NCCL_IB_GID_INDEX=3 NCCL_IB_DISABLE=0 NCCL_NET=IB S5_PROCESS_PREFETCH='$S5_PROCESS_PREFETCH' S5_NCCL_TIMEOUT_SECONDS='$S5_NCCL_TIMEOUT_SECONDS'; '$PYTHON_BIN' -m torch.distributed.run --nnodes=2 --nproc_per_node=8 --node_rank=1 --master_addr='$MASTER_ADDR' --master_port='$MASTER_PORT' -m muse_encoder.s5.train --config '$CONFIG' --run-dir '$RUN_DIR' --stop-after '$TARGET' $MODE_TEXT > '$REMOTE_LOG' 2>&1 < /dev/null; rc=\$?; echo \$rc > '$REMOTE_RC_FILE'; exit \$rc"
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
  "$PYTHON_BIN" -m torch.distributed.run --nnodes=2 --nproc_per_node=7 \
    --node_rank=0 --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    -m muse_encoder.s5.train --config "$CONFIG" --run-dir "$RUN_DIR" \
    --stop-after "$TARGET" "${MODE[@]}" > "$LOCAL_LOG" 2>&1 &
  LOCAL_PID=$!
  for _ in $(seq 1 60); do
    [[ -f "$REMOTE_PID_FILE" ]] && break
    kill -0 "$LOCAL_PID" 2>/dev/null || break
    sleep 1
  done
  [[ -f "$REMOTE_PID_FILE" ]] && REMOTE_PID=$(cat "$REMOTE_PID_FILE")
  while true; do
    if [[ -f "$REMOTE_RC_FILE" ]] && [[ $(cat "$REMOTE_RC_FILE") -ne 0 ]]; then
      stop_local
    fi
    if ! kill -0 "$LOCAL_PID" 2>/dev/null; then
      set +e
      wait "$LOCAL_PID"
      LOCAL_RC=$?
      set -e
      if [[ $LOCAL_RC -ne 0 && ! -f "$REMOTE_RC_FILE" ]]; then
        stop_remote
      fi
      break
    fi
    sleep 5
  done
  for _ in $(seq 1 120); do
    [[ -f "$REMOTE_RC_FILE" ]] && break
    [[ $LOCAL_RC -eq 0 ]] || break
    sleep 5
  done
  if [[ -f "$REMOTE_RC_FILE" ]]; then
    REMOTE_RC=$(cat "$REMOTE_RC_FILE")
  else
    stop_remote
    REMOTE_RC=124
  fi
  if [[ $LOCAL_RC -eq 0 && $REMOTE_RC -eq 0 ]]; then
    break
  fi
  echo "第 $attempt 次双机训练失败：node0=$LOCAL_RC node1=$REMOTE_RC" >&2
  if "$PYTHON_BIN" - "$LOCAL_LOG" "$REMOTE_LOG" <<'PY'
import sys
from pathlib import Path

markers = (
    "官方 3D 新增读取失败",
    "数据清单",
    "data manifest",
    "data_manifest",
    "sha256 mismatch",
    "SHA256 mismatch",
)
for name in sys.argv[1:]:
    path = Path(name)
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    matched = next((marker for marker in markers if marker in text), None)
    if matched is not None:
        print(f"检测到不可自动重试错误：{path}: {matched}", file=sys.stderr)
        raise SystemExit(0)
raise SystemExit(1)
PY
  then
    echo '确定性数据或清单/hash 错误，停止自动重启并保留现场' >&2
    exit 1
  fi
  failures=$((failures + 1))
  [[ $failures -le $MAX_FAILURES ]] || { echo '连续失败超过上限，停止自动重启并保留现场' >&2; exit 1; }
  stop_local
  stop_remote
  sleep $((failures * 30))
  attempt=$((attempt + 1))
done
