#!/usr/bin/env bash
set -euo pipefail
CKPT=${1:?frozen checkpoint}; RUN=${2:?new A-test run directory}; NAME=${3:?submission name}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
REPAIRS=${4:-$ROOT/data/pretrain/manifest/qm9_official_graph_repairs.v3.json}
export REPO_ROOT=$ROOT
export PYTHONPATH="$ROOT/model_design:$ROOT"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=7
PY=/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python
DATA=$ROOT/data/muse-a/MUSE-A/data
EVAL=$ROOT/data/muse-a/eval/MUSE-eval
CACHE_ROOT=${CACHE_ROOT:-$ROOT/data/pretrain/cache/frozen_eval}
mkdir -p "$RUN"
"$PY" "$ROOT/model_design/launches/eval16-20261027/fast_eval.py" \
  --checkpoint "$CKPT" --output-dir "$RUN" \
  --cache-root "$CACHE_ROOT" \
  --external-cache "$ROOT/data/pretrain/cache/omc25" \
  --device cuda:0 --devices 0 --cpu-workers 14 --batch-size 64 \
  --train-cap 50000 --second-split test --single-seed \
  --official-graph-repairs "$REPAIRS" \
  --team zhaoyao --name "$NAME"
"$PY" "$EVAL/score.py" tasks --data-root "$DATA" --board A --out "$RUN/tasks.json"
"$PY" "$EVAL/score.py" score --predictions "$RUN/predictions" \
  --data-root "$DATA" --board A --split test --out "$RUN/scores.json"
"$PY" "$EVAL/score.py" check --scores "$RUN/scores.json" \
  --data-root "$DATA" --board A > "$RUN/check.json"
"$PY" - "$RUN" "$CKPT" "$REPAIRS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
scores = json.loads((run / "scores.json").read_text())
check = json.loads((run / "check.json").read_text())
manifest = json.loads((run / "predictions/manifest.json").read_text())
assert scores["schema_version"] == "muse.scores.v1"
assert scores["evaluation"]["board"] == "A"
assert scores["evaluation"]["split"] == "test"
assert scores["num_domains"] == 4 and scores["num_datasets"] == 13
assert scores["num_tasks"] == scores["scored_tasks"] == 90
assert scores["complete"] is True
assert scores["missing_predictions"] == [] and scores["invalid_predictions"] == []
assert check == {"consistent": True, "problems": []}
assert manifest["board"] == "A" and manifest["split"] == "test"
assert not manifest.get("heads")
result = {
    "checkpoint": str(Path(sys.argv[2]).resolve()),
    "checkpoint_sha256": hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest(),
    "input_repairs": str(Path(sys.argv[3]).resolve()),
    "input_repairs_sha256": hashlib.sha256(Path(sys.argv[3]).read_bytes()).hexdigest(),
    "train_cap": 50000,
    "single_seed": True,
    "final_score": scores["final_score"],
    "complete": scores["complete"],
    "scored_tasks": scores["scored_tasks"],
    "check": check,
    "scores": str(run / "scores.json"),
}
(run / "a_test_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False), flush=True)
PY
