from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from muse_encoder.pretrain.trainer import sha256_file
from s4_common import RUN_ROOT, PARENT, PARENT_SHA256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    evaluation = RUN_ROOT / "evaluation"
    parent_full_path = evaluation / "step0/parent_full.json"
    shadow_path = evaluation / "step0/shadow_parent_v3/shadow_results.json"
    stopped_path = RUN_ROOT / "stopped.json"
    parent_full = read(parent_full_path)
    shadow = read(shadow_path)
    stopped = read(stopped_path)
    result_paths = [
        evaluation / "step0/gate_parent/results.json",
        evaluation / "step0/full_remaining/results.json",
    ]
    tasks = {}
    heads = {}
    encoding_failures = []
    for result_path in result_paths:
        payload = read(result_path)
        if payload["checkpoint"]["sha256"] != PARENT_SHA256:
            raise ValueError("父模型 checkpoint 身份不一致")
        for task_id, row in payload["tasks"].items():
            if row.get("status") == "not_run":
                continue
            if task_id in tasks:
                raise ValueError(f"重复任务: {task_id}")
            tasks[task_id] = row
            heads[task_id] = row["status"]
            if row.get("missing_predictions"):
                encoding_failures.append({
                    "task_id": task_id, "split": "prediction",
                    "fallback_predictions": row["missing_predictions"],
                })
    if set(tasks) != set(parent_full["tasks"]):
        raise ValueError("90项分数与任务运行记录不对齐")
    for result_path in result_paths:
        for path in (result_path.parent / "embeddings").glob("*/*/*.encoding.json"):
            payload = read(path)
            for failure in payload.get("failed") or []:
                encoding_failures.append({
                    "task_subset": str(path.parent.relative_to(result_path.parent / "embeddings")),
                    "split": path.name.split(".")[0], **failure,
                })
    status_counts = dict(Counter(heads.values()))
    report = {
        "schema_version": "muse.s4.final_report.v1",
        "run_name": "s4-overnight-polymer-electronic-consolidation-v1",
        "training_outcome": "stopped_at_50_gate_failed",
        "selected_checkpoint": str(PARENT),
        "selected_checkpoint_sha256": PARENT_SHA256,
        "selection_reason": "No new candidate accepted; treatment gap_mean did not exceed replay-only control",
        "candidate_full_validation_run": False,
        "candidate_full_validation_skip_reason": "Rejected by mandatory step-50 gate; no later training was run",
        "parent_validation": {
            "path": str(parent_full_path),
            "sha256": sha256_file(parent_full_path),
            "J_val": parent_full["J_val"],
            "num_tasks": parent_full["num_tasks"],
            "scored_tasks": parent_full["scored_tasks"],
            "complete": parent_full["complete"],
            "datasets": parent_full["datasets"],
            "gap_mean": parent_full["gap_mean"],
        },
        "shadow_B": {
            "path": str(shadow_path), "sha256": sha256_file(shadow_path), **shadow["summary"],
        },
        "gate_50": {
            "path": str(stopped_path), "passed": stopped["decision"]["passed"],
            "checks": stopped["decision"]["checks"],
            "dbaasp_rescue": stopped["decision"]["dbaasp_rescue"],
        },
        "head_status_counts": status_counts,
        "all_90_heads_ok": status_counts == {"ok": 90},
        "encoding_or_prediction_failures": encoding_failures,
        "num_failure_audit_entries": len(encoding_failures),
        "test_used_for_selection": False,
        "logs": {
            "agent_log": str(RUN_ROOT / "KIMI_AGENT_LOG.md"),
            "events": str(RUN_ROOT / "events.jsonl"),
            "code_hashes": str(RUN_ROOT / "code_sha256.txt"),
        },
    }
    output = RUN_ROOT / "final_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "J_val": report["parent_validation"]["J_val"],
        "scored_tasks": 90, "all_90_heads_ok": report["all_90_heads_ok"],
        "shadow_all_ok": report["shadow_B"]["all_ok"],
        "head_status_counts": status_counts,
        "num_failure_audit_entries": len(encoding_failures),
        "selected_checkpoint": report["selected_checkpoint"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
