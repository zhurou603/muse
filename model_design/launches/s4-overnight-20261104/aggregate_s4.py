from __future__ import annotations

import argparse
import json
from pathlib import Path

from s4_common import DATASET_DOMAINS, DATASET_TASK_COUNTS, GAPS


def official_weighted_score(dataset_scores):
    if set(dataset_scores) != set(DATASET_TASK_COUNTS):
        raise ValueError("J_val 需要全部13个数据集")
    return (
        sum(dataset_scores[name] for name in DATASET_DOMAINS["small_molecule"]) / 20
        + sum(dataset_scores[name] for name in DATASET_DOMAINS["polymer"]) / 16
        + dataset_scores["OMC25"] / 4
        + sum(dataset_scores[name] for name in DATASET_DOMAINS["biomacromolecule"]) / 12
    )


def aggregate(paths, require="gate"):
    tasks = {}
    sources = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("evaluation", {}).get("board") != "A" \
                or payload.get("evaluation", {}).get("split") != "validation":
            raise ValueError(f"只允许 A 榜 validation scores: {path}")
        sources.append({"path": str(path), "num_tasks": payload.get("num_tasks"),
                        "scored_tasks": payload.get("scored_tasks")})
        for task_id, row in payload.get("tasks", {}).items():
            if row.get("status") != "scored":
                continue
            if task_id in tasks:
                raise ValueError(f"重复任务: {task_id}")
            tasks[task_id] = float(row["score"]) * 100.0
    expected = 35 + 1 if require == "gate" else 90
    if len(tasks) != expected:
        raise ValueError(f"{require} 应评分 {expected} 项，实际 {len(tasks)}")
    dataset_scores = {}
    for dataset, count in DATASET_TASK_COUNTS.items():
        selected = [score for task_id, score in tasks.items() if task_id.startswith(dataset + "/")]
        if selected and len(selected) != count:
            raise ValueError(f"{dataset}: 任务不完整 {len(selected)}/{count}")
        if selected:
            dataset_scores[dataset] = sum(selected) / len(selected)
    if require == "gate":
        needed = set(DATASET_DOMAINS["polymer"]) | {"dbaasp_v4"}
        if set(dataset_scores) != needed:
            raise ValueError(f"门控数据集不匹配: {sorted(dataset_scores)}")
        j_val = None
    else:
        j_val = official_weighted_score(dataset_scores)
    polymer_sum = sum(dataset_scores.get(name, 0.0) for name in DATASET_DOMAINS["polymer"])
    return {
        "schema_version": "muse.s4.official_validation_scores.v1",
        "scope": require,
        "sources": sources,
        "evaluation": {"board": "A", "split": "validation"},
        "complete": require == "full",
        "scored_tasks": len(tasks),
        "final_score": j_val,
        "num_tasks": len(tasks),
        "datasets": dataset_scores,
        "tasks": tasks,
        "gap_scores": {name: tasks[name] for name in GAPS},
        "gap_mean": sum(tasks[name] for name in GAPS) / 2,
        "dbaasp_default": tasks["dbaasp_v4/target_activity/target_activity"],
        "polymer_dataset_sum": polymer_sum,
        "J_val": j_val,
        "test_used": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--require", choices=("gate", "full"), default="gate")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.scores, args.require)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({
        "scope": result["scope"], "num_tasks": result["num_tasks"],
        "gap_mean": result["gap_mean"], "dbaasp_default": result["dbaasp_default"],
        "polymer_dataset_sum": result["polymer_dataset_sum"], "J_val": result["J_val"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
