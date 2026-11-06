"""Aggregate DBAASP, full PEER, and full CATH into the official J_bio objective."""
import argparse
import json
from pathlib import Path

PEER_TASKS = 16
CATH_TASKS = 4


def dataset_scores(peer_paths, cath_path):
    tasks = {}
    for path in peer_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("evaluation", {}).get("split") != "validation":
            raise ValueError(f"只允许 validation: {path}")
        for task_id, row in payload["tasks"].items():
            if not task_id.startswith("PEER/") or row.get("status") != "scored":
                continue
            if task_id in tasks:
                raise ValueError(f"重复 PEER 任务: {task_id}")
            tasks[task_id] = float(row["score"])
    if len(tasks) != PEER_TASKS:
        raise ValueError(f"PEER 应有{PEER_TASKS}项计分，实际{len(tasks)}")
    payload = json.loads(Path(cath_path).read_text(encoding="utf-8"))
    if payload.get("evaluation", {}).get("split") != "validation":
        raise ValueError(f"只允许 validation: {cath_path}")
    cath = [float(row["score"]) for task_id, row in payload["tasks"].items()
            if task_id.startswith("CATH/") and row.get("status") == "scored"]
    if len(cath) != CATH_TASKS:
        raise ValueError(f"CATH 应有{CATH_TASKS}项计分，实际{len(cath)}")
    return {"PEER": 100.0 * sum(tasks.values()) / PEER_TASKS,
            "CATH": 100.0 * sum(cath) / CATH_TASKS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbaasp", type=Path, required=True)
    parser.add_argument("--peer", type=Path, nargs="+", required=True)
    parser.add_argument("--cath", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dbaasp = json.loads(args.dbaasp.read_text(encoding="utf-8"))
    if dbaasp.get("test_used") is not False:
        raise ValueError("J_bio 只允许 validation")
    scores = dataset_scores(args.peer, args.cath)
    scores["DBAASP"] = 100.0 * float(dbaasp["validation_metrics"]["score"])
    result = {
        "schema_version": "muse.stage2.bio_scores.v1",
        "datasets": scores, "J_bio": sum(scores.values()), "test_used": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
