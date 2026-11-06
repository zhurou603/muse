"""Aggregate official validation scores for the four polymer datasets."""
import argparse
import json
from pathlib import Path

DATASETS = (
    "Polymer-Genome-2016",
    "TransPolymer",
    "Aldeghi-Coley-Copolymer",
    "PolyGraphMT",
)
GAPS = (
    "Polymer-Genome-2016/DFT-HSE06/electronic_band_gap",
    "Polymer-Genome-2016/DFT-rPW86/electronic_band_gap",
)


def aggregate(paths):
    tasks = {}
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("evaluation", {}).get("split") != "validation":
            raise ValueError(f"只允许 validation scores: {path}")
        for task_id, row in payload.get("tasks", {}).items():
            if task_id.split("/", 1)[0] not in DATASETS:
                continue
            if row.get("status") != "scored":
                continue
            if task_id in tasks:
                raise ValueError(f"重复任务: {task_id}")
            tasks[task_id] = float(row["score"]) * 100.0
    if len(tasks) != 35:
        raise ValueError(f"聚合物任务应为35项，实际{len(tasks)}")
    dataset_scores = {
        dataset: sum(score for task_id, score in tasks.items() if task_id.startswith(dataset + "/")) \
            / sum(task_id.startswith(dataset + "/") for task_id in tasks)
        for dataset in DATASETS
    }
    return {
        "schema_version": "muse.stage3.polymer_scores.v1",
        "datasets": dataset_scores,
        "gap_scores": {task_id: tasks[task_id] for task_id in GAPS},
        "gap_mean": sum(tasks[task_id] for task_id in GAPS) / 2,
        "J_polymer": sum(dataset_scores.values()),
        "tasks": tasks,
        "num_tasks": len(tasks),
        "test_used": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.scores)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
