"""为未完成的子集生成官方基线兜底预测（常数模型，归一化 0 分）。

仅补齐缺失的 prediction 文件；已存在的真实预测不动。兜底值走 dashboard
原有的 _fallback_value 路径（回归=官方 train_median，二分类=0.5，多分类=均匀）。
"""
import json
import sys
from pathlib import Path

from muse_encoder.pretrain import dashboard

ROOT = Path("/yinghepool/tech-dcu/zhaoyao")
STAGE = ROOT / "model_design/launches/eval16-20261027"


def main():
    mode, shard_root = sys.argv[1], Path(sys.argv[2])
    shard_map = {}
    for shard in json.loads((STAGE / "shards.json").read_text())["shards"]:
        for key in shard["subsets"]:
            shard_map[key] = shard["name"]
    board = dashboard.load_board(dashboard.DATA_ROOT, "A")
    subset_map = {f"{s.dataset}/{s.subset}": s for s in board.subsets.values()}
    made = []
    for task in board.tasks:
        key = f"{task.dataset}/{task.subset}"
        pred_dir = shard_root / shard_map[key] / "predictions"
        path = pred_dir / task.prediction_file
        if path.exists():
            continue
        subset = subset_map[key]
        records = dashboard._read_jsonl(dashboard._split_file(board.data_root, subset, mode))
        dashboard.write_task_predictions(path, task, records, [], {})
        if not (pred_dir / "manifest.json").exists():
            (pred_dir / "manifest.json").write_text(json.dumps({
                "team": "zhaoyao", "name": "muse-encoder-fixed-probe", "board": "A",
                "split": mode, "heads": {}, "smoke": False, "limit_per_split": None,
            }, indent=2), encoding="utf-8")
        made.append(task.task_id)
    print(json.dumps({"mode": mode, "fallback_tasks": made, "count": len(made)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
