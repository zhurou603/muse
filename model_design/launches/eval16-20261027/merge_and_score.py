"""合并 16 个分片的 predictions，调用官方 scorer 产出全量 90 任务官方分。

不修改 dashboard.py / score.py；表格复用 dashboard 的内部函数。
"""
import argparse
import json
import os
from pathlib import Path

from muse_encoder.pretrain import dashboard


def collect_validated_predictions(shard_dirs, tasks, split):
    expected = {task.prediction_file: task for task in tasks}
    if len(expected) != len(tasks):
        raise ValueError("官方任务清单位置重复")
    predictions = {}
    task_runs = {}
    provenance = []
    checkpoint_sha = None
    model_config = None
    probe_config = None
    for shard in shard_dirs:
        state = json.loads((shard / "run_state.json").read_text(encoding="utf-8"))
        current_sha = state["checkpoint"]["sha256"]
        if state["probe_config"].get("seeds") != [0]:
            raise ValueError(f"{shard}: 正式合并只接受单种子 MLP")
        if checkpoint_sha is None:
            checkpoint_sha = current_sha
            model_config = state["model_config"]
            probe_config = state["probe_config"]
        elif current_sha != checkpoint_sha:
            raise ValueError(f"{shard}: checkpoint 与其他分片不一致")
        if state["model_config"] != model_config or state["probe_config"] != probe_config:
            raise ValueError(f"{shard}: 模型或 MLP 配置与其他分片不一致")
        if probe_config.get("seeds") != [0]:
            raise ValueError(f"{shard}: 正式合并只接受单种子 MLP")
        manifest_path = shard / "predictions" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("board") != "A" or manifest.get("split") != split:
            raise ValueError(f"{shard}: manifest 必须是 A/{split}")
        if manifest.get("heads"):
            raise ValueError(f"{shard}: 存在异常 MLP 状态")
        shard_files = {}
        for path in (shard / "predictions").rglob("*.csv.gz"):
            relative = path.relative_to(shard / "predictions").as_posix()
            if relative not in expected:
                raise ValueError(f"{shard}: 非官方任务预测 {relative}")
            if relative in predictions:
                raise ValueError(f"重复任务预测: {expected[relative].task_id}")
            shard_files[relative] = path
            predictions[relative] = path
        for cache in (shard / "task_cache").glob("*.json"):
            result = json.loads(cache.read_text(encoding="utf-8"))
            task_runs[result["task_id"]] = result
        provenance.append({
            "path": str(shard),
            "checkpoint_sha256": current_sha,
            "prediction_count": len(shard_files),
        })
    missing = [task.task_id for task in tasks if task.prediction_file not in predictions]
    if missing:
        raise ValueError(f"缺少 {len(missing)} 个任务预测: {missing}")
    if len(predictions) != len(tasks):
        raise ValueError("预测任务数与官方清单不一致")
    return predictions, task_runs, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--team", default="zhaoyao")
    parser.add_argument("--name", default="muse-encoder-fixed-probe")
    parser.add_argument("--train-cap", type=int, default=50_000)
    args = parser.parse_args()
    board = dashboard.load_board(dashboard.DATA_ROOT, "A")
    merged = args.output_dir / "predictions"
    if merged.exists() and any(merged.iterdir()):
        raise FileExistsError(f"合并目录必须为空，禁止混入旧预测: {merged}")
    shard_dirs = sorted(p for p in args.shards_root.iterdir() if (p / "predictions").is_dir())
    if not shard_dirs:
        raise ValueError("没有可合并分片")
    predictions, task_runs, provenance = collect_validated_predictions(
        shard_dirs, board.tasks, args.split)
    merged.mkdir(parents=True, exist_ok=False)
    for relative, source in predictions.items():
        target = merged / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)
    dashboard._write_json(merged / "manifest.json", {
        "team": args.team, "name": args.name, "board": "A",
        "split": args.split, "heads": {}, "smoke": False, "limit_per_split": None,
    })
    official = dashboard.official_scorer().score_predictions(merged, board.data_root, "A", split=args.split)
    dashboard._write_json(args.output_dir / "official_scores.json", official)
    rows = dashboard._merge_official_scores(board.tasks, official, task_runs)
    summary = dict(official)
    summary["scope"] = f"full_{args.split}_train_capped_{args.train_cap}"
    summary["is_full_a_board"] = True
    dashboard._write_tables(args.output_dir, rows, summary)
    dashboard._write_json(args.output_dir / "results.json", {
        "schema_version": "muse_encoder.dashboard.merged.v2",
        "shards": provenance,
        "prediction_files": len(predictions),
        "summary": summary,
        "tasks": rows,
    })
    print(json.dumps({
        "final_score": official.get("final_score"),
        "num_tasks": official.get("num_tasks"),
        "scored_tasks": official.get("scored_tasks"),
        "complete": official.get("complete"),
        "missing_predictions": official.get("missing_predictions"),
        "invalid_predictions": official.get("invalid_predictions"),
        "domains": official.get("domains"),
    }, ensure_ascii=False, indent=2))
    assert official.get("num_tasks") == 90, "任务数不是 90"
    assert official.get("scored_tasks") == 90, "有任务未得分，禁止标全量通过"
    assert official.get("complete") is True
    assert not official.get("missing_predictions") and not official.get("invalid_predictions")


if __name__ == "__main__":
    main()
