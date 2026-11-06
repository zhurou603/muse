"""分片评估入口：train 侧截断、validation 全量；不修改 dashboard.py 源码。

run_dashboard 对 train/validation 共用 limit_per_split；本包装仅把 train 的
limit 换成 train_cap，validation 保持全量，其余逻辑（探针、预测、官方打分）
完全走原代码。注意：run_key/task_cache 不含 train_cap，换 cap 必须换输出目录。

SDF 读取改为逐条容错：原始实现中任何一条 SDF 严格解析失败都会让 prepare()
抛错、整个 split 全部记录零编码（全量数据必现，64 条试点碰不到）。这里跳过
坏记录，让其走 normalize 阶段的逐条失败 + 官方 fallback 路径，并落盘审计。
"""
import argparse
import io
import json
from pathlib import Path

from muse_encoder.pretrain import dashboard

TRAIN_CAP = 200_000
SECOND_SPLIT = "validation"
SDF_SKIPPED = []
_orig_encode = dashboard.encode_split_cached


def _encode_with_train_cap(encoder, featurizer, cfg, subset, data_root, split, run_root,
                           checkpoint_sha, limit_per_split, token_level, batch_size, device,
                           probe_config=dashboard.PROBE_CONFIG):
    if split == "train":
        limit_per_split = TRAIN_CAP
    elif split == "validation" and SECOND_SPLIT == "test":
        split = "test"
    return _orig_encode(encoder, featurizer, cfg, subset, data_root, split, run_root,
                        checkpoint_sha, limit_per_split, token_level, batch_size, device,
                        probe_config)


def _read_sdf_selected_tolerant(path, selected):
    """与 dashboard._read_sdf_selected 相同，但单条坏记录不再拖垮整个 split。"""
    from rdkit import Chem

    found = {}
    seen = set()
    with Path(path).open("rb") as stream:
        block = []
        for line_number, line in enumerate(stream, start=1):
            block.append(line)
            if line.strip() != b"$$$$":
                continue
            if not block or not block[0].strip():
                raise ValueError(f"{path}:{line_number}: SDF 记录没有名称")
            key = block[0].decode("utf-8").strip()
            if key in seen:
                raise ValueError(f"SDF 结构编号重复: {key}")
            seen.add(key)
            if key in selected:
                try:
                    supplier = Chem.ForwardSDMolSupplier(
                        io.BytesIO(b"".join(block)), removeHs=False, sanitize=True)
                    mol = next(supplier)
                except Exception:
                    mol = None
                if mol is None:
                    SDF_SKIPPED.append(key)
                else:
                    found[key] = mol
                    if len(found) == len(selected):
                        return found
            block = []
    return found


def _make_test_scorer(original):
    class _TestScorer:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def score_predictions(self, predictions_root, data_root, board, *, split="validation", **kw):
            return self._inner.score_predictions(predictions_root, data_root, board, split="test", **kw)

    def factory(path=dashboard.OFFICIAL_SCORER_PATH):
        return _TestScorer(original(path))

    return factory


def main():
    global TRAIN_CAP, SECOND_SPLIT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--subsets", required=True)
    parser.add_argument("--train-cap", type=int, default=TRAIN_CAP)
    parser.add_argument("--second-split", choices=["validation", "test"], default="validation",
                        help="探针预测的目标切分；test 用于 A 榜正式提交链路")
    args = parser.parse_args()
    TRAIN_CAP = args.train_cap
    SECOND_SPLIT = args.second_split
    dashboard.encode_split_cached = _encode_with_train_cap
    dashboard._read_sdf_selected = _read_sdf_selected_tolerant
    if args.second_split == "test":
        dashboard.official_scorer = _make_test_scorer(dashboard.official_scorer)
    result = dashboard.run_dashboard(
        args.checkpoint, args.output_dir, device=args.device, batch_size=args.batch_size,
        limit_per_split=None, selected_subsets=args.subsets.split(","), run_shadow=False)
    (args.output_dir / "sdf_skipped.json").write_text(
        json.dumps({"count": len(SDF_SKIPPED), "ids": SDF_SKIPPED}, ensure_ascii=False),
        encoding="utf-8")
    summary = result["summary"]
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "train_cap": args.train_cap,
        "sdf_skipped": len(SDF_SKIPPED),
        "final_score": summary.get("final_score"),
        "scored_tasks": summary.get("scored_tasks"),
        "num_tasks": summary.get("num_tasks"),
        "status": "SHARD-DONE",
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
