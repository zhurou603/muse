"""官方计分桥接：把 processed 缓存里的预测转成官方 predictions 格式并调用官方 score.py。

关键点：官方规定一个子任务只要有一个被评测 id 缺预测就记 0 分。
转换时被隔离的记录没有模型预测，这里按无信息回退补齐：
- 回归：train_median；二分类：训练集正类频率；多分类：均匀分布。

用法：
  python -m muse.official --tag v0 --split validation --team zhaoyao --name v0-dev
"""

import argparse
import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from muse import tasks as T
from muse.run import safe_task_name as _safe

class R:
    safe_name = staticmethod(_safe)

EVAL_DIR = T.ROOT / "data/muse-a/eval/MUSE-eval"
TASKS_CACHE = T.ROOT / "muse/tasks_A.json"
RESULTS = T.ROOT / "muse/results"


def load_task_defs():
    if not TASKS_CACHE.exists():
        subprocess.run(
            [sys.executable, str(EVAL_DIR / "score.py"), "tasks",
             "--data-root", str(T.DATA_DIR), "--board", "A", "--out", str(TASKS_CACHE)],
            check=True)
    return json.loads(TASKS_CACHE.read_text(encoding="utf-8"))["tasks"]


def split_ids_in_order(subset: T.Subset, split):
    return [json.loads(l)["id"] for l in
            subset.split_file(split).open(encoding="utf-8")]


def train_positive_rate(subset: T.Subset, task_name):
    m = json.loads(T.MANIFEST_PATH.read_text(encoding="utf-8"))
    lab = m["datasets"][subset.dataset]["subsets"][subset.subset]["labels"][task_name]
    counts = lab.get("class_counts", {}).get("train", {})
    pos = [k for k in counts if int(float(T.normalize_label_value(k))) == 1]
    if not pos:
        return 0.5
    total = sum(counts.values())
    return counts[pos[0]] / total if total else 0.5


def _seq_len_of(subset, split, rid):
    """逐残基任务缺预测时的回退：从原始记录取序列长度。"""
    with subset.split_file(split).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] == rid:
                return len(r["inputs"]["sequence"])
    raise ValueError(f"记录不存在: {rid}")


def write_predictions(tag, split, team, name, out_root):
    task_defs = load_task_defs()
    subsets = T.load_subsets()
    pred_root = Path(out_root) / "predictions"
    written, skipped = 0, []
    for td in task_defs:
        dataset, sname, tname = td["task_id"].split("/", 2)
        subset = subsets[(dataset, sname)]
        cache = subset.cache_dir(tag)
        pred_npy = cache / f"{split}.{R.safe_name(tname)}.pred.npy"
        meta_json = cache / f"{split}.{R.safe_name(tname)}.meta.json"
        if not pred_npy.exists():
            skipped.append(td["task_id"])
            continue
        preds = np.load(pred_npy)
        if meta_json.exists():
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
            if meta.get("token_level"):
                # 逐残基：按 token_counts 把 [ΣL, C] 切成每记录 [L, C]
                counts = meta["token_counts"]
                pred_by_id, off = {}, 0
                for rid, c in zip(meta["pred_ids"], counts):
                    pred_by_id[rid] = preds[off:off + c].tolist()
                    off += c
                if off != len(preds):
                    raise ValueError(f"{td['task_id']}: token 计数与预测行数不符")
            else:
                pred_by_id = dict(zip(meta["pred_ids"], preds.tolist()))
            my_classes = [str(c) for c in meta.get("classes") or []]
        else:
            # 旧缓存没有 meta：从缓存 jsonl 重建有标签记录的 id 列表
            rows = {}
            with (cache / f"{split}.jsonl").open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if r.get("labels", {}).get(tname):
                            rows[r["id"]] = True
            ids = (cache / f"{split}.ids.txt").read_text(encoding="utf-8").split()
            keep = [rid for rid in ids if rid in rows]
            if len(keep) != len(preds):
                raise ValueError(f"{td['task_id']}: 重建 id 数与预测数不符")
            pred_by_id = dict(zip(keep, preds.tolist()))
            with (cache / f"train.jsonl").open(encoding="utf-8") as f:
                cls = sorted({T.normalize_label_value(o["value"])
                              for line in f if line.strip()
                              for o in (json.loads(line).get("labels", {}).get(tname) or [])})
            my_classes = cls
        task = next(t for t in subset.tasks if t.name == tname)
        all_ids = split_ids_in_order(subset, split)

        # 回退值
        if task.task_type == "regression":
            fallback = float(task.train_median)
        elif task.task_type == "binary_classification":
            fallback = train_positive_rate(subset, tname)
        elif task.task_type in ("multiclass_classification", "sequence_classification"):
            n_cls = len(td["classes"])
            fallback = [1.0 / n_cls] * n_cls
        else:
            skipped.append(td["task_id"])
            continue

        official_classes = [str(c) for c in td.get("classes", [])]
        col = td["prediction_column"]
        out_file = pred_root / td["prediction_file"]
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(out_file, "wt", encoding="utf-8") as f:
            f.write(f"id,{col}\n")
            for rid in all_ids:
                if rid in pred_by_id:
                    val = pred_by_id[rid]
                else:
                    val = fallback
                if task.task_type == "regression":
                    f.write(f"{rid},{float(val)}\n")
                elif task.task_type == "binary_classification":
                    f.write(f"{rid},{float(val)}\n")
                elif task.task_type == "sequence_classification":
                    # val: [L, C] 逐残基概率（my_classes 顺序），重排到官方类别顺序
                    if isinstance(val, list) and val and isinstance(val[0], list):
                        remap = [my_classes.index(c) for c in official_classes]
                        mat = [[row[i] for i in remap] for row in val]
                    else:
                        # 无预测：均匀分布，残基数从原始序列长度取
                        L = _seq_len_of(subset, split, rid)
                        mat = [fallback] * L
                    f.write(f"{rid},{json.dumps(mat)}\n")
                else:
                    # 多分类：按官方类别顺序重排概率
                    prob = dict(zip(my_classes, val)) if isinstance(val, list) else None
                    if prob is None:
                        arr = fallback
                    else:
                        arr = [float(prob.get(c, 0.0)) for c in official_classes]
                        s = sum(arr)
                        if s <= 0:
                            arr = fallback
                        else:
                            arr = [a / s for a in arr]
                    f.write(f"{rid},{json.dumps(arr)}\n")
        written += 1
    (pred_root / "manifest.json").write_text(json.dumps(
        {"team": team, "name": name, "board": "A", "split": split},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return written, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="v0")
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--team", default="zhaoyao")
    ap.add_argument("--name", default="v0-dev")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    run_name = args.run_name or f"{args.name}-{args.split}"
    out_root = RESULTS / run_name
    out_root.mkdir(parents=True, exist_ok=True)
    written, skipped = write_predictions(args.tag, args.split, args.team, args.name, out_root)
    print(f"预测文件: {written} 个任务已写出; 无预测的 {len(skipped)} 个任务", flush=True)
    if skipped:
        print("缺预测:", ", ".join(skipped[:10]), "..." if len(skipped) > 10 else "", flush=True)

    scores_out = out_root / "scores.json"
    subprocess.run(
        [sys.executable, str(EVAL_DIR / "score.py"), "score",
         "--predictions", str(out_root / "predictions"),
         "--data-root", str(T.DATA_DIR), "--board", "A",
         "--split", args.split, "--out", str(scores_out)],
        check=True)
    s = json.loads(scores_out.read_text(encoding="utf-8"))
    print(f"\n官方总分({args.split}): {s['final_score']}  complete={s['complete']}")
    for dom, info in s["domains"].items():
        print(f"  {dom}: {info['score']}")
    print(f"得分文件: {scores_out}")


if __name__ == "__main__":
    main()
