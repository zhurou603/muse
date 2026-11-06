"""管线驱动：convert → encode → train → score，结果追加 scoreboard.csv。

用法示例（QM9 全量）：
  python -m muse.run --dataset QM9 --subset qm9 --mode sdf --device 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from muse import tasks as T


def safe_task_name(name):
    """任务名转安全文件名（字母数字 ._- 以外替换为 _），与官方 prediction_file 规则一致。"""
    import re
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)
from muse import score as S
from muse import heads as H
from muse.encoders import unimol2_mol as U

SCOREBOARD = T.ROOT / "muse/scoreboard.csv"
METRIC_NAME = {
    "regression": "norm_mae_clip01",
    "binary_classification": "roc_auc",
    "multiclass_classification": "macro_ovo_auc",
    "sequence_classification": "macro_ovo_auc",
}


def _convert_encode_special(subset, mode, cache_dir, device):
    if mode == "copolymer":
        from muse.encoders import copolymer as C
        C.encode_subset(subset, cache_dir, device_id=device)
    elif mode == "electrolyte":
        from muse.encoders import electrolyte as L
        L.encode_subset(subset, cache_dir, device_id=device)
    elif mode == "bio":
        from muse.encoders import esm2_seq as E
        E.encode_subset(subset, cache_dir, device_id=device,
                        unimol2_embed_fn=_ligand_embedder(device))
    else:
        raise ValueError(mode)


def _ligand_embedder(device):
    """protein_ligand 任务的配体 SMILES → Uni-Mol2 表征。"""
    state = {}

    def embed(smiles_list):
        if "model" not in state:
            state["model"], state["dev"] = U.load_model(device_id=device)
        rows, bad = U.smiles_to_rows(smiles_list)
        ok = [s for i, s in enumerate(smiles_list) if i not in set(bad)]
        embs = U.encode_rows(rows, state["model"], state["dev"])
        m = dict(zip(ok, embs))
        dim = embs.shape[1] if len(embs) else 1024
        return np.stack([m.get(s, np.zeros(dim, np.float32)) for s in smiles_list])

    return embed


def load_cache(cache_dir, split):
    cache_dir = Path(cache_dir)
    ids = (cache_dir / f"{split}.ids.txt").read_text(encoding="utf-8").split()
    emb = np.load(cache_dir / f"{split}.emb.npy")
    if emb.shape[0] != len(ids):
        raise ValueError(f"{split}: 嵌入行数 {emb.shape[0]} != id 数 {len(ids)}")
    rows = {}
    with (cache_dir / f"{split}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r.get("labels", {})
    return ids, emb, rows


def targets_from_cache(labels_rows, ids, task: T.LabelTask):
    """从缓存 labels 提取某任务的观测。返回 id->观测值列表（仅含该任务有标签的记录）。"""
    out = {}
    for rid in ids:
        obs = labels_rows.get(rid, {}).get(task.name)
        if obs:
            out[rid] = [o["value"] for o in obs]
    return out


def run_subset(subset: T.Subset, mode, cache_dir, stages, device, batch_size,
               limit_train=None, limit_eval=None, seeds=(0, 1, 2, 3)):
    cache_dir = Path(cache_dir)
    splits = T.SPLITS

    if "convert" in stages:
        if mode in ("bio", "copolymer", "electrolyte", "composition"):
            cache_dir.mkdir(parents=True, exist_ok=True)  # 特殊模式允许断点续跑
        else:
            cache_dir.mkdir(parents=True, exist_ok=False)
        t0 = time.monotonic()
        if mode in ("bio", "copolymer", "electrolyte"):
            # 序列/共聚物：转换与编码合一，直接产出 jsonl + emb.npy
            _convert_encode_special(subset, mode, cache_dir, device)
        elif mode == "composition":
            from muse.encoders import composition as X
            X.encode_subset(subset, cache_dir, limit_train=limit_train)
        else:
            for split in splits:
                # 大文件（GEOM 23GB）流式读取，train 到限量即停
                records = []
                lim = limit_train if split == "train" else limit_eval
                with subset.split_file(split).open(encoding="utf-8") as fs:
                    for line in fs:
                        records.append(json.loads(line))
                        if lim and len(records) >= lim:
                            break
                out_j = cache_dir / f"{split}.jsonl"
                rej_j = cache_dir / f"{split}.rejected.jsonl"
                if mode == "sdf":
                    ref = [r for r in records[0]["external_inputs"]
                           if r["role"] == "molecular_structure"][0]
                    sdf_path = subset.external_base() / ref["source_path"]
                    n_ok, n_rej = U.convert_split_sdf(records, sdf_path, out_j, rej_j)
                elif mode == "pickle":
                    n_ok, n_rej = U.convert_split_pickle(
                        records, subset.external_base(), out_j, rej_j)
                elif mode == "smiles":
                    n_ok, n_rej = U.convert_split_smiles(records, out_j, rej_j)
                else:
                    raise ValueError(mode)
                print(f"CONVERT {subset.dataset}/{subset.subset}/{split}: "
                      f"ok={n_ok} rejected={n_rej}", flush=True)
        manifest = {
            "dataset": subset.dataset, "subset": subset.subset, "mode": mode,
            "limit_train": limit_train,
            "script_sha256": U.sha256_file(Path(__file__)),
            "elapsed_seconds": round(time.monotonic() - t0, 2),
        }
        (cache_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if "encode" in stages and mode not in ("bio", "copolymer", "composition", "electrolyte"):
        t0 = time.monotonic()
        dim = U.encode_splits(cache_dir, splits, batch_size=batch_size, device_id=device)
        print(f"ENCODE done dim={dim} elapsed={time.monotonic()-t0:.1f}s", flush=True)

    if "train" not in stages and "score" not in stages:
        return []

    # train + score
    ids_tr, Xtr_all, lab_tr = load_cache(cache_dir, "train")
    ids_va, Xva_all, lab_va = load_cache(cache_dir, "validation")
    has_test = (cache_dir / "test.emb.npy").exists()
    if has_test:
        ids_te, Xte_all, lab_te = load_cache(cache_dir, "test")

    results = []
    for task in subset.tasks:
        if task.task_type == "sequence_classification":
            row = _run_token_level(subset, task, cache_dir, device, seeds, has_test)
            results.extend(row)
            continue
        obs_tr = targets_from_cache(lab_tr, ids_tr, task)
        obs_va = targets_from_cache(lab_va, ids_va, task)
        obs_te = targets_from_cache(lab_te, ids_te, task) if has_test else {}

        def pick(obs, ids, X):
            keep = [i for i, rid in enumerate(ids) if rid in obs]
            return ([ids[i] for i in keep], X[keep],
                    [obs[ids[i]] for i in keep])

        tr_ids, Xtr, ytr_obs = pick(obs_tr, ids_tr, Xtr_all)
        va_ids, Xva, yva_obs = pick(obs_va, ids_va, Xva_all)
        te_ids, Xte, yte_obs = pick(obs_te, ids_te, Xte_all) if has_test else ([], None, [])

        if task.task_type == "regression":
            ytr = [float(np.mean(v)) for v in ytr_obs]
        elif task.task_type == "binary_classification":
            ytr = [int(float(T.normalize_label_value(v[0]))) for v in ytr_obs]
        elif task.task_type == "multiclass_classification":
            ytr = [T.normalize_label_value(v[0]) for v in ytr_obs]
        else:
            print(f"SKIP {task.name}: 暂不支持 {task.task_type}", flush=True)
            continue

        val_pred, test_pred, meta = H.train_and_predict(
            Xtr, ytr, Xva, Xte, task.task_type,
            class_labels=task.class_labels, seeds=seeds,
            device=f"cuda:{device}",
        )

        for split, pids, preds, yobs in (
            ("validation", va_ids, val_pred, yva_obs),
            ("test", te_ids, test_pred, yte_obs),
        ):
            if preds is None or not pids:
                continue
            np.save(cache_dir / f"{split}.{safe_task_name(task.name)}.pred.npy", preds)
            meta_out = {k: v for k, v in meta.items()}
            meta_out["pred_ids"] = pids
            (cache_dir / f"{split}.{safe_task_name(task.name)}.meta.json").write_text(
                json.dumps(meta_out, ensure_ascii=False), encoding="utf-8")
            if task.task_type == "regression":
                pred_by_id = dict(zip(pids, preds.tolist()))
                obs_by_id = dict(zip(pids, [[float(v) for v in vo] for vo in yobs]))
                score, mae, base = S.score_regression(
                    obs_by_id, pred_by_id, task.train_median)
                extra = {"mae": mae, "baseline_mae": base}
            elif task.task_type == "binary_classification":
                y_true = [int(float(T.normalize_label_value(v[0]))) for v in yobs]
                score = S.score_binary(y_true, preds.tolist())
                extra = {}
            else:
                y_true = [T.normalize_label_value(v[0]) for v in yobs]
                score = S.score_multiclass(y_true, preds, meta["classes"])
                extra = {}
            row = {
                "dataset": subset.dataset, "subset": subset.subset,
                "task": task.name, "task_type": task.task_type,
                "split": split, "metric": METRIC_NAME[task.task_type],
                "score": round(score, 6), "n_train": meta["n_train"],
                "n_eval": len(pids), "params_per_head": meta["params_per_head"],
                "seeds": len(meta["seeds"]), "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                **extra,
            }
            results.append(row)
            print(f"SCORE {subset.dataset}/{subset.subset}/{task.name}/{split}: "
                  f"{row['metric']}={score:.4f}", flush=True)
    return results


def _run_token_level(subset, task, cache_dir, device, seeds, has_test):
    """逐残基分类：在展平的残基表征上训练，预测按记录重新分组。"""
    cache_dir = Path(cache_dir)

    def load_flat(split):
        ids = (cache_dir / f"{split}.ids.txt").read_text(encoding="utf-8").split()
        tokens = np.load(cache_dir / f"{split}.tokens.npy")
        offsets = np.load(cache_dir / f"{split}.token_offsets.npy")
        labels = {}
        with (cache_dir / f"{split}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    obs = r.get("labels", {}).get(task.name)
                    if obs:
                        labels[r["id"]] = obs[0]["value"]
        return ids, tokens, offsets, labels

    def flat_xy(split):
        ids, tokens, offsets, labels = load_flat(split)
        xs, ys, kept_ids, kept_tokens = [], [], [], []
        for i, rid in enumerate(ids):
            if rid not in labels:
                continue
            seg = tokens[offsets[i]:offsets[i + 1]]
            y = [int(v) for v in labels[rid]]
            if len(y) != seg.shape[0]:
                raise ValueError(f"{split}/{rid}: 残基数 {seg.shape[0]} != 标签数 {len(y)}")
            xs.append(seg)
            ys.extend(y)
            kept_ids.append(rid)
            kept_tokens.append(seg)
        return (np.concatenate(xs, axis=0) if xs else None), ys, kept_ids, kept_tokens

    Xtr, ytr, _, _ = flat_xy("train")
    Xva, yva, va_ids, va_tok = flat_xy("validation")
    Xte, yte, te_ids, te_tok = flat_xy("test") if has_test else (None, [], [], [])

    val_pred, test_pred, meta = H.train_and_predict(
        Xtr, ytr, Xva, Xte, "multiclass_classification",
        class_labels=task.class_labels, seeds=seeds, device=f"cuda:{device}")

    rows = []
    for split, y_true, k_ids, k_tok, preds in (
        ("validation", yva, va_ids, va_tok, val_pred),
        ("test", yte, te_ids, te_tok, test_pred),
    ):
        if preds is None or not k_ids:
            continue
        np.save(cache_dir / f"{split}.{safe_task_name(task.name)}.pred.npy", preds)
        meta_out = {k: v for k, v in meta.items() if k != "n_train"}
        meta_out.update({"pred_ids": k_ids, "token_level": True,
                         "token_counts": [t.shape[0] for t in k_tok],
                         "n_train": len(ytr)})
        (cache_dir / f"{split}.{safe_task_name(task.name)}.meta.json").write_text(
            json.dumps(meta_out, ensure_ascii=False), encoding="utf-8")
        score = S.score_multiclass([str(v) for v in y_true], preds, meta["classes"])
        row = {
            "dataset": subset.dataset, "subset": subset.subset,
            "task": task.name, "task_type": task.task_type, "split": split,
            "metric": METRIC_NAME[task.task_type], "score": round(score, 6),
            "n_train": len(ytr), "n_eval": len(y_true),
            "params_per_head": meta["params_per_head"], "seeds": len(meta["seeds"]),
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        rows.append(row)
        print(f"SCORE {subset.dataset}/{subset.subset}/{task.name}/{split}: "
              f"macro_ovo_auc={score:.4f}", flush=True)
    return rows


def append_scoreboard(rows):
    if not rows:
        return
    keys = ["dataset", "subset", "task", "task_type", "split", "metric", "score",
            "n_train", "n_eval", "params_per_head", "seeds", "ts", "mae", "baseline_mae"]
    new = not SCOREBOARD.exists()
    with SCOREBOARD.open("a", encoding="utf-8") as f:
        if new:
            f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print(f"scoreboard += {len(rows)} 行 -> {SCOREBOARD}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--mode", choices=["sdf", "smiles", "pickle", "bio", "copolymer", "electrolyte", "composition"], required=True)
    ap.add_argument("--stages", default="convert,encode,train,score")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-eval", type=int, default=None,
                    help="冒烟测试用：截断 validation/test 记录数")
    ap.add_argument("--tag", default="v0")
    ap.add_argument("--seeds", default="0,1,2,3")
    args = ap.parse_args()

    subsets = T.load_subsets()
    subset = subsets[(args.dataset, args.subset)]
    cache_dir = subset.cache_dir(args.tag)
    stages = set(args.stages.split(","))
    seeds = tuple(int(s) for s in args.seeds.split(","))
    rows = run_subset(subset, args.mode, cache_dir, stages, args.device,
                      args.batch_size, args.limit_train, args.limit_eval, seeds)
    append_scoreboard(rows)


if __name__ == "__main__":
    main()
