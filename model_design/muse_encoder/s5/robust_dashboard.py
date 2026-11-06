"""Full 90-task fixed probe with train-only median/MAD regression scaling.

This reuses the reviewed dashboard's encoding, official scorer, MLP budget and
early stopping.  Regression uses robust location/scale, and every task uses one
declared seed: post-activation seed ensembles are not a compliant two-layer MLP.
Candidate comparisons therefore use one explicit robust fixed-head protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch

from muse_encoder.pretrain import dashboard as D


def fit_probe_robust(X_train: np.ndarray, y_train_raw: Sequence[Any], X_validation: np.ndarray,
                     task_type: str, class_labels: Optional[Sequence[Any]] = None,
                     positive_label: Optional[str] = None, device="cpu",
                     config: D.ProbeConfig = D.PROBE_CONFIG) -> Tuple[np.ndarray, dict[str, Any]]:
    X_train = np.asarray(X_train, dtype=np.float32)
    X_validation = np.asarray(X_validation, dtype=np.float32)
    if X_train.ndim != 2 or X_validation.ndim != 2 or X_train.shape[1] != X_validation.shape[1]:
        raise ValueError("probe 输入必须是特征维度一致的二维矩阵")
    if len(X_train) != len(y_train_raw) or len(X_train) < 2:
        raise ValueError("probe 训练特征/标签数量异常")
    device = torch.device(device)
    classes = None
    if task_type == "regression":
        y_raw = np.asarray(y_train_raw, dtype=np.float64)
        if not np.isfinite(y_raw).all():
            raise ValueError("回归训练标签含 NaN/Inf")
        y_mean = float(np.median(y_raw))
        mad = float(np.median(np.abs(y_raw - y_mean)))
        std = float(y_raw.std())
        y_std = mad if mad > 1e-8 else (std if std > 1e-8 else 1.0)
        y_train = ((y_raw - y_mean) / y_std).astype(np.float32)
        scale_kind = "mad" if mad > 1e-8 else ("std_fallback" if std > 1e-8 else "constant_fallback")
        d_out = 1
    elif task_type == "binary_classification":
        if positive_label is None:
            raise ValueError("二分类 probe 需要官方 positive_label")
        y_train = np.asarray([
            1.0 if D.official_scorer().encode_class_label(value) == positive_label else 0.0
            for value in y_train_raw
        ], dtype=np.float32)
        if len(set(y_train.tolist())) < 2:
            raise ValueError("二分类训练集只有一个类别")
        y_mean = y_std = None
        scale_kind = None
        d_out = 1
    elif task_type in ("multiclass_classification", "sequence_classification"):
        if not class_labels:
            raise ValueError("多分类/逐残基 probe 需要官方 class_labels")
        classes = [str(value) for value in class_labels]
        lookup = {value: index for index, value in enumerate(classes)}
        encoded = [D.official_scorer().encode_class_label(value) for value in y_train_raw]
        unknown = sorted(set(encoded) - set(lookup))
        if unknown:
            raise ValueError(f"训练标签不在官方类别表: {unknown[:5]}")
        y_train = np.asarray([lookup[value] for value in encoded], dtype=np.int64)
        if len(set(y_train.tolist())) < 2:
            raise ValueError("分类训练集少于两个类别")
        y_mean = y_std = None
        scale_kind = None
        d_out = len(classes)
    else:
        raise NotImplementedError(task_type)

    hidden, adapted, params = D.probe_hidden_dim(X_train.shape[1], d_out, config)
    # A post-sigmoid/post-softmax seed ensemble is not a two-layer MLP, and
    # merging four regression heads would also exceed the 100K budget.  The S5
    # fixed protocol therefore uses exactly one declared seed per task.
    if not config.seeds:
        raise ValueError("固定 probe 至少需要一个种子")
    seeds = (int(config.seeds[0]),)
    rng = np.random.default_rng(config.early_stop_seed)
    order = rng.permutation(len(X_train))
    n_stop = max(1, int(config.early_stop_fraction * len(X_train)))
    stop_index, train_index = order[:n_stop], order[n_stop:]
    if len(train_index) < 1:
        raise ValueError("early-stop 切分后没有训练样本")
    seed_predictions = []
    for seed in seeds:
        model = D._train_probe_seed(
            X_train[train_index], y_train[train_index], X_train[stop_index], y_train[stop_index],
            task_type, len(classes) if classes else None, seed, device, config)
        seed_predictions.append(D._predict_probe(model, X_validation, task_type, device))
    prediction = np.mean(seed_predictions, axis=0) if seed_predictions else np.empty((0,))
    if task_type == "regression":
        prediction = prediction * y_std + y_mean
    meta = {
        "protocol": "muse_encoder.s5_fixed_robust_probe.v1",
        "regression_scaling": "train-only median/MAD with std-or-1 fallback",
        "scale_kind": scale_kind, "d_in": int(X_train.shape[1]),
        "hidden_requested": config.hidden, "hidden": int(hidden),
        "budget_adapted": bool(adapted), "params_per_head": int(params),
        "seeds": list(seeds), "seed_ensemble": False,
        "mlp_compliance": "single Linear->ReLU->Linear head within 100K parameters",
        "max_epochs": config.max_epochs,
        "patience": config.patience, "lr": config.lr,
        "early_stop_fraction": config.early_stop_fraction,
        "early_stop_seed": config.early_stop_seed,
        "n_train": int(len(X_train)), "classes": classes,
    }
    return prediction.astype(np.float32), meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", default=None)
    parser.add_argument("--cpu-workers", type=int, default=14)
    parser.add_argument("--cache-root", type=Path, default=Path("data/pretrain/cache/frozen_eval"))
    parser.add_argument("--external-cache", type=Path, default=Path("data/pretrain/cache/omc25"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-cap", type=int, default=None)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--subsets", default=None)
    parser.add_argument("--team", default="zhaoyao")
    parser.add_argument("--name", default="s5-fixed-robust-probe")
    parser.add_argument("--skip-shadow", action="store_true")
    parser.add_argument("--shadow-limit", type=int, default=None)
    parser.add_argument("--official-graph-repairs", type=Path, default=None)
    args = parser.parse_args()
    D.configure_official_graph_repairs(args.official_graph_repairs)
    if args.devices:
        import time
        from muse_encoder.pretrain.parallel_dashboard import encode_records_parallel
        devices = tuple(args.devices.split(","))
        devices = tuple(device if device.startswith(("cuda:", "cpu")) else f"cuda:{device}"
                        for device in devices)
        data_version = D.sha256_file(D.DATA_ROOT / "manifest.json")

        def encode_cached(encoder, featurizer, cfg, subset, data_root, split, run_root,
                          checkpoint_sha, limit_per_split, token_level, batch_size, device,
                          probe_config=D.PROBE_CONFIG):
            if split == "train" and args.train_cap is not None:
                limit_per_split = args.train_cap
            raw = D._read_jsonl(D._split_file(data_root, subset, split), limit_per_split)
            started = time.monotonic()
            encoded, stats = encode_records_parallel(
                raw, checkpoint_path=args.checkpoint, checkpoint_sha=checkpoint_sha, cfg=cfg,
                cache_root=args.cache_root,
                external_base=D._external_base(data_root, subset),
                external_cache=args.external_cache / "ase_lmdb",
                namespace=f"{subset.dataset}/{subset.subset}/{split}",
                data_version=data_version, devices=devices, cpu_workers=args.cpu_workers,
                batch_size=batch_size, token_level=token_level, probe_config=probe_config,
                graph_repairs=args.official_graph_repairs)
            cache = D._subset_cache(run_root / "embeddings", subset)
            cache.mkdir(parents=True, exist_ok=True)
            np.save(cache / f"{split}.embeddings.npy", encoded.embeddings)
            if token_level:
                np.save(cache / f"{split}.tokens.npy", encoded.tokens)
                np.save(cache / f"{split}.token_offsets.npy", encoded.offsets)
            (cache / f"{split}.ids.json").write_text(
                json.dumps(encoded.ids, ensure_ascii=False), encoding="utf-8")
            D._write_json(cache / f"{split}.encoding.json", {
                "checkpoint_sha256": checkpoint_sha,
                "model_config_sha256": D._config_sha(cfg.to_dict()),
                "split": split, "limit_per_split": limit_per_split,
                "token_level": token_level, "ids": [record["id"] for record in raw],
                "probe_config_sha256": D._config_sha(probe_config.to_json()),
                "failed": encoded.failed, "elapsed_seconds": round(time.monotonic() - started, 3),
                "parallel": {"devices": devices, "cpu_workers": args.cpu_workers, **stats},
            })
            return encoded, raw, stats["central_hit"]

        D.encode_split_cached = encode_cached

        def encode_shadow(encoder, featurizer, cfg, records, batch_size, token_level, device,
                          namespace):
            source = namespace.split("::", 1)[0]
            external_base = D._AUX_EXTERNAL_BASES.get(source, D.EXTERNAL_ROOT)
            encoded, _stats = encode_records_parallel(
                records, checkpoint_path=args.checkpoint, checkpoint_sha=D.sha256_file(args.checkpoint),
                cfg=cfg, cache_root=args.cache_root, external_base=external_base,
                external_cache=args.external_cache / "ase_lmdb", namespace=f"shadow/{namespace}",
                data_version=data_version, devices=devices, cpu_workers=args.cpu_workers,
                batch_size=batch_size, token_level=token_level,
                graph_repairs=args.official_graph_repairs)
            return encoded

        original_run_shadow = D.run_shadow_probes

        def run_shadow_parallel(encoder, featurizer, cfg, output_dir, checkpoint_sha,
                                limit_per_property, batch_size, device, train_cap=None):
            return original_run_shadow(
                encoder, featurizer, cfg, output_dir, checkpoint_sha, limit_per_property,
                batch_size, device, record_encoder=encode_shadow, train_cap=train_cap)

        D.run_shadow_probes = run_shadow_parallel
    else:
        original_encode_cached = D.encode_split_cached

        def cap_encode_cached(encoder, featurizer, cfg, subset, data_root, split, run_root,
                              checkpoint_sha, limit_per_split, token_level, batch_size, device,
                              probe_config=D.PROBE_CONFIG):
            if split == "train" and args.train_cap is not None:
                limit_per_split = args.train_cap
            return original_encode_cached(
                encoder, featurizer, cfg, subset, data_root, split, run_root,
                checkpoint_sha, limit_per_split, token_level, batch_size, device, probe_config)

        D.encode_split_cached = cap_encode_cached
    original_shadow_with_encoding = D.run_shadow_probes

    def run_shadow_capped(encoder, featurizer, cfg, output_dir, checkpoint_sha,
                          limit_per_property, batch_size, device):
        return original_shadow_with_encoding(
            encoder, featurizer, cfg, output_dir, checkpoint_sha, limit_per_property,
            batch_size, device, train_cap=args.train_cap)

    D.run_shadow_probes = run_shadow_capped
    D.fit_probe = fit_probe_robust
    result = D.run_dashboard(
        args.checkpoint, args.output_dir, args.device, args.batch_size,
        args.limit_per_split, args.subsets.split(",") if args.subsets else None,
        D.DATA_ROOT, args.team, args.name, not args.skip_shadow, args.shadow_limit)
    result["fixed_probe_protocol"] = "muse_encoder.s5_fixed_robust_probe.v1"
    D._write_json(args.output_dir / "results.json", result)
    print(json.dumps({
        "scope": result["summary"].get("scope"), "smoke": result["smoke"],
        "complete": result["summary"].get("complete"),
        "final_score": result["summary"].get("final_score"),
        "scored_tasks": result["summary"].get("scored_tasks"),
        "num_tasks": result["summary"].get("num_tasks"),
        "fixed_probe_protocol": result["fixed_probe_protocol"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
