"""多设备冻结表征评估；预处理与表征使用跨 run 持久缓存。"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from muse_encoder.pretrain import dashboard
from muse_encoder.pretrain.parallel_dashboard import encode_records_parallel
from shard_eval import _make_test_scorer, _read_sdf_selected_tolerant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path,
                        default=Path("data/pretrain/cache/frozen_eval"))
    parser.add_argument("--external-cache", type=Path,
                        default=Path("data/pretrain/cache/omc25"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", default=None,
                        help="逗号分隔，例如 0,1,2,3；默认只用 --device")
    parser.add_argument("--cpu-workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--official-graph-repairs", type=Path, default=None)
    parser.add_argument("--subsets", default=None,
                        help="默认全部 A 榜子集；调试用逗号分隔列表")
    parser.add_argument("--train-cap", type=int, default=200_000)
    parser.add_argument("--second-split", choices=["validation", "test"], default="validation")
    parser.add_argument("--single-seed", action="store_true",
                        help="正式提交使用单个 MLP，避免多种子合并超过参数预算")
    parser.add_argument("--team", default="zhaoyao")
    parser.add_argument("--name", default="muse-encoder-fixed-probe")
    args = parser.parse_args()
    devices = tuple(args.devices.split(",")) if args.devices else (args.device,)
    devices = tuple(device if device.startswith(("cuda:", "cpu")) else f"cuda:{device}"
                    for device in devices)
    data_version = dashboard.sha256_file(dashboard.DATA_ROOT / "manifest.json")
    probe_config = dashboard.ProbeConfig(seeds=(0,)) if args.single_seed else dashboard.PROBE_CONFIG

    def encode_cached(encoder, featurizer, cfg, subset, data_root, split, run_root,
                      checkpoint_sha, limit_per_split, token_level, batch_size, device,
                      probe_config=dashboard.PROBE_CONFIG):
        if split == "train":
            limit_per_split = args.train_cap
        elif split == "validation" and args.second_split == "test":
            split = "test"
        raw = dashboard._read_jsonl(
            dashboard._split_file(data_root, subset, split), limit_per_split)
        namespace = f"{subset.dataset}/{subset.subset}/{split}"
        started = time.monotonic()
        encoded, stats = encode_records_parallel(
            raw,
            checkpoint_path=args.checkpoint,
            checkpoint_sha=checkpoint_sha,
            cfg=cfg,
            cache_root=args.cache_root,
            external_base=dashboard._external_base(data_root, subset),
            external_cache=args.external_cache / "ase_lmdb",
            namespace=namespace,
            data_version=data_version,
            devices=devices,
            cpu_workers=args.cpu_workers,
            batch_size=batch_size,
            token_level=token_level,
            probe_config=probe_config,
            graph_repairs=args.official_graph_repairs,
        )
        cache = dashboard._subset_cache(run_root / "embeddings", subset)
        cache.mkdir(parents=True, exist_ok=True)
        np.save(cache / f"{split}.embeddings.npy", encoded.embeddings)
        if token_level:
            np.save(cache / f"{split}.tokens.npy", encoded.tokens)
            np.save(cache / f"{split}.token_offsets.npy", encoded.offsets)
        (cache / f"{split}.ids.json").write_text(
            json.dumps(encoded.ids, ensure_ascii=False), encoding="utf-8")
        metadata = {
            "checkpoint_sha256": checkpoint_sha,
            "model_config_sha256": dashboard._config_sha(cfg.to_dict()),
            "split": split,
            "limit_per_split": limit_per_split,
            "token_level": token_level,
            "ids": [record["id"] for record in raw],
            "probe_config_sha256": dashboard._config_sha(probe_config.to_json()),
            "failed": encoded.failed,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "parallel": {"devices": devices, "cpu_workers": args.cpu_workers, **stats},
        }
        dashboard._write_json(cache / f"{split}.encoding.json", metadata)
        dashboard._write_json(cache / f"{split}.failed.json", encoded.failed)
        return encoded, raw, stats["central_hit"]

    dashboard.encode_split_cached = encode_cached
    dashboard._read_sdf_selected = _read_sdf_selected_tolerant
    if args.second_split == "test":
        dashboard.official_scorer = _make_test_scorer(dashboard.official_scorer)
    result = dashboard.run_dashboard(
        args.checkpoint, args.output_dir, device=args.device, batch_size=args.batch_size,
        limit_per_split=None,
        selected_subsets=args.subsets.split(",") if args.subsets else None,
        team=args.team, name=args.name, run_shadow=False, probe_config=probe_config)
    if args.second_split == "test":
        manifest_path = args.output_dir / "predictions" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["split"] = "test"
        dashboard._write_json(manifest_path, manifest)
        official_scores = dashboard.official_scorer().score_predictions(
            args.output_dir / "predictions", dashboard.DATA_ROOT, "A", split="test")
        dashboard._write_json(args.output_dir / "official_scores.json", official_scores)
        summary = dict(official_scores)
        summary["scope"] = "full_test"
        summary["is_full_a_board"] = not args.subsets
        if args.subsets:
            summary["complete"] = False
            summary["note"] = "selected-subset 结果只用于调试，不能作为完整 A 榜提交"
        result["summary"] = summary
        result["split"] = "test"
        dashboard._write_json(args.output_dir / "results.json", result)
    summary = result["summary"]
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "train_cap": args.train_cap,
        "devices": devices,
        "cached_splits": result["cached_splits"],
        "final_score": summary.get("final_score"),
        "scored_tasks": summary.get("scored_tasks"),
        "num_tasks": summary.get("num_tasks"),
        "status": "FAST-DONE",
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
