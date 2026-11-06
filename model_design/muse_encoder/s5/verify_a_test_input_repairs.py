"""Encode every packaged failure with registered repairs and no labels."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from muse_encoder.pretrain import dashboard as D
from muse_encoder.pretrain.parallel_dashboard import encode_records_parallel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--repairs", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    root = args.handoff_root.resolve()
    failures = [json.loads(line) for line in (root / "failures.jsonl").read_text().splitlines()
                if line.strip()]
    repairs = json.loads(args.repairs.read_text(encoding="utf-8"))
    by_id = {entry["record_id"]: entry for entry in repairs["repairs"]}
    if set(by_id) != {failure["id"] for failure in failures}:
        raise ValueError("修复清单与失败 ID 不一致")
    board = D.load_board(D.DATA_ROOT, "A")
    _model, cfg, _info = D.load_checkpoint(args.checkpoint, "cpu")
    del _model
    data_version = D.sha256_file(D.DATA_ROOT / "manifest.json")
    groups = defaultdict(list)
    for failure in failures:
        groups[(failure["dataset"], failure["subset"], failure["split"])].append(failure)
    results = []
    for (dataset, subset, split), rows in sorted(groups.items()):
        source_path = root / rows[0]["records_file"]
        records = [json.loads(line) for line in source_path.read_text().splitlines()
                   if line.strip()]
        if any("labels" in record or "targets" in record for record in records):
            raise ValueError("验证器拒绝含标签输入")
        ids = [record["id"] for record in records]
        if set(ids) != {row["id"] for row in rows} or len(ids) != len(rows):
            raise ValueError(f"{dataset}/{subset}/{split}: 输入 ID 不一致")
        encoded, stats = encode_records_parallel(
            records, checkpoint_path=args.checkpoint,
            checkpoint_sha=D.sha256_file(args.checkpoint), cfg=cfg,
            cache_root=args.cache_root,
            external_base=D._external_base(D.DATA_ROOT, board.subsets[(dataset, subset)]),
            external_cache=D.PROJECT_ROOT / "data/pretrain/cache/omc25/ase_lmdb",
            namespace=f"input_repairs/{dataset}/{subset}/{split}",
            data_version=data_version, devices=(args.device,),
            cpu_workers=args.cpu_workers, batch_size=args.batch_size,
            token_level=False, graph_repairs=args.repairs,
        )
        failed_ids = {row["id"] for row in encoded.failed}
        ids_match = len(encoded.ids) == len(ids) and set(encoded.ids) == set(ids)
        result = {
            "dataset": dataset, "subset": subset, "split": split,
            "expected": len(ids), "encoded": len(encoded.ids),
            "failed": encoded.failed,
            "ids_match": ids_match, "order_matches": encoded.ids == ids,
            "finite": bool(np.isfinite(encoded.embeddings).all()),
            "shape": list(encoded.embeddings.shape),
            "representations": dict(Counter(
                by_id[record_id].get("representation", "official_graph") for record_id in ids)),
            "cache_stats": stats,
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if failed_ids or not ids_match or not result["finite"]:
            raise RuntimeError(f"{dataset}/{subset}/{split}: 修复后仍有编码失败")
    payload = {
        "schema": "muse_encoder.input_repair_verification.v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": D.sha256_file(args.checkpoint),
        "repairs": str(args.repairs.resolve()),
        "repairs_sha256": D.sha256_file(args.repairs),
        "selected": len(failures), "encoded": sum(row["encoded"] for row in results),
        "failed": [], "all_ids_match": all(row["ids_match"] for row in results),
        "all_finite": all(row["finite"] for row in results),
        "id_alignment": "Official dashboard maps features and observations by record id; windowing may change order",
        "labels_used": False, "groups": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "groups"},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
