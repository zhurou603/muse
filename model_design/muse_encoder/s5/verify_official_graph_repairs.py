"""Verify registered official-graph repairs through the production encoder path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from muse_encoder.pretrain import dashboard as D


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--cpu-workers", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cache-root", type=Path, default=Path("data/pretrain/cache/frozen_eval"))
    args = parser.parse_args()
    payload = json.loads(args.repairs.read_text(encoding="utf-8"))
    ids = [entry["record_id"] for entry in payload["repairs"]]
    D.configure_official_graph_repairs(args.repairs)
    try:
        model, cfg, checkpoint = D.load_checkpoint(args.checkpoint, "cpu")
        board = D.load_board(D.DATA_ROOT, "A")
        subset = board.subsets[("QM9", "qm9")]
        wanted = set(ids)
        raw = [record for record in D._read_jsonl(D._split_file(D.DATA_ROOT, subset, "validation"))
               if record["id"] in wanted]
        if {record["id"] for record in raw} != wanted:
            raise RuntimeError("注册记录与官方 validation 输入不一致")
        if args.devices:
            del model
            from muse_encoder.pretrain.parallel_dashboard import encode_records_parallel
            devices = tuple(device if device.startswith(("cuda:", "cpu")) else f"cuda:{device}"
                            for device in args.devices.split(","))
            encoded, stats = encode_records_parallel(
                raw, checkpoint_path=args.checkpoint, checkpoint_sha=checkpoint["sha256"], cfg=cfg,
                cache_root=args.cache_root, external_base=D._AUX_EXTERNAL_BASES["qm9"],
                external_cache=Path("data/pretrain/cache/omc25/ase_lmdb"),
                namespace="official_graph_repairs/qm9/validation",
                data_version=D.sha256_file(D.DATA_ROOT / "manifest.json"), devices=devices,
                cpu_workers=args.cpu_workers, batch_size=args.batch_size, token_level=False,
                graph_repairs=args.repairs)
        else:
            normalized = [D.normalize_record(record) for record in raw]
            encoded = D.encode_records(model, D.RecordFeaturizer(cfg), cfg, normalized,
                                       args.batch_size, False, "cpu")
            stats = {"parallel": False}
    finally:
        D.configure_official_graph_repairs(None)
    summary = {
        "checkpoint_sha256": checkpoint["sha256"],
        "official_graph_repairs_sha256": D.sha256_file(args.repairs),
        "registered": len(ids), "input": len(raw), "encoded": len(encoded.ids),
        "failed": encoded.failed, "stats": stats,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if encoded.failed or set(encoded.ids) != wanted:
        raise RuntimeError("官方图修复生产编码验收失败")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖已有验证结果: {args.output}")
    D._write_json(args.output, summary)


if __name__ == "__main__":
    main()
