"""Verify registered A-test repairs on CPU without changing model weights."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger

from muse_encoder.config import MuseConfig
from muse_encoder.featurize import RecordFeaturizer, collate
from muse_encoder.model import MuseEncoder
from muse_encoder.pretrain import dashboard as D
from muse_encoder.structures import mol_from_sdf_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--repairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.handoff_root.resolve()
    failures = [json.loads(line) for line in (root / "failures.jsonl").read_text().splitlines()
                if line.strip()]
    records = {}
    for path in (root / "records").glob("*.jsonl"):
        for record in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
            if "labels" in record or "targets" in record:
                raise ValueError(f"{path}: 验证器拒绝标签")
            records[record["id"]] = record
    if len(records) != len(failures) or {row["id"] for row in failures} != set(records):
        raise ValueError("失败清单与无标签输入不一致")
    cfg = MuseConfig(
        d_model=32, n_layers=2, n_heads=4, d_ffn=64,
        num_experts=2, mole_every=2, embed_dim=32,
        contrastive_dim=16, max_seq_len=4096, gradient_checkpointing=False,
    )
    featurizer = RecordFeaturizer(cfg, D.load_tokenizers())
    torch.manual_seed(4100)
    model = MuseEncoder(cfg).eval()
    D.configure_official_graph_repairs(args.repairs)

    class LocalResolver:
        def __init__(self, failure):
            self.failure = failure

        def molecular_mol(self, _record):
            path = root / self.failure["structure_file"]
            if self.failure["dataset"] == "QM9":
                return mol_from_sdf_bytes(path.read_bytes(), self.failure["id"])
            return Chem.Mol(path.read_bytes())

    results, errors = [], []
    try:
        for index, failure in enumerate(failures, start=1):
            raw = records[failure["id"]]
            before = json.dumps(raw, sort_keys=True, ensure_ascii=False)
            try:
                resolver = LocalResolver(failure) if raw.get("external_inputs") else None
                normalized = D.normalize_record(raw, resolver)
                record = D._prepare_featurized(featurizer, normalized)
                if record.n_real_tokens < 1 or len(record.token_ids) != record.n_real_tokens + 1:
                    raise ValueError("token 数量与记录不一致")
                arrays = collate([record], cfg)
                batch = {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                         for key, value in arrays.items()}
                with torch.no_grad():
                    encoded = model.encode(batch)
                if tuple(encoded["embedding"].shape) != (1, cfg.embed_dim) \
                        or not torch.isfinite(encoded["embedding"]).all() \
                        or not torch.isfinite(encoded["tokens"]).all():
                    raise ValueError("Encoder 前向产生非法表征")
                if json.dumps(raw, sort_keys=True, ensure_ascii=False) != before:
                    raise ValueError("原始输入被特征化修改")
                repair = record.meta.get("input_repair", {})
                results.append({
                    "dataset": failure["dataset"], "split": failure["split"],
                    "id": failure["id"],
                    "representation": repair.get("representation", "official_graph"),
                    "parse_mode": record.meta.get("parse_mode"),
                    "coordinates": record.coords is not None,
                    "degraded": bool(record.meta.get("degraded", False)),
                })
            except Exception as exc:
                errors.append({"dataset": failure["dataset"], "split": failure["split"],
                               "id": failure["id"], "error": f"{type(exc).__name__}: {exc}"})
            if index % 100 == 0:
                print(f"[cpu-verify] processed={index}/{len(failures)} errors={len(errors)}",
                      flush=True)
    finally:
        D.configure_official_graph_repairs(None)
    summary = {
        "schema": "muse_encoder.input_repair_cpu_verification.v1",
        "selected": len(failures), "encoded": len(results),
        "failed": errors, "all_finite": not errors,
        "with_coordinates": sum(row["coordinates"] for row in results),
        "without_coordinates": sum(not row["coordinates"] for row in results),
        "degraded": sum(row["degraded"] for row in results),
        "by_representation": dict(collections.Counter(
            row["representation"] for row in results)),
        "by_dataset": {
            dataset: {
                "selected": sum(row["dataset"] == dataset for row in failures),
                "encoded": sum(row["dataset"] == dataset for row in results),
                "failed": sum(row["dataset"] == dataset for row in errors),
            } for dataset in sorted({row["dataset"] for row in failures})
        },
        "labels_used": False, "third_party_weights_loaded": False,
        "model": "randomly initialized two-layer CPU smoke model; no A-test score",
        "repairs_sha256": D.sha256_file(args.repairs),
        "code_sha256": {
            "structures": D.sha256_file(D.PROJECT_ROOT / "model_design/muse_encoder/structures.py"),
            "featurize": D.sha256_file(D.PROJECT_ROOT / "model_design/muse_encoder/featurize.py"),
            "dashboard": D.sha256_file(Path(D.__file__)),
        },
        "errors_detail": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "errors_detail"},
                     ensure_ascii=False, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    RDLogger.DisableLog("rdApp.*")
    main()
