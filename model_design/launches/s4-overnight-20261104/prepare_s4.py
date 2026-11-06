from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from dbaasp import robust_stats
from muse_encoder.pretrain.a_train import (
    _aggregate_label,
    compute_a_train_target_stats,
    iter_train_records,
    load_source_specs,
)
from muse_encoder.pretrain.dashboard import DATA_ROOT
from muse_encoder.pretrain.trainer import sha256_file

from s4_common import DATA, PARENT, PARENT_SHA256, RUN_NAME, treatment_weights, domain_source_weights


def atomic_json(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    output = DATA / "bio_targets.json"
    if output.exists() or (DATA / "manifest.json").exists():
        raise FileExistsError(f"S4 数据目录已经准备，拒绝覆盖: {DATA}")
    registered = load_source_specs()
    sources = [spec for spec in registered.values() if spec.dataset in ("PEER", "CATH")]
    if len(sources) != 17:
        raise RuntimeError(f"PEER/CATH source 数应为17，实际{len(sources)}")
    payload = compute_a_train_target_stats(sources, DATA / "bio_targets.base.json")
    for source in sources:
        values = {task.name: [] for task in source.tasks if task.task_type == "regression"}
        for record in iter_train_records(source):
            for task in source.tasks:
                if task.task_type != "regression":
                    continue
                value, _, _ = _aggregate_label(record, task)
                if value is not None:
                    values[task.name].append(value)
        for task in source.tasks:
            if task.task_type != "regression":
                continue
            median, scale = robust_stats(np.asarray(values[task.name], dtype=np.float32))
            stat = payload["stats"][task.key]
            stat["mean"] = [float(median)]
            stat["std"] = [float(scale)]
            stat["robust_scaling"] = "median/MAD with std-or-1 fallback"
    payload["label_policy"]["regression"] = (
        "mean of finite observations within each train record; "
        "train-only per-task median/MAD; Huber beta=1")
    atomic_json(output, payload)
    inputs = {}
    root = Path(__file__).resolve().parents[3]
    for path in (
        DATA_ROOT / "manifest.json",
        root / "data/pretrain/stage2/dbaasp/targets.json",
        root / "data/pretrain/stage2/dbaasp/unsupported_train.rows.txt",
        root / "data/pretrain/stage3-polymer/targets.json",
        root / "data/pretrain/targets/aux_targets.json",
        root / "data/pretrain/targets/mtr_norm.json",
        root / "data/pretrain/manifest/shadow_properties.json",
        PARENT,
    ):
        inputs[str(path.relative_to(root))] = {"path": str(path), "sha256": sha256_file(path)}
    if inputs[str(PARENT.relative_to(root))]["sha256"] != PARENT_SHA256:
        raise RuntimeError("父 checkpoint 哈希变化")
    weights = treatment_weights()
    manifest = {
        "schema_version": "muse.s4.consolidation_data.v1",
        "run_name": RUN_NAME,
        "split_used_for_targets": "train",
        "validation_test_labels_used": False,
        "new_external_data": False,
        "uses_existing_replay_corpora": True,
        "crystal_training": False,
        "sources": {"PEER_CATH": [spec.name for spec in sources]},
        "treatment_weights": weights,
        "domain_weights": domain_source_weights(weights),
        "inputs": inputs,
        "bio_targets_sha256": sha256_file(output),
    }
    atomic_json(DATA / "manifest.json", manifest)
    print(json.dumps({
        "status": "prepared",
        "bio_tasks": len(payload["specs"]),
        "bio_targets": str(output),
        "bio_targets_sha256": manifest["bio_targets_sha256"],
        "manifest": str(DATA / "manifest.json"),
        "domain_weights": manifest["domain_weights"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
