from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from muse_encoder.featurize import RecordFeaturizer
from muse_encoder.pretrain import dashboard as D
from muse_encoder.tokenizers import load_tokenizers
from shard_eval import _read_sdf_selected_tolerant


def restore_shadow_labels(record):
    labels = record.get("labels")
    if labels is None:
        labels = {
            name: [{"task_type": task_type, "value": value} for value in values]
            for name, (task_type, values) in (record.get("targets") or {}).items()
        }
    output = {**record, "labels": labels}
    if output.get("sample_type") == "protein_construct":
        output["sample_type"] = "protein"
    return output


def evaluate_with_labels(encoder, featurizer, cfg, property_name, records, *args, **kwargs):
    restored = [restore_shadow_labels(record) for record in records]
    return evaluate_shadow_records_original(
        encoder, featurizer, cfg, property_name, restored, *args, **kwargs)


def collect_shadow_records(source, property_name, limit):
    if (source, property_name) != ("qmugs", "gfn2_dipole"):
        return collect_shadow_records_original(source, property_name, limit)
    from muse_encoder.pretrain.aux import iter_source
    records = []
    for namespaced in iter_source(source):
        observations = (namespaced.payload.get("labels") or {}).get(property_name, [])
        for observation in observations:
            value = observation.get("value")
            if observation.get("task_type") != "vector_regression" \
                    or not isinstance(value, list) or len(value) != 4 \
                    or not all(math.isfinite(float(item)) for item in value):
                continue
            payload = dict(namespaced.payload)
            payload["labels"] = dict(payload.get("labels") or {})
            payload["labels"][property_name] = [{
                "task_type": "regression", "value": float(value[3]),
                "shadow_source_task_type": "vector_regression",
                "shadow_component": "total_dipole",
            }]
            records.append(payload)
            break
        if limit is not None and len(records) >= limit:
            break
    return records


evaluate_shadow_records_original = D.evaluate_shadow_records
collect_shadow_records_original = D._collect_shadow_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.limit < 20:
        raise ValueError("shadow 每属性至少需要20条记录")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    D.evaluate_shadow_records = evaluate_with_labels
    D._collect_shadow_records = collect_shadow_records
    D._read_sdf_selected = _read_sdf_selected_tolerant
    encoder, cfg, info = D.load_checkpoint(args.checkpoint, args.device)
    featurizer = RecordFeaturizer(cfg, load_tokenizers())
    result = D.run_shadow_probes(
        encoder, featurizer, cfg, args.output_dir, info["sha256"],
        args.limit, args.batch_size, args.device)
    result["checkpoint"] = str(args.checkpoint.resolve())
    result["checkpoint_sha256"] = info["sha256"]
    result["probe_config"] = D.PROBE_CONFIG.to_json()
    valid = [row["normalized_score"] for row in result["properties"].values()
             if row.get("status") == "ok" and row.get("normalized_score") is not None]
    result["summary"] = {
        "num_properties": len(result["properties"]),
        "num_ok": len(valid),
        "mean_normalized_score": sum(valid) / len(valid) if valid else None,
        "all_ok": len(valid) == 8 and len(result["properties"]) == 8,
    }
    D._write_json(args.output_dir / "shadow_results.json", result)
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
