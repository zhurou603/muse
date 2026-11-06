"""Train-only median aggregation and median/MAD scaling, compatible with v1 payloads."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from muse_encoder.pretrain.a_train import (
    ATrainSourceSpec,
    ATrainTargetNormalizer,
    ATrainTaskSpec,
    compute_a_train_target_stats,
    iter_train_records,
    load_source_specs,
)


def regression_value(record: Mapping[str, Any], spec: ATrainTaskSpec,
                     policy: str = "median") -> float | None:
    if spec.task_type != "regression":
        raise ValueError(f"{spec.task_id}: 不是回归任务")
    values = []
    for observation in (record.get("labels") or {}).get(spec.name) or []:
        if observation.get("task_type") != spec.task_type:
            raise ValueError(f"{record['id']}/{spec.name}: observation task_type 不匹配")
        if "split" in observation and observation["split"] != "train":
            raise ValueError(f"{record['id']}/{spec.name}: 拒绝非 train observation")
        value = observation.get("value")
        if value is None:
            continue
        number = float(value)
        if np.isfinite(number):
            values.append(number)
    if not values:
        return None
    if policy == "median":
        return float(np.median(values))
    if policy == "mean":
        return float(np.mean(values))
    raise ValueError(f"未知回归聚合策略: {policy}")


def robust_location_scale(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("robust_location_scale 需要非空有限一维数组")
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    std = float(array.std())
    scale = mad if mad > 1e-8 else (std if std > 1e-8 else 1.0)
    return median, scale


def compute_s5_target_stats(sources: Sequence[str | ATrainSourceSpec], output: Path,
                            data_root: Path) -> dict[str, Any]:
    """Build the legacy payload first, then replace only regression statistics.

    Classification encoding/class discovery and all manifest/count checks stay in
    the reviewed base implementation.  The second pass touches train.jsonl only.
    """
    output = Path(output)
    base_path = output.with_suffix(".base.json")
    payload = compute_a_train_target_stats(sources, base_path, data_root=data_root, split="train")
    registered = load_source_specs(data_root)
    resolved = [registered[item] if isinstance(item, str) else item for item in sources]
    for source in resolved:
        tasks = [task for task in source.tasks if task.task_type == "regression"]
        if not tasks:
            continue
        values = {task.name: [] for task in tasks}
        for record in iter_train_records(source):
            for task in tasks:
                value = regression_value(record, task, "median")
                if value is not None:
                    values[task.name].append(value)
        for task in tasks:
            median, scale = robust_location_scale(values[task.name])
            stat = payload["stats"][task.key]
            if len(values[task.name]) != stat["n_labeled_records"]:
                raise RuntimeError(f"{task.task_id}: 两次 train 扫描的有效记录数不一致")
            stat.update({
                "mean": [median],
                "std": [scale],
                "mask": [True],
                "center": "median",
                "scale": "MAD with std-or-1 fallback",
                "record_aggregation": "median of finite train observations",
            })
    payload["label_policy"]["regression"] = (
        "median of finite observations within each train record; "
        "train-only per-task median/MAD; Huber delta=1"
    )
    payload["s5_extension"] = {
        "schema_version": "muse_encoder.s5_robust_targets.v1",
        "legacy_schema_preserved": True,
        "validation_test_labels_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, output)
    return payload


class S5TargetNormalizer(ATrainTargetNormalizer):
    """Keep base classification behavior; replace regression values/statistics."""

    def aux_records(self, source, raw, record):
        output = super().aux_records(source, raw, record)
        by_name = {spec.key: spec for spec in self.specs.values()
                   if spec.task_type == "regression"}
        for item in output:
            spec = by_name.get(item["name"])
            if spec is None:
                continue
            value = regression_value(raw, spec, "median")
            stat = self.stats[spec.key]
            normalized = np.zeros(1, dtype=np.float32)
            mask = np.zeros(1, dtype=bool)
            if value is not None and stat["mask"][0]:
                normalized[0] = (value - stat["mean"][0]) / stat["std"][0]
                mask[0] = bool(np.isfinite(normalized[0]))
                normalized[~mask] = 0.0
            item["value"] = normalized
            item["mask"] = mask
        return output
