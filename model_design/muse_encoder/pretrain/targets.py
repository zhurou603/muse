from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .aux import NamespacedRecord, iter_source

KIND_BY_TASK = {
    "regression": "graph",
    "vector_regression": "graph",
    "atom_regression": "atom",
    "bond_regression": "bond",
    "binary_classification": "graph",
    "multiclass_classification": "graph",
}


@dataclass(frozen=True)
class LabelSpec:
    source: str
    name: str
    unit: str | None
    kind: str
    task_type: str
    dim: int
    shadow: bool = False

    @property
    def key(self):
        unit = "" if self.unit is None else self.unit
        return f"{self.source}::{self.name}::{unit}"


def infer_dim(value, kind: str) -> int:
    array = np.asarray(value)
    if array.dtype == object:
        raise ValueError("标签不能是不规则 object 数组")
    if kind == "graph":
        return 1 if array.ndim == 0 else int(array.shape[-1])
    if array.ndim == 0:
        raise ValueError(f"{kind} 标签至少需要一维")
    return 1 if array.ndim == 1 else int(array.shape[-1])


def specs_from_record(record: NamespacedRecord, shadow_keys: set[str] | None = None):
    shadow_keys = shadow_keys or set()
    specs = []
    for name, observations in (record.payload.get("labels") or {}).items():
        for observation in observations:
            task_type = observation["task_type"]
            if task_type not in KIND_BY_TASK:
                continue
            kind = KIND_BY_TASK[task_type]
            value = observation.get("value")
            if value is None:
                continue
            unit = observation.get("unit")
            probe = LabelSpec(record.source, name, unit, kind, task_type,
                              infer_dim(value, kind))
            if probe.key in shadow_keys or f"{record.source}::{name}" in shadow_keys:
                probe = LabelSpec(record.source, name, unit, kind, task_type,
                                  probe.dim, True)
            specs.append(probe)
    return specs


class RunningStats:
    def __init__(self, dim: int):
        self.n = np.zeros(dim, dtype=np.int64)
        self.total = np.zeros(dim, dtype=np.float64)
        self.total_sq = np.zeros(dim, dtype=np.float64)
        self.bad = 0

    def add(self, value, kind: str):
        array = np.asarray(value, dtype=np.float64)
        if self.n.size == 1:
            array = array.reshape(-1, 1)
        elif kind == "graph":
            array = array.reshape(-1, self.n.size)
        else:
            array = array.reshape(-1, self.n.size)
        if array.shape[-1] != self.n.size:
            raise ValueError(f"标签维度 {array.shape[-1]} != {self.n.size}")
        good = np.isfinite(array)
        self.bad += int((~good).sum())
        self.n += good.sum(axis=0)
        safe = np.where(good, array, 0.0)
        self.total += safe.sum(axis=0)
        self.total_sq += np.square(safe).sum(axis=0)

    def result(self):
        if (self.n == 0).any():
            raise ValueError("部分标签维度没有有效观测")
        mean = self.total / self.n
        variance = np.maximum(self.total_sq / self.n - np.square(mean), 0.0)
        std = np.sqrt(variance)
        return {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "mask": (std > 1e-12).tolist(),
            "n": self.n.tolist(),
            "n_bad": self.bad,
        }


def compute_target_stats(sources, output: Path, shadow_keys: set[str] | None = None):
    shadow_keys = shadow_keys or set()
    specs = {}
    stats = {}
    for source in sources:
        for record in iter_source(source):
            for spec in specs_from_record(record, shadow_keys):
                previous = specs.get(spec.key)
                if previous is not None and previous != spec:
                    raise ValueError(f"标签 schema 冲突: {previous} vs {spec}")
                specs[spec.key] = spec
                stats.setdefault(spec.key, RunningStats(spec.dim))
                observations = record.payload["labels"][spec.name]
                for observation in observations:
                    if (observation.get("unit") == spec.unit
                            and observation["task_type"] == spec.task_type
                            and observation.get("value") is not None):
                        stats[spec.key].add(observation["value"], spec.kind)
    payload = {
        "specs": {key: asdict(spec) for key, spec in sorted(specs.items())},
        "stats": {key: stats[key].result() for key in sorted(stats)},
        "shadow_keys": sorted(shadow_keys),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


class TargetNormalizer:
    def __init__(self, payload: dict):
        self.specs = {key: LabelSpec(**value) for key, value in payload["specs"].items()}
        self.stats = payload["stats"]

    @classmethod
    def from_path(cls, path: Path):
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def normalize(self, key: str, value) -> tuple[np.ndarray, np.ndarray]:
        spec = self.specs[key]
        stat = self.stats[key]
        mean = np.asarray(stat["mean"], dtype=np.float32)
        std = np.asarray(stat["std"], dtype=np.float32)
        valid_dimensions = np.asarray(stat["mask"], dtype=bool)
        if value is None:
            return np.zeros(spec.dim, dtype=np.float32), np.zeros(spec.dim, dtype=bool)
        array = np.asarray(value, dtype=np.float32)
        array = array.reshape(-1, spec.dim) if spec.dim > 1 else array.reshape(-1, 1)
        finite = np.isfinite(array)
        if spec.task_type.endswith("classification"):
            normalized = np.where(finite, array, 0.0).astype(np.float32)
            mask = finite
        else:
            safe_std = np.where(valid_dimensions, std, 1.0)
            normalized = (array - mean) / safe_std
            normalized = np.where(finite & valid_dimensions, normalized, 0.0).astype(np.float32)
            mask = finite & valid_dimensions
        if spec.dim == 1 and spec.kind == "graph":
            normalized = normalized.reshape(-1)
            mask = mask.reshape(-1)
        return normalized, mask

    def aux_records(self, record: NamespacedRecord, keys=None) -> list[dict]:
        keys = list(keys) if keys is not None else [
            key for key, spec in self.specs.items() if spec.source == record.source and not spec.shadow
        ]
        output = []
        for key in keys:
            spec = self.specs[key]
            if spec.source != record.source or spec.shadow:
                continue
            value = None
            for observation in (record.payload.get("labels") or {}).get(spec.name, []):
                if (observation.get("unit") == spec.unit
                        and observation["task_type"] == spec.task_type):
                    value = observation.get("value")
                    break
            normalized, mask = self.normalize(key, value)
            output.append({"name": key, "value": normalized, "mask": mask})
        return output
