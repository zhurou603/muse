"""Train-only OMC25 element references and residual target statistics."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..featurize import _ELEMENT_INDEX


SCHEMA_VERSION = "muse.omc25.element_references.v1"
RESIDUAL_MODE = "element_reference_residual"
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)([1-9][0-9]*)?")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def formula_counts(formula: str) -> dict[int, int]:
    if not isinstance(formula, str) or not formula:
        raise ValueError("chemical_formula 必须为非空字符串")
    counts: dict[int, int] = {}
    position = 0
    for match in _FORMULA_TOKEN.finditer(formula):
        if match.start() != position:
            raise ValueError(f"不支持或非法的 chemical_formula: {formula!r}")
        symbol, count = match.groups()
        atomic_number = _ELEMENT_INDEX.get(symbol)
        if atomic_number is None:
            raise ValueError(f"未知元素: {symbol!r}")
        counts[atomic_number] = counts.get(atomic_number, 0) + int(count or 1)
        position = match.end()
    if position != len(formula) or not counts:
        raise ValueError(f"不支持或非法的 chemical_formula: {formula!r}")
    return counts


def load_heap_exclusions(path: Path | str | None) -> frozenset[str]:
    if path is None:
        return frozenset()
    heaps = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            heap = record.get("starting_heap")
            if not isinstance(heap, str) or not heap:
                raise ValueError(f"{path}:{line_number}: 缺少 starting_heap")
            heaps.add(heap)
    return frozenset(heaps)


@dataclass(frozen=True)
class ElementReferences:
    element_energies: tuple[float | None, ...]
    intercept: float
    sha256: str = ""

    def __post_init__(self) -> None:
        if len(self.element_energies) != 119:
            raise ValueError("element_energies 必须按原子序数 0–118 排列")
        if not math.isfinite(self.intercept):
            raise ValueError("参考能截距必须有限")
        for value in self.element_energies:
            if value is not None and not math.isfinite(value):
                raise ValueError("元素参考能必须为有限值或 null")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], sha256: str = "") -> "ElementReferences":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("元素参考能 schema_version 不符")
        return cls(
            element_energies=tuple(payload["element_energies_ev"]),
            intercept=float(payload["intercept_ev"]),
            sha256=sha256,
        )

    @classmethod
    def from_json(cls, path: Path | str) -> "ElementReferences":
        raw = Path(path).read_bytes()
        return cls.from_mapping(json.loads(raw), hashlib.sha256(raw).hexdigest())

    def reference_energy(self, atomic_numbers: Sequence[int] | np.ndarray) -> float:
        numbers = np.asarray(atomic_numbers)
        if numbers.ndim != 1 or not len(numbers) or numbers.dtype.kind not in "iu":
            raise ValueError("atomic_numbers 必须为非空整数一维数组")
        total = float(self.intercept)
        for number in np.unique(numbers):
            atomic_number = int(number)
            if not 1 <= atomic_number <= 118:
                raise ValueError(f"非法原子序数: {atomic_number}")
            coefficient = self.element_energies[atomic_number]
            if coefficient is None:
                raise ValueError(f"训练参考能未覆盖元素 Z={atomic_number}")
            total += coefficient * int((numbers == atomic_number).sum())
        if not math.isfinite(total):
            raise ValueError("元素参考能求和结果非有限")
        return total


def fit_element_references(
    train_jsonl: Path | str,
    max_records: int | None = 100_000,
    exclude_heaps: Sequence[str] | frozenset[str] = (),
) -> dict[str, Any]:
    """Fit E = sum_Z(n_Z e_Z) + b using only records explicitly split as train."""
    train_jsonl = Path(train_jsonl)
    if max_records is not None and max_records < 1:
        raise ValueError("max_records 必须为正数或 None")
    excluded = frozenset(exclude_heaps)

    def rows():
        with train_jsonl.open("rb") as stream:
            read = 0
            for line_number, raw in enumerate(stream, start=1):
                if max_records is not None and read >= max_records:
                    break
                read += 1
                if not raw.strip():
                    raise ValueError(f"{train_jsonl}:{line_number}: 空行")
                record = json.loads(raw)
                sid = record["external_inputs"][0]["selector"]["sid"]
                if sid.rpartition("-")[0] in excluded:
                    continue
                observations = (record.get("labels") or {}).get("energy") or []
                if len(observations) != 1:
                    raise ValueError(f"{train_jsonl}:{line_number}: 期望一个 energy 标签")
                observation = observations[0]
                if observation.get("split") != "train" or observation.get("unit") != "eV":
                    raise ValueError(f"{train_jsonl}:{line_number}: 只允许 train split 的 eV 能量")
                energy = observation.get("value")
                if isinstance(energy, bool) or not isinstance(energy, (int, float)) \
                        or not math.isfinite(energy):
                    raise ValueError(f"{train_jsonl}:{line_number}: 非法能量")
                counts = formula_counts(record["inputs"]["chemical_formula"])
                if sum(counts.values()) != record["inputs"]["atom_count"]:
                    raise ValueError(f"{train_jsonl}:{line_number}: 化学式与 atom_count 不符")
                features = np.zeros(120, dtype=np.float64)
                for atomic_number, count in counts.items():
                    features[atomic_number] = count
                features[119] = 1.0
                yield raw, features, float(energy), read

    fit_features = []
    fit_energies = []
    used = read = 0
    records_digest = hashlib.sha256()
    for raw, features, energy, read in rows():
        records_digest.update(raw)
        fit_features.append(features)
        fit_energies.append(energy)
        used += 1
    if used < 2:
        raise ValueError("参考能拟合至少需要两条合法 train 记录")
    design = np.asarray(fit_features)
    targets = np.asarray(fit_energies)
    coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]

    residual_count = 0
    residual_mean = residual_m2 = 0.0
    for _raw, features, energy, _read in rows():
        residual = energy - float(features @ coefficients)
        residual_count += 1
        delta = residual - residual_mean
        residual_mean += delta / residual_count
        residual_m2 += delta * (residual - residual_mean)
    if residual_count != used:
        raise RuntimeError("两遍参考能拟合的记录数不一致")
    residual_variance = max(residual_m2 / residual_count, 0.0)
    residual_std = math.sqrt(residual_variance)
    if not math.isfinite(residual_std) or residual_std <= 0:
        raise ValueError("残差标准差必须有限且大于零")
    seen = np.any(design != 0, axis=0)
    element_energies = [None] + [
        float(coefficients[index]) if seen[index] else None
        for index in range(1, 119)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "element_energies_ev": element_energies,
        "intercept_ev": float(coefficients[119]),
        "fit": {
            "source_jsonl": str(train_jsonl.resolve()),
            "source_split": "train",
            "records_read": read,
            "records_used": used,
            "max_records": max_records,
            "records_sha256": records_digest.hexdigest(),
            "excluded_heap_count": len(excluded),
            "residual_mean_ev": residual_mean,
            "residual_std_ev": residual_std,
            "residual_variance_ev2": residual_variance,
        },
    }


def write_references_and_stats(
    references_path: Path | str,
    stats_output: Path | str,
    reference_payload: Mapping[str, Any],
    frame_stats_path: Path | str,
) -> None:
    references_path = Path(references_path)
    _write_json_atomic(references_path, reference_payload)
    reference_sha256 = hashlib.sha256(references_path.read_bytes()).hexdigest()
    fit = reference_payload["fit"]
    stats = copy.deepcopy(json.loads(Path(frame_stats_path).read_text(encoding="utf-8")))
    stats["energy"] = {
        "count": fit["records_used"],
        "ddof": 0,
        "mean": fit["residual_mean_ev"],
        "std": fit["residual_std_ev"],
        "variance": fit["residual_variance_ev2"],
        "nonfinite": 0,
        "unit": "eV (total residual after element references)",
        "mode": RESIDUAL_MODE,
        "reference_sha256": reference_sha256,
        "reference_fit_records": fit["records_used"],
    }
    _write_json_atomic(Path(stats_output), stats)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--records", type=int, default=100_000,
                        help="最多读取的 train 记录数；0 表示全量")
    parser.add_argument("--exclude-heaps", type=Path)
    parser.add_argument("--frame-stats", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    args = parser.parse_args()
    payload = fit_element_references(
        args.train,
        max_records=args.records or None,
        exclude_heaps=load_heap_exclusions(args.exclude_heaps),
    )
    write_references_and_stats(args.out, args.stats_output, payload, args.frame_stats)
    print(json.dumps(payload["fit"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
