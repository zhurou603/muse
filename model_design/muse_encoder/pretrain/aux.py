"""官方 JSONL aux 的合规排除、流式读取与标签辅助。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Set, Tuple, Union


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUX_ROOT = PROJECT_ROOT / "data/muse-a/MUSE-A/aux_labels/A_datasets"
AUDIT_ROOT = PROJECT_ROOT / "data/pretrain/audit"


@dataclass(frozen=True)
class AuxSource:
    name: str
    paths: Tuple[Path, ...]
    hits_path: Path
    expected_raw: int
    expected_excluded: int
    expected_kept: int


@dataclass(frozen=True)
class NamespacedRecord:
    source: str
    id: str
    payload: Dict[str, Any]


@dataclass(frozen=True)
class SourceCounts:
    raw: int
    excluded: int
    kept: int


def _make_source(name: str, paths: Iterable[str], raw: int,
                 excluded: int, kept: int) -> AuxSource:
    return AuxSource(
        name=name,
        paths=tuple(AUX_ROOT / path for path in paths),
        hits_path=AUDIT_ROOT / f"aux_{name}.hits.jsonl",
        expected_raw=raw,
        expected_excluded=excluded,
        expected_kept=kept,
    )


AUX_SOURCES = {
    source.name: source
    for source in (
        _make_source(
            "geom_drugs",
            [f"small_molecule/GEOM-Drugs/rdkit_folder/aux-{i:02d}.jsonl"
             for i in range(5)],
            32305220, 592688, 31712532,
        ),
        _make_source(
            "qmugs",
            [f"small_molecule/QMugs/qmugs/aux-{i:02d}.jsonl" for i in range(2)],
            1912850, 14785, 1898065,
        ),
        _make_source(
            "qm9", ["small_molecule/QM9/qm9/aux.jsonl"],
            90965, 1431, 89534,
        ),
        _make_source(
            "pubchemqc", ["small_molecule/PubChemQC/b3lyp/aux.jsonl"],
            3970758, 192995, 3777763,
        ),
        _make_source(
            "pcqm4v2", ["small_molecule/pcqm4v2/pcqm4v2/aux.jsonl"],
            3445326, 1148, 3444178,
        ),
        _make_source(
            "megascale_d1",
            ["biomacromolecule/MegaScale/dataset_1/aux.jsonl"],
            1838174, 346534, 1491640,
        ),
        _make_source(
            "megascale_d23",
            ["biomacromolecule/MegaScale/dataset_2_3/aux.jsonl"],
            775516, 300751, 474765,
        ),
        _make_source(
            "dbaasp_hemo",
            ["biomacromolecule/dbaasp_v4/hemolytic_cytotoxic/aux.jsonl"],
            13848, 2441, 11407,
        ),
        _make_source(
            "dbaasp_nonmic",
            ["biomacromolecule/dbaasp_v4/target_activity_non_mic/aux.jsonl"],
            24568, 4569, 19999,
        ),
    )
}

SourceRef = Union[str, AuxSource]
LabelKey = Tuple[str, Optional[str]]


def _resolve_source(source: SourceRef) -> AuxSource:
    return AUX_SOURCES[source] if isinstance(source, str) else source


def load_hit_ids(path: Union[str, Path]) -> Set[str]:
    """逐行加载 hits 的顶层 id；reason 不参与筛选。"""
    ids = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


def _iter_jsonl(paths: Iterable[Path]) -> Iterator[Dict[str, Any]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def iter_source(source: SourceRef,
                hit_ids: Optional[Set[str]] = None) -> Iterator[NamespacedRecord]:
    """流式产出一个源的未命中记录，不修改或跨记录去重 payload。"""
    spec = _resolve_source(source)
    hits = load_hit_ids(spec.hits_path) if hit_ids is None else hit_ids
    for record in _iter_jsonl(spec.paths):
        record_id = record["id"]
        if record_id not in hits:
            yield NamespacedRecord(spec.name, record_id, record)


def iter_sources(sources: Iterable[SourceRef]) -> Iterator[NamespacedRecord]:
    """顺序合并合规记录，始终保留 (source, id) 命名空间。"""
    for source in sources:
        yield from iter_source(source)


def validate_source(source: SourceRef,
                    hit_ids: Optional[Set[str]] = None) -> SourceCounts:
    """全量扫描一个源，并断言原始、排除、保留计数均符合注册表。"""
    spec = _resolve_source(source)
    hits = load_hit_ids(spec.hits_path) if hit_ids is None else hit_ids
    raw = excluded = 0
    for record in _iter_jsonl(spec.paths):
        raw += 1
        if record["id"] in hits:
            excluded += 1
    counts = SourceCounts(raw, excluded, raw - excluded)
    expected = SourceCounts(
        spec.expected_raw, spec.expected_excluded, spec.expected_kept
    )
    if counts != expected:
        raise ValueError(
            f"{spec.name} aux count mismatch: actual={counts}, expected={expected}"
        )
    return counts


def label_key(name: str, unit: Optional[str] = None) -> LabelKey:
    """以原始标签名和精确单位构成分组键。"""
    return name, unit


def extract_labels(record: Union[NamespacedRecord, Mapping[str, Any]],
                   keys: Iterable[LabelKey]
                   ) -> Tuple[Dict[LabelKey, Any], Dict[LabelKey, bool]]:
    """按 (名称, 单位) 提取标签；missing_mask 为 True 表示缺失。"""
    payload = record.payload if isinstance(record, NamespacedRecord) else record
    labels = payload.get("labels") or {}
    values = {}
    missing_mask = {}
    for key in keys:
        name, unit = key
        value = None
        missing = True
        for observation in labels.get(name) or ():
            if observation.get("unit") == unit:
                candidate = observation.get("value")
                if candidate is not None:
                    value = candidate
                    missing = False
                break
        values[key] = value
        missing_mask[key] = missing
    return values, missing_mask


__all__ = [
    "AUX_SOURCES", "AuxSource", "LabelKey", "NamespacedRecord", "SourceCounts",
    "extract_labels", "iter_source", "iter_sources", "label_key", "load_hit_ids",
    "validate_source",
]
