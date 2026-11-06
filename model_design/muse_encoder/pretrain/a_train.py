"""MUSE-A 官方 train split 的轻量监督数据源。

只从 manifest 的 ``files.train`` 和对应 ``train.jsonl`` 读取记录；validation/test
标签不参与采样、统计或归一化。回归在一条记录内对有限观测取均值，统计也使用
该记录级均值。分类使用官方 scorer 的 JSON 类别编码和类别顺序；逐残基类别仅从
train 标签发现。输出可直接交给 ``BatchBuilder(..., items=...)``。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from ..featurize import FeaturizedRecord, RecordFeaturizer
from ..tokenizers import load_tokenizers
from .corpora import PRETRAIN_ROOT
from .dashboard import (
    DATA_ROOT,
    ExternalStructureResolver,
    _prepare_featurized,
    normalize_record,
    official_scorer,
)
from .sampling import SamplePointer, SourceSampler, SourceSpec
from .targets import RunningStats

TARGETS_SCHEMA = "muse_encoder.a_train_targets.v1"
TASK_TYPES = {
    "regression",
    "binary_classification",
    "multiclass_classification",
    "sequence_classification",
}
DOMAIN_MAP = {
    "small_molecule": "small_molecule",
    "polymer": "polymer",
    "organic_crystal": "crystal",
    "biomacromolecule": "bio",
}


@dataclass(frozen=True)
class ATrainTaskSpec:
    source: str
    name: str
    dataset: str
    subset: str
    domain: str
    kind: str
    task_type: str
    dim: int
    train_observations: int
    class_labels: tuple[str, ...] = ()
    positive_label: str | None = None

    @property
    def task_id(self) -> str:
        return f"{self.dataset}/{self.subset}/{self.name}"

    @property
    def key(self) -> str:
        return "a_train::" + self.task_id.replace(".", "%2E")

    @property
    def shadow(self) -> bool:
        return False

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "ATrainTaskSpec":
        raw = dict(value)
        raw["class_labels"] = tuple(raw.get("class_labels", ()))
        return cls(**raw)


@dataclass(frozen=True)
class ATrainSourceSpec:
    source: SourceSpec
    kind: str
    dataset: str
    subset: str
    domain: str
    train_path: Path
    input_keys: tuple[str, ...]
    external_formats: Mapping[str, int]
    sample_types: Mapping[str, int]
    tasks: tuple[ATrainTaskSpec, ...]

    @property
    def name(self) -> str:
        return self.source.name

    @property
    def records(self) -> int:
        return self.source.records

    @property
    def train_observations(self) -> int:
        return sum(task.train_observations for task in self.tasks)


def _encoded_classes(values: Iterable[Any]) -> tuple[str, ...]:
    scorer = official_scorer()
    encoded = []
    for value in values:
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                decoded = value
            value = decoded
        encoded.append(scorer.encode_class_label(value))
    return tuple(sorted(encoded, key=scorer._class_sort_key))


def load_source_specs(data_root: Path = DATA_ROOT) -> dict[str, ATrainSourceSpec]:
    data_root = Path(data_root).resolve()
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "muse.release.v2":
        raise ValueError("A-train 需要 muse.release.v2 manifest")
    output: dict[str, ATrainSourceSpec] = {}
    for dataset, dataset_meta in manifest["datasets"].items():
        for subset, meta in dataset_meta["subsets"].items():
            parts = PurePosixPath(str(meta["directory"])).parts
            if len(parts) != 4 or parts[0] != "A_datasets":
                continue
            domain = str(meta["domain"])
            if domain != parts[1] or domain not in DOMAIN_MAP:
                raise ValueError(f"{dataset}/{subset}: 非法 A 榜 domain {domain}")
            train_meta = (meta.get("files") or {}).get("train")
            if train_meta is None:
                raise ValueError(f"{dataset}/{subset}: 缺少官方 train.jsonl")
            relative = PurePosixPath(str(train_meta["file"]))
            expected = PurePosixPath(str(meta["directory"])) / "train.jsonl"
            if relative != expected or relative.name != "train.jsonl":
                raise ValueError(f"{dataset}/{subset}: 只允许 manifest 官方 train.jsonl")
            train_path = (data_root / Path(relative)).resolve()
            if data_root != train_path and data_root not in train_path.parents:
                raise ValueError(f"{dataset}/{subset}: train 路径越出 data_root")
            records = int(train_meta["records"])
            if records < 1:
                raise ValueError(f"{dataset}/{subset}: train records 必须为正数")
            source_name = f"{dataset}/{subset}"
            tasks = []
            for name, label_meta in (meta.get("labels") or {}).items():
                types = list((label_meta.get("task_types") or {}).keys())
                if len(types) != 1 or types[0] not in TASK_TYPES:
                    raise ValueError(f"{source_name}/{name}: 非法任务类型 {types}")
                task_type = types[0]
                observations = int(label_meta["split"]["train"])
                if observations < 1:
                    raise ValueError(f"{source_name}/{name}: train observations 必须为正数")
                if task_type == "sequence_classification":
                    classes = ()
                    positive = None
                    dim = 0
                else:
                    classes = _encoded_classes(label_meta.get("class_labels", ()))
                    positive = None
                    if task_type in ("binary_classification", "multiclass_classification"):
                        if len(classes) < 2:
                            raise ValueError(f"{source_name}/{name}: 分类类别少于两个")
                        dim = 1 if task_type == "binary_classification" else len(classes)
                    else:
                        dim = 1
                    if task_type == "binary_classification":
                        if len(classes) != 2:
                            raise ValueError(f"{source_name}/{name}: 二分类必须恰好两类")
                        positive = official_scorer()._positive_label(classes)
                        if positive is None:
                            raise ValueError(f"{source_name}/{name}: 无法确定 positive_label")
                tasks.append(ATrainTaskSpec(
                    source_name, name, dataset, subset, domain,
                    "atom" if task_type == "sequence_classification" else "graph",
                    task_type, dim, observations, classes, positive,
                ))
            if not tasks:
                raise ValueError(f"{source_name}: 没有 A-train 任务")
            output[source_name] = ATrainSourceSpec(
                SourceSpec(source_name, DOMAIN_MAP[domain], records, 1),
                "a_train", dataset, subset, domain, train_path,
                tuple(meta.get("input_keys") or ()),
                dict(meta.get("external_formats") or {}),
                dict(train_meta.get("sample_types") or {}),
                tuple(tasks),
            )
    if not output:
        raise ValueError(f"{data_root}: 没有 A 榜 train source")
    if data_root == Path(DATA_ROOT).resolve():
        records = sum(spec.records for spec in output.values())
        observations = sum(spec.train_observations for spec in output.values())
        tasks = sum(len(spec.tasks) for spec in output.values())
        if (len(output), tasks, records, observations) != (78, 90, 60_868_251, 70_909_310):
            raise RuntimeError(
                f"A 榜 train 清单异常: sources={len(output)}, tasks={tasks}, "
                f"records={records}, observations={observations}"
            )
    return output


def source_spec(name: str, data_root: Path = DATA_ROOT) -> ATrainSourceSpec:
    specs = load_source_specs(data_root)
    if name not in specs:
        raise KeyError(f"未知 A-train source: {name}")
    return specs[name]


def _validate_train_record(
    record: Mapping[str, Any], task_names: set[str], context: str,
) -> None:
    if not isinstance(record, Mapping) or "id" not in record:
        raise ValueError(f"{context}: 记录缺少 id")
    if "split" in record and record["split"] != "train":
        raise ValueError(f"{context}: 拒绝 split={record['split']!r}")
    labels = record.get("labels") or {}
    if not isinstance(labels, Mapping):
        raise ValueError(f"{context}: labels 必须是 mapping")
    unknown = set(labels) - task_names
    if unknown:
        raise ValueError(f"{context}: 未注册标签 {sorted(unknown)}")
    for name, observations in labels.items():
        if not isinstance(observations, list):
            raise ValueError(f"{context}/{name}: observations 必须是 list")
        for observation in observations:
            if not isinstance(observation, Mapping):
                raise ValueError(f"{context}/{name}: observation 必须是 mapping")
            if "split" in observation and observation["split"] != "train":
                raise ValueError(
                    f"{context}/{name}: 拒绝 observation split={observation['split']!r}"
                )


def iter_train_records(spec: ATrainSourceSpec) -> Iterator[dict[str, Any]]:
    task_names = {task.name for task in spec.tasks}
    with spec.train_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{spec.train_path}:{line_number}: 空行")
            record = json.loads(line)
            _validate_train_record(
                record, task_names, f"{spec.train_path}:{line_number}"
            )
            yield record


def _sequence_length(record: Mapping[str, Any]) -> int:
    inputs = record.get("inputs") or {}
    value = inputs.get("sequence") or inputs.get("protein_sequence")
    if not isinstance(value, str):
        raise ValueError(f"{record['id']}: 逐残基任务缺少单条 sequence")
    return sum(not char.isspace() for char in value.strip())


def _aggregate_label(
    record: Mapping[str, Any], spec: ATrainTaskSpec,
    sequence_length: int | None = None,
    class_universe: Counter | None = None,
) -> tuple[Any, int, int]:
    observations = (record.get("labels") or {}).get(spec.name) or []
    scorer = official_scorer()
    for observation in observations:
        if observation.get("task_type") != spec.task_type:
            raise ValueError(
                f"{record['id']}/{spec.name}: observation task_type "
                f"{observation.get('task_type')!r} != {spec.task_type!r}"
            )
    if spec.task_type == "regression":
        values = []
        bad = 0
        for observation in observations:
            value = observation.get("value")
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{record['id']}/{spec.name}: 非数值回归标签") from exc
            if np.isfinite(number):
                values.append(number)
            else:
                bad += 1
        return (float(np.mean(values)) if values else None), bad, len(observations)

    if spec.task_type == "sequence_classification":
        if sequence_length is None:
            raise ValueError(f"{record['id']}/{spec.name}: 缺少逐残基序列长度")
        encoded_observations = []
        for observation in observations:
            value = observation.get("value")
            if value is None:
                encoded_observations.append(None)
                continue
            if not isinstance(value, (list, tuple)) or len(value) != sequence_length:
                raise ValueError(
                    f"{record['id']}/{spec.name}: 残基标签数必须等于 {sequence_length}"
                )
            encoded = tuple(
                scorer.encode_class_label(item) if item is not None else None
                for item in value
            )
            if class_universe is not None:
                class_universe.update(item for item in encoded if item is not None)
            encoded_observations.append(encoded)
        first = encoded_observations[0] if encoded_observations else None
        value = first if first is not None and any(item is not None for item in first) else None
        return value, 0, len(observations)

    if not observations or observations[0].get("value") is None:
        return None, 0, len(observations)
    return scorer.encode_class_label(observations[0]["value"]), 0, len(observations)


def _check_known_classes(spec: ATrainTaskSpec, value: Any) -> None:
    known = set(spec.class_labels)
    values = value if spec.task_type == "sequence_classification" else (value,)
    unknown = {item for item in values if item is not None and item not in known}
    if unknown:
        raise ValueError(f"{spec.task_id}: train 标签不在类别表: {sorted(unknown)[:5]}")


def compute_a_train_target_stats(
    sources: Sequence[ATrainSourceSpec | str], output: Path,
    data_root: Path = DATA_ROOT, split: str = "train",
) -> dict[str, Any]:
    """流式统计所选 source；不会建立特征缓存或读取其他 split。"""
    if split != "train":
        raise ValueError(f"A-train 统计只允许 train，收到 {split!r}")
    if not sources:
        raise ValueError("sources 不能为空")
    registered = load_source_specs(data_root)
    resolved = [registered[name] if isinstance(name, str) else name for name in sources]
    names = [spec.name for spec in resolved]
    if len(set(names)) != len(names):
        raise ValueError("source 重复")
    for spec in resolved:
        if spec.name not in registered or registered[spec.name] != spec:
            raise ValueError(f"拒绝未注册或非官方 train source: {spec.name}")
    all_specs: dict[str, ATrainTaskSpec] = {}
    all_stats: dict[str, dict[str, Any]] = {}
    source_payload = {}
    for spec in resolved:
        states = {
            task.name: {
                "observations": 0,
                "labeled_records": 0,
                "missing_records": 0,
                "bad": 0,
                "running": RunningStats(1) if task.task_type == "regression" else None,
                "classes": Counter(),
                "class_universe": Counter(),
            }
            for task in spec.tasks
        }
        by_name = {task.name: task for task in spec.tasks}
        records = 0
        for records, record in enumerate(iter_train_records(spec), 1):
            sequence_length = (
                _sequence_length(record)
                if any(task.task_type == "sequence_classification" for task in spec.tasks)
                else None
            )
            for name, task in by_name.items():
                state = states[name]
                value, bad, observations = _aggregate_label(
                    record, task, sequence_length, state["class_universe"]
                )
                state["observations"] += observations
                state["bad"] += bad
                if value is None:
                    state["missing_records"] += 1
                    continue
                state["labeled_records"] += 1
                if task.task_type == "regression":
                    state["running"].add(value, "graph")
                elif task.task_type == "sequence_classification":
                    state["classes"].update(item for item in value if item is not None)
                else:
                    state["classes"].update((value,))
        if records != spec.records:
            raise ValueError(f"{spec.name}: train 记录数 {records} != manifest {spec.records}")
        source_payload[spec.name] = {
            "records": records,
            "observations": {},
            "train_file": str(spec.train_path),
        }
        for name, task in by_name.items():
            state = states[name]
            if state["observations"] != task.train_observations:
                raise ValueError(
                    f"{task.task_id}: train 观测数 {state['observations']} != "
                    f"manifest {task.train_observations}"
                )
            source_payload[spec.name]["observations"][name] = state["observations"]
            if task.task_type == "sequence_classification":
                classes = tuple(sorted(state["class_universe"], key=official_scorer()._class_sort_key))
                if len(classes) < 2:
                    raise ValueError(f"{task.task_id}: train 逐残基类别少于两个")
                task = replace(task, class_labels=classes, dim=len(classes))
            elif task.task_type != "regression":
                unknown = set(state["classes"]) - set(task.class_labels)
                if unknown:
                    raise ValueError(f"{task.task_id}: train 标签不在官方类别表: {sorted(unknown)[:5]}")
                classes = task.class_labels
            else:
                classes = ()
            if state["labeled_records"] == 0:
                raise ValueError(f"{task.task_id}: train 没有有效标签")
            if task.task_type == "regression":
                stat = state["running"].result()
            else:
                stat = {
                    "n": state["labeled_records"],
                    "class_counts": [int(state["classes"][label]) for label in classes],
                }
            stat.update({
                "n_observations": state["observations"],
                "n_labeled_records": state["labeled_records"],
                "n_missing_records": state["missing_records"],
                "n_bad": state["bad"],
            })
            if task.key in all_specs:
                raise ValueError(f"任务 head 名冲突: {task.key}")
            all_specs[task.key] = task
            all_stats[task.key] = stat
        source_payload[spec.name]["observation_count"] = sum(
            source_payload[spec.name]["observations"].values()
        )
    payload = {
        "schema_version": TARGETS_SCHEMA,
        "split": "train",
        "label_policy": {
            "regression": "mean of finite observations within each train record",
            "classification": "official JSON encoding; first observation per train record",
            "sequence": "first observation for loss; classes from all train observations; null masked",
        },
        "sources": source_payload,
        "specs": {key: asdict(all_specs[key]) for key in sorted(all_specs)},
        "stats": {key: all_stats[key] for key in sorted(all_stats)},
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return payload


def _validate_sequence_featurized(record: FeaturizedRecord, length: int, task_id: str) -> None:
    indices = np.asarray(record.meta.get("output_token_indices", ()), dtype=np.int64)
    if (len(record.token_ids) != length + 1 or record.n_real_tokens != length
            or not np.array_equal(indices, np.arange(1, length + 1))):
        raise ValueError(
            f"{task_id}: 逐残基 aux 需要无分隔符单序列且标签/token 一一对齐；"
            "序列对或蛋白配体不能走当前 atom collate"
        )


class ATrainTargetNormalizer:
    def __init__(self, payload: Mapping[str, Any]):
        if payload.get("schema_version") != TARGETS_SCHEMA or payload.get("split") != "train":
            raise ValueError("A-train normalizer 只接受 train-only targets payload")
        self.specs = {
            key: ATrainTaskSpec.from_payload(value)
            for key, value in payload["specs"].items()
        }
        for key, spec in self.specs.items():
            if key != spec.key:
                raise ValueError(f"任务 key 不一致: {key} != {spec.key}")
            if spec.dim < 1:
                raise ValueError(f"{spec.task_id}: 归一化任务 dim 必须已完成 train 类别发现")
        self.stats = dict(payload["stats"])
        if set(self.stats) != set(self.specs):
            raise ValueError("specs/stats 任务集合不一致")
        self.sources = dict(payload.get("sources") or {})
        self.payload = dict(payload)

    @classmethod
    def from_path(cls, path: Path) -> "ATrainTargetNormalizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def model_specs(self, source_names: Iterable[str] | None = None) -> dict[str, dict[str, Any]]:
        selected = set(source_names) if source_names is not None else None
        return {
            key: {"kind": spec.kind, "dim": spec.dim, "task_type": spec.task_type}
            for key, spec in self.specs.items()
            if selected is None or spec.source in selected
        }

    def aux_records(
        self, source: ATrainSourceSpec | str, raw: Mapping[str, Any],
        record: FeaturizedRecord,
    ) -> list[dict[str, Any]]:
        source_name = source.name if isinstance(source, ATrainSourceSpec) else str(source)
        specs = sorted(
            (spec for spec in self.specs.values() if spec.source == source_name),
            key=lambda spec: spec.task_id,
        )
        if not specs:
            raise KeyError(f"normalizer 不含 source: {source_name}")
        _validate_train_record(raw, {spec.name for spec in specs}, str(raw.get("id")))
        sequence_length = (
            _sequence_length(raw)
            if any(spec.task_type == "sequence_classification" for spec in specs)
            else None
        )
        output = []
        for spec in specs:
            value, _bad, _observations = _aggregate_label(raw, spec, sequence_length)
            if value is not None and spec.task_type != "regression":
                _check_known_classes(spec, value)
            if spec.task_type == "regression":
                stat = self.stats[spec.key]
                normalized = np.zeros(1, dtype=np.float32)
                mask = np.zeros(1, dtype=bool)
                if value is not None and stat["mask"][0]:
                    normalized[0] = (value - stat["mean"][0]) / stat["std"][0]
                    mask[0] = bool(np.isfinite(normalized[0]))
                    normalized[~mask] = 0.0
            elif spec.task_type == "binary_classification":
                normalized = np.zeros(1, dtype=np.float32)
                mask = np.zeros(1, dtype=bool)
                if value is not None:
                    normalized[0] = float(value == spec.positive_label)
                    mask[0] = True
            elif spec.task_type == "multiclass_classification":
                normalized = np.zeros(spec.dim, dtype=np.float32)
                mask = np.zeros(spec.dim, dtype=bool)
                if value is not None:
                    normalized[spec.class_labels.index(value)] = 1.0
                    mask[:] = True
            else:
                _validate_sequence_featurized(record, sequence_length, spec.task_id)
                normalized = np.zeros((sequence_length, spec.dim), dtype=np.float32)
                mask = np.zeros((sequence_length, spec.dim), dtype=bool)
                if value is not None:
                    for index, label in enumerate(value):
                        if label is not None:
                            normalized[index, spec.class_labels.index(label)] = 1.0
                            mask[index, :] = True
            output.append({"name": spec.key, "value": normalized, "mask": mask})
        return output


def _safe_source(name: str) -> str:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=6).hexdigest()
    return re.sub(r"[^A-Za-z0-9._-]", "_", name) + "-" + digest


class _TrainJsonlStore:
    def __init__(self, spec: ATrainSourceSpec, index_path: Path):
        self.spec = spec
        self.index_path = Path(index_path)
        self.metadata_path = self.index_path.with_suffix(".json")
        if not self.index_path.exists() and not self.metadata_path.exists():
            self._build_index()
        if not self.index_path.is_file() or not self.metadata_path.is_file():
            raise RuntimeError(f"{spec.name}: train 索引与 metadata 必须同时存在")
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        size = spec.train_path.stat().st_size
        expected = {
            "schema_version": "muse_encoder.a_train_index.v1",
            "split": "train",
            "source": spec.name,
            "train_file": str(spec.train_path),
            "records": spec.records,
            "bytes": size,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{spec.name}: train 索引 provenance/count 不匹配")
        self.offsets = np.load(self.index_path, mmap_mode="r")
        if len(self.offsets) != spec.records + 1 or int(self.offsets[-1]) != size:
            raise ValueError(f"{spec.name}: train offsets 与 manifest/文件大小不一致")
        self.data = np.memmap(spec.train_path, dtype=np.uint8, mode="r")

    def _build_index(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.index_path.with_suffix(".tmp.npy")
        offsets = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.int64, shape=(self.spec.records + 1,)
        )
        offsets[0] = 0
        count = 0
        with self.spec.train_path.open("rb") as stream:
            for line in stream:
                if not line.strip():
                    raise ValueError(f"{self.spec.train_path}:{count + 1}: 空行")
                count += 1
                if count > self.spec.records:
                    raise ValueError(f"{self.spec.name}: train 记录数超过 manifest")
                offsets[count] = stream.tell()
        if count != self.spec.records:
            raise ValueError(f"{self.spec.name}: train 记录数 {count} != {self.spec.records}")
        offsets.flush()
        del offsets
        metadata = {
            "schema_version": "muse_encoder.a_train_index.v1",
            "split": "train",
            "source": self.spec.name,
            "train_file": str(self.spec.train_path),
            "records": count,
            "bytes": self.spec.train_path.stat().st_size,
        }
        metadata_tmp = self.metadata_path.with_suffix(".tmp.json")
        metadata_tmp.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.index_path)
        os.replace(metadata_tmp, self.metadata_path)

    def get(self, row: int) -> dict[str, Any]:
        if row < 0 or row >= self.spec.records:
            raise IndexError(row)
        raw = bytes(self.data[self.offsets[row]:self.offsets[row + 1]]).decode("utf-8")
        record = json.loads(raw)
        _validate_train_record(record, {task.name for task in self.spec.tasks}, str(record.get("id")))
        return record


class ATrainDataset:
    """一个 official subset 的随机访问 train dataset，可配 SourceSampler 使用。"""

    def __init__(
        self, name: str, cfg, normalizer: ATrainTargetNormalizer | None = None,
        data_root: Path = DATA_ROOT, tokenizers=None, seed: int = 42,
        index_root: Path | None = None, cache_root: Path | None = None,
        split: str = "train", external_mol_cache_size: int = 128,
    ):
        if split != "train":
            raise ValueError(f"ATrainDataset 只允许 split='train'，收到 {split!r}")
        if external_mol_cache_size < 0:
            raise ValueError("external_mol_cache_size 必须非负")
        self.name = name
        self.spec = source_spec(name, data_root)
        self.kind = self.spec.kind
        self.cfg = cfg
        self.normalizer = normalizer
        self.tokenizers = tokenizers or load_tokenizers()
        self.featurizer = RecordFeaturizer(cfg, self.tokenizers)
        self.seed = seed
        if normalizer is not None:
            expected = {task.task_id for task in self.spec.tasks}
            actual = {
                spec.task_id for spec in normalizer.specs.values() if spec.source == name
            }
            if actual != expected:
                raise ValueError(f"{name}: normalizer 任务集合与 source 不一致")
        index_root = Path(index_root) if index_root is not None else PRETRAIN_ROOT / "index" / "a_train"
        cache_root = Path(cache_root) if cache_root is not None else PRETRAIN_ROOT / "cache" / "a_train"
        safe = _safe_source(name)
        self._store = _TrainJsonlStore(self.spec, index_root / f"{safe}.offsets.npy")
        self.external_base = (
            Path(data_root).resolve().parent / "external_inputs" / "A_datasets"
            / self.spec.domain / self.spec.dataset
        )
        self.external_cache = cache_root / safe
        self.external_mol_cache_size = int(external_mol_cache_size)
        self._resolver = None
        self._resolver_lock = threading.Lock()
        self._sampler = SourceSampler(self.spec.source)

    def __len__(self) -> int:
        return self.spec.records

    def sample(self, rng) -> SamplePointer:
        return self._sampler.sample(rng)

    def _trim_external_mols(self) -> None:
        if self._resolver is None:
            return
        for cache in (self._resolver.sdf_mols, self._resolver.geom_mols):
            while len(cache) > self.external_mol_cache_size:
                cache.pop(next(iter(cache)))

    def get(self, pointer: SamplePointer) -> tuple[FeaturizedRecord, dict[str, Any]]:
        if pointer.source != self.name or pointer.domain != self.spec.source.domain:
            raise ValueError(f"pointer 不属于 {self.name}: {pointer}")
        if pointer.view != 0:
            raise ValueError("A-train 没有增强 view，view 必须为 0")
        if self.normalizer is None:
            raise RuntimeError(f"{self.name}: 需要 train-only ATrainTargetNormalizer")
        raw = self._store.get(pointer.row)
        if raw.get("external_inputs"):
            with self._resolver_lock:
                if self._resolver is None:
                    self._resolver = ExternalStructureResolver(
                        self.external_base, self.external_cache
                    )
                self._resolver.prepare([raw])
                normalized = normalize_record(raw, self._resolver)
        else:
            normalized = normalize_record(raw)
        try:
            record = _prepare_featurized(self.featurizer, normalized)
        except ValueError as exc:
            inputs = normalized.get("inputs", {})
            if normalized.get("sample_type") not in ("molecule", "conformer") \
                    or inputs.get("sdf_mol") is None \
                    or "非立体化学结构不一致" not in str(exc):
                raise
            from rdkit import Chem
            from .molecule3d import _official_graph_variant
            variant = _official_graph_variant(inputs["smiles"], inputs["sdf_mol"])
            if variant is None:
                raise
            graph, coords, mapping = variant
            adjusted = dict(normalized)
            adjusted_inputs = dict(inputs)
            adjusted_inputs["smiles"] = Chem.MolToSmiles(graph, isomericSmiles=True)
            adjusted_inputs["coords"] = coords
            adjusted_inputs.pop("sdf_mol", None)
            adjusted["inputs"] = adjusted_inputs
            record = _prepare_featurized(self.featurizer, adjusted)
            source_bonds = [
                (int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx()))
                for bond in inputs["sdf_mol"].GetBonds()
            ]
            record.meta["structure_provenance"] = {
                "graph_source": "official_conformer",
                "alignment_note": str(exc),
                "n_atoms": int(graph.GetNumAtoms()),
                "heavy_atom_mapping": [int(value) for value in mapping],
                "source_bond_pairs": source_bonds,
            }
            record.meta["sdf_mapping"] = np.asarray(mapping, dtype=np.int64)
            record.meta["sdf_bond_pairs"] = np.asarray(source_bonds, dtype=np.int64)
        if raw.get("external_inputs"):
            with self._resolver_lock:
                self._trim_external_mols()
        record.meta.update({
            "a_train_source": self.name,
            "a_train_record_id": str(raw["id"]),
            "a_train_row": int(pointer.row),
            "a_train_split": "train",
        })
        metadata = {
            "aux_records": self.normalizer.aux_records(self.spec, raw, record),
            "a_train": {
                "source": self.name,
                "id": str(raw["id"]),
                "row": int(pointer.row),
                "split": "train",
                "tasks": [spec.task_id for spec in self.normalizer.specs.values()
                          if spec.source == self.name],
            },
        }
        return record, metadata

    def close(self) -> None:
        self._store.data = None
        self._resolver = None


__all__ = [
    "ATrainDataset",
    "ATrainSourceSpec",
    "ATrainTargetNormalizer",
    "ATrainTaskSpec",
    "TARGETS_SCHEMA",
    "compute_a_train_target_stats",
    "iter_train_records",
    "load_source_specs",
    "source_spec",
]
