"""可复现的 RDKit 3D 构象离线生成与按 aux 行号随机读取。

仅为 PubChemQC / PCQM4Mv2 这类没有官方坐标的 aux 源生成数据。输出位于
独立目录，不是官方 molecule3d 索引，也不会修改 QM9 / QMugs / GEOM 坐标。
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import operator
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..structures import require_rdkit
from .aux import AUX_SOURCES, AuxSource, iter_source
from .corpora import PRETRAIN_ROOT
from .molecule3d import AlignedStructure

GENERATED_SOURCES = ("pubchemqc", "pcqm4v2")
CONFORMER_ROOT = PRETRAIN_ROOT / "generated_conformers"
SCHEMA = "muse.generated_conformers.v1"
SHARD_SCHEMA = "muse.generated_conformer_shard.v1"
METHOD = "ETKDGv3+MMFFOptimizeMolecule(MMFF94)"
MMFF_MAX_ITERATIONS = 200
COORDINATE_DTYPE = np.dtype("float32")

SourceRef = Union[str, AuxSource]


@dataclass(frozen=True)
class _ShardJob:
    directory: Path
    shard_index: int
    records: Tuple[Tuple[int, str, Any], ...]
    input_sha256: str
    config: Dict[str, Any]
    config_sha256: str
    rdkit_version: str
    resume: bool

    @property
    def path(self) -> Path:
        return self.directory / "shards" / f"shard-{self.shard_index:06d}.npz"


class _GenerationFailure(ValueError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def _integer(value: Any, name: str, minimum: int, maximum: Optional[int] = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是整数")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if result < minimum or (maximum is not None and result > maximum):
        upper = "" if maximum is None else f" 且 <= {maximum}"
        raise ValueError(f"{name} 必须 >= {minimum}{upper}")
    return int(result)


def derive_rdkit_seed(seed: int, record_id: str) -> int:
    """以 blake2b(little-endian uint64 seed || UTF-8 id) 派生有符号正整数种子。"""
    seed = _integer(seed, "seed", 0, 2**64 - 1)
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("顶层 id 必须是非空字符串")
    payload = seed.to_bytes(8, "little") + record_id.encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def _resolve_source(source: SourceRef) -> AuxSource:
    if isinstance(source, str):
        if source not in GENERATED_SOURCES:
            raise ValueError(f"仅允许为 {GENERATED_SOURCES} 生成构象: {source}")
        return AUX_SOURCES[source]
    if source.name not in GENERATED_SOURCES:
        raise ValueError(f"仅允许为 {GENERATED_SOURCES} 生成构象: {source.name}")
    return source


def _output_directory(name: str, root: Optional[Union[str, Path]]) -> Path:
    if name not in GENERATED_SOURCES:
        raise ValueError(f"未知生成构象源 {name!r}；仅支持 {GENERATED_SOURCES}")
    base = CONFORMER_ROOT if root is None else Path(root)
    return base / name


def _rdkit_version() -> str:
    return str(require_rdkit().rdBase.rdkitVersion)


def _build_config(seed: int, shard_size: int) -> Dict[str, Any]:
    return {
        "format_version": 1,
        "seed": seed,
        "shard_size": shard_size,
        "method": METHOD,
        "embed": "ETKDGv3",
        "optimizer": "MMFFOptimizeMolecule",
        "mmff_variant": "MMFF94",
        "mmff_max_iterations": MMFF_MAX_ITERATIONS,
        "coordinate_dtype": COORDINATE_DTYPE.name,
        "seed_derivation": "blake2b-8(uint64-le seed || utf8 id), low 31 bits",
    }


def _config_digest(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_savez(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _update_input_hash(digest: Any, row: int, record_id: str, smiles: Any) -> None:
    encoded = json.dumps(
        [row, record_id, smiles], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _check_identity_graph(target: Any, source: Any) -> None:
    if (source.GetNumAtoms(), source.GetNumBonds()) != (
        target.GetNumAtoms(), target.GetNumBonds()
    ):
        raise ValueError("去氢前后重原子数或键数不一致")
    for index in range(target.GetNumAtoms()):
        expected = target.GetAtomWithIdx(index)
        actual = source.GetAtomWithIdx(index)
        expected_signature = (
            expected.GetAtomicNum(), expected.GetIsotope(), expected.GetFormalCharge(),
            expected.GetIsAromatic(), expected.GetTotalNumHs(),
            expected.GetNumRadicalElectrons(), expected.GetChiralTag(),
        )
        actual_signature = (
            actual.GetAtomicNum(), actual.GetIsotope(), actual.GetFormalCharge(),
            actual.GetIsAromatic(), actual.GetTotalNumHs(),
            actual.GetNumRadicalElectrons(), actual.GetChiralTag(),
        )
        if actual_signature != expected_signature:
            raise ValueError(f"重原子 {index} 的属性或顺序与输入 SMILES 不一致")
    for bond in target.GetBonds():
        actual = source.GetBondBetweenAtoms(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        if actual is None or (
            actual.GetBondType(), actual.GetIsAromatic(), actual.GetStereo()
        ) != (bond.GetBondType(), bond.GetIsAromatic(), bond.GetStereo()):
            raise ValueError("化学键顺序或属性与输入 SMILES 不一致")


def _generate_coordinates(smiles: Any, seed: int, max_iterations: int) -> np.ndarray:
    chem = require_rdkit()
    from rdkit.Chem import AllChem

    if not isinstance(smiles, str) or not smiles:
        raise _GenerationFailure("smiles", "inputs.smiles 必须是非空字符串")
    try:
        molecule = chem.MolFromSmiles(smiles)
    except Exception as exc:
        raise _GenerationFailure("smiles", str(exc)) from exc
    if molecule is None:
        raise _GenerationFailure("smiles", "RDKit 无法解析 SMILES")
    try:
        molecule = chem.RemoveAllHs(molecule)
        if molecule.GetNumAtoms() < 1:
            raise ValueError("分子没有重原子")
        target = chem.Mol(molecule)
        molecule = chem.AddHs(molecule)
    except Exception as exc:
        raise _GenerationFailure("add_hs", str(exc)) from exc

    try:
        parameters = AllChem.ETKDGv3()
        parameters.randomSeed = int(seed)
        parameters.numThreads = 1
        status = AllChem.EmbedMolecule(molecule, parameters)
    except Exception as exc:
        raise _GenerationFailure("embed", str(exc)) from exc
    if status != 0:
        raise _GenerationFailure("embed", f"ETKDGv3 EmbedMolecule 返回 {status}")

    try:
        properties = AllChem.MMFFGetMoleculeProperties(molecule)
    except Exception as exc:
        raise _GenerationFailure("mmff_support", str(exc)) from exc
    if properties is None:
        raise _GenerationFailure("mmff_support", "MMFF94 不支持该分子；禁止 UFF fallback")
    try:
        status = AllChem.MMFFOptimizeMolecule(
            molecule, mmffVariant="MMFF94", maxIters=int(max_iterations)
        )
    except Exception as exc:
        raise _GenerationFailure("mmff_optimize", str(exc)) from exc
    if status != 0:
        raise _GenerationFailure(
            "mmff_optimize", f"MMFFOptimizeMolecule 未收敛，返回 {status}"
        )

    try:
        heavy = chem.RemoveAllHs(molecule)
        if heavy.GetNumConformers() != 1:
            raise ValueError("优化后构象数不等于 1")
        _check_identity_graph(target, heavy)
        coordinates = np.asarray(heavy.GetConformer().GetPositions(), dtype=np.float64)
        if coordinates.shape != (target.GetNumAtoms(), 3):
            raise ValueError(f"坐标形状非法: {coordinates.shape}")
        if not np.isfinite(coordinates).all():
            raise ValueError("坐标含 NaN/Inf")
        coordinates = np.ascontiguousarray(coordinates, dtype=COORDINATE_DTYPE)
        if not np.isfinite(coordinates).all():
            raise ValueError("坐标转换为 float32 后含 NaN/Inf")
        return coordinates
    except Exception as exc:
        raise _GenerationFailure("validation", str(exc)) from exc


def _failure(record: Tuple[int, str, Any], seed: int, stage: str, error: Any) -> Dict[str, Any]:
    row, record_id, smiles = record
    return {
        "id": record_id,
        "aux_row": int(row),
        "smiles": smiles if isinstance(smiles, str) else None,
        "seed": int(seed),
        "stage": stage,
        "error": str(error),
    }


def _shard_info(job: _ShardJob, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "shard": int(job.shard_index),
        "path": f"shards/{job.path.name}",
        "aux_row_start": int(manifest["aux_row_start"]),
        "aux_row_end": int(manifest["aux_row_end"]),
        "attempted": int(manifest["attempted"]),
        "count": int(manifest["count"]),
        "failed": int(manifest["failed"]),
        "input_sha256": str(manifest["input_sha256"]),
        "sha256": _sha256_file(job.path),
    }


def _generate_shard(job: _ShardJob) -> Dict[str, Any]:
    ids: List[str] = []
    aux_rows: List[int] = []
    smiles_values: List[str] = []
    flat_coordinates: List[np.ndarray] = []
    offsets = [0]
    failures: List[Dict[str, Any]] = []
    for record in job.records:
        row, record_id, _ = record
        seed = derive_rdkit_seed(job.config["seed"], record_id)
        try:
            coordinates = _generate_coordinates(
                record[2], seed, job.config["mmff_max_iterations"]
            )
        except _GenerationFailure as exc:
            failures.append(_failure(record, seed, exc.stage, exc))
            continue
        except Exception as exc:
            failures.append(_failure(record, seed, "unexpected", exc))
            continue
        ids.append(record_id)
        aux_rows.append(int(row))
        smiles_values.append(record[2])
        flat_coordinates.append(coordinates.reshape(-1))
        offsets.append(offsets[-1] + int(coordinates.size))

    coordinates = (
        np.concatenate(flat_coordinates).astype(COORDINATE_DTYPE, copy=False)
        if flat_coordinates else np.empty(0, dtype=COORDINATE_DTYPE)
    )
    manifest = {
        "schema": SHARD_SCHEMA,
        "done": True,
        "source": AUX_SOURCES[job.config.get("source", "")].name
        if job.config.get("source", "") in AUX_SOURCES else None,
        "shard": int(job.shard_index),
        "aux_row_start": int(job.records[0][0]),
        "aux_row_end": int(job.records[-1][0]) + 1,
        "attempted": len(job.records),
        "count": len(ids),
        "failed": len(failures),
        "input_sha256": job.input_sha256,
        "config_sha256": job.config_sha256,
        "rdkit_version": job.rdkit_version,
    }
    # source 由任务配置外的目录名恢复；自定义 AuxSource 与注册表同名也可复现。
    manifest["source"] = job.directory.name
    _atomic_savez(
        job.path,
        ids=np.asarray(ids, dtype=np.str_),
        aux_rows=np.asarray(aux_rows, dtype=np.int64),
        smiles=np.asarray(smiles_values, dtype=np.str_),
        offsets=np.asarray(offsets, dtype=np.int64),
        coords=coordinates,
        failures_json=np.asarray(json.dumps(failures, ensure_ascii=False)),
        manifest=np.asarray(json.dumps(manifest, sort_keys=True)),
    )
    return _shard_info(job, manifest)


def _read_shard(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest"].item()))
        failures = json.loads(str(archive["failures_json"].item()))
        arrays = {
            key: archive[key]
            for key in ("ids", "aux_rows", "smiles", "offsets", "coords")
        }
    return manifest, failures, arrays


def _validate_existing_shard(job: _ShardJob) -> Dict[str, Any]:
    manifest, failures, arrays = _read_shard(job.path)
    expected_fields = {
        "schema": SHARD_SCHEMA,
        "done": True,
        "source": job.directory.name,
        "shard": job.shard_index,
        "aux_row_start": job.records[0][0],
        "aux_row_end": job.records[-1][0] + 1,
        "attempted": len(job.records),
        "input_sha256": job.input_sha256,
        "config_sha256": job.config_sha256,
        "rdkit_version": job.rdkit_version,
    }
    for key, expected in expected_fields.items():
        if manifest.get(key) != expected:
            raise ValueError(f"{job.path}: resume 字段 {key} 不一致")
    by_row = {record[0]: record for record in job.records}
    failed_rows = [int(entry["aux_row"]) for entry in failures]
    if failed_rows != sorted(set(failed_rows)):
        raise ValueError(f"{job.path}: 失败审计行号重复或无序")
    for entry in failures:
        row = int(entry["aux_row"])
        if row not in by_row:
            raise ValueError(f"{job.path}: 失败审计含分片外行号 {row}")
        record = by_row[row]
        expected_smiles = record[2] if isinstance(record[2], str) else None
        if entry.get("id") != record[1] or entry.get("smiles") != expected_smiles:
            raise ValueError(f"{job.path}: 失败审计与 aux 输入不一致")
        if entry.get("seed") != derive_rdkit_seed(job.config["seed"], record[1]):
            raise ValueError(f"{job.path}: 失败审计种子不一致")
        if not entry.get("stage") or not entry.get("error"):
            raise ValueError(f"{job.path}: 失败审计缺少 stage/error")
    failed_set = set(failed_rows)
    expected_success = [record for record in job.records if record[0] not in failed_set]
    if manifest.get("count") != len(expected_success) or manifest.get("failed") != len(failures):
        raise ValueError(f"{job.path}: 成功/失败计数不一致")
    if arrays["aux_rows"].tolist() != [record[0] for record in expected_success]:
        raise ValueError(f"{job.path}: aux_rows 与 resume 输入不一致")
    if arrays["ids"].tolist() != [record[1] for record in expected_success]:
        raise ValueError(f"{job.path}: ids 与 resume 输入不一致")
    if arrays["smiles"].tolist() != [record[2] for record in expected_success]:
        raise ValueError(f"{job.path}: SMILES 与 resume 输入不一致")
    offsets = arrays["offsets"]
    coordinates = arrays["coords"]
    if offsets.dtype != np.dtype("int64") or offsets.shape != (len(expected_success) + 1,):
        raise ValueError(f"{job.path}: offsets 形状或 dtype 非法")
    if coordinates.dtype != COORDINATE_DTYPE or coordinates.ndim != 1:
        raise ValueError(f"{job.path}: coords 形状或 dtype 非法")
    if len(offsets) == 0 or offsets[0] != 0 or offsets[-1] != coordinates.size:
        raise ValueError(f"{job.path}: offsets 与 coords 长度不一致")
    if np.any(np.diff(offsets) <= 0) or np.any(np.diff(offsets) % 3 != 0):
        raise ValueError(f"{job.path}: offsets 非递增或原子坐标不是三维")
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{job.path}: coords 含 NaN/Inf")
    return _shard_info(job, manifest)


def _process_shard(job: _ShardJob) -> Dict[str, Any]:
    if job.path.exists():
        if not job.resume:
            raise FileExistsError(f"分片已存在，未启用 resume: {job.path}")
        return _validate_existing_shard(job)
    return _generate_shard(job)


def _iter_jobs(
    spec: AuxSource,
    directory: Path,
    config: Dict[str, Any],
    config_sha256: str,
    rdkit_version: str,
    resume: bool,
) -> Iterator[_ShardJob]:
    shard_size = config["shard_size"]
    records: List[Tuple[int, str, Any]] = []
    digest = hashlib.sha256()
    shard_index = 0
    for aux_row, record in enumerate(iter_source(spec)):
        record_id = record.id
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"aux_row={aux_row}: 顶层 id 必须是非空字符串")
        if record.payload.get("id") != record_id:
            raise ValueError(f"aux_row={aux_row}: NamespacedRecord id 与顶层 id 不一致")
        inputs = record.payload.get("inputs") or {}
        smiles = inputs.get("smiles") if isinstance(inputs, Mapping) else None
        item = (aux_row, record_id, smiles)
        _update_input_hash(digest, *item)
        records.append(item)
        if len(records) == shard_size:
            yield _ShardJob(
                directory, shard_index, tuple(records), digest.hexdigest(),
                config, config_sha256, rdkit_version, resume,
            )
            shard_index += 1
            records = []
            digest = hashlib.sha256()
    if records:
        yield _ShardJob(
            directory, shard_index, tuple(records), digest.hexdigest(),
            config, config_sha256, rdkit_version, resume,
        )


def _execute_jobs(jobs: Iterator[_ShardJob], workers: int) -> Iterator[Dict[str, Any]]:
    if workers == 1:
        for job in jobs:
            yield _process_shard(job)
        return
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        pending: Dict[concurrent.futures.Future, int] = {}
        for job in jobs:
            pending[executor.submit(_process_shard, job)] = job.shard_index
            if len(pending) >= workers * 2:
                done, _ = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    pending.pop(future)
                    yield future.result()
        while pending:
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                pending.pop(future)
                yield future.result()


def _prepare_run(
    directory: Path,
    name: str,
    config: Dict[str, Any],
    config_sha256: str,
    rdkit_version: str,
    resume: bool,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "shards").mkdir(exist_ok=True)
    run_path = directory / "config.json"
    if run_path.exists():
        if not resume:
            raise FileExistsError(f"{directory}: 已有构建状态；续跑请显式使用 --resume")
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        expected = {
            "schema": SCHEMA,
            "source": name,
            "config": config,
            "config_sha256": config_sha256,
            "rdkit_version": rdkit_version,
        }
        if previous != expected:
            raise ValueError(f"{run_path}: resume 的输入源、配置或 RDKit 版本不一致")
        return
    if any(directory.iterdir()) and any(path.name != "shards" for path in directory.iterdir()):
        raise FileExistsError(f"{directory}: 目录含无 config.json 的文件，拒绝覆盖")
    if (directory / "shards").exists() and any((directory / "shards").iterdir()):
        raise FileExistsError(f"{directory}: 存在无 config.json 的分片，拒绝覆盖")
    _atomic_json({
        "schema": SCHEMA,
        "source": name,
        "config": config,
        "config_sha256": config_sha256,
        "rdkit_version": rdkit_version,
    }, run_path)


def build_conformers(
    source: SourceRef,
    root: Optional[Union[str, Path]] = None,
    workers: int = 1,
    seed: int = 42,
    shard_size: int = 50_000,
    resume: bool = False,
) -> Path:
    """从已去泄漏 aux 构建生成构象库；返回最终 metadata.json 路径。

    aux_row 是 ``iter_source`` 过滤后的从 0 开始行号。已发布分片只在
    ``resume=True`` 且输入、配置和 RDKit 版本完全一致时复用。
    """
    spec = _resolve_source(source)
    workers = _integer(workers, "workers", 1)
    seed = _integer(seed, "seed", 0, 2**64 - 1)
    shard_size = _integer(shard_size, "shard_size", 1)
    config = _build_config(seed, shard_size)
    config_sha256 = _config_digest(config)
    version = _rdkit_version()
    directory = _output_directory(spec.name, root)
    _prepare_run(directory, spec.name, config, config_sha256, version, bool(resume))

    shard_info: Dict[int, Dict[str, Any]] = {}
    jobs = _iter_jobs(spec, directory, config, config_sha256, version, bool(resume))
    for info in _execute_jobs(jobs, workers):
        index = int(info["shard"])
        if index in shard_info:
            raise RuntimeError(f"重复完成分片 {index}")
        shard_info[index] = info
    ordered = [shard_info[index] for index in sorted(shard_info)]
    if [info["shard"] for info in ordered] != list(range(len(ordered))):
        raise RuntimeError("分片编号不连续，拒绝发布 master")
    total = sum(int(info["attempted"]) for info in ordered)
    expected_row = 0
    for info in ordered:
        if info["aux_row_start"] != expected_row:
            raise RuntimeError("分片 aux_row 不连续，拒绝发布 master")
        expected_row = info["aux_row_end"]
    if expected_row != total:
        raise RuntimeError("分片 aux_row 与输入记录数不一致")

    aux_rows = np.arange(total, dtype=np.int64)
    locations = np.full((total, 2), -1, dtype=np.int64)
    failures: List[Dict[str, Any]] = []
    input_digest = hashlib.sha256()
    for info in ordered:
        input_digest.update(bytes.fromhex(info["input_sha256"]))
        path = directory / info["path"]
        _, shard_failures, arrays = _read_shard(path)
        rows = arrays["aux_rows"]
        if len(rows) != info["count"]:
            raise ValueError(f"{path}: 坐标条数与分片清单不一致")
        locations[rows, 0] = int(info["shard"])
        locations[rows, 1] = np.arange(len(rows), dtype=np.int64)
        failures.extend(shard_failures)
    count = int(np.count_nonzero(locations[:, 0] >= 0))
    if count != sum(int(info["count"]) for info in ordered):
        raise RuntimeError("master 成功计数不一致")
    if len(failures) != total - count:
        raise RuntimeError("master 失败计数不一致")

    master_path = directory / "master.npz"
    _atomic_savez(master_path, aux_rows=aux_rows, locations=locations)
    metadata = {
        "schema": SCHEMA,
        "source": spec.name,
        "kind": "generated_rdkit_conformer_store",
        "generated": True,
        "official_index": False,
        "done": True,
        "config": config,
        "config_sha256": config_sha256,
        "rdkit_version": version,
        "aux_records": int(total),
        "count": int(count),
        "records": int(count),
        "failed": len(failures),
        "shards": ordered,
        "master": {
            "path": master_path.name,
            "sha256": _sha256_file(master_path),
            "aux_rows": "all filtered aux rows, 0..aux_records-1",
            "locations": "int64 [aux_records,2]; shard,item or -1,-1 for failure",
        },
        "hashes": {
            "algorithm": "sha256",
            "input": input_digest.hexdigest(),
            "master": _sha256_file(master_path),
        },
        "failures": failures,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    metadata_path = directory / "metadata.json"
    _atomic_json(metadata, metadata_path)
    return metadata_path


class GeneratedConformerStore:
    """生成构象的只读随机访问；``get`` 的入参始终是过滤后的 aux 行号。

    ``len(store) == store.count == metadata['count']`` 是成功构象数。失败行
    仍保留在 master 行号空间中，读取时抛 KeyError，并可在 ``meta['failures']``
    查看原因。默认 root 与官方 molecule3d 索引完全分离。
    """

    def __init__(self, name: str, root: Optional[Union[str, Path]] = None):
        self.name = name
        self.directory = _output_directory(name, root)
        metadata_path = self.directory / "metadata.json"
        master_path = self.directory / "master.npz"
        if not metadata_path.is_file() or not master_path.is_file():
            raise FileNotFoundError(
                f"{name}: 生成构象库尚未完整发布于 {self.directory}；这不是官方 3D 索引"
            )
        self.meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            self.meta.get("schema") != SCHEMA
            or self.meta.get("source") != name
            or self.meta.get("generated") is not True
            or self.meta.get("official_index") is not False
            or self.meta.get("done") is not True
        ):
            raise ValueError(f"{metadata_path}: schema/source/generated/done 状态非法")
        self.count = _integer(self.meta.get("count"), "metadata.count", 0)
        self.total = _integer(self.meta.get("aux_records"), "metadata.aux_records", 0)
        if self.meta.get("records") != self.count or self.meta.get("failed") != self.total - self.count:
            raise ValueError(f"{metadata_path}: count/records/failed 不一致")
        if len(self.meta.get("failures", [])) != self.total - self.count:
            raise ValueError(f"{metadata_path}: 失败清单长度不一致")
        with np.load(master_path, allow_pickle=False) as master:
            self.aux_rows = master["aux_rows"]
            self.locations = master["locations"]
        if self.aux_rows.shape != (self.total,) or self.locations.shape != (self.total, 2):
            raise ValueError(f"{master_path}: master 形状与 metadata 不一致")
        if self.aux_rows.dtype != np.dtype("int64") or self.locations.dtype != np.dtype("int64"):
            raise ValueError(f"{master_path}: master dtype 非法")
        if not np.array_equal(self.aux_rows, np.arange(self.total, dtype=np.int64)):
            raise ValueError(f"{master_path}: aux_rows 必须覆盖全部过滤后行号")
        if int(np.count_nonzero(self.locations[:, 0] >= 0)) != self.count:
            raise ValueError(f"{master_path}: locations 成功数与 metadata 不一致")
        self.shards = sorted(self.meta.get("shards", []), key=lambda item: item["shard"])
        if [item["shard"] for item in self.shards] != list(range(len(self.shards))):
            raise ValueError(f"{metadata_path}: 分片清单不连续")
        if sum(int(item["count"]) for item in self.shards) != self.count:
            raise ValueError(f"{metadata_path}: 分片计数与 metadata 不一致")
        self._cache: "OrderedDict[int, Dict[str, np.ndarray]]" = OrderedDict()
        self._cache_size = 2
        self._lock = threading.Lock()
        self.aux_rows.setflags(write=False)
        self.locations.setflags(write=False)

    def __len__(self) -> int:
        return self.count

    def _shard(self, index: int) -> Dict[str, np.ndarray]:
        with self._lock:
            cached = self._cache.get(index)
            if cached is not None:
                self._cache.move_to_end(index)
                return cached
        info = self.shards[index]
        relative = Path(info["path"])
        path = (self.directory / relative).resolve()
        if self.directory.resolve() not in path.parents:
            raise ValueError(f"分片路径越出 Store 根目录: {relative}")
        manifest, _, arrays = _read_shard(path)
        if manifest.get("source") != self.name or manifest.get("shard") != index:
            raise ValueError(f"{path}: 分片 source/index 不一致")
        if manifest.get("rdkit_version") != self.meta["rdkit_version"]:
            raise ValueError(f"{path}: 分片 RDKit 版本不一致")
        if len(arrays["aux_rows"]) != int(info["count"]):
            raise ValueError(f"{path}: 分片条数与 master 清单不一致")
        for array in arrays.values():
            array.setflags(write=False)
        with self._lock:
            self._cache[index] = arrays
            self._cache.move_to_end(index)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return arrays

    def get(self, aux_row: int) -> AlignedStructure:
        row = _integer(aux_row, "aux_row", 0)
        if row >= self.total:
            raise IndexError(aux_row)
        shard_index, item_index = (int(value) for value in self.locations[row])
        if shard_index < 0:
            raise KeyError(f"{self.name}: aux_row={row} 构象生成失败，请查看 failures 审计")
        if shard_index >= len(self.shards):
            raise ValueError(f"aux_row={row}: master 分片下标越界")
        arrays = self._shard(shard_index)
        rows = arrays["aux_rows"]
        offsets = arrays["offsets"]
        if item_index < 0 or item_index >= len(rows) or int(rows[item_index]) != row:
            raise ValueError(f"aux_row={row}: master location 与分片记录不一致")
        if item_index + 1 >= len(offsets):
            raise ValueError(f"aux_row={row}: offsets 越界")
        start, end = int(offsets[item_index]), int(offsets[item_index + 1])
        if start < 0 or end <= start or end > arrays["coords"].size or (end - start) % 3:
            raise ValueError(f"aux_row={row}: 坐标 offset 非法")
        coordinates = arrays["coords"][start:end].reshape(-1, 3).astype(np.float64)
        if not np.isfinite(coordinates).all():
            raise ValueError(f"aux_row={row}: 坐标含 NaN/Inf")
        record_id = str(arrays["ids"][item_index])
        smiles = str(arrays["smiles"][item_index])
        atom_count = int(coordinates.shape[0])
        seed = derive_rdkit_seed(self.meta["config"]["seed"], record_id)
        provenance = {
            "generated": True,
            "method": self.meta["config"]["method"],
            "seed": seed,
            "version": self.meta["rdkit_version"],
            "rdkit_version": self.meta["rdkit_version"],
            "base_seed": int(self.meta["config"]["seed"]),
            "source": self.name,
            "aux_row": row,
            "shard": shard_index,
            "format": "generated_npz",
            "n_atoms": atom_count,
            "heavy_atom_mapping": list(range(atom_count)),
        }
        return AlignedStructure(
            source=self.name,
            id=record_id,
            smiles=smiles,
            coords=coordinates,
            mapping=tuple(range(atom_count)),
            provenance=provenance,
        )

    def close(self) -> None:
        with self._lock:
            self._cache.clear()

    def __enter__(self) -> "GeneratedConformerStore":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="可复现 RDKit 3D 构象离线生成")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="分片构建；metadata.json 原子发布后才可读")
    build.add_argument("name", choices=GENERATED_SOURCES)
    build.add_argument("--workers", type=int, default=1)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--shard-size", type=int, default=50_000)
    build.add_argument("--resume", action="store_true")
    build.add_argument("--root", type=Path, default=None, help="输出父目录（默认 data/pretrain）")
    args = parser.parse_args(argv)
    if args.command == "build":
        metadata_path = build_conformers(
            args.name,
            root=args.root,
            workers=args.workers,
            seed=args.seed,
            shard_size=args.shard_size,
            resume=args.resume,
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        print(
            f"{args.name}: done aux={metadata['aux_records']} "
            f"generated={metadata['count']} failed={metadata['failed']} "
            f"metadata={metadata_path}",
            flush=True,
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFORMER_ROOT",
    "GeneratedConformerStore",
    "GENERATED_SOURCES",
    "METHOD",
    "build_conformers",
    "derive_rdkit_seed",
    "main",
]
