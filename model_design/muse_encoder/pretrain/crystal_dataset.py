"""OMC25 选中帧训练数据读取器（抽样 JSONL + 正式 SQLite index → 晶体训练样本）。

数据流：

1. ``omc25.sample_frames`` 产出抽样 JSONL（每行含 id/starting_heap/shard_id/frame_row）。
2. ``build_frame_cache`` 把抽样帧按 shard 归组，借助正式索引（energy_records +
   aux_shards/aux_frames）定位 aux tar 成员，用 ``os.pread`` 按 offsets 直接抽取
   选中帧的 forces/stress，落盘为可断点续跑的分片帧缓存。每帧在 frames.jsonl
   中记录 energy 值、forces/stress 形状以及 blake2b digest（能量+应力+力原字节）。
   每个 tar 只打开一次，绝不因随机样本顺序反复全扫 tar，也绝不触碰未选中帧区间。
3. ``OMC25CrystalDataset`` 运行时只读缓存（mmap npy）与 ASE-LMDB 结构：
   LMDB member 已解包则直接使用（不重复解包、不再扫描结构 tar），缺失时才调用
   ``omc25.extract_lmdb_member`` 原子解包。结构经
   ``RecordFeaturizer.featurize_crystal`` 特征化，返回
   ``(FeaturizedRecord, item)``；item 携带按给定均值/std 做 z-score 的
   energy/force/stress 及 mask，与 ``batching.collate_metadata`` 完全兼容。
4. ``compute_frame_stats`` 只在缓存（即排除后的允许训练帧）上流式计算
   energy/force/stress 统计，供生成 ``FrameTargetStats`` 归一化参数。

严格性约定：

- 排除清单（suggested_exclude_heaps 或显式 heap/id）只在构建缓存时生效；
  数据集加载时以 exclusion digest 握手，排除参数与缓存不一致即拒绝，
  不允许静默使用未排除的缓存。
- 默认要求每个排除条目至少命中一个抽样帧（防止拼写错误被静默忽略）；
  若上游抽样已用 prior_exclusions 排除过同一清单，须显式
  ``allow_unmatched_exclusions=True`` 才允许零命中。
- 默认 ``strict_targets=True``：能量/力/应力出现非有限值直接报错，不静默 mask。
- 默认 ``verify_digests=True``：每次取帧校验 digest，缓存损坏立即报错。
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import sqlite3
import struct
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import quote

import numpy as np

from ..config import MuseConfig
from ..featurize import FeaturizedRecord, RecordFeaturizer
from .crystal_references import RESIDUAL_MODE, ElementReferences
from .omc25 import (
    SCHEMA_VERSION as INDEX_SCHEMA_VERSION,
    OnlineStats,
    extract_lmdb_member,
    read_ase_lmdb_row,
)


CACHE_SCHEMA_VERSION = "omc25.frame_cache.v1"
MASTER_SCHEMA_VERSION = "omc25.frame_cache.master.v1"
STATS_SCHEMA_VERSION = "omc25.frame_stats.v1"

_DIGEST_SIZE = 16
_MAX_RUN_ATOMS = 4_000_000


# ---------------------------------------------------------------------------
# 小工具（与 omc25.py 内部例程保持同一套严格校验语义）


def _new_temp_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    os.unlink(name)
    return Path(name)


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = _new_temp_path(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _validate_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if (not name or "\x00" in name or "\\" in name or path.is_absolute()
            or any(part in ("", ".", "..") for part in path.parts)
            or path.as_posix() != name):
        raise ValueError(f"不安全或不规范的 tar member: {name!r}")


def _pread_exact(fd: int, offset: int, size: int, name: str) -> bytes:
    if size < 0 or offset < 0:
        raise ValueError(f"{name}: 非法 pread 区间")
    chunks = []
    remaining = size
    while remaining:
        block = os.pread(fd, remaining, offset)
        if not block:
            raise ValueError(f"{name}: pread 截断")
        chunks.append(block)
        offset += len(block)
        remaining -= len(block)
    return b"".join(chunks)


def _npy_header_at(fd: int, base: int, name: str, member_size: int) -> tuple[tuple[int, ...], np.dtype, int]:
    """从 tar 原始字节的 member 起点解析 NPY header；校验语义同 omc25._npy_header。"""
    if _pread_exact(fd, base, 6, name) != b"\x93NUMPY":
        raise ValueError(f"{name}: 不是 NPY 文件")
    version = _pread_exact(fd, base + 6, 2, name)
    major = version[0]
    if major == 1:
        length_format, length_size = "<H", 2
    elif major in (2, 3):
        length_format, length_size = "<I", 4
    else:
        raise ValueError(f"{name}: 不支持的 NPY version {major}.{version[1]}")
    header_length = struct.unpack(length_format, _pread_exact(fd, base + 8, length_size, name))[0]
    if header_length > 1_048_576:
        raise ValueError(f"{name}: NPY header 异常大")
    raw = _pread_exact(fd, base + 8 + length_size, header_length, name)
    encoding = "utf-8" if major == 3 else "latin1"
    try:
        values = ast.literal_eval(raw.decode(encoding).strip())
    except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"{name}: 非法 NPY header") from exc
    if not isinstance(values, dict):
        raise ValueError(f"{name}: 非法 NPY header")
    try:
        shape = tuple(int(value) for value in values["shape"])
        dtype = np.dtype(values["descr"])
        fortran = values["fortran_order"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{name}: 非法 NPY shape/dtype") from exc
    if (fortran is not False or any(value < 0 for value in shape)
            or dtype.hasobject or dtype.subdtype is not None):
        raise ValueError(f"{name}: 仅支持 C-order、非 object NPY")
    data_offset = base + 8 + length_size + header_length
    data_bytes = math.prod(shape) * dtype.itemsize
    if data_offset + data_bytes != base + member_size:
        raise ValueError(f"{name}: NPY 数据大小与 member 不符")
    return shape, dtype, data_offset


def _tar_member(tf: tarfile.TarFile, name: str) -> tarfile.TarInfo:
    _validate_member_name(name)
    try:
        member = tf.getmember(name)
    except KeyError as exc:
        raise KeyError(f"{tf.name}: 找不到 member {name}") from exc
    if not member.isfile() or member.linkname:
        raise ValueError(f"{name}: 必须是普通文件")
    return member


def _target_digest(energy: float, stress: np.ndarray, forces: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    digest.update(struct.pack("<d", float(energy)))
    digest.update(np.ascontiguousarray(stress).tobytes())
    digest.update(np.ascontiguousarray(forces).tobytes())
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_int(value: Any, name: str, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where}: {name} 必须是非负整数")
    return value


# ---------------------------------------------------------------------------
# 排除清单


@dataclass(frozen=True)
class ExclusionSpec:
    """按 starting_heap 或记录 id 排除抽样帧；digest 用于缓存握手。"""

    heaps: frozenset[str]
    ids: frozenset[str]

    @property
    def empty(self) -> bool:
        return not self.heaps and not self.ids

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {"heaps": sorted(self.heaps), "ids": sorted(self.ids)},
            ensure_ascii=False, separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def matches(self, record_id: str, starting_heap: str) -> bool:
        return record_id in self.ids or starting_heap in self.heaps


EMPTY_EXCLUSIONS = ExclusionSpec(frozenset(), frozenset())


def load_exclusion_spec(
    suggested_exclude_heaps: Path | str | None = None,
    exclude_heaps: Sequence[str] = (),
    exclude_ids: Sequence[str] = (),
) -> ExclusionSpec:
    """合并 suggested_exclude_heaps JSONL 与显式 heap/id 清单，全部严格校验。"""
    heaps: set[str] = set()
    ids: set[str] = set()

    def add(target: set[str], value: Any, kind: str, where: str) -> None:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{where}: 非法 {kind}: {value!r}")
        target.add(value)

    for value in exclude_heaps:
        add(heaps, value, "starting_heap", "exclude_heaps")
    for value in exclude_ids:
        add(ids, value, "id", "exclude_ids")
    if suggested_exclude_heaps is not None:
        path = Path(suggested_exclude_heaps)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: 非法 JSON") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: 排除行必须是对象")
                where = f"{path}:{line_number}"
                if record.get("starting_heap") is None and record.get("id") is None:
                    raise ValueError(f"{where}: 排除行缺少 starting_heap/id")
                if record.get("starting_heap") is not None:
                    add(heaps, record["starting_heap"], "starting_heap", where)
                if record.get("id") is not None:
                    add(ids, record["id"], "id", where)
    return ExclusionSpec(frozenset(heaps), frozenset(ids))


# ---------------------------------------------------------------------------
# 归一化参数


@dataclass(frozen=True)
class FrameTargetStats:
    """energy/force/stress 的 z-score 参数；std 必须全部为有限正数。"""

    energy_mean: float
    energy_std: float
    force_mean: float
    force_std: float
    stress_mean: tuple[float, ...]
    stress_std: tuple[float, ...]
    energy_mode: str = "total"
    reference_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in ("energy_mean", "energy_std", "force_mean", "force_std"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} 必须为有限数值")
        for name in ("energy_std", "force_std"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为正数，禁止静默回退")
        for name in ("stress_mean", "stress_std"):
            values = tuple(float(v) for v in getattr(self, name))
            if len(values) < 1 or not all(math.isfinite(v) for v in values):
                raise ValueError(f"{name} 必须为非空有限序列")
            object.__setattr__(self, name, values)
        if any(v <= 0 for v in self.stress_std):
            raise ValueError("stress_std 必须全部分量为正，禁止静默回退")
        if len(self.stress_mean) != len(self.stress_std):
            raise ValueError("stress_mean/stress_std 长度不一致")
        if self.energy_mode not in ("total", RESIDUAL_MODE):
            raise ValueError(f"未知 energy_mode: {self.energy_mode!r}")
        if self.energy_mode == RESIDUAL_MODE and (
                not isinstance(self.reference_sha256, str) or len(self.reference_sha256) != 64):
            raise ValueError("残差能量统计必须绑定 reference_sha256")

    def stress_arrays(self, width: int) -> tuple[np.ndarray, np.ndarray]:
        if width < 1:
            raise ValueError("stress 宽度必须为正")
        if len(self.stress_mean) not in (1, width):
            raise ValueError(f"stress 统计维度 {len(self.stress_mean)} 与宽度 {width} 不符")
        mean = np.asarray(self.stress_mean, dtype=np.float32)
        std = np.asarray(self.stress_std, dtype=np.float32)
        if len(self.stress_mean) == 1 and width != 1:
            mean = np.full(width, mean[0], dtype=np.float32)
            std = np.full(width, std[0], dtype=np.float32)
        return mean, std

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FrameTargetStats":
        try:
            energy = payload["energy"]
            forces = payload["forces"]
            stress = payload["stress"]
        except KeyError as exc:
            raise ValueError(f"统计缺少字段: {exc}") from exc

        def scalar(section: Mapping[str, Any], key: str, name: str) -> float:
            value = section.get(key)
            if isinstance(value, Sequence) and not isinstance(value, str):
                if len(value) != 1:
                    raise ValueError(f"{name}.{key} 必须是标量")
                value = value[0]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name}.{key} 必须是数值")
            return float(value)

        def vector(section: Mapping[str, Any], key: str) -> tuple[float, ...]:
            value = section.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return (float(value),)
            if isinstance(value, Sequence) and not isinstance(value, str) and value:
                return tuple(float(v) for v in value)
            raise ValueError(f"stress.{key} 必须是数值或非空序列")

        return cls(
            energy_mean=scalar(energy, "mean", "energy"),
            energy_std=scalar(energy, "std", "energy"),
            force_mean=scalar(forces, "mean", "forces"),
            force_std=scalar(forces, "std", "forces"),
            stress_mean=vector(stress, "mean"),
            stress_std=vector(stress, "std"),
            energy_mode=str(energy.get("mode", "total")),
            reference_sha256=energy.get("reference_sha256"),
        )

    @classmethod
    def from_json(cls, path: Path | str) -> "FrameTargetStats":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# 抽样 JSONL


def _parse_sampled_frame(record: Any, where: str) -> tuple[str, str, int, int]:
    if not isinstance(record, dict):
        raise ValueError(f"{where}: 抽样行必须是对象")
    record_id = record.get("id")
    heap = record.get("starting_heap")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError(f"{where}: 缺少有效 id")
    if not isinstance(heap, str) or not heap:
        raise ValueError(f"{where}: 缺少有效 starting_heap")
    shard_id = _require_int(record.get("shard_id"), "shard_id", where)
    frame_row = _require_int(record.get("frame_row"), "frame_row", where)
    return record_id, heap, shard_id, frame_row


# ---------------------------------------------------------------------------
# LMDB 路径解析


def resolve_archive_path(external_root: Path | str, source_path: str) -> Path:
    """把 energy_records.source_path 约束解析到 external_root 内。"""
    root = Path(external_root).resolve()
    relative = Path(source_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"非法 source_path: {source_path}")
    archive = (root / relative).resolve()
    if root != archive and root not in archive.parents:
        raise ValueError(f"source_path 越过 external_root: {source_path}")
    return archive


def lmdb_cache_destination(lmdb_cache_root: Path | str, source_path: str, member_name: str) -> Path:
    """与 omc25.extract_lmdb_member 相同的目标路径公式。"""
    _validate_member_name(member_name)
    root = Path(lmdb_cache_root).resolve()
    destination = root / Path(source_path).stem / PurePosixPath(member_name)
    resolved = destination.resolve()
    if resolved != destination or root not in resolved.parents:
        raise ValueError("LMDB 目标路径逃逸 lmdb_cache_root")
    return destination


# ---------------------------------------------------------------------------
# 分片帧缓存构建


def _open_readonly_index(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def _index_schema_version(connection: sqlite3.Connection) -> str:
    row = connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
    if row is None:
        raise ValueError("索引缺少 metadata.schema_version")
    return json.loads(row[0])


def _load_manifest(cache_dir: Path, require_complete: bool = True) -> dict[str, Any]:
    path = cache_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"帧缓存缺少 manifest.json: {cache_dir}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version 不符")
    if require_complete and not manifest.get("complete"):
        raise ValueError(f"{cache_dir}: 帧缓存未构建完成（manifest.complete != true）")
    return manifest


def _write_manifest(cache_dir: Path, manifest: Mapping[str, Any]) -> None:
    _write_json_atomic(cache_dir / "manifest.json", manifest)


def _stage_selection(
    sample_jsonl: Path,
    work: sqlite3.Connection,
    exclusions: ExclusionSpec,
    allow_unmatched_exclusions: bool,
) -> dict[str, Any]:
    """单遍流式读抽样 JSONL：哈希原始字节、应用排除、写入 staging 表。"""
    digest = hashlib.sha256()
    matched_heaps: set[str] = set()
    matched_ids: set[str] = set()
    total = kept = 0
    batch: list[tuple[Any, ...]] = []
    with sample_jsonl.open("rb") as handle:
        for line_number, raw in enumerate(handle):
            digest.update(raw)
            if not raw.strip():
                raise ValueError(f"{sample_jsonl}:{line_number + 1}: 空行")
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{sample_jsonl}:{line_number + 1}: 非法 JSON") from exc
            record_id, heap, shard_id, frame_row = _parse_sampled_frame(
                record, f"{sample_jsonl}:{line_number + 1}"
            )
            total += 1
            if exclusions.matches(record_id, heap):
                if heap in exclusions.heaps:
                    matched_heaps.add(heap)
                if record_id in exclusions.ids:
                    matched_ids.add(record_id)
                continue
            batch.append((record_id, line_number, heap, shard_id, frame_row))
            kept += 1
            if len(batch) >= 10_000:
                work.executemany("INSERT INTO selections VALUES (?,?,?,?,?)", batch)
                batch.clear()
    if batch:
        work.executemany("INSERT INTO selections VALUES (?,?,?,?,?)", batch)
        batch.clear()
    unmatched_heaps = sorted(exclusions.heaps - matched_heaps)
    unmatched_ids = sorted(exclusions.ids - matched_ids)
    if (unmatched_heaps or unmatched_ids) and not allow_unmatched_exclusions:
        preview = (unmatched_heaps + unmatched_ids)[:10]
        raise ValueError(
            f"排除条目未命中任何抽样帧（可能拼写错误或已被上游排除）: {preview}；"
            "确认上游已排除时显式使用 allow_unmatched_exclusions=True"
        )
    return {
        "sample_sha256": digest.hexdigest(),
        "total_rows": total,
        "kept_rows": kept,
        "excluded_rows": total - kept,
        "matched_heaps": len(matched_heaps),
        "matched_ids": len(matched_ids),
        "unmatched_heaps": unmatched_heaps,
        "unmatched_ids": unmatched_ids,
    }


def _shard_selection_rows(
    work: sqlite3.Connection, shard_id: int
) -> tuple[list[tuple[Any, ...]], str]:
    """选中帧 JOIN 正式索引；校验 id/heap/原子数一致，返回行与分片选择 digest。"""
    rows = work.execute(
        """SELECT s.line_no, s.id, s.frame_row, s.starting_heap, a.id, a.atom_start, a.atom_end,
                  e.energy, e.atom_count, e.starting_heap, e.sid,
                  e.source_path, e.archive_member, e.lmdb_row_id
           FROM selections s
           JOIN idx.aux_frames a ON a.shard_id=s.shard_id AND a.frame_row=s.frame_row
           JOIN idx.energy_records e ON e.id=s.id
           WHERE s.shard_id=? ORDER BY s.frame_row""",
        (shard_id,),
    ).fetchall()
    expected = work.execute(
        "SELECT count(*) FROM selections WHERE shard_id=?", (shard_id,)
    ).fetchone()[0]
    if len(rows) != expected:
        missing = work.execute(
            """SELECT s.frame_row, s.id FROM selections s
               LEFT JOIN idx.aux_frames a ON a.shard_id=s.shard_id AND a.frame_row=s.frame_row
               WHERE s.shard_id=? AND a.id IS NULL LIMIT 10""",
            (shard_id,),
        ).fetchall()
        raise ValueError(f"shard {shard_id}: {expected - len(rows)} 个抽样帧不在索引中: {missing}")
    digest = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    for line_no, select_id, frame_row, select_heap, aux_id, atom_start, atom_end, energy, atom_count, heap, sid, *_ in rows:
        if aux_id != select_id:
            raise ValueError(
                f"shard {shard_id} frame {frame_row}: 抽样 id {select_id!r} != 索引 id {aux_id!r}"
            )
        if select_heap != heap:
            raise ValueError(f"{select_id}: 抽样 starting_heap 与索引不一致，抽样文件可能已过期")
        if atom_end - atom_start != atom_count:
            raise ValueError(f"{select_id}: aux 原子区间与 energy_records.atom_count 不一致")
        digest.update(f"{select_id},{frame_row}\n".encode("utf-8"))
    return rows, digest.hexdigest()


def _read_offsets(fd: int, member: tarfile.TarInfo, entry: Mapping[str, Any]) -> tuple[np.ndarray, int]:
    shape, dtype, data_offset = _npy_header_at(fd, member.offset_data, member.name, member.size)
    if len(shape) != 1 or dtype.kind not in "iu":
        raise ValueError(f"{member.name}: offsets 必须为整数一维数组")
    frame_count = int(entry["frame_count"])
    if shape != (frame_count + 1,):
        raise ValueError(f"{member.name}: offsets 长度 {shape} != frame_count+1 {frame_count + 1}")
    if dtype.str != entry["offsets_dtype"]:
        raise ValueError(f"{member.name}: offsets dtype {dtype.str} != 索引 {entry['offsets_dtype']}")
    raw = _pread_exact(fd, data_offset, shape[0] * dtype.itemsize, member.name)
    offsets = np.frombuffer(raw, dtype=dtype)
    if int(offsets[0]) != 0 or (offsets[1:] <= offsets[:-1]).any():
        raise ValueError(f"{member.name}: offsets 必须从 0 开始且严格递增")
    if int(offsets[-1]) != int(entry["atom_count"]):
        raise ValueError(f"{member.name}: offsets[-1] 与索引 atom_count 不符")
    return offsets.astype(np.int64), data_offset


def _gather_forces(
    fd: int, data_offset: int, dtype: np.dtype, offsets: np.ndarray,
    frame_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """按 frame_rows 的原子区间抽取 forces；相邻帧合并为连续 run 后 pread。"""
    starts = offsets[frame_rows]
    ends = offsets[frame_rows + 1]
    cache_offsets = np.zeros(len(frame_rows) + 1, dtype=np.int64)
    np.cumsum(ends - starts, out=cache_offsets[1:])
    total_atoms = int(cache_offsets[-1])
    forces = np.empty((total_atoms, 3), dtype=dtype)
    row_bytes = 3 * dtype.itemsize
    run_start = 0
    index = 1
    while index <= len(frame_rows):
        contiguous = index < len(frame_rows) and starts[index] == ends[index - 1]
        under_cap = cache_offsets[index] - cache_offsets[run_start] < _MAX_RUN_ATOMS
        if index < len(frame_rows) and contiguous and under_cap:
            index += 1
            continue
        first, last = run_start, index
        byte_start = data_offset + int(starts[first]) * row_bytes
        byte_count = (int(ends[last - 1]) - int(starts[first])) * row_bytes
        raw = _pread_exact(fd, byte_start, byte_count, "forces.npy")
        block = np.frombuffer(raw, dtype=dtype).reshape(-1, 3)
        forces[cache_offsets[first]:cache_offsets[last]] = block
        run_start = index
        index += 1
    return forces, cache_offsets


def _gather_stress(
    fd: int, data_offset: int, dtype: np.dtype, width: int,
    frame_rows: np.ndarray,
) -> np.ndarray:
    row_bytes = width * dtype.itemsize
    stress = np.empty((len(frame_rows), width), dtype=dtype)
    run_start = 0
    index = 1
    while index <= len(frame_rows):
        if index < len(frame_rows) and int(frame_rows[index]) == int(frame_rows[index - 1]) + 1:
            index += 1
            continue
        first, last = run_start, index
        byte_start = data_offset + int(frame_rows[first]) * row_bytes
        byte_count = (int(frame_rows[last - 1]) - int(frame_rows[first]) + 1) * row_bytes
        raw = _pread_exact(fd, byte_start, byte_count, "stress.npy")
        stress[first:last] = np.frombuffer(raw, dtype=dtype).reshape(-1, width)
        run_start = index
        index += 1
    return stress


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    temporary = _new_temp_path(path)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _materialize_shard(
    cache_dir: Path,
    shard_id: int,
    shard_entry: Mapping[str, Any],
    rows: list[tuple[Any, ...]],
    selection_digest: str,
    tf: tarfile.TarFile,
    fd: int,
) -> dict[str, Any]:
    """抽取一个 shard 的选中帧，写 forces.npy/stress.npy/frames.jsonl（原子落盘）。"""
    forces_member = _tar_member(tf, str(shard_entry["forces_member"]))
    offsets_member = _tar_member(tf, str(shard_entry["offsets_member"]))
    stress_member = _tar_member(tf, str(shard_entry["stress_member"]))
    offsets, _ = _read_offsets(fd, offsets_member, shard_entry)

    forces_shape, forces_dtype, forces_offset = _npy_header_at(
        fd, forces_member.offset_data, forces_member.name, forces_member.size
    )
    if len(forces_shape) != 2 or forces_shape[1] != 3 or forces_dtype.kind != "f":
        raise ValueError(f"{forces_member.name}: forces 必须为浮点 [n_atoms,3]")
    if forces_shape[0] != int(shard_entry["atom_count"]):
        raise ValueError(f"{forces_member.name}: 原子数与索引不符")
    if forces_dtype.str != shard_entry["forces_dtype"]:
        raise ValueError(f"{forces_member.name}: dtype 与索引不符")
    stress_shape, stress_dtype, stress_offset = _npy_header_at(
        fd, stress_member.offset_data, stress_member.name, stress_member.size
    )
    if stress_shape != (int(shard_entry["frame_count"]), int(shard_entry["stress_width"])):
        raise ValueError(f"{stress_member.name}: stress 形状与索引不符")
    if stress_dtype.str != shard_entry["stress_dtype"]:
        raise ValueError(f"{stress_member.name}: dtype 与索引不符")

    frame_rows = np.asarray([int(row[2]) for row in rows], dtype=np.int64)
    forces, cache_offsets = _gather_forces(fd, forces_offset, forces_dtype, offsets, frame_rows)
    stress = _gather_stress(fd, stress_offset, stress_dtype, stress_shape[1], frame_rows)

    shard_dir = cache_dir / f"shard_{shard_id:06d}"
    frames_path = shard_dir / "frames.jsonl"
    lines = []
    for cache_row, row in enumerate(rows):
        (line_no, record_id, frame_row, _select_heap, _aux_id, atom_start, atom_end,
         energy, atom_count, heap, sid, source_path, archive_member, lmdb_row_id) = row
        if int(offsets[frame_row]) != atom_start or int(offsets[frame_row + 1]) != atom_end:
            raise ValueError(f"shard {shard_id} frame {frame_row}: 索引与 offsets 不一致")
        frame_forces = forces[cache_offsets[cache_row]:cache_offsets[cache_row + 1]]
        frame_stress = stress[cache_row]
        lines.append(json.dumps({
            "cache_row": cache_row,
            "line_no": line_no,
            "id": record_id,
            "starting_heap": heap,
            "frame_row": frame_row,
            "atom_start": int(cache_offsets[cache_row]),
            "atom_end": int(cache_offsets[cache_row + 1]),
            "atom_count": atom_count,
            "energy": energy,
            "stress_row": cache_row,
            "force_shape": [atom_end - atom_start, 3],
            "stress_shape": [stress_shape[1]],
            "target_digest": _target_digest(energy, frame_stress, frame_forces),
            "sid": sid,
            "source_path": source_path,
            "archive_member": archive_member,
            "lmdb_row_id": lmdb_row_id,
        }, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    _save_npy_atomic(shard_dir / "forces.npy", forces)
    _save_npy_atomic(shard_dir / "stress.npy", stress)
    _write_text_atomic(frames_path, "".join(line + "\n" for line in lines))
    return {
        "shard_id": shard_id,
        "member_prefix": shard_entry["member_prefix"],
        "tar_path": shard_entry["tar_path"],
        "frame_count": len(rows),
        "atom_count": int(cache_offsets[-1]),
        "stress_width": int(shard_entry["stress_width"]),
        "forces_dtype": str(shard_entry["forces_dtype"]),
        "stress_dtype": str(shard_entry["stress_dtype"]),
        "selection_digest": selection_digest,
        "forces_file": f"shard_{shard_id:06d}/forces.npy",
        "stress_file": f"shard_{shard_id:06d}/stress.npy",
        "frames_file": f"shard_{shard_id:06d}/frames.jsonl",
    }


def _resumable_entry(cache_dir: Path, entry: Mapping[str, Any], selection_digest: str) -> bool:
    if entry.get("selection_digest") != selection_digest:
        return False
    for key in ("forces_file", "stress_file", "frames_file"):
        if not (cache_dir / str(entry[key])).is_file():
            return False
    frames_lines = 0
    with (cache_dir / str(entry["frames_file"])).open("rb") as handle:
        for _ in handle:
            frames_lines += 1
    if frames_lines != int(entry["frame_count"]):
        return False
    try:
        forces = np.load(cache_dir / str(entry["forces_file"]), mmap_mode="r", allow_pickle=False)
        stress = np.load(cache_dir / str(entry["stress_file"]), mmap_mode="r", allow_pickle=False)
    except Exception:
        return False
    return (
        forces.shape == (int(entry["atom_count"]), 3)
        and forces.dtype.str == entry["forces_dtype"]
        and stress.shape == (int(entry["frame_count"]), int(entry["stress_width"]))
        and stress.dtype.str == entry["stress_dtype"]
    )


def build_frame_cache(
    sample_jsonl: Path | str,
    index_db: Path | str,
    cache_dir: Path | str,
    exclusions: ExclusionSpec | None = None,
    allow_unmatched_exclusions: bool = False,
) -> dict[str, Any]:
    """按抽样 JSONL 构建可断点续跑的分片帧缓存；只读取选中帧区间。"""
    started = time.monotonic()
    exclusions = exclusions or EMPTY_EXCLUSIONS
    sample_jsonl = Path(sample_jsonl)
    index_db = Path(index_db)
    cache_dir = Path(cache_dir)
    if not sample_jsonl.is_file():
        raise FileNotFoundError(f"抽样 JSONL 不存在: {sample_jsonl}")
    if not index_db.is_file():
        raise FileNotFoundError(f"索引不存在: {index_db}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    sample_sha256 = _sha256_path(sample_jsonl)

    existing: dict[str, Any] | None = None
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError(f"{manifest_path}: schema_version 不符，拒绝复用")
        if existing.get("sample_sha256") != sample_sha256:
            raise ValueError("抽样 JSONL 内容与已有缓存不符，拒绝续跑；请使用新 cache_dir")
        if existing.get("exclusion_digest") != exclusions.digest:
            raise ValueError("排除清单与已有缓存不符，拒绝续跑；请使用新 cache_dir")
        files_ok = (cache_dir / "master.npz").is_file() and (cache_dir / "master.json").is_file()
        for shard_entry in existing.get("shards", {}).values():
            for key in ("forces_file", "stress_file", "frames_file"):
                if not (cache_dir / str(shard_entry[key])).is_file():
                    files_ok = False
        if existing.get("complete") and files_ok:
            master = np.load(cache_dir / "master.npz", allow_pickle=False)
            frames = int(master["shard_id"].shape[0])
            return {
                "schema_version": CACHE_SCHEMA_VERSION,
                "cache_dir": str(cache_dir),
                "sample_sha256": sample_sha256,
                "exclusion_digest": exclusions.digest,
                "shards_built": 0,
                "shards_resumed": len(existing["shards"]),
                "frames": frames,
                "atoms": int(existing["selected_atoms"]),
                "resumed_only": True,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }

    index = _open_readonly_index(index_db)
    try:
        version = _index_schema_version(index)
        if version != INDEX_SCHEMA_VERSION:
            raise ValueError(f"索引 schema_version {version} != {INDEX_SCHEMA_VERSION}")
    finally:
        index.close()

    work_path = _new_temp_path(cache_dir / "staging.sqlite3")
    work = sqlite3.connect(str(work_path), uri=True)
    work.execute("PRAGMA journal_mode=OFF")
    work.execute("PRAGMA synchronous=OFF")
    work.execute("PRAGMA temp_store=FILE")
    work.execute(
        """CREATE TABLE selections(
               id TEXT PRIMARY KEY, line_no INTEGER NOT NULL UNIQUE,
               starting_heap TEXT NOT NULL, shard_id INTEGER NOT NULL,
               frame_row INTEGER NOT NULL
           ) WITHOUT ROWID"""
    )
    work.execute(
        """CREATE TABLE master(
               line_no INTEGER PRIMARY KEY, shard_id INTEGER NOT NULL,
               cache_row INTEGER NOT NULL, frame_row INTEGER NOT NULL
           ) WITHOUT ROWID"""
    )
    built = resumed = 0
    selected_atoms = 0
    try:
        staging = _stage_selection(sample_jsonl, work, exclusions, allow_unmatched_exclusions)
        if staging["kept_rows"] < 1:
            raise ValueError("排除后没有任何可训练帧")
        work.execute("CREATE INDEX selections_shard ON selections(shard_id, frame_row)")
        work.execute(f"ATTACH DATABASE 'file:{quote(str(index_db.resolve()))}?mode=ro' AS idx")
        shard_ids = [row[0] for row in work.execute(
            "SELECT DISTINCT shard_id FROM selections ORDER BY shard_id"
        )]
        shard_entries = {}
        for shard_id in shard_ids:
            row = work.execute(
                """SELECT tar_path,member_prefix,forces_member,offsets_member,stress_member,
                          frame_count,atom_count,stress_width,forces_dtype,offsets_dtype,stress_dtype
                   FROM idx.aux_shards WHERE shard_id=?""",
                (shard_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"索引中没有 aux shard_id={shard_id}")
            shard_entries[shard_id] = {
                "tar_path": row[0], "member_prefix": row[1], "forces_member": row[2],
                "offsets_member": row[3], "stress_member": row[4], "frame_count": row[5],
                "atom_count": row[6], "stress_width": row[7], "forces_dtype": row[8],
                "offsets_dtype": row[9], "stress_dtype": row[10],
            }
        by_tar: dict[str, list[int]] = {}
        for shard_id in shard_ids:
            by_tar.setdefault(shard_entries[shard_id]["tar_path"], []).append(shard_id)

        manifest: dict[str, Any] = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "complete": False,
            "sample_jsonl": str(sample_jsonl.resolve()),
            "sample_sha256": sample_sha256,
            "index_db": str(index_db.resolve()),
            "exclusion_digest": exclusions.digest,
            "exclusions": {
                "heaps": len(exclusions.heaps),
                "ids": len(exclusions.ids),
                "matched_heaps": staging["matched_heaps"],
                "matched_ids": staging["matched_ids"],
                "unmatched_heaps": staging["unmatched_heaps"],
                "unmatched_ids": staging["unmatched_ids"],
                "allow_unmatched": bool(allow_unmatched_exclusions),
            },
            "sample_rows": staging["total_rows"],
            "excluded_rows": staging["excluded_rows"],
            "selected_frames": staging["kept_rows"],
            "selected_atoms": 0,
            "selection_digest": None,
            "shards": dict((existing or {}).get("shards", {})),
        }
        _write_manifest(cache_dir, manifest)

        for tar_path in sorted(by_tar):
            members = sorted(by_tar[tar_path])
            with tarfile.open(tar_path, "r:") as tf:
                fd = os.open(tar_path, os.O_RDONLY)
                try:
                    for shard_id in members:
                        rows, selection_digest = _shard_selection_rows(work, shard_id)
                        shard_entry = shard_entries[shard_id]
                        previous = manifest["shards"].get(str(shard_id))
                        if previous is not None and _resumable_entry(
                            cache_dir, previous, selection_digest
                        ):
                            resumed += 1
                            selected_atoms += int(previous["atom_count"])
                            master_rows = work.execute(
                                """SELECT line_no, frame_row FROM selections
                                   WHERE shard_id=? ORDER BY frame_row""",
                                (shard_id,),
                            ).fetchall()
                            work.executemany(
                                "INSERT INTO master VALUES (?,?,?,?)",
                                [(line_no, shard_id, cache_row, frame_row)
                                 for cache_row, (line_no, frame_row) in enumerate(master_rows)],
                            )
                            continue
                        entry = _materialize_shard(
                            cache_dir, shard_id, shard_entry, rows, selection_digest, tf, fd
                        )
                        work.executemany(
                            "INSERT INTO master VALUES (?,?,?,?)",
                            [(int(row[0]), shard_id, cache_row, int(row[2]))
                             for cache_row, row in enumerate(rows)],
                        )
                        manifest["shards"][str(shard_id)] = entry
                        _write_manifest(cache_dir, manifest)
                        built += 1
                        selected_atoms += entry["atom_count"]
                finally:
                    os.close(fd)

        frame_ids = np.zeros(staging["kept_rows"], dtype=np.int32)
        frame_cache_rows = np.zeros(staging["kept_rows"], dtype=np.int32)
        frame_rows_master = np.zeros(staging["kept_rows"], dtype=np.int32)
        selection_digest = hashlib.blake2b(digest_size=_DIGEST_SIZE)
        cursor = work.execute(
            """SELECT m.shard_id, m.cache_row, m.frame_row, s.id
               FROM master m JOIN selections s ON s.line_no=m.line_no
               ORDER BY m.line_no"""
        )
        offset = 0
        while True:
            chunk = cursor.fetchmany(100_000)
            if not chunk:
                break
            for index, (shard_id, cache_row, frame_row, record_id) in enumerate(chunk):
                frame_ids[offset + index] = shard_id
                frame_cache_rows[offset + index] = cache_row
                frame_rows_master[offset + index] = frame_row
                selection_digest.update(
                    f"{record_id},{shard_id},{frame_row}\n".encode("utf-8")
                )
            offset += len(chunk)
        if offset != staging["kept_rows"]:
            raise RuntimeError("master 行数与选中帧数不一致")
        _save_npy_atomic_master(cache_dir, frame_ids, frame_cache_rows, frame_rows_master)
        manifest["complete"] = True
        manifest["selected_atoms"] = selected_atoms
        manifest["selection_digest"] = selection_digest.hexdigest()
        _write_manifest(cache_dir, manifest)
        _write_json_atomic(cache_dir / "master.json", {
            "schema_version": MASTER_SCHEMA_VERSION,
            "frames": staging["kept_rows"],
            "selection_digest": selection_digest.hexdigest(),
            "sample_sha256": sample_sha256,
            "exclusion_digest": exclusions.digest,
        })
    finally:
        work.close()
        work_path.unlink(missing_ok=True)
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_dir": str(cache_dir),
        "sample_sha256": sample_sha256,
        "exclusion_digest": exclusions.digest,
        "shards_built": built,
        "shards_resumed": resumed,
        "frames": staging["kept_rows"],
        "atoms": selected_atoms,
        "excluded_rows": staging["excluded_rows"],
        "unmatched_exclusions": staging["unmatched_heaps"] + staging["unmatched_ids"],
        "resumed_only": False,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def _save_npy_atomic_master(
    cache_dir: Path, shard_ids: np.ndarray, cache_rows: np.ndarray, frame_rows: np.ndarray
) -> None:
    temporary = _new_temp_path(cache_dir / "master.npz")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, shard_id=shard_ids, cache_row=cache_rows, frame_row=frame_rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, cache_dir / "master.npz")
    finally:
        temporary.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 缓存统计（只覆盖允许训练帧）


class _ShardView:
    """单个 shard 缓存的懒加载视图：frames.jsonl 元数据 + mmap npy。"""

    def __init__(self, cache_dir: Path, entry: Mapping[str, Any]):
        self._cache_dir = cache_dir
        self.entry = entry
        self._meta: dict[str, Any] | None = None
        self._forces: np.ndarray | None = None
        self._stress: np.ndarray | None = None

    def meta(self) -> dict[str, Any]:
        if self._meta is None:
            path = self._cache_dir / str(self.entry["frames_file"])
            columns: dict[str, list[Any]] = {
                key: [] for key in (
                    "cache_row", "line_no", "id", "starting_heap", "frame_row",
                    "atom_start", "atom_end", "atom_count", "energy", "stress_row",
                    "target_digest", "sid", "source_path", "archive_member", "lmdb_row_id",
                )
            }
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle):
                    record = json.loads(line)
                    for key in columns:
                        if key not in record:
                            raise ValueError(f"{path}:{line_number + 1}: frames.jsonl 缺少 {key}")
                        columns[key].append(record[key])
            expected = int(self.entry["frame_count"])
            if len(columns["cache_row"]) != expected:
                raise ValueError(f"{path}: 行数 {len(columns['cache_row'])} != manifest {expected}")
            if columns["cache_row"] != list(range(expected)):
                raise ValueError(f"{path}: cache_row 必须等于行号")
            meta: dict[str, Any] = dict(columns)
            for key in ("cache_row", "line_no", "frame_row", "atom_start", "atom_end",
                        "atom_count", "stress_row", "lmdb_row_id"):
                meta[key] = np.asarray(columns[key], dtype=np.int64)
            meta["energy"] = np.asarray(columns["energy"], dtype=np.float64)
            self._meta = meta
        return self._meta

    def forces(self) -> np.ndarray:
        if self._forces is None:
            array = np.load(
                self._cache_dir / str(self.entry["forces_file"]),
                mmap_mode="r", allow_pickle=False,
            )
            if array.shape != (int(self.entry["atom_count"]), 3):
                raise ValueError(f"{self.entry['forces_file']}: 形状与 manifest 不符")
            if array.dtype.str != self.entry["forces_dtype"]:
                raise ValueError(f"{self.entry['forces_file']}: dtype 与 manifest 不符")
            self._forces = array
        return self._forces

    def stress(self) -> np.ndarray:
        if self._stress is None:
            array = np.load(
                self._cache_dir / str(self.entry["stress_file"]),
                mmap_mode="r", allow_pickle=False,
            )
            expected = (int(self.entry["frame_count"]), int(self.entry["stress_width"]))
            if array.shape != expected:
                raise ValueError(f"{self.entry['stress_file']}: 形状与 manifest 不符")
            if array.dtype.str != self.entry["stress_dtype"]:
                raise ValueError(f"{self.entry['stress_file']}: dtype 与 manifest 不符")
            self._stress = array
        return self._stress


def compute_frame_stats(
    cache_dir: Path | str,
    output_path: Path | str | None = None,
    chunk_rows: int = 65_536,
) -> dict[str, Any]:
    """只在缓存覆盖的允许训练帧上流式计算 energy/force/stress 统计。"""
    if chunk_rows < 1:
        raise ValueError("chunk_rows 必须为正数")
    cache_dir = Path(cache_dir)
    manifest = _load_manifest(cache_dir, require_complete=True)
    energy = OnlineStats()
    forces = OnlineStats()
    stress_stats: list[OnlineStats] | None = None
    frames = atoms = 0
    for shard_key in sorted(manifest["shards"], key=int):
        view = _ShardView(cache_dir, manifest["shards"][shard_key])
        energy.update(view.meta()["energy"])
        frames += int(view.entry["frame_count"])
        atoms += int(view.entry["atom_count"])
        width = int(view.entry["stress_width"])
        if stress_stats is None:
            stress_stats = [OnlineStats() for _ in range(width)]
        elif len(stress_stats) != width:
            raise ValueError("stress 宽度跨 shard 不一致")
        forces_array = view.forces()
        for start in range(0, forces_array.shape[0], chunk_rows):
            forces.update(forces_array[start:start + chunk_rows])
        stress_array = view.stress()
        for start in range(0, stress_array.shape[0], chunk_rows):
            block = stress_array[start:start + chunk_rows]
            for column in range(width):
                stress_stats[column].update(block[:, column])
    if not frames or energy.count != frames or energy.nonfinite:
        raise ValueError("缓存能量存在缺失或非有限值，统计中止")
    if forces.count != atoms * 3 or forces.nonfinite:
        raise ValueError("缓存 forces 存在缺失或非有限值，统计中止")
    assert stress_stats is not None
    stress_mean = []
    stress_std = []
    for column, stats in enumerate(stress_stats):
        summary = stats.summary()
        if summary["count"] != frames or stats.nonfinite:
            raise ValueError(f"stress 第 {column} 列存在缺失或非有限值")
        stress_mean.append(summary["mean"])
        stress_std.append(summary["std"])
    result = {
        "schema_version": STATS_SCHEMA_VERSION,
        "cache_dir": str(cache_dir),
        "selection_digest": manifest["selection_digest"],
        "exclusion_digest": manifest["exclusion_digest"],
        "frames": frames,
        "atoms": atoms,
        "energy": {**energy.summary(), "unit": "eV (total energy, stored unit)"},
        "forces": {**forces.summary(), "mode": "component", "unit": "eV/Angstrom"},
        "stress": {
            "count": frames,
            "mean": stress_mean,
            "std": stress_std,
            "width": len(stress_stats),
            "unit": "stored unit (officially eV/Angstrom^3), Voigt 6 分量",
        },
    }
    if output_path is not None:
        _write_json_atomic(Path(output_path), result)
    return result


# ---------------------------------------------------------------------------
# 数据集


class OMC25CrystalDataset:
    """OMC25 选中帧数据集：返回 (FeaturizedRecord, item)，与 collate_metadata 兼容。

    运行时只读帧缓存与 ASE-LMDB；index 仅在 ``build_frame_cache`` 阶段使用。
    """

    def __init__(
        self,
        cache_dir: Path | str,
        cfg: MuseConfig,
        stats: FrameTargetStats,
        external_root: Path | str,
        lmdb_cache_root: Path | str,
        sample_jsonl: Path | str | None = None,
        exclusions: ExclusionSpec | None = None,
        tokenizers=None,
        featurizer: RecordFeaturizer | None = None,
        spacegroups: Mapping[str, int] | None = None,
        strict_targets: bool = True,
        verify_digests: bool = True,
        references: ElementReferences | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cfg = cfg
        self.stats = stats
        if references is None and stats.energy_mode != "total":
            raise ValueError("残差能量统计必须同时提供元素参考能")
        if references is not None:
            if stats.energy_mode != RESIDUAL_MODE:
                raise ValueError("提供元素参考能时，统计必须声明残差 energy_mode")
            if references.sha256 != stats.reference_sha256:
                raise ValueError("元素参考能与残差统计的 reference_sha256 不符")
        self.references = references
        self.featurizer = featurizer or RecordFeaturizer(cfg, tokenizers)
        self.spacegroups = spacegroups
        self.strict_targets = bool(strict_targets)
        self.verify_digests = bool(verify_digests)
        self.external_root = Path(external_root)
        self.lmdb_cache_root = Path(lmdb_cache_root).resolve()

        self._manifest = _load_manifest(self.cache_dir, require_complete=True)
        exclusions = exclusions or EMPTY_EXCLUSIONS
        if exclusions.digest != self._manifest["exclusion_digest"]:
            raise ValueError(
                "排除清单与缓存构建时不符，拒绝静默使用；"
                "请传入构建缓存时相同的 exclusions（或重建缓存）"
            )
        if sample_jsonl is not None:
            sample_jsonl = Path(sample_jsonl)
            if _sha256_path(sample_jsonl) != self._manifest["sample_sha256"]:
                raise ValueError(f"{sample_jsonl}: 与缓存构建时的抽样 JSONL 不符")
        master_meta = json.loads((self.cache_dir / "master.json").read_text(encoding="utf-8"))
        if master_meta.get("schema_version") != MASTER_SCHEMA_VERSION:
            raise ValueError("master.json schema_version 不符")
        if master_meta.get("selection_digest") != self._manifest["selection_digest"]:
            raise ValueError("master 与 manifest 的 selection_digest 不符")
        master = np.load(self.cache_dir / "master.npz", allow_pickle=False)
        self._master_shard = master["shard_id"].astype(np.int64)
        self._master_row = master["cache_row"].astype(np.int64)
        self._master_frame_row = master["frame_row"].astype(np.int64)
        if not (len(self._master_shard) == len(self._master_row)
                == len(self._master_frame_row) == int(master_meta["frames"])):
            raise ValueError("master.npz 长度与 master.json 不符")
        if int(master_meta["frames"]) != int(self._manifest["selected_frames"]):
            raise ValueError("master 帧数与 manifest 不符")
        for entry in self._manifest["shards"].values():
            if int(entry["stress_width"]) != 6:
                raise ValueError(
                    f"shard {entry['shard_id']}: stress 宽度 {entry['stress_width']} != 6；"
                    "collate_metadata 仅支持 Voigt 6 分量，禁止静默截断"
                )
        stats.stress_arrays(6)  # 提前校验归一化参数与 stress 宽度匹配

        self._views: dict[int, _ShardView] = {}
        self._lmdb_paths: dict[tuple[str, str], Path] = {}

    def __len__(self) -> int:
        return int(len(self._master_shard))

    def _view(self, shard_id: int) -> _ShardView:
        view = self._views.get(shard_id)
        if view is None:
            entry = self._manifest["shards"].get(str(shard_id))
            if entry is None:
                raise ValueError(f"缓存缺少 shard {shard_id}，缓存不完整")
            view = _ShardView(self.cache_dir, entry)
            self._views[shard_id] = view
        return view

    def _lmdb_path(self, source_path: str, member_name: str) -> Path:
        key = (source_path, member_name)
        cached = self._lmdb_paths.get(key)
        if cached is not None:
            return cached
        destination = lmdb_cache_destination(self.lmdb_cache_root, source_path, member_name)
        if destination.is_file():
            # 已解包：直接使用，绝不为确认而重扫结构 tar。
            path = destination
        else:
            archive = resolve_archive_path(self.external_root, source_path)
            if not archive.is_file():
                raise FileNotFoundError(f"结构 tar 不存在: {archive}")
            path = extract_lmdb_member(archive, member_name, self.lmdb_cache_root)
        self._lmdb_paths[key] = path
        return path

    def _normalize(self, meta: Mapping[str, Any], cache_row: int,
                   forces: np.ndarray, stress: np.ndarray,
                   reference_energy: float = 0.0) -> dict[str, Any]:
        energy = float(meta["energy"][cache_row])
        stats = self.stats
        energy_z = (energy - reference_energy - stats.energy_mean) / stats.energy_std
        force_z = ((forces.astype(np.float32) - np.float32(stats.force_mean))
                   / np.float32(stats.force_std))
        mean, std = stats.stress_arrays(int(stress.shape[-1]))
        stress_z = (stress.astype(np.float32) - mean) / std
        energy_ok = math.isfinite(energy_z)
        force_ok = np.isfinite(force_z).all(axis=1)
        stress_ok = np.isfinite(stress_z)
        if self.strict_targets and not (energy_ok and force_ok.all() and stress_ok.all()):
            raise ValueError(f"{meta['id'][cache_row]}: 能量/力/应力含非有限值")
        return {
            "energy": (np.asarray([energy_z if energy_ok else 0.0], dtype=np.float32),
                       np.asarray([energy_ok], dtype=bool)),
            "force": (np.where(force_ok[:, None], force_z, 0.0).astype(np.float32), force_ok),
            "stress": (np.where(stress_ok, stress_z, 0.0).astype(np.float32), stress_ok),
        }

    def get(self, row: int) -> tuple[FeaturizedRecord, dict[str, Any]]:
        row = int(row)
        if row < 0 or row >= len(self):
            raise IndexError(f"行号越界: {row}")
        shard_id = int(self._master_shard[row])
        cache_row = int(self._master_row[row])
        view = self._view(shard_id)
        meta = view.meta()
        frame_row = int(meta["frame_row"][cache_row])
        if frame_row != int(self._master_frame_row[row]):
            raise ValueError(f"master 与 frames.jsonl 不一致: row {row}")
        atom_start = int(meta["atom_start"][cache_row])
        atom_end = int(meta["atom_end"][cache_row])
        atom_count = int(meta["atom_count"][cache_row])
        if atom_end - atom_start != atom_count:
            raise ValueError(f"{meta['id'][cache_row]}: 缓存原子区间与 atom_count 不符")
        forces = np.asarray(view.forces()[atom_start:atom_end])
        stress = np.asarray(view.stress()[int(meta["stress_row"][cache_row])])
        if forces.shape != (atom_count, 3):
            raise ValueError(f"{meta['id'][cache_row]}: forces 形状与 atom_count 不符")
        if self.verify_digests:
            actual = _target_digest(meta["energy"][cache_row], stress, forces)
            if actual != meta["target_digest"][cache_row]:
                raise ValueError(f"{meta['id'][cache_row]}: 缓存 digest 校验失败")

        record_id = str(meta["id"][cache_row])
        heap = str(meta["starting_heap"][cache_row])
        sid = str(meta["sid"][cache_row])
        lmdb_path = self._lmdb_path(
            str(meta["source_path"][cache_row]), str(meta["archive_member"][cache_row])
        )
        structure = read_ase_lmdb_row(lmdb_path, int(meta["lmdb_row_id"][cache_row]))
        actual_sid = (structure.raw.get("data") or {}).get("sid")
        if actual_sid != sid:
            raise ValueError(f"{record_id}: LMDB sid {actual_sid!r} != 索引 sid {sid!r}")
        atoms = structure.atoms
        if len(atoms) != atom_count:
            raise ValueError(f"{record_id}: LMDB 原子数 {len(atoms)} != {atom_count}")
        lattice = np.asarray(atoms.cell.array, dtype=np.float64)
        frac_coords = np.asarray(atoms.get_scaled_positions(wrap=False), dtype=np.float64)
        if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
            raise ValueError(f"{record_id}: 非法晶胞")
        if frac_coords.shape != (atom_count, 3) or not np.isfinite(frac_coords).all():
            raise ValueError(f"{record_id}: 非法分数坐标")
        spacegroup_id = 0
        if self.spacegroups is not None:
            spacegroup_id = int(self.spacegroups.get(heap, 0))
        atomic_numbers = atoms.get_atomic_numbers()
        reference_energy = (
            self.references.reference_energy(atomic_numbers)
            if self.references is not None else 0.0
        )
        record = self.featurizer.featurize_crystal(
            lattice, frac_coords, atoms.get_chemical_symbols(), spacegroup_id=spacegroup_id
        )
        if record.n_real_tokens != atom_count:
            raise ValueError(f"{record_id}: 特征化原子数与标签不一致")
        item = self._normalize(meta, cache_row, forces, stress, reference_energy)
        item.update({
            "id": record_id,
            "energy_reference_ev": reference_energy,
            "energy_residual_ev": float(meta["energy"][cache_row]) - reference_energy,
            "starting_heap": heap,
            "shard_id": shard_id,
            "frame_row": frame_row,
            "sid": sid,
        })
        return record, item

    def __getitem__(self, row: int) -> tuple[FeaturizedRecord, dict[str, Any]]:
        return self.get(row)


__all__ = [
    "CACHE_SCHEMA_VERSION", "EMPTY_EXCLUSIONS", "ExclusionSpec", "FrameTargetStats",
    "MASTER_SCHEMA_VERSION", "OMC25CrystalDataset", "STATS_SCHEMA_VERSION",
    "build_frame_cache", "compute_frame_stats", "load_exclusion_spec",
    "lmdb_cache_destination", "resolve_archive_path",
]


# ---------------------------------------------------------------------------
# CLI


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-cache", help="按抽样 JSONL 构建分片帧缓存（可断点续跑）")
    build.add_argument("--sample-jsonl", type=Path, required=True)
    build.add_argument("--index-db", type=Path, required=True)
    build.add_argument("--cache-dir", type=Path, required=True)
    build.add_argument("--suggested-exclude-heaps", type=Path)
    build.add_argument("--exclude-heap", action="append", default=[])
    build.add_argument("--exclude-id", action="append", default=[])
    build.add_argument("--allow-unmatched-exclusions", action="store_true")
    stats = subparsers.add_parser("stats", help="只在缓存（允许训练帧）上计算归一化统计")
    stats.add_argument("--cache-dir", type=Path, required=True)
    stats.add_argument("--output", type=Path, required=True)
    stats.add_argument("--chunk-rows", type=int, default=65_536)
    args = parser.parse_args()
    if args.command == "build-cache":
        exclusions = load_exclusion_spec(
            args.suggested_exclude_heaps, args.exclude_heap, args.exclude_id
        )
        result = build_frame_cache(
            args.sample_jsonl, args.index_db, args.cache_dir,
            exclusions=exclusions,
            allow_unmatched_exclusions=args.allow_unmatched_exclusions,
        )
    else:
        result = compute_frame_stats(args.cache_dir, args.output, args.chunk_rows)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
