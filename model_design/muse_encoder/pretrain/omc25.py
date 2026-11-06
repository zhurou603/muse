"""OMC25 正式索引、去泄漏审计、离群过滤与分层抽样工具。

索引使用 SQLite，避免把全部帧 ID 或 forces 装入内存。真实 ASE-LMDB
使用 zlib 压缩的旧式 ASE JSON 编码；这里只读解码指定行，不执行 pickle。
"""
from __future__ import annotations

import argparse
import ast
import atexit
import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import tarfile
import tempfile
import threading
import zlib
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


EXPECTED_ENERGY_RECORDS = 25_117_577
EXPECTED_AUX_FRAMES = 25_108_033
SCHEMA_VERSION = "omc25.index.v1"


@dataclass(frozen=True)
class NpyHeader:
    shape: tuple[int, ...]
    dtype: np.dtype
    data_offset: int
    member_size: int

    @property
    def values(self) -> int:
        return math.prod(self.shape)

    @property
    def data_bytes(self) -> int:
        return self.values * self.dtype.itemsize


@dataclass(frozen=True)
class StructureRow:
    row_id: int
    atoms: Any
    raw: dict[str, Any]


@dataclass(frozen=True)
class SamplingConfig:
    target_frames: int = 7_000_000
    minimum_frames: int = 6_000_000
    maximum_frames: int = 8_000_000
    seed: int = 42
    z_threshold: float = 10.0
    force_mode: str = "component"
    chunk_atoms: int = 65_536

    def validate(self) -> None:
        if not 0 <= self.minimum_frames <= self.target_frames <= self.maximum_frames:
            raise ValueError("抽样目标必须位于 minimum_frames/maximum_frames 内")
        if self.target_frames < 1 or self.chunk_atoms < 1:
            raise ValueError("target_frames/chunk_atoms 必须为正数")
        if not math.isfinite(self.z_threshold) or self.z_threshold <= 0:
            raise ValueError("z_threshold 必须为有限正数")
        if self.force_mode not in ("component", "norm"):
            raise ValueError("force_mode 必须为 component 或 norm")


_SCHEMA = """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE energy_records (
    id TEXT PRIMARY KEY,
    jsonl_row INTEGER NOT NULL UNIQUE,
    chemical_formula TEXT NOT NULL,
    atom_count INTEGER NOT NULL,
    energy REAL NOT NULL,
    energy_unit TEXT NOT NULL,
    source_path TEXT NOT NULL,
    archive_member TEXT NOT NULL,
    lmdb_row_id INTEGER NOT NULL,
    sid TEXT NOT NULL UNIQUE,
    starting_heap TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE aux_shards (
    shard_id INTEGER PRIMARY KEY,
    tar_path TEXT NOT NULL,
    member_prefix TEXT NOT NULL,
    forces_member TEXT NOT NULL,
    offsets_member TEXT NOT NULL,
    stress_member TEXT NOT NULL,
    ids_member TEXT NOT NULL,
    frame_count INTEGER NOT NULL,
    atom_count INTEGER NOT NULL,
    stress_width INTEGER NOT NULL,
    forces_dtype TEXT NOT NULL,
    offsets_dtype TEXT NOT NULL,
    stress_dtype TEXT NOT NULL,
    UNIQUE(tar_path, member_prefix)
);
CREATE TABLE aux_frames (
    shard_id INTEGER NOT NULL REFERENCES aux_shards(shard_id),
    frame_row INTEGER NOT NULL,
    id TEXT NOT NULL UNIQUE,
    atom_start INTEGER NOT NULL,
    atom_end INTEGER NOT NULL,
    PRIMARY KEY(shard_id, frame_row)
) WITHOUT ROWID;
"""


def initialize_index(connection: sqlite3.Connection) -> None:
    connection.executescript(_SCHEMA)
    connection.execute(
        "INSERT INTO metadata(key, value) VALUES ('schema_version', ?)",
        (json.dumps(SCHEMA_VERSION),),
    )
    connection.commit()


def _open_index(path: Path, readonly: bool = False) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path))
    if readonly:
        connection.execute("PRAGMA query_only = ON")
    else:
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = OFF")
        connection.execute("PRAGMA temp_store = FILE")
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _new_temp_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    os.unlink(name)
    return Path(name)


@contextmanager
def _atomic_text(path: Path, mode: str = "w"):
    temporary = _new_temp_path(path)
    try:
        with temporary.open(mode, encoding="utf-8", newline="") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    with _atomic_text(path) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _validate_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if (not name or "\x00" in name or "\\" in name or path.is_absolute()
            or any(part in ("", ".", "..") for part in path.parts)
            or path.as_posix() != name):
        raise ValueError(f"不安全或不规范的 tar member: {name!r}")


def _npy_header(stream: Any, member: tarfile.TarInfo) -> NpyHeader:
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError(f"{member.name}: 不是 NPY 文件")
    version = stream.read(2)
    if len(version) != 2:
        raise ValueError(f"{member.name}: NPY version 截断")
    major = version[0]
    if major == 1:
        length_raw = stream.read(2)
        length_format = "<H"
    elif major in (2, 3):
        length_raw = stream.read(4)
        length_format = "<I"
    else:
        raise ValueError(f"{member.name}: 不支持的 NPY version {major}.{version[1]}")
    if len(length_raw) != struct.calcsize(length_format):
        raise ValueError(f"{member.name}: NPY header 截断")
    header_length = struct.unpack(length_format, length_raw)[0]
    if header_length > 1_048_576:
        raise ValueError(f"{member.name}: NPY header 异常大")
    raw = stream.read(header_length)
    if len(raw) != header_length:
        raise ValueError(f"{member.name}: NPY header 截断")
    encoding = "utf-8" if major == 3 else "latin1"
    try:
        values = ast.literal_eval(raw.decode(encoding).strip())
    except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"{member.name}: 非法 NPY header") from exc
    if not isinstance(values, dict):
        raise ValueError(f"{member.name}: 非法 NPY header")
    try:
        shape = tuple(int(value) for value in values["shape"])
        dtype = np.dtype(values["descr"])
        fortran = values["fortran_order"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{member.name}: 非法 NPY shape/dtype") from exc
    if (fortran is not False or any(value < 0 for value in shape)
            or dtype.hasobject or dtype.subdtype is not None):
        raise ValueError(f"{member.name}: 仅支持 C-order、非 object NPY")
    result = NpyHeader(shape, dtype, stream.tell(), member.size)
    if result.data_offset + result.data_bytes != member.size:
        raise ValueError(
            f"{member.name}: NPY 大小不符，header={result.data_offset}, "
            f"data={result.data_bytes}, member={member.size}"
        )
    return result


def _read_npy_array(stream: Any, member: tarfile.TarInfo) -> tuple[np.ndarray, NpyHeader]:
    header = _npy_header(stream, member)
    raw = stream.read(header.data_bytes)
    if len(raw) != header.data_bytes:
        raise ValueError(f"{member.name}: NPY data 截断")
    array = np.frombuffer(raw, dtype=header.dtype).reshape(header.shape)
    return array, header


def _header_for(tf: tarfile.TarFile, member: tarfile.TarInfo) -> NpyHeader:
    stream = tf.extractfile(member)
    if stream is None:
        raise ValueError(f"无法读取 tar member: {member.name}")
    with stream:
        return _npy_header(stream, member)


def _validate_aux_headers(
    forces: NpyHeader, offsets: np.ndarray, offsets_header: NpyHeader,
    stress: NpyHeader, names: Sequence[str],
) -> tuple[int, int, int]:
    if len(forces.shape) != 2 or forces.shape[1] != 3 or forces.dtype.kind != "f":
        raise ValueError(f"{names[0]}: forces 必须为浮点 [n_atoms,3]")
    if len(offsets_header.shape) != 1 or offsets_header.dtype.kind not in "iu":
        raise ValueError(f"{names[1]}: offsets 必须为整数一维数组")
    if len(offsets) < 2 or int(offsets[0]) != 0:
        raise ValueError(f"{names[1]}: offsets 必须从 0 开始且至少含一帧")
    if (offsets[1:] <= offsets[:-1]).any():
        raise ValueError(f"{names[1]}: offsets 必须严格递增（不接受空帧）")
    if int(offsets[-1]) != forces.shape[0]:
        raise ValueError(f"{names[1]}: offsets[-1] 与 forces 原子数不符")
    frame_count = len(offsets) - 1
    if len(stress.shape) != 2 or stress.shape[0] != frame_count or stress.shape[1] not in (6, 9):
        raise ValueError(f"{names[2]}: stress 必须为 [n_frames,6] 或 [n_frames,9]")
    if stress.dtype.kind != "f":
        raise ValueError(f"{names[2]}: stress 必须为浮点数组")
    return frame_count, int(forces.shape[0]), int(stress.shape[1])


def index_aux_tar(
    connection: sqlite3.Connection,
    tar_path: Path | str,
    only_prefixes: Iterable[str] | None = None,
) -> dict[str, int]:
    """索引一个 aux tar；forces/stress 只读 header，不读完整 forces。"""
    tar_path = Path(tar_path).resolve()
    selected = set(only_prefixes) if only_prefixes is not None else None
    total_frames = total_atoms = shards = 0
    with tarfile.open(tar_path, "r:") as tf, connection:
        members: dict[str, tarfile.TarInfo] = {}
        groups: dict[str, dict[str, tarfile.TarInfo]] = {}
        for member in tf:
            if member.name in members:
                raise ValueError(f"{tar_path}: 重复 member {member.name}")
            members[member.name] = member
            parts = PurePosixPath(member.name).parts
            if len(parts) == 3 and parts[-1] in ("forces.npy", "offsets.npy", "stress.npy", "ids.txt"):
                _validate_member_name(member.name)
                if not member.isfile() or member.linkname:
                    raise ValueError(f"{tar_path}:{member.name}: 必须是普通文件")
                prefix = "/".join(parts[:-1])
                groups.setdefault(prefix, {})[parts[-1]] = member
        if selected is not None:
            unknown = selected - set(groups)
            if unknown:
                raise ValueError(f"{tar_path}: 找不到 aux shards {sorted(unknown)}")
        for prefix in sorted(groups):
            if selected is not None and prefix not in selected:
                continue
            group = groups[prefix]
            if set(group) != {"forces.npy", "offsets.npy", "stress.npy", "ids.txt"}:
                raise ValueError(f"{tar_path}:{prefix}: 必须精确包含四个 aux 文件")
            forces_member = group["forces.npy"]
            offsets_member = group["offsets.npy"]
            stress_member = group["stress.npy"]
            ids_member = group["ids.txt"]
            forces_header = _header_for(tf, forces_member)
            stress_header = _header_for(tf, stress_member)
            with tf.extractfile(offsets_member) as stream:
                offsets, offsets_header = _read_npy_array(stream, offsets_member)
            names = (forces_member.name, offsets_member.name, stress_member.name)
            frame_count, atom_count, stress_width = _validate_aux_headers(
                forces_header, offsets, offsets_header, stress_header, names
            )
            cursor = connection.execute(
                """INSERT INTO aux_shards(
                       tar_path, member_prefix, forces_member, offsets_member,
                       stress_member, ids_member, frame_count, atom_count,
                       stress_width, forces_dtype, offsets_dtype, stress_dtype
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (str(tar_path), prefix, forces_member.name, offsets_member.name,
                 stress_member.name, ids_member.name, frame_count, atom_count,
                 stress_width, forces_header.dtype.str, offsets_header.dtype.str,
                 stress_header.dtype.str),
            )
            shard_id = int(cursor.lastrowid)
            batch = []
            rows = 0
            with tf.extractfile(ids_member) as stream:
                for row, raw in enumerate(stream):
                    if row >= frame_count:
                        raise ValueError(f"{ids_member.name}: IDs 多于 offsets/stress 帧数")
                    raw = raw.rstrip(b"\r\n")
                    try:
                        record_id = raw.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise ValueError(f"{ids_member.name}:{row + 1}: 非 UTF-8 ID") from exc
                    if not record_id or record_id != record_id.strip():
                        raise ValueError(f"{ids_member.name}:{row + 1}: 空 ID 或首尾空白")
                    batch.append((shard_id, row, record_id, int(offsets[row]), int(offsets[row + 1])))
                    rows += 1
                    if len(batch) >= 10_000:
                        connection.executemany(
                            "INSERT INTO aux_frames VALUES (?,?,?,?,?)", batch
                        )
                        batch.clear()
            if batch:
                connection.executemany("INSERT INTO aux_frames VALUES (?,?,?,?,?)", batch)
            if rows != frame_count:
                raise ValueError(
                    f"{ids_member.name}: IDs={rows}，offsets/stress 帧数={frame_count}"
                )
            shards += 1
            total_frames += frame_count
            total_atoms += atom_count
    return {"shards": shards, "frames": total_frames, "atoms": total_atoms}


def starting_heap_id(sid: str) -> str:
    """OMC25 sid 的最后一段是帧号，其余部分唯一标识起始堆积。"""
    heap, separator, frame = sid.rpartition("-")
    if not separator or not heap or not frame.isdigit():
        raise ValueError(f"非法 OMC25 sid: {sid!r}")
    return heap


def _energy_row(record: Mapping[str, Any], jsonl_row: int) -> tuple[Any, ...]:
    record_id = record.get("id")
    inputs = record.get("inputs") or {}
    formula = inputs.get("chemical_formula")
    atom_count = inputs.get("atom_count")
    if (not isinstance(record_id, str) or not record_id
            or not isinstance(formula, str) or not formula
            or isinstance(atom_count, bool) or not isinstance(atom_count, int)
            or atom_count < 1):
        raise ValueError(f"JSONL 第 {jsonl_row + 1} 行缺少有效 id/formula/atom_count")
    observations = (record.get("labels") or {}).get("energy") or []
    if len(observations) != 1:
        raise ValueError(f"{record_id}: 期望恰好一个 energy 标签")
    energy = observations[0].get("value")
    unit = observations[0].get("unit")
    if (isinstance(energy, bool) or not isinstance(energy, (int, float))
            or not math.isfinite(energy) or not isinstance(unit, str) or not unit):
        raise ValueError(f"{record_id}: 非法 energy/unit")
    structures = [
        item for item in record.get("external_inputs") or []
        if item.get("role") == "periodic_structure" and item.get("format") == "ase_lmdb"
    ]
    if len(structures) != 1:
        raise ValueError(f"{record_id}: 期望恰好一个 ASE-LMDB structure selector")
    structure = structures[0]
    selector = structure.get("selector") or {}
    source_path = structure.get("source_path")
    archive_member = selector.get("archive_member")
    lmdb_row_id = selector.get("lmdb_row_id")
    sid = selector.get("sid")
    if structure.get("id", record_id) != record_id:
        raise ValueError(f"{record_id}: external_inputs.id 不一致")
    if (not isinstance(source_path, str) or not source_path
            or not isinstance(archive_member, str)
            or isinstance(lmdb_row_id, bool) or not isinstance(lmdb_row_id, int)
            or lmdb_row_id < 1 or not isinstance(sid, str) or not sid):
        raise ValueError(f"{record_id}: 非法 structure selector")
    _validate_member_name(archive_member)
    return (record_id, jsonl_row, formula, atom_count, float(energy), unit,
            source_path, archive_member, lmdb_row_id, sid, starting_heap_id(sid))


def index_energy_jsonl(
    connection: sqlite3.Connection, path: Path | str, limit: int | None = None
) -> int:
    path = Path(path)
    count = 0
    batch = []
    with path.open("r", encoding="utf-8") as handle, connection:
        for line_number, line in enumerate(handle):
            if limit is not None and count >= limit:
                break
            if not line.strip():
                raise ValueError(f"{path}:{line_number + 1}: 空行")
            record = json.loads(line)
            batch.append(_energy_row(record, line_number))
            count += 1
            if len(batch) >= 10_000:
                connection.executemany("INSERT INTO energy_records VALUES (?,?,?,?,?,?,?,?,?,?,?)", batch)
                batch.clear()
        if batch:
            connection.executemany("INSERT INTO energy_records VALUES (?,?,?,?,?,?,?,?,?,?,?)", batch)
    return count


def build_index(
    train_jsonl: Path | str,
    aux_tars: Iterable[Path | str],
    output_db: Path | str,
    expected_energy: int | None = None,
    expected_aux: int | None = None,
    energy_limit: int | None = None,
) -> dict[str, Any]:
    """流式构建正式索引，并完整输出双向 missing ID 清单。"""
    output_db = Path(output_db)
    temporary = _new_temp_path(output_db)
    report_path = Path(str(output_db) + ".report.json")
    missing_aux_path = Path(str(output_db) + ".missing_aux.jsonl")
    missing_energy_path = Path(str(output_db) + ".missing_energy.jsonl")
    connection = None
    try:
        connection = _open_index(temporary)
        initialize_index(connection)
        energy_count = index_energy_jsonl(connection, train_jsonl, energy_limit)
        aux_summary = {"shards": 0, "frames": 0, "atoms": 0}
        for tar_path in sorted(Path(path) for path in aux_tars):
            summary = index_aux_tar(connection, tar_path)
            for key in aux_summary:
                aux_summary[key] += summary[key]
        aux_count = aux_summary["frames"]
        matched = connection.execute(
            "SELECT count(*) FROM energy_records e JOIN aux_frames a ON a.id=e.id"
        ).fetchone()[0]
        energy_only = energy_count - matched
        aux_only = aux_count - matched
        mismatches = connection.execute(
            """SELECT count(*) FROM energy_records e JOIN aux_frames a ON a.id=e.id
               WHERE e.atom_count != a.atom_end-a.atom_start"""
        ).fetchone()[0]
        report = {
            "schema_version": SCHEMA_VERSION,
            "train_jsonl": str(Path(train_jsonl).resolve()),
            "energy_records": energy_count,
            "aux_frames": aux_count,
            "aux_atoms": aux_summary["atoms"],
            "aux_shards": aux_summary["shards"],
            "matched_by_id": matched,
            "count_difference_energy_minus_aux": energy_count - aux_count,
            "energy_without_aux": energy_only,
            "aux_without_energy": aux_only,
            "matched_atom_count_mismatches": mismatches,
            "join_key": "id (never JSONL/aux row number)",
            "missing_aux_jsonl": str(missing_aux_path),
            "missing_energy_jsonl": str(missing_energy_path),
        }
        if expected_energy is not None and energy_count != expected_energy:
            raise ValueError(f"energy 记录数 {energy_count} != {expected_energy}: {report}")
        if expected_aux is not None and aux_count != expected_aux:
            raise ValueError(f"aux 帧数 {aux_count} != {expected_aux}: {report}")
        with _atomic_text(missing_aux_path) as handle:
            for row in connection.execute(
                """SELECT e.id,e.jsonl_row,e.sid FROM energy_records e
                   LEFT JOIN aux_frames a ON a.id=e.id WHERE a.id IS NULL
                   ORDER BY e.jsonl_row"""
            ):
                handle.write(json.dumps(
                    {"id": row[0], "jsonl_row": row[1], "sid": row[2], "reason": "missing_aux"},
                    ensure_ascii=False, separators=(",", ":")) + "\n")
        with _atomic_text(missing_energy_path) as handle:
            for row in connection.execute(
                """SELECT a.id,s.tar_path,s.member_prefix,a.frame_row FROM aux_frames a
                   JOIN aux_shards s ON s.shard_id=a.shard_id
                   LEFT JOIN energy_records e ON e.id=a.id WHERE e.id IS NULL
                   ORDER BY s.tar_path,s.member_prefix,a.frame_row"""
            ):
                handle.write(json.dumps(
                    {"id": row[0], "tar_path": row[1], "member_prefix": row[2],
                     "frame_row": row[3], "reason": "missing_energy"},
                    ensure_ascii=False, separators=(",", ":")) + "\n")
        connection.close()
        connection = None
        os.replace(temporary, output_db)
        _write_json_atomic(report_path, report)
        return report
    finally:
        if connection is not None:
            connection.close()
        temporary.unlink(missing_ok=True)


def extract_lmdb_member(
    archive_path: Path | str, member_name: str, cache_root: Path | str
) -> Path:
    """顺序、安全地提取一个普通 .aselmdb 文件，并以 os.replace 原子落盘。"""
    archive_path = Path(archive_path)
    cache_root = Path(cache_root).resolve()
    _validate_member_name(member_name)
    if not member_name.endswith(".aselmdb"):
        raise ValueError("只允许提取 .aselmdb member")
    destination = cache_root / archive_path.stem / PurePosixPath(member_name)
    resolved = destination.resolve()
    if resolved != destination or cache_root not in resolved.parents:
        raise ValueError("LMDB 目标路径逃逸 cache_root")
    found = False
    with tarfile.open(archive_path, "r|") as tf:
        for member in tf:
            if member.name != member_name:
                continue
            found = True
            if not member.isfile() or member.linkname:
                raise ValueError(f"拒绝提取链接或非普通文件: {member_name}")
            if destination.exists():
                if destination.is_file() and destination.stat().st_size == member.size:
                    return destination
                raise FileExistsError(f"已有大小不符的目标: {destination}")
            source = tf.extractfile(member)
            if source is None:
                raise ValueError(f"无法读取 {member_name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = _new_temp_path(destination)
            try:
                with source, temporary.open("xb") as target:
                    remaining = member.size
                    while remaining:
                        block = source.read(min(8 * 1024 * 1024, remaining))
                        if not block:
                            raise ValueError(f"{member_name}: 提取截断")
                        target.write(block)
                        remaining -= len(block)
                    if source.read(1):
                        raise ValueError(f"{member_name}: member 大小异常")
                    target.flush()
                    os.fsync(target.fileno())
                if temporary.stat().st_size != member.size:
                    raise ValueError(f"{member_name}: 落盘大小不符")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            break
    if not found:
        raise KeyError(f"{archive_path}: 找不到 {member_name}")
    return destination


def _decode_ase_json(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_ase_json(item) for item in value]
    if isinstance(value, dict):
        if "__ndarray__" in value:
            shape, dtype, values = value["__ndarray__"]
            if not isinstance(values, list):
                raise ValueError("仅支持 OMC25 的 inline ASE ndarray 编码")
            array = np.asarray(values, dtype=np.dtype(dtype))
            if array.size != math.prod(shape):
                raise ValueError("ASE ndarray shape 与值数不符")
            return array.reshape(shape)
        return {key: _decode_ase_json(item) for key, item in value.items()}
    return value


_LMDB_READ_LOCKS = {}
_LMDB_READ_LOCKS_GUARD = threading.Lock()
_LMDB_ENVIRONMENTS = {}


def _lmdb_read_lock(path: Path | str) -> threading.Lock:
    key = str(Path(path).resolve())
    with _LMDB_READ_LOCKS_GUARD:
        return _LMDB_READ_LOCKS.setdefault(key, threading.Lock())


def _lmdb_environment(path: Path | str):
    key = str(Path(path).resolve())
    environment = _LMDB_ENVIRONMENTS.get(key)
    if environment is None:
        import lmdb
        environment = lmdb.open(
            key, subdir=False, readonly=True, lock=False,
            readahead=False, meminit=False, max_readers=4,
        )
        _LMDB_ENVIRONMENTS[key] = environment
    return environment


def read_ase_lmdb_row(lmdb_path: Path | str, row_id: int) -> StructureRow:
    """只读获取一行并构造 ASE Atoms；环境按进程和文件复用。"""
    with _lmdb_read_lock(lmdb_path):
        return _read_ase_lmdb_row_unlocked(lmdb_path, row_id)


def _read_ase_lmdb_row_unlocked(lmdb_path: Path | str, row_id: int) -> StructureRow:
    """LMDB key 是 1-based ASCII 整数。"""
    if isinstance(row_id, bool) or not isinstance(row_id, int) or row_id < 1:
        raise ValueError("lmdb_row_id 必须是 >=1 的整数")
    from ase.db.row import AtomsRow

    environment = _lmdb_environment(lmdb_path)
    with environment.begin() as transaction:
        value = transaction.get(str(row_id).encode("ascii"))
    if value is None:
        raise KeyError(f"{lmdb_path}: 没有 LMDB row {row_id}")
    try:
        payload = zlib.decompress(value)
    except zlib.error:
        payload = value
    try:
        decoded = _decode_ase_json(json.loads(payload.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{lmdb_path}: row {row_id} 不是 ASE JSON") from exc
    if not isinstance(decoded, dict) or "numbers" not in decoded:
        raise ValueError(f"{lmdb_path}: row {row_id} 缺少 ASE numbers")
    atoms = AtomsRow(decoded).toatoms(add_additional_information=True)
    return StructureRow(row_id, atoms, decoded)


def close_lmdb_environments():
    with _LMDB_READ_LOCKS_GUARD:
        environments = list(_LMDB_ENVIRONMENTS.values())
        _LMDB_ENVIRONMENTS.clear()
    for environment in environments:
        try:
            environment.close()
        except Exception:
            pass


atexit.register(close_lmdb_environments)


def row_field_report(row: StructureRow) -> dict[str, Any]:
    scalar_paths = []

    def visit(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{prefix}.{key}" if prefix else str(key))
        elif not isinstance(value, np.ndarray):
            scalar_paths.append(prefix)

    visit(row.raw)
    spacegroup = [
        path for path in scalar_paths
        if any(token in path.lower() for token in ("spacegroup", "space_group", "spg"))
    ]
    parent = [
        path for path in scalar_paths
        if (any(token in path.lower() for token in ("parent", "mother"))
            and any(token in path.lower() for token in ("smiles", "inchi", "mol", "structure")))
        or path.lower().split(".")[-1] in ("smiles", "inchikey", "inchi_key")
    ]
    geometry = [key for key in ("numbers", "positions", "cell", "pbc") if key in row.raw]
    return {
        "row_id": row.row_id,
        "top_level_fields": sorted(row.raw),
        "data_fields": sorted((row.raw.get("data") or {}).keys()),
        "scalar_fields": sorted(scalar_paths),
        "geometry_fields": geometry,
        "spacegroup_fields": spacegroup,
        "potential_parent_molecule_fields": parent,
        "sid": (row.raw.get("data") or {}).get("sid"),
        "triple_audit_ready": bool(spacegroup and parent),
    }


_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)([1-9][0-9]*)?")


def reduced_formula(formula: str) -> str:
    """解析无括号经验式，聚合同元素并除以计数最大公约数。"""
    if not isinstance(formula, str) or not formula:
        raise ValueError("空化学式")
    position = 0
    counts: dict[str, int] = {}
    for match in _FORMULA_TOKEN.finditer(formula):
        if match.start() != position:
            raise ValueError(f"不支持或非法的化学式: {formula!r}")
        element, count = match.groups()
        counts[element] = counts.get(element, 0) + int(count or 1)
        position = match.end()
    if position != len(formula) or not counts:
        raise ValueError(f"不支持或非法的化学式: {formula!r}")
    divisor = math.gcd(*counts.values())
    counts = {element: count // divisor for element, count in counts.items()}
    if "C" in counts:
        order = ["C"] + (["H"] if "H" in counts else []) + sorted(
            element for element in counts if element not in ("C", "H")
        )
    else:
        order = sorted(counts)
    return "".join(element + (str(counts[element]) if counts[element] != 1 else "") for element in order)


def triple_key(formula: str, spacegroup: Any, parent_inchikey: str) -> tuple[str, str, str]:
    if spacegroup is None or not str(spacegroup).strip() or not parent_inchikey:
        raise ValueError("三元组需要空间群和母分子 InChIKey")
    return reduced_formula(formula), str(spacegroup).strip(), str(parent_inchikey).strip()


def audit_triples(
    records: Iterable[Mapping[str, Any]], references: Iterable[Mapping[str, Any]]
) -> Iterator[dict[str, Any]]:
    """通用三元组审计；输入必须显式提供 spacegroup/parent_inchikey。"""
    reference_keys = {
        triple_key(row["chemical_formula"], row["spacegroup"], row["parent_inchikey"])
        for row in references
    }
    for row in records:
        key = triple_key(row["chemical_formula"], row["spacegroup"], row["parent_inchikey"])
        if key in reference_keys:
            yield {"id": row["id"], "reason": "reduced_formula_spacegroup_parent_inchikey",
                   "triple": key}


def audit_valtest_conservative(
    db_path: Path | str,
    crystals_tsv: Path | str,
    output_jsonl: Path | str,
    report_path: Path | str | None = None,
) -> dict[str, Any]:
    """真实数据可执行的保守替代审计，绝不声称完成三元组审计。

    任一 reduced formula、CSD refcode、起始堆积或 sid 与 validation/test
    相同即输出命中。组成命中可能很保守，需人工决定排除策略。
    """
    formulas: set[str] = set()
    refcodes: set[str] = set()
    heaps: set[str] = set()
    sids: set[str] = set()
    references = 0
    with Path(crystals_tsv).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 4:
                raise ValueError(f"{crystals_tsv}:{line_number}: 非法 crystals.tsv")
            formula, _atom_count, sid, _source = fields
            formulas.add(reduced_formula(formula))
            if sid:
                sids.add(sid)
                heaps.add(starting_heap_id(sid))
                refcodes.add(sid.split("-", 1)[0])
            references += 1
    connection = _open_index(Path(db_path), readonly=True)
    output_jsonl = Path(output_jsonl)
    train_records = hits = 0
    reason_counts: Counter[str] = Counter()
    try:
        with _atomic_text(output_jsonl) as handle:
            cursor = connection.execute(
                "SELECT id,chemical_formula,sid,starting_heap FROM energy_records ORDER BY jsonl_row"
            )
            for record_id, formula, sid, heap in cursor:
                train_records += 1
                reasons = []
                if reduced_formula(formula) in formulas:
                    reasons.append("reduced_formula")
                if sid.split("-", 1)[0] in refcodes:
                    reasons.append("csd_refcode")
                if heap in heaps:
                    reasons.append("starting_heap")
                if sid in sids:
                    reasons.append("sid")
                if reasons:
                    hits += 1
                    reason_counts.update(reasons)
                    handle.write(json.dumps(
                        {"id": record_id, "sid": sid, "reasons": reasons},
                        ensure_ascii=False, separators=(",", ":")) + "\n")
    finally:
        connection.close()
    report = {
        "schema_version": "omc25.audit.v1",
        "audit_kind": "conservative_reduced_formula_or_csd_refcode_or_starting_heap_or_sid",
        "reference_rows": references,
        "reference_reduced_formulas": len(formulas),
        "train_records": train_records,
        "hits": hits,
        "reason_counts": dict(sorted(reason_counts.items())),
        "hits_jsonl": str(output_jsonl),
        "triple_audit": {
            "completed": False,
            "status": "blocked_missing_source_fields",
            "missing": ["spacegroup", "parent_inchikey", "parent_molecule_structure"],
            "note": "该保守命中清单不是约化化学式+空间群+母分子 InChIKey 三元组审计。",
        },
    }
    if report_path is not None:
        _write_json_atomic(Path(report_path), report)
    return report


class OnlineStats:
    """按数据块合并的流式总体统计（population variance, ddof=0）。"""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.nonfinite = 0

    def update(self, values: Any) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if not array.size:
            return
        finite = np.isfinite(array)
        self.nonfinite += int((~finite).sum())
        if not finite.all():
            array = array[finite]
        if not array.size:
            return
        count = int(array.size)
        mean = float(array.mean())
        m2 = float(np.square(array - mean).sum())
        if self.count:
            delta = mean - self.mean
            combined = self.count + count
            self.m2 += m2 + delta * delta * self.count * count / combined
            self.mean += delta * count / combined
            self.count = combined
        else:
            self.count, self.mean, self.m2 = count, mean, m2

    def summary(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "nonfinite": self.nonfinite,
            "mean": self.mean if self.count else None,
            "std": math.sqrt(max(self.m2 / self.count, 0.0)) if self.count else None,
            "variance": max(self.m2 / self.count, 0.0) if self.count else None,
            "ddof": 0,
        }


def _iter_npy_blocks(
    stream: Any, header: NpyHeader, chunk_rows: int
) -> Iterator[np.ndarray]:
    if len(header.shape) != 2:
        raise ValueError("流式 NPY block 只支持二维数组")
    rows = header.shape[0]
    width = header.shape[1]
    row_bytes = width * header.dtype.itemsize
    start = 0
    while start < rows:
        count = min(chunk_rows, rows - start)
        raw = stream.read(count * row_bytes)
        if len(raw) != count * row_bytes:
            raise ValueError("NPY 流式读取截断")
        yield np.frombuffer(raw, dtype=header.dtype).reshape(count, width)
        start += count


def _iter_force_frame_blocks(
    stream: Any, header: NpyHeader, offsets: np.ndarray, chunk_atoms: int
) -> Iterator[tuple[int, int, np.ndarray]]:
    """产出完整帧块，避免逐帧小读，也不把 shard forces 全量装入内存。"""
    frame_start = 0
    frame_count = len(offsets) - 1
    while frame_start < frame_count:
        atom_start = int(offsets[frame_start])
        limit = atom_start + chunk_atoms
        if int(offsets[frame_start + 1]) > limit:
            frame_end = frame_start + 1
        else:
            frame_end = min(
                frame_count,
                int(np.searchsorted(offsets, limit, side="right")) - 1,
            )
        if frame_end <= frame_start:
            raise RuntimeError("force frame block 无法前进")
        atom_end = int(offsets[frame_end])
        atom_count = atom_end - atom_start
        raw = stream.read(atom_count * 3 * header.dtype.itemsize)
        expected = atom_count * 3 * header.dtype.itemsize
        if len(raw) != expected:
            raise ValueError("forces 帧块读取截断")
        values = np.frombuffer(raw, dtype=header.dtype).reshape(atom_count, 3)
        yield frame_start, frame_end, values
        frame_start = frame_end


def _shards_by_tar(connection: sqlite3.Connection) -> dict[str, list[tuple[Any, ...]]]:
    grouped: dict[str, list[tuple[Any, ...]]] = {}
    rows = connection.execute(
        """SELECT shard_id,tar_path,forces_member,offsets_member,frame_count,atom_count
           FROM aux_shards ORDER BY tar_path,member_prefix"""
    )
    for row in rows:
        grouped.setdefault(row[1], []).append(row)
    return grouped


def compute_statistics(
    db_path: Path | str,
    output_path: Path | str | None = None,
    force_mode: str = "component",
    chunk_atoms: int = 65_536,
) -> dict[str, Any]:
    """全量流式计算 energy 与 force component/norm 的过滤统计。"""
    if force_mode not in ("component", "norm") or chunk_atoms < 1:
        raise ValueError("非法 force_mode/chunk_atoms")
    connection = _open_index(Path(db_path), readonly=True)
    energy = OnlineStats()
    forces = OnlineStats()
    try:
        cursor = connection.execute("SELECT energy FROM energy_records ORDER BY jsonl_row")
        while True:
            rows = cursor.fetchmany(50_000)
            if not rows:
                break
            energy.update(np.asarray(rows, dtype=np.float64))
        for tar_path, shards in _shards_by_tar(connection).items():
            with tarfile.open(tar_path, "r:") as tf:
                for _shard_id, _path, forces_member, _offsets, _frames, _atoms in shards:
                    member = tf.getmember(forces_member)
                    with tf.extractfile(member) as stream:
                        header = _npy_header(stream, member)
                        if len(header.shape) != 2 or header.shape[1] != 3:
                            raise ValueError(f"{forces_member}: 非法 forces shape")
                        for block in _iter_npy_blocks(stream, header, chunk_atoms):
                            forces.update(block if force_mode == "component" else np.linalg.norm(block, axis=1))
    finally:
        connection.close()
    if not energy.count or not forces.count:
        raise ValueError("没有足够标签计算过滤统计")
    result = {
        "schema_version": "omc25.filter_stats.v1",
        "energy": {**energy.summary(), "definition": "total energy; stored unit (officially eV)"},
        "forces": {
            **forces.summary(),
            "mode": force_mode,
            "definition": ("all Cartesian force components" if force_mode == "component"
                           else "per-atom Euclidean force norms"),
            "unit": "eV/Angstrom",
        },
        "chunk_atoms": chunk_atoms,
    }
    if output_path is not None:
        _write_json_atomic(Path(output_path), result)
    return result


def sampling_priority(seed: int, starting_heap: str, record_id: str) -> int:
    digest = hashlib.blake2b(
        f"{seed}\0{starting_heap}\0{record_id}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little", signed=True)


def stratified_quotas(counts: Mapping[str, int], target: int) -> dict[str, int]:
    """每堆积先保一帧，再按剩余容量比例最大余数分配，精确达到 target。"""
    if not counts or any(count < 1 for count in counts.values()):
        raise ValueError(" strata counts 必须为非空正数")
    total = sum(counts.values())
    if target < len(counts) or target > total:
        raise ValueError(f"target={target} 无法覆盖 {len(counts)} 个 strata / {total} 个候选")
    if target == total:
        return dict(counts)
    quotas = {heap: 1 for heap in counts}
    residual = {heap: count - 1 for heap, count in counts.items()}
    remaining = target - len(counts)
    residual_total = sum(residual.values())
    fractions = {}
    for heap, capacity in residual.items():
        raw = remaining * capacity / residual_total if residual_total else 0.0
        addition = min(int(math.floor(raw)), capacity)
        quotas[heap] += addition
        fractions[heap] = raw - math.floor(raw)
    left = target - sum(quotas.values())
    order = sorted(counts, key=lambda heap: (-fractions[heap], heap))
    while left:
        progressed = False
        for heap in order:
            if quotas[heap] < counts[heap]:
                quotas[heap] += 1
                left -= 1
                progressed = True
                if not left:
                    break
        if not progressed:
            raise RuntimeError("stratified quota 容量不足")
    return quotas


def _z_scores(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std > 0:
        return np.abs((values - mean) / std)
    return np.where(values == mean, 0.0, np.inf)


def _json_float(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _load_prior_exclusions(path: Path | str | None) -> tuple[set[str], set[str]]:
    if path is None:
        return set(), set()
    ids = set()
    heaps = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("id"):
                ids.add(record["id"])
            if record.get("starting_heap"):
                heaps.add(record["starting_heap"])
    return ids, heaps


def sample_frames(
    db_path: Path | str,
    statistics: Mapping[str, Any],
    output_jsonl: Path | str,
    config: SamplingConfig | None = None,
    exclusions_jsonl: Path | str | None = None,
    strata_jsonl: Path | str | None = None,
    report_path: Path | str | None = None,
    prior_exclusions_jsonl: Path | str | None = None,
) -> dict[str, Any]:
    """过滤全部帧并在起始堆积内按 seed 哈希优先级精确抽样。"""
    config = config or SamplingConfig()
    config.validate()
    if statistics.get("forces", {}).get("mode") != config.force_mode:
        raise ValueError("统计与抽样配置的 force_mode 不一致")
    energy_stats = statistics["energy"]
    force_stats = statistics["forces"]
    energy_mean = float(energy_stats["mean"])
    energy_std = float(energy_stats["std"])
    force_mean = float(force_stats["mean"])
    force_std = float(force_stats["std"])
    prior_ids, prior_heaps = _load_prior_exclusions(prior_exclusions_jsonl)
    output_jsonl = Path(output_jsonl)
    exclusions_jsonl = Path(exclusions_jsonl or (str(output_jsonl) + ".exclusions.jsonl"))
    strata_jsonl = Path(strata_jsonl or (str(output_jsonl) + ".strata.jsonl"))
    report_path = Path(report_path or (str(output_jsonl) + ".report.json"))
    work_path = _new_temp_path(output_jsonl)
    work = sqlite3.connect(str(work_path), isolation_level=None)
    work.execute("PRAGMA journal_mode=OFF")
    work.execute("PRAGMA synchronous=OFF")
    work.execute("PRAGMA temp_store=FILE")
    work.execute(
        """CREATE TABLE candidates(
               id TEXT PRIMARY KEY, heap TEXT NOT NULL, priority INTEGER NOT NULL,
               shard_id INTEGER NOT NULL, frame_row INTEGER NOT NULL
           ) WITHOUT ROWID"""
    )
    source = _open_index(Path(db_path), readonly=True)
    reason_counts: Counter[str] = Counter()
    selected = excluded = processed = 0
    try:
        with (_atomic_text(output_jsonl) as selected_handle,
              _atomic_text(exclusions_jsonl) as exclusions_handle,
              _atomic_text(strata_jsonl) as strata_handle):

            def reject(record_id: str, heap: str | None, reasons: list[str], **extra: Any) -> None:
                nonlocal excluded
                excluded += 1
                reason_counts.update(reasons)
                payload = {"id": record_id, "starting_heap": heap,
                           "reasons": reasons, "seed": config.seed, **extra}
                exclusions_handle.write(json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                ) + "\n")

            for record_id, heap, energy in source.execute(
                """SELECT e.id,e.starting_heap,e.energy FROM energy_records e
                   LEFT JOIN aux_frames a ON a.id=e.id WHERE a.id IS NULL ORDER BY e.jsonl_row"""
            ):
                energy_z = abs((energy - energy_mean) / energy_std) if energy_std > 0 else (
                    0.0 if energy == energy_mean else math.inf
                )
                reasons = ["missing_aux"]
                if record_id in prior_ids or heap in prior_heaps:
                    reasons.append("prior_deleak")
                if energy_z > config.z_threshold:
                    reasons.append("energy_z")
                reject(record_id, heap, reasons, energy_z=_json_float(energy_z))
                processed += 1

            batch = []
            for tar_path, shards in _shards_by_tar(source).items():
                with tarfile.open(tar_path, "r:") as tf:
                    for shard_id, _path, forces_member, offsets_member, frame_count, _atoms in shards:
                        offsets_info = tf.getmember(offsets_member)
                        with tf.extractfile(offsets_info) as stream:
                            offsets, offsets_header = _read_npy_array(stream, offsets_info)
                        if len(offsets) != frame_count + 1 or offsets_header.dtype.kind not in "iu":
                            raise ValueError(f"{offsets_member}: 索引后 offsets 已改变")
                        forces_info = tf.getmember(forces_member)
                        cursor = source.execute(
                            """SELECT a.frame_row,a.id,a.atom_start,a.atom_end,
                                      e.energy,e.atom_count,e.starting_heap
                                   FROM aux_frames a LEFT JOIN energy_records e ON e.id=a.id
                                   WHERE a.shard_id=? ORDER BY a.frame_row""",
                            (shard_id,),
                        )
                        with tf.extractfile(forces_info) as stream:
                            header = _npy_header(stream, forces_info)
                            for frame_start, frame_end, block in _iter_force_frame_blocks(
                                stream, header, offsets, config.chunk_atoms
                            ):
                                rows = cursor.fetchmany(frame_end - frame_start)
                                if len(rows) != frame_end - frame_start:
                                    raise ValueError(f"shard {shard_id}: frame cursor 截断")
                                local_offsets = offsets[frame_start:frame_end + 1] - int(offsets[frame_start])
                                nonfinite_atoms = ~np.isfinite(block)
                                nonfinite_frames = np.logical_or.reduceat(
                                    nonfinite_atoms, local_offsets[:-1], axis=0
                                ).any(axis=1)
                                if config.force_mode == "component":
                                    values = _z_scores(block, force_mean, force_std)
                                    frame_z = np.maximum.reduceat(
                                        values, local_offsets[:-1], axis=0
                                    ).max(axis=1)
                                else:
                                    norms = np.linalg.norm(block, axis=1)
                                    frame_z = np.maximum.reduceat(
                                        _z_scores(norms, force_mean, force_std), local_offsets[:-1]
                                    )
                                for index, row in enumerate(rows):
                                    frame_row, record_id, atom_start, atom_end, energy, atom_count, heap = row
                                    if frame_row != frame_start + index or atom_start != int(offsets[frame_row]) or atom_end != int(offsets[frame_row + 1]):
                                        raise ValueError(f"shard {shard_id}: 索引与 offsets 不一致")
                                    reasons = []
                                    if record_id in prior_ids or heap in prior_heaps:
                                        reasons.append("prior_deleak")
                                    energy_z = None
                                    if energy is None:
                                        reasons.append("missing_energy")
                                    else:
                                        energy_z = float(_z_scores(
                                            np.asarray(energy, dtype=np.float64), energy_mean, energy_std
                                        ))
                                        if atom_count != atom_end - atom_start:
                                            reasons.append("atom_count_mismatch")
                                        if energy_z > config.z_threshold:
                                            reasons.append("energy_z")
                                    if nonfinite_frames[index]:
                                        reasons.append("force_nonfinite")
                                    if frame_z[index] > config.z_threshold:
                                        reasons.append("force_z")
                                    if reasons:
                                        reject(record_id, heap, reasons,
                                               energy_z=_json_float(energy_z) if energy_z is not None else None,
                                               force_max_abs_z=_json_float(float(frame_z[index])))
                                    else:
                                        priority = sampling_priority(config.seed, heap, record_id)
                                        batch.append((record_id, heap, priority, shard_id, frame_row))
                                        if len(batch) >= 10_000:
                                            work.executemany("INSERT INTO candidates VALUES (?,?,?,?,?)", batch)
                                            batch.clear()
                                    processed += 1
                        if cursor.fetchone() is not None:
                            raise ValueError(f"shard {shard_id}: 未消费全部 frames")
            if batch:
                work.executemany("INSERT INTO candidates VALUES (?,?,?,?,?)", batch)
                batch.clear()
            union_count = source.execute(
                "SELECT (SELECT count(*) FROM energy_records)+(SELECT count(*) FROM aux_frames)"
            ).fetchone()[0]
            # 匹配帧在 union 中只需计一次。
            union_count -= source.execute(
                "SELECT count(*) FROM energy_records e JOIN aux_frames a ON a.id=e.id"
            ).fetchone()[0]
            if processed != union_count:
                raise RuntimeError(f"抽样处理数 {processed} != ID union {union_count}")

            work.execute("CREATE INDEX candidates_order ON candidates(heap,priority,id)")
            counts = dict(work.execute("SELECT heap,count(*) FROM candidates GROUP BY heap"))
            quotas = stratified_quotas(counts, config.target_frames)
            for heap in sorted(quotas):
                strata_handle.write(json.dumps(
                    {"starting_heap": heap, "eligible": counts[heap], "quota": quotas[heap]},
                    ensure_ascii=False, separators=(",", ":")) + "\n")
            current_heap = None
            rank = 0
            for record_id, heap, priority, shard_id, frame_row in work.execute(
                "SELECT id,heap,priority,shard_id,frame_row FROM candidates ORDER BY heap,priority,id"
            ):
                if heap != current_heap:
                    current_heap = heap
                    rank = 0
                rank += 1
                if rank <= quotas[heap]:
                    selected += 1
                    selected_handle.write(json.dumps(
                        {"id": record_id, "starting_heap": heap, "seed": config.seed,
                         "priority": priority, "shard_id": shard_id, "frame_row": frame_row,
                         "stratum_rank": rank, "selection_reason": "stratified_hash_quota"},
                        ensure_ascii=False, separators=(",", ":")) + "\n")
                else:
                    reject(record_id, heap, ["stratum_quota"], priority=priority,
                           shard_id=shard_id, frame_row=frame_row)
        if selected != config.target_frames:
            raise RuntimeError(f"抽样数 {selected} != target {config.target_frames}")
        if not config.minimum_frames <= selected <= config.maximum_frames:
            raise RuntimeError("抽样数不在配置范围内")
        if selected + excluded != processed:
            raise RuntimeError("selected+excluded 与输入 ID union 不一致")
        report = {
            "schema_version": "omc25.sample.v1",
            "config": asdict(config),
            "statistics": statistics,
            "quota_method": "minimum_one_per_starting_heap_then_proportional_largest_remainder",
            "strata": len(quotas),
            "eligible_frames": sum(counts.values()),
            "input_id_union": processed,
            "selected": selected,
            "excluded": excluded,
            "reason_counts": dict(sorted(reason_counts.items())),
            "prior_exclusions_jsonl": (
                str(Path(prior_exclusions_jsonl).resolve()) if prior_exclusions_jsonl else None
            ),
            "prior_exclusion_ids": len(prior_ids),
            "prior_exclusion_heaps": len(prior_heaps),
            "prior_deleak_excluded_frames": reason_counts.get("prior_deleak", 0),
            "sample_jsonl": str(output_jsonl),
            "exclusions_jsonl": str(exclusions_jsonl),
            "strata_jsonl": str(strata_jsonl),
        }
        _write_json_atomic(report_path, report)
        return report
    finally:
        source.close()
        work.close()
        work_path.unlink(missing_ok=True)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--train-jsonl", type=Path, required=True)
    build.add_argument("--aux-tars", type=Path, nargs="+", required=True)
    build.add_argument("--output-db", type=Path, required=True)
    build.add_argument("--expected-energy", type=int)
    build.add_argument("--expected-aux", type=int)
    stats = subparsers.add_parser("stats")
    stats.add_argument("--db", type=Path, required=True)
    stats.add_argument("--output", type=Path, required=True)
    stats.add_argument("--force-mode", choices=("component", "norm"), default="component")
    stats.add_argument("--chunk-atoms", type=int, default=65_536)
    sample = subparsers.add_parser("sample")
    sample.add_argument("--db", type=Path, required=True)
    sample.add_argument("--stats", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--target", type=int, default=7_000_000)
    sample.add_argument("--minimum", type=int, default=6_000_000)
    sample.add_argument("--maximum", type=int, default=8_000_000)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--z-threshold", type=float, default=10.0)
    sample.add_argument("--force-mode", choices=("component", "norm"), default="component")
    sample.add_argument("--chunk-atoms", type=int, default=65_536)
    sample.add_argument("--prior-exclusions", type=Path, default=None,
                        help="先验去泄漏 JSONL（按顶层 id 与 starting_heap 排除，先于配额生成）")
    audit = subparsers.add_parser("audit")
    audit.add_argument("--db", type=Path, required=True)
    audit.add_argument("--crystals-tsv", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--report", type=Path)
    extract = subparsers.add_parser("extract")
    extract.add_argument("--archive", type=Path, required=True)
    extract.add_argument("--member", required=True)
    extract.add_argument("--cache-root", type=Path, required=True)
    inspect = subparsers.add_parser("inspect-row")
    inspect.add_argument("--lmdb", type=Path, required=True)
    inspect.add_argument("--row-id", type=int, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_index(args.train_jsonl, args.aux_tars, args.output_db,
                             args.expected_energy, args.expected_aux)
    elif args.command == "stats":
        result = compute_statistics(args.db, args.output, args.force_mode, args.chunk_atoms)
    elif args.command == "sample":
        statistics = json.loads(args.stats.read_text(encoding="utf-8"))
        result = sample_frames(
            args.db, statistics, args.output,
            SamplingConfig(args.target, args.minimum, args.maximum, args.seed,
                           args.z_threshold, args.force_mode, args.chunk_atoms),
            prior_exclusions_jsonl=args.prior_exclusions,
        )
    elif args.command == "audit":
        result = audit_valtest_conservative(args.db, args.crystals_tsv, args.output, args.report)
    elif args.command == "extract":
        result = {"lmdb": str(extract_lmdb_member(args.archive, args.member, args.cache_root))}
    else:
        row = read_ase_lmdb_row(args.lmdb, args.row_id)
        result = row_field_report(row)
        result.update({"atom_count": len(row.atoms),
                       "chemical_formula": row.atoms.get_chemical_formula()})
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


__all__ = [
    "EXPECTED_AUX_FRAMES", "EXPECTED_ENERGY_RECORDS", "OnlineStats", "SamplingConfig",
    "StructureRow", "audit_triples", "audit_valtest_conservative", "build_index",
    "compute_statistics", "extract_lmdb_member", "index_aux_tar", "index_energy_jsonl",
    "initialize_index", "read_ase_lmdb_row", "reduced_formula", "row_field_report",
    "sample_frames", "sampling_priority", "starting_heap_id", "stratified_quotas", "triple_key",
]


if __name__ == "__main__":
    _main()
