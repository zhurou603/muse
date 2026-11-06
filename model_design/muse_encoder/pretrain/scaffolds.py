from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from itertools import islice
from multiprocessing import Pool, TimeoutError as PoolTimeoutError
from pathlib import Path
from typing import Iterator

import numpy as np

from .aux import AUX_SOURCES, iter_source
from .corpora import CORPORA, PRETRAIN_ROOT, iter_corpus

CORPUS_NAMES = frozenset(("pubchem", "chembl", "pi1m"))
AUX_NAMES = frozenset(("qm9", "qmugs", "geom_drugs", "pubchemqc", "pcqm4v2"))
SUPPORTED_SOURCES = CORPUS_NAMES | AUX_NAMES
SCAFFOLD_ROOT = PRETRAIN_ROOT / "index" / "scaffolds"
HASH_ALGORITHM = "blake2b"
HASH_DIGEST_BYTES = 16
SCHEMA_VERSION = 1
_STATE_SCHEMA_VERSION = 1

WorkItem = tuple[str, int, str, str, str | None]

# scaffold 解析失败的记录不丢弃、不并入他组：各自落入以记录 ID 命名的单例组，
# 保证温度采样把它们当作互不相同的新骨架；总数计入 metadata.fallback_records。
FALLBACK_PREFIX = "__row_scaffold_fallback__:"


def _check_name(name: str) -> None:
    if name not in SUPPORTED_SOURCES:
        raise ValueError(f"不支持的 scaffold 数据源: {name}")


def _source_kind(name: str) -> str:
    _check_name(name)
    return "corpus" if name in CORPUS_NAMES else "aux"


def _expected_records(name: str) -> int:
    if _source_kind(name) == "corpus":
        return int(CORPORA[name].records)
    return int(AUX_SOURCES[name].expected_kept)


def _source_files(name: str) -> list[dict]:
    if _source_kind(name) == "corpus":
        paths = (CORPORA[name].path,)
    else:
        spec = AUX_SOURCES[name]
        paths = (*spec.paths, spec.hits_path)
    output = []
    for path in paths:
        stat = path.stat()
        output.append({"path": str(path), "size": stat.st_size,
                       "mtime_ns": stat.st_mtime_ns})
    return output


def _aux_work_item(name: str, row: int, record) -> WorkItem:
    if record.source != name:
        raise ValueError(f"{name}: iter_source 在 row={row} 返回了 {record.source}")
    inputs = record.payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"{name}: row={row}, id={record.id!r}: inputs 必须是对象")
    smiles = inputs.get("smiles")
    if not isinstance(smiles, str) or not smiles:
        raise ValueError(f"{name}: row={row}, id={record.id!r}: inputs.smiles 缺失或非法")
    parent = inputs.get("parent_molecule_id") if name == "geom_drugs" else None
    if parent is not None and (not isinstance(parent, str) or not parent):
        raise ValueError(
            f"{name}: row={row}, id={record.id!r}: parent_molecule_id 必须是非空字符串"
        )
    return name, row, str(record.id), smiles, parent


def _iter_work(name: str) -> Iterator[WorkItem]:
    if _source_kind(name) == "corpus":
        for expected_row, record in enumerate(iter_corpus(name)):
            if record.source != name or record.row != expected_row:
                raise ValueError(
                    f"{name}: iter_corpus 行错位 expected={expected_row}, "
                    f"actual={record.source}/{record.row}"
                )
            yield name, expected_row, record.key, record.text, None
    else:
        for row, record in enumerate(iter_source(name)):
            yield _aux_work_item(name, row, record)


def scaffold_smiles(smiles: str, allow_unsanitized: bool = False) -> str:
    """返回 canonical Bemis-Murcko scaffold；无环分子的 scaffold 为空字符串。"""
    if not isinstance(smiles, str) or not smiles:
        raise ValueError("SMILES/PSMILES 必须是非空字符串")
    from rdkit import Chem, rdBase
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = None
    parse_error = None
    try:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smiles)
    except Exception as exc:
        parse_error = exc
    if mol is None and allow_unsanitized:
        try:
            with rdBase.BlockLogs():
                mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None:
                mol.UpdatePropertyCache(strict=False)
                Chem.FastFindRings(mol)
        except Exception as exc:
            if mol is not None:
                mol = None
            parse_error = exc
    if mol is None:
        detail = f": {parse_error}" if parse_error is not None else ""
        raise ValueError(f"RDKit 无法解析 SMILES/PSMILES{detail}")
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)
    except Exception as exc:
        raise ValueError(f"RDKit 无法计算 Bemis-Murcko scaffold: {exc}") from exc


def _scaffold_worker(item: WorkItem) -> tuple[int, str]:
    source, row, record_id, smiles, parent = item
    if parent is not None:
        return row, parent
    try:
        return row, scaffold_smiles(smiles, allow_unsanitized=source == "pi1m")
    except Exception:
        return row, f"{FALLBACK_PREFIX}{record_id}"


def _salvage_item(item: WorkItem, timeout: float, worker=None):
    """单条独立进程硬超时打捞；超时/异常落单例 fallback 组。

    返回 ((row, key), timed_out)。毒化输入（RDKit C 层长时/死循环）只能靠
    进程终止解决，因此每条用独立 Pool(1) 跑，超时即 terminate。
    """
    if timeout <= 0:
        raise ValueError("打捞超时必须为正数")
    worker = worker or _scaffold_worker
    pool = Pool(1)
    try:
        result = pool.map_async(worker, [item])
        return result.get(timeout=timeout)[0], False
    except PoolTimeoutError:
        return (item[1], f"{FALLBACK_PREFIX}{item[2]}"), True
    finally:
        pool.terminate()
        pool.join()


def _salvage_batch(batch: list[WorkItem], timeout: float,
                   sub_chunk: int = 64) -> tuple[list[tuple[int, str]], int]:
    """批内毒化行隔离：先按 sub_chunk 子块跑小池，超时子块再细分到单条。

    逐条 Pool(1) 在 4096 规模下开销不可接受，子块级先隔离可把干净行快速放行；
    结果严格保持批内行序。返回 (results, timeout_fallbacks)。
    """
    if sub_chunk < 1:
        raise ValueError("sub_chunk 必须为正数")
    results: list[tuple[int, str]] = []
    timeouts = 0

    def run_items(items: list[WorkItem]):
        pool = Pool(min(4, len(items)))
        try:
            async_result = pool.map_async(_scaffold_worker, items, chunksize=1)
            return async_result.get(timeout=timeout)
        except PoolTimeoutError:
            return None
        finally:
            pool.terminate()
            pool.join()

    for start in range(0, len(batch), sub_chunk):
        chunk = batch[start:start + sub_chunk]
        out = run_items(chunk)
        if out is not None:
            results.extend(out)
            continue
        for item in chunk:
            out = run_items([item])
            if out is None:
                timeouts += 1
                results.append((item[1], f"{FALLBACK_PREFIX}{item[2]}"))
            else:
                results.extend(out)
    return results, timeouts


def group_hash(group_key: str) -> tuple[int, int]:
    """把实际分组键映射为稳定的两字 uint64 指纹。"""
    if not isinstance(group_key, str):
        raise TypeError("group_key 必须是字符串")
    digest = hashlib.blake2b(
        group_key.encode("utf-8"), digest_size=HASH_DIGEST_BYTES
    ).digest()
    return (int.from_bytes(digest[:8], "little"),
            int.from_bytes(digest[8:], "little"))


def _paths(name: str, root: Path) -> dict[str, Path]:
    return {
        "row_groups": root / f"{name}.row_groups.npy",
        "group_hashes": root / f"{name}.group_hashes.npy",
        "metadata": root / f"{name}.json",
        "pending": root / f"{name}.row_groups.pending.npy",
        "state": root / f"{name}.state.sqlite3",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _open_state(path: Path):
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def _create_state(path: Path, name: str, kind: str, expected: int,
                  source_files: list[dict], rdkit_version: str):
    connection = _open_state(path)
    with connection:
        connection.execute(
            "CREATE TABLE state (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE registry (group_id INTEGER PRIMARY KEY, "
            "group_key TEXT NOT NULL UNIQUE)"
        )
        values = {
            "state_schema": str(_STATE_SCHEMA_VERSION),
            "source": name,
            "source_kind": kind,
            "expected_records": str(expected),
            "source_files": json.dumps(source_files, sort_keys=True),
            "rdkit_version": rdkit_version,
            "processed": "0",
            "groups": "0",
        }
        connection.executemany(
            "INSERT INTO state(key, value) VALUES (?, ?)", values.items()
        )
    return connection


def _load_state(path: Path, name: str, kind: str, expected: int,
                source_files: list[dict], rdkit_version: str):
    connection = _open_state(path)
    try:
        state = dict(connection.execute("SELECT key, value FROM state"))
        expected_state = {
            "state_schema": str(_STATE_SCHEMA_VERSION),
            "source": name,
            "source_kind": kind,
            "expected_records": str(expected),
            "source_files": json.dumps(source_files, sort_keys=True),
            "rdkit_version": rdkit_version,
        }
        for key, value in expected_state.items():
            if state.get(key) != value:
                raise ValueError(f"{name}: checkpoint {key} 与当前输入/实现不一致")
        processed = int(state["processed"])
        stored_groups = int(state["groups"])
        if not 0 <= processed <= expected or stored_groups < 0:
            raise ValueError(f"{name}: checkpoint 进度越界")
        registry = {}
        count = 0
        rows = connection.execute(
            "SELECT group_id, group_key FROM registry ORDER BY group_id"
        )
        for expected_id, (group_id, group_key) in enumerate(rows):
            if group_id != expected_id or group_key in registry:
                raise ValueError(f"{name}: checkpoint group registry 不连续或含重复键")
            registry[group_key] = group_id
            count += 1
        if count != stored_groups:
            raise ValueError(f"{name}: checkpoint groups={stored_groups}，实际 registry={count}")
        return connection, registry, processed
    except Exception:
        connection.close()
        raise


def _checkpoint(connection, row_groups, processed: int, groups: int,
                pending: list[tuple[int, str]]) -> None:
    row_groups.flush()
    with connection:
        connection.executemany(
            "INSERT INTO registry(group_id, group_key) VALUES (?, ?)", pending
        )
        connection.executemany(
            "INSERT OR REPLACE INTO state(key, value) VALUES (?, ?)",
            (("processed", str(processed)), ("groups", str(groups))),
        )
    pending.clear()


def _validate_final_arrays(connection, row_groups, records: int, groups: int) -> None:
    registry_count, minimum, maximum = connection.execute(
        "SELECT COUNT(*), MIN(group_id), MAX(group_id) FROM registry"
    ).fetchone()
    if registry_count != groups or (groups and (minimum != 0 or maximum != groups - 1)):
        raise ValueError("scaffold registry 的组 ID 不连续")
    if row_groups.shape != (records,) or row_groups.dtype != np.dtype(np.int32):
        raise ValueError("临时 row_groups 的形状或 dtype 不符")
    if records == 0 or groups == 0:
        raise ValueError("空数据源不能构建 ScaffoldIndex")
    minimum_group = int(row_groups.min())
    maximum_group = int(row_groups.max())
    if minimum_group < 0 or maximum_group >= groups:
        raise ValueError("row_groups 含未写入或越界的组 ID")
    counts = np.bincount(row_groups, minlength=groups)
    if len(counts) != groups or not np.all(counts > 0):
        raise ValueError("row_groups 的组 ID 不连续或存在空组")


def _write_group_hashes(connection, path: Path, groups: int) -> None:
    hashes = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.uint64, shape=(groups, 2)
    )
    count = 0
    for group_id, group_key in connection.execute(
            "SELECT group_id, group_key FROM registry ORDER BY group_id"):
        hashes[group_id] = group_hash(group_key)
        count += 1
    if count != groups:
        raise ValueError("发布前 registry 组数发生变化")
    hashes.flush()
    del hashes


def _remove_state(paths: dict[str, Path]) -> None:
    state = paths["state"]
    for path in (state, Path(str(state) + "-journal"),
                 Path(str(state) + "-wal"), Path(str(state) + "-shm")):
        path.unlink(missing_ok=True)


def build_scaffold_arrays(name: str, root: Path | str | None = None,
                          workers: int = 1, resume: bool = False,
                          checkpoint_interval: int = 100_000,
                          batch_size: int = 4096,
                          pool_timeout: float = 45.0,
                          item_timeout: float = 20.0) -> dict:
    """构建并在完整校验后发布一个源的 row_groups/group_hashes。

    多进程主循环对每批设 pool_timeout 硬上限：单条毒化输入（RDKit C 层
    长时计算）会让按序交付的 map 卡死，超时即终止整池、逐条打捞、重建池；
    打捞超时的行落单例 fallback 组并计入 metadata["timeout_fallbacks"]。
    """
    _check_name(name)
    if workers < 1:
        raise ValueError("workers 必须为正数")
    if checkpoint_interval < 1 or batch_size < 1:
        raise ValueError("checkpoint_interval/batch_size 必须为正数")
    if pool_timeout <= 0 or item_timeout <= 0:
        raise ValueError("pool_timeout/item_timeout 必须为正数")
    root = Path(root) if root is not None else SCAFFOLD_ROOT
    root.mkdir(parents=True, exist_ok=True)
    paths = _paths(name, root)
    final_artifacts = [paths[key] for key in ("row_groups", "group_hashes", "metadata")]
    if paths["metadata"].exists():
        if not resume:
            raise FileExistsError(f"{name}: 已完成索引存在: {paths['metadata']}")
        _, _, metadata = load_scaffold_arrays(name, root, required=True)
        return metadata
    if not resume and any(path.exists() for path in
                          (*final_artifacts, paths["pending"], paths["state"])):
        raise FileExistsError(f"{name}: 已有未完成输出；使用 resume=True 继续")

    kind = _source_kind(name)
    expected = _expected_records(name)
    if expected < 1:
        raise ValueError(f"{name}: 预期记录数必须为正数")
    source_files = _source_files(name)
    import rdkit
    rdkit_version = rdkit.__version__

    state_exists = paths["state"].exists()
    pending_exists = paths["pending"].exists()
    if state_exists != pending_exists:
        raise RuntimeError(f"{name}: checkpoint 与临时 memmap 不成对，拒绝覆盖")

    connection = None
    try:
        if state_exists:
            connection, registry, processed = _load_state(
                paths["state"], name, kind, expected, source_files, rdkit_version
            )
            row_groups = np.load(paths["pending"], mmap_mode="r+", allow_pickle=False)
            if (row_groups.dtype != np.dtype(np.int32)
                    or row_groups.shape != (expected,)):
                raise ValueError(f"{name}: 临时 row_groups 与 checkpoint 不匹配")
        else:
            registry = {}
            processed = 0
            row_groups = np.lib.format.open_memmap(
                paths["pending"], mode="w+", dtype=np.int32, shape=(expected,)
            )
            row_groups.fill(-1)
            row_groups.flush()
            connection = _create_state(
                paths["state"], name, kind, expected, source_files, rdkit_version
            )

        work = _iter_work(name)
        skipped = 0
        for item in islice(work, processed):
            if item[1] != skipped:
                raise ValueError(f"{name}: resume 前缀行号错位 {item[1]} != {skipped}")
            skipped += 1
        if skipped != processed:
            raise ValueError(f"{name}: 输入仅有 {skipped} 行，少于 checkpoint {processed} 行")

        pending_groups: list[tuple[int, str]] = []
        timeout_fallbacks = 0

        def consume(results) -> None:
            nonlocal processed
            for result_row, group_key in results:
                if result_row != processed:
                    raise RuntimeError(
                        f"{name}: worker 行错位 {result_row} != {processed}"
                    )
                if processed >= expected:
                    raise ValueError(f"{name}: 实际记录数超过预期 {expected}")
                if not isinstance(group_key, str):
                    raise TypeError(f"{name}: row={processed} 的分组键不是字符串")
                group_id = registry.get(group_key)
                if group_id is None:
                    group_id = len(registry)
                    if group_id > np.iinfo(np.int32).max:
                        raise OverflowError(f"{name}: 组数超过 int32 容量")
                    registry[group_key] = group_id
                    pending_groups.append((group_id, group_key))
                row_groups[processed] = group_id
                processed += 1
                if processed % checkpoint_interval == 0:
                    _checkpoint(connection, row_groups, processed,
                                len(registry), pending_groups)

        if workers == 1:
            while True:
                batch = list(islice(work, batch_size))
                if not batch:
                    break
                consume(map(_scaffold_worker, batch))
        else:
            pool = Pool(workers)
            try:
                while True:
                    batch = list(islice(work, batch_size))
                    if not batch:
                        break
                    async_result = pool.map_async(_scaffold_worker, batch, chunksize=1)
                    try:
                        consume(async_result.get(timeout=pool_timeout))
                    except PoolTimeoutError:
                        pool.terminate()
                        pool.join()
                        salvaged, new_timeouts = _salvage_batch(batch, item_timeout)
                        timeout_fallbacks += new_timeouts
                        consume(salvaged)
                        print(f"{name}: 批超时，子块打捞完成"
                              f"（累计超时 {timeout_fallbacks} 条）", flush=True)
                        pool = Pool(workers)
            finally:
                pool.terminate()
                pool.join()

        if processed != expected:
            raise ValueError(f"{name}: 实际保留记录数 {processed} != 预期 {expected}")
        _checkpoint(connection, row_groups, processed, len(registry), pending_groups)
        groups = len(registry)
        _validate_final_arrays(connection, row_groups, expected, groups)
        fallback_records = sum(1 for key in registry if key.startswith(FALLBACK_PREFIX))
        row_groups.flush()
        del row_groups

        hash_temporary = root / f".{name}.group_hashes.{os.getpid()}.tmp.npy"
        _write_group_hashes(connection, hash_temporary, groups)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "source": name,
            "source_kind": kind,
            "records": expected,
            "groups": groups,
            "fallback_records": fallback_records,
            "timeout_fallbacks": timeout_fallbacks,
            "complete": True,
            "hash_algorithm": HASH_ALGORITHM,
            "hash_digest_bytes": HASH_DIGEST_BYTES,
            "hash_byteorder": "little",
            "hash_input": "utf-8 group key",
            "scaffold_algorithm": "RDKit Bemis-Murcko",
            "canonical_smiles": True,
            "group_key": ("inputs.parent_molecule_id, otherwise canonical scaffold"
                          if name == "geom_drugs" else "canonical scaffold"),
            "rdkit_version": rdkit_version,
            "source_files": source_files,
            "row_groups": {
                "file": paths["row_groups"].name,
                "dtype": "int32",
                "shape": [expected],
                "sha256": _sha256_file(paths["pending"]),
            },
            "group_hashes": {
                "file": paths["group_hashes"].name,
                "dtype": "uint64",
                "shape": [groups, 2],
                "sha256": _sha256_file(hash_temporary),
            },
        }
        os.replace(paths["pending"], paths["row_groups"])
        os.replace(hash_temporary, paths["group_hashes"])
        _atomic_json(paths["metadata"], metadata)
        connection.close()
        connection = None
        _remove_state(paths)
        return metadata
    finally:
        if connection is not None:
            connection.close()


def load_scaffold_arrays(name: str, root: Path | str | None = None,
                         required: bool = False):
    """返回 (row_groups, group_hashes, metadata)；可选索引完全缺失时返回 None。"""
    _check_name(name)
    root = Path(root) if root is not None else SCAFFOLD_ROOT
    paths = _paths(name, root)
    artifacts = [paths[key] for key in ("row_groups", "group_hashes", "metadata")]
    missing = [path for path in artifacts if not path.is_file()]
    if missing:
        if not required and len(missing) == len(artifacts):
            return None
        raise FileNotFoundError(f"{name}: scaffold 索引缺失或不完整: {missing}")

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{name}: scaffold metadata schema 不符")
    if metadata.get("source") != name or metadata.get("complete") is not True:
        raise ValueError(f"{name}: scaffold metadata source/complete 不符")
    if (metadata.get("hash_algorithm") != HASH_ALGORITHM
            or metadata.get("hash_digest_bytes") != HASH_DIGEST_BYTES
            or metadata.get("hash_byteorder") != "little"):
        raise ValueError(f"{name}: scaffold 指纹算法 metadata 不符")
    records = metadata.get("records")
    groups = metadata.get("groups")
    if type(records) is not int or type(groups) is not int or records < 1 or groups < 1:
        raise ValueError(f"{name}: records/groups 必须是正整数")

    row_info = metadata.get("row_groups")
    hash_info = metadata.get("group_hashes")
    if not isinstance(row_info, dict) or not isinstance(hash_info, dict):
        raise ValueError(f"{name}: 缺少数组 metadata")
    expected_row_info = {"file": paths["row_groups"].name, "dtype": "int32",
                         "shape": [records]}
    expected_hash_info = {"file": paths["group_hashes"].name, "dtype": "uint64",
                          "shape": [groups, 2]}
    for key, value in expected_row_info.items():
        if row_info.get(key) != value:
            raise ValueError(f"{name}: row_groups metadata {key} 不符")
    for key, value in expected_hash_info.items():
        if hash_info.get(key) != value:
            raise ValueError(f"{name}: group_hashes metadata {key} 不符")
    if _sha256_file(paths["row_groups"]) != row_info.get("sha256"):
        raise ValueError(f"{name}: row_groups SHA-256 校验失败")
    if _sha256_file(paths["group_hashes"]) != hash_info.get("sha256"):
        raise ValueError(f"{name}: group_hashes SHA-256 校验失败")

    row_groups = np.load(paths["row_groups"], mmap_mode="r", allow_pickle=False)
    group_hashes = np.load(paths["group_hashes"], mmap_mode="r", allow_pickle=False)
    if row_groups.dtype != np.dtype(np.int32) or row_groups.shape != (records,):
        raise ValueError(f"{name}: 期望 int32[{records}] row_groups")
    if group_hashes.dtype != np.dtype(np.uint64) or group_hashes.shape != (groups, 2):
        raise ValueError(f"{name}: 期望 uint64[{groups},2] group_hashes")
    minimum = int(row_groups.min())
    maximum = int(row_groups.max())
    if minimum < 0 or maximum >= groups:
        raise ValueError(f"{name}: row_groups 含负数或越界组 ID")
    counts = np.bincount(row_groups, minlength=groups)
    if len(counts) != groups or not np.all(counts > 0):
        raise ValueError(f"{name}: row_groups 组 ID 不连续或存在空组")
    return row_groups, group_hashes, metadata


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="构建 MUSE scaffold/母分子离线索引")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("name", choices=sorted(SUPPORTED_SOURCES))
    build.add_argument("--workers", type=int, default=1)
    build.add_argument("--resume", action="store_true")
    build.add_argument("--root", type=Path, default=SCAFFOLD_ROOT)
    build.add_argument("--checkpoint-interval", type=int, default=100_000)
    build.add_argument("--batch-size", type=int, default=4096)
    build.add_argument("--pool-timeout", type=float, default=45.0,
                       help="单批硬超时秒数，超时终止整池并逐条打捞")
    build.add_argument("--item-timeout", type=float, default=20.0,
                       help="打捞阶段单条硬超时秒数")
    args = parser.parse_args(argv)
    if args.command == "build":
        metadata = build_scaffold_arrays(
            args.name, root=args.root, workers=args.workers, resume=args.resume,
            checkpoint_interval=args.checkpoint_interval, batch_size=args.batch_size,
            pool_timeout=args.pool_timeout, item_timeout=args.item_timeout,
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
