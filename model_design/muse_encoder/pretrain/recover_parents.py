"""可断点续跑的 OMC25 首末帧母分子恢复与 MUSE 三元组审计执行器。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .crystal_audit import (
    StartingCrystal,
    StartingCrystalIndex,
    StructureSample,
    audit_crystal_triples,
    build_triple_record,
)
from .omc25 import (
    extract_lmdb_member,
    index_energy_jsonl,
    initialize_index,
    read_ase_lmdb_row,
    starting_heap_id,
)


EXECUTOR_VERSION = "omc25.parent_recovery.v1"


@dataclass(frozen=True)
class FrameSelector:
    record_id: str
    sid: str
    source_path: str
    archive_member: str
    lmdb_row_id: int
    chemical_formula: str
    atom_count: int


@dataclass(frozen=True)
class HeapSelectors:
    starting_heap: str
    first: FrameSelector
    last: FrameSelector
    frame_count: int

    @property
    def distinct_representatives(self) -> int:
        return 1 if self.first == self.last else 2


@dataclass(frozen=True)
class HeapRecoveryTask:
    selectors: HeapSelectors
    metadata: StartingCrystal | None
    first_lmdb: Path
    last_lmdb: Path


def _temporary_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    os.unlink(name)
    return Path(name)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl_atomic(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_energy_index(
    jsonl_paths: Iterable[str | Path],
    output_db: str | Path,
    expected_counts: Iterable[int | None] | None = None,
    limits: Iterable[int | None] | None = None,
) -> dict[str, Any]:
    """原子构建一个或多个 JSONL 的 energy_records，行号跨文件保持唯一。"""
    paths = [Path(path) for path in jsonl_paths]
    expected = list(expected_counts) if expected_counts is not None else [None] * len(paths)
    limits = list(limits) if limits is not None else [None] * len(paths)
    if len(paths) != len(expected) or len(paths) != len(limits):
        raise ValueError("expected_counts/limits 必须与 jsonl_paths 等长")
    output_db = Path(output_db)
    temporary = _temporary_sibling(output_db)
    connection = None
    offset = 0
    file_reports = []
    try:
        connection = sqlite3.connect(temporary)
        initialize_index(connection)
        for path, expected_count, limit in zip(paths, expected, limits):
            staging = sqlite3.connect(":memory:")
            try:
                initialize_index(staging)
                count = index_energy_jsonl(staging, path, limit=limit)
                if expected_count is not None and count != expected_count:
                    raise ValueError(f"{path}: 记录数 {count} != {expected_count}")
                rows = staging.execute(
                    """SELECT id,jsonl_row,chemical_formula,atom_count,energy,energy_unit,
                              source_path,archive_member,lmdb_row_id,sid,starting_heap
                       FROM energy_records ORDER BY jsonl_row"""
                )
                batch = []
                for row in rows:
                    batch.append((row[0], row[1] + offset, *row[2:]))
                    if len(batch) >= 5000:
                        connection.executemany("INSERT INTO energy_records VALUES (?,?,?,?,?,?,?,?,?,?,?)", batch)
                        batch.clear()
                if batch:
                    connection.executemany("INSERT INTO energy_records VALUES (?,?,?,?,?,?,?,?,?,?,?)", batch)
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"{path}: ID/sid/row 重复或与前一 split 冲突") from exc
            finally:
                staging.close()
            file_reports.append({"path": str(path), "records": count, "row_offset": offset})
            offset += count
        connection.commit()
        connection.close()
        connection = None
        os.replace(temporary, output_db)
    finally:
        if connection is not None:
            connection.close()
        temporary.unlink(missing_ok=True)
    return {"records": offset, "files": file_reports, "output_db": str(output_db)}


def build_valtest_index(
    validation_jsonl: str | Path,
    test_jsonl: str | Path,
    output_db: str | Path,
    expected_validation: int | None = None,
    expected_test: int | None = None,
) -> dict[str, Any]:
    report = build_energy_index(
        [validation_jsonl, test_jsonl], output_db,
        expected_counts=[expected_validation, expected_test],
    )
    report["split_source"] = "MUSE validation.jsonl + test.jsonl"
    return report


def _frame_from_row(row: Sequence[Any]) -> FrameSelector:
    return FrameSelector(*row)


def collect_heap_selectors(
    db_path: str | Path, limit: int | None = None
) -> list[HeapSelectors]:
    """流式扫描 energy_records，仅保留每 heap 首、末 selector。"""
    if limit is not None and limit < 1:
        raise ValueError("limit 必须为正数")
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA query_only=ON")
    query = """SELECT starting_heap,id,sid,source_path,archive_member,lmdb_row_id,
                      chemical_formula,atom_count,jsonl_row
               FROM energy_records ORDER BY jsonl_row"""
    selected_heaps = None
    if limit is not None:
        selected_heaps = []
        seen = set()
        for row in connection.execute("SELECT starting_heap FROM energy_records ORDER BY jsonl_row"):
            heap = row[0]
            if heap not in seen:
                seen.add(heap)
                selected_heaps.append(heap)
                if len(selected_heaps) == limit:
                    break
    selected_set = set(selected_heaps) if selected_heaps is not None else None
    collected: dict[str, dict[str, Any]] = {}
    try:
        for row in connection.execute(query):
            heap = row[0]
            if selected_set is not None and heap not in selected_set:
                continue
            sid = row[2]
            if starting_heap_id(sid) != heap:
                raise ValueError(f"{sid}: starting_heap 与索引不一致")
            selector = _frame_from_row(row[1:8])
            if heap not in collected:
                collected[heap] = {"first": selector, "last": selector, "count": 1}
            else:
                collected[heap]["last"] = selector
                collected[heap]["count"] += 1
    finally:
        connection.close()
    if selected_heaps is not None:
        missing = set(selected_heaps) - set(collected)
        if missing:
            raise ValueError(f"选定 heaps 无记录: {sorted(missing)}")
        order = selected_heaps
    else:
        order = list(collected)
    return [
        HeapSelectors(heap, values["first"], values["last"], values["count"])
        for heap in order for values in [collected[heap]]
    ]


def _archive_path(external_root: Path, source_path: str) -> Path:
    root = external_root.resolve()
    relative = Path(source_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"非法 source_path: {source_path}")
    archive = (root / relative).resolve()
    if root not in archive.parents or not archive.is_file():
        raise ValueError(f"结构 tar 不存在或越过 external_root: {source_path}")
    return archive


def ensure_lmdb_cache(
    selectors: Iterable[HeapSelectors],
    external_root: str | Path,
    cache_root: str | Path,
) -> dict[tuple[str, str], Path]:
    """仅解包选中 heaps 所需 member；重复 member 只处理一次。"""
    external_root = Path(external_root)
    cache_root = Path(cache_root)
    required = {}
    for heap in selectors:
        for frame in (heap.first, heap.last):
            required[(frame.source_path, frame.archive_member)] = frame
    cached = {}
    for key, frame in sorted(required.items()):
        source_path, member = key
        archive = _archive_path(external_root, source_path)
        cached[key] = extract_lmdb_member(archive, member, cache_root)
    return cached


def _task_from_selectors(
    selectors: HeapSelectors,
    index: StartingCrystalIndex,
    cached: Mapping[tuple[str, str], Path],
) -> HeapRecoveryTask:
    first = cached[(selectors.first.source_path, selectors.first.archive_member)]
    last = cached[(selectors.last.source_path, selectors.last.archive_member)]
    return HeapRecoveryTask(selectors, index.get(selectors.starting_heap), first, last)


def _options_signature(options: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {"version": EXECUTOR_VERSION, "options": options},
        sort_keys=True, separators=(",", ":"), default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _selectors_signature(selectors: HeapSelectors) -> str:
    payload = {
        "heap": selectors.starting_heap,
        "frame_count": selectors.frame_count,
        "first": selectors.first.__dict__,
        "last": selectors.last.__dict__,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _recovery_signature(task: HeapRecoveryTask, options: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        f"{_options_signature(options)}:{_selectors_signature(task.selectors)}".encode()
    ).hexdigest()


def _base_record(task: HeapRecoveryTask) -> dict[str, Any]:
    selectors = task.selectors
    metadata = task.metadata
    return {
        "schema_version": EXECUTOR_VERSION,
        "id": selectors.starting_heap,
        "starting_heap": selectors.starting_heap,
        "csd_refcode": metadata.csd_refcode if metadata else None,
        "chemical_formula": metadata.xtal_formula if metadata else selectors.first.chemical_formula,
        "parent_composition": metadata.mol_formula if metadata else None,
        "spacegroup": metadata.spacegroup if metadata else None,
        "parent_inchikey": None,
        "parent_smiles": None,
        "missing": [],
    }


def _summarize_recovery(recovery: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if recovery is None:
        return None
    return {
        "status": recovery.get("status"),
        "failure_reason": recovery.get("failure_reason"),
        "matching_fragments": recovery.get("matching_fragments"),
        "successful_fragments": recovery.get("successful_fragments"),
        "fragment_inchikeys": recovery.get("fragment_inchikeys", ()),
        "attempts": recovery.get("attempts", ()),
        "fragments": recovery.get("fragments", ()),
        "method": recovery.get("method"),
    }


def _recover_heap(task: HeapRecoveryTask, options: Mapping[str, Any]) -> dict[str, Any]:
    record = _base_record(task)
    selectors = task.selectors
    metadata = task.metadata
    if metadata is None:
        record["missing"] = ["starting_heap_metadata", "spacegroup", "parent_inchikey"]
        record["recovery"] = {"status": "failed", "consensus": "no_metadata", "frames": []}
        return record
    frames = []
    for role, selector, lmdb_path in zip(
        ("first", "last"), (selectors.first, selectors.last),
        (task.first_lmdb, task.last_lmdb),
    ):
        frame: dict[str, Any] = {
            "role": role,
            "record_id": selector.record_id,
            "sid": selector.sid,
            "lmdb_row_id": selector.lmdb_row_id,
            "lmdb_path": str(lmdb_path),
        }
        try:
            row = read_ase_lmdb_row(lmdb_path, selector.lmdb_row_id)
            actual_sid = (row.raw.get("data") or {}).get("sid")
            if actual_sid != selector.sid:
                raise ValueError(f"LMDB sid {actual_sid!r} != selector {selector.sid!r}")
            sample = StructureSample(selector.record_id, selectors.starting_heap, row.atoms, selector.sid)
            built = build_triple_record(sample, metadata, options)
            frame.update({
                "parent_inchikey": built.get("parent_inchikey"),
                "parent_smiles": built.get("parent_smiles"),
                "missing": built.get("missing", []),
                "recovery": _summarize_recovery(built.get("recovery")),
            })
        except Exception as exc:
            frame.update({
                "parent_inchikey": None,
                "missing": ["parent_inchikey", "frame_execution_error"],
                "error": f"{type(exc).__name__}: {exc}",
                "recovery": None,
            })
        frames.append(frame)
    keys = [frame.get("parent_inchikey") for frame in frames]
    missing = []
    for frame in frames:
        for reason in frame.get("missing", []):
            if reason not in missing:
                missing.append(reason)
    if all(keys) and keys[0] == keys[1]:
        record["parent_inchikey"] = keys[0]
    else:
        if "parent_inchikey" not in missing:
            missing.append("parent_inchikey")
        if all(keys) and keys[0] != keys[1]:
            missing.append("cross_frame_inchikey_mismatch")
    if metadata.spacegroup is None and "spacegroup" not in missing:
        missing.append("spacegroup")
    record["missing"] = missing
    if record["parent_inchikey"]:
        record["parent_smiles"] = next(
            (frame.get("parent_smiles") for frame in frames if frame.get("parent_smiles")), None
        )
    singleton = selectors.distinct_representatives == 1
    consensus = (
        "singleton_same_row_consistent" if singleton and all(keys) and keys[0] == keys[1]
        else "two_frame_consistent" if all(keys) and keys[0] == keys[1]
        else "cross_frame_mismatch" if all(keys)
        else "unavailable"
    )
    record["recovery"] = {
        "status": "success" if record["parent_inchikey"] else "failed",
        "frame_count": selectors.frame_count,
        "distinct_representatives": selectors.distinct_representatives,
        "consensus": consensus,
        "frames": frames,
    }
    return record


def _recover_heap_worker(arguments: tuple[HeapRecoveryTask, Mapping[str, Any]]) -> dict[str, Any]:
    task, options = arguments
    try:
        record = _recover_heap(task, options)
    except Exception as exc:
        record = _base_record(task)
        record["missing"] = ["parent_inchikey", "executor_error"]
        if task.metadata is None:
            record["missing"].insert(0, "starting_heap_metadata")
        if task.metadata is None or task.metadata.spacegroup is None:
            record["missing"].append("spacegroup")
        record["recovery"] = {
            "status": "failed", "consensus": "executor_error",
            "frame_count": task.selectors.frame_count,
            "distinct_representatives": task.selectors.distinct_representatives,
            "error": f"{type(exc).__name__}: {exc}", "frames": [],
        }
    record["recovery_signature"] = _recovery_signature(task, options)
    return record


def _load_resume(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: resume JSONL 损坏") from exc
            heap = record.get("starting_heap")
            signature = record.get("recovery_signature")
            if not heap or not signature:
                raise ValueError(f"{path}:{line_number}: resume 缺少 heap/signature")
            records[heap] = record
    return records


def recover_heaps(
    tasks: Iterable[HeapRecoveryTask],
    output_jsonl: str | Path,
    recovery_options: Mapping[str, Any] | None = None,
    workers: int = 1,
    resume: bool = True,
) -> dict[str, int]:
    """按任务顺序写出；每行 flush，重复执行复用相同 selector/config 的记录。"""
    if workers < 1:
        raise ValueError("workers 必须为正数")
    options = dict(recovery_options or {})
    tasks = list(tasks)
    heaps = [task.selectors.starting_heap for task in tasks]
    if len(set(heaps)) != len(heaps):
        raise ValueError("tasks 含重复 starting_heap")
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    previous = _load_resume(output_jsonl) if resume else {}
    pending = []
    for task in tasks:
        heap = task.selectors.starting_heap
        signature = _recovery_signature(task, options)
        if heap in previous:
            if previous[heap]["recovery_signature"] != signature:
                raise ValueError(f"{heap}: resume 与当前 selector/config 不符；请使用新输出或 no-resume")
        else:
            pending.append(task)
    summary = Counter({
        key: 0 for key in (
            "processed", "resumed", "complete", "missing",
            "parent_recovered", "cross_frame_mismatch",
        )
    })
    summary["resumed"] = len(tasks) - len(pending)
    for task in tasks:
        if task.selectors.starting_heap in previous:
            old = previous[task.selectors.starting_heap]
            summary["complete"] += int(not old.get("missing"))
            summary["missing"] += int(bool(old.get("missing")))
            summary["parent_recovered"] += int(bool(old.get("parent_inchikey")))
            summary["cross_frame_mismatch"] += int(
                "cross_frame_inchikey_mismatch" in old.get("missing", [])
            )
    mode = "a" if resume else "w"
    with output_jsonl.open(mode, encoding="utf-8") as handle:
        def write(record: dict[str, Any]) -> None:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            summary["processed"] += 1
            summary["complete"] += int(not record.get("missing"))
            summary["missing"] += int(bool(record.get("missing")))
            summary["parent_recovered"] += int(bool(record.get("parent_inchikey")))
            summary["cross_frame_mismatch"] += int("cross_frame_inchikey_mismatch" in record.get("missing", []))
            if summary["processed"] % 16 == 0:
                os.fsync(handle.fileno())
        if workers == 1:
            for task in pending:
                write(_recover_heap_worker((task, options)))
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                arguments = ((task, options) for task in pending)
                for record in executor.map(_recover_heap_worker, arguments, chunksize=1):
                    write(record)
        handle.flush()
        os.fsync(handle.fileno())
    return dict(summary)


def iter_heap_recovery(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: 非法 recovery JSONL") from exc


def audit_recovery_files(
    train_recovery: str | Path,
    reference_recovery: str | Path,
    output_dir: str | Path,
    conservative_missing: bool = True,
) -> dict[str, Any]:
    training = list(iter_heap_recovery(train_recovery))
    references = list(iter_heap_recovery(reference_recovery))
    audit = audit_crystal_triples(training, references, conservative_missing)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "triple_hits": output_dir / "triple_hits.jsonl",
        "conservative_hits": output_dir / "conservative_hits.jsonl",
        "missing_training": output_dir / "missing_training.jsonl",
        "missing_reference": output_dir / "missing_reference.jsonl",
        "suggested_exclude_heaps": output_dir / "suggested_exclude_heaps.jsonl",
        "summary": output_dir / "summary.json",
    }
    triple_hits = [
        {"starting_heap": hit["id"], **hit} for hit in audit["triple_hits"]
    ]
    _write_jsonl_atomic(paths["triple_hits"], triple_hits)
    _write_jsonl_atomic(paths["conservative_hits"], audit["conservative_hits"])
    _write_jsonl_atomic(paths["missing_training"], audit["missing_training"])
    _write_jsonl_atomic(paths["missing_reference"], audit["missing_reference"])
    suggestions: dict[str, set[str]] = {}
    for hit in triple_hits:
        suggestions.setdefault(hit["starting_heap"], set()).add("triple_hit")
    for hit in audit["conservative_hits"]:
        suggestions.setdefault(hit["starting_heap"], set()).update(hit["reasons"])
    for record in audit["missing_training"]:
        reasons = record.get("missing") or ["unspecified"]
        suggestions.setdefault(record["starting_heap"], set()).update(
            f"missing_training:{reason}" for reason in reasons
        )
    suggested = [
        {"starting_heap": heap, "reasons": sorted(reasons)}
        for heap, reasons in sorted(suggestions.items())
    ]
    _write_jsonl_atomic(paths["suggested_exclude_heaps"], suggested)
    summary = {
        "schema_version": EXECUTOR_VERSION + ".audit",
        **audit["summary"],
        "reference_split_source": "MUSE validation/test SQLite; never starting-crystals CSV split",
        "suggested_exclude_heaps": len(suggested),
        "training_missing_ignored": False,
        "reference_missing_requires_resolution": bool(audit["missing_reference"]),
        "recommended_action": "排除建议 heaps 或完成人工复核；training/reference missing 均不得静默忽略。",
        "files": {key: str(path) for key, path in paths.items()},
    }
    _write_json_atomic(paths["summary"], summary)
    return summary


def run_pipeline(
    train_db: str | Path,
    valtest_db: str | Path,
    starting_index: StartingCrystalIndex,
    external_root: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
    workers: int = 1,
    limit: int | None = None,
    resume: bool = True,
    recovery_options: Mapping[str, Any] | None = None,
    conservative_missing: bool = True,
) -> dict[str, Any]:
    if Path(train_db).resolve() == Path(valtest_db).resolve():
        raise ValueError("train_db 与 MUSE valtest_db 必须是不同文件")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_selectors = collect_heap_selectors(train_db, limit)
    reference_selectors = collect_heap_selectors(valtest_db, limit)
    cached = ensure_lmdb_cache(
        [*train_selectors, *reference_selectors], external_root, cache_root
    )
    options = dict(recovery_options or {})
    train_tasks = [_task_from_selectors(item, starting_index, cached) for item in train_selectors]
    reference_tasks = [_task_from_selectors(item, starting_index, cached) for item in reference_selectors]
    train_path = output_dir / "train.heap_recovery.jsonl"
    reference_path = output_dir / "reference.heap_recovery.jsonl"
    train_summary = recover_heaps(train_tasks, train_path, options, workers, resume)
    reference_summary = recover_heaps(reference_tasks, reference_path, options, workers, resume)
    audit_summary = audit_recovery_files(
        train_path, reference_path, output_dir, conservative_missing
    )
    return {
        "schema_version": EXECUTOR_VERSION,
        "limit_per_side": limit,
        "workers": workers,
        "train_recovery": train_summary,
        "reference_recovery": reference_summary,
        "audit": audit_summary,
        "train_recovery_jsonl": str(train_path),
        "reference_recovery_jsonl": str(reference_path),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-valtest")
    build.add_argument("--validation-jsonl", type=Path, required=True)
    build.add_argument("--test-jsonl", type=Path, required=True)
    build.add_argument("--output-db", type=Path, required=True)
    build.add_argument("--expected-validation", type=int)
    build.add_argument("--expected-test", type=int)
    run = subparsers.add_parser("run")
    run.add_argument("--train-db", type=Path, required=True)
    run.add_argument("--valtest-db", type=Path, required=True)
    run.add_argument("--validation-jsonl", type=Path)
    run.add_argument("--test-jsonl", type=Path)
    run.add_argument("--starting-csv", type=Path, required=True)
    run.add_argument("--external-root", type=Path, required=True)
    run.add_argument("--cache-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--limit", type=int)
    run.add_argument("--no-resume", action="store_true")
    run.add_argument("--exhaustive", action="store_true")
    run.add_argument("--no-conservative", action="store_true")
    run.add_argument("--expected-starting-heaps", type=int, default=218841)
    run.add_argument("--expected-validation", type=int)
    run.add_argument("--expected-test", type=int)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--train-recovery", type=Path, required=True)
    audit.add_argument("--reference-recovery", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument("--no-conservative", action="store_true")
    args = parser.parse_args()
    if args.command == "build-valtest":
        result = build_valtest_index(
            args.validation_jsonl, args.test_jsonl, args.output_db,
            args.expected_validation, args.expected_test,
        )
    elif args.command == "run":
        if not args.valtest_db.exists():
            if not args.validation_jsonl or not args.test_jsonl:
                parser.error("valtest db 不存在时必须提供 --validation-jsonl 和 --test-jsonl")
            build_valtest_index(
                args.validation_jsonl, args.test_jsonl, args.valtest_db,
                args.expected_validation, args.expected_test,
            )
        starting_index = StartingCrystalIndex.from_csv(
            args.starting_csv, expected_count=args.expected_starting_heaps
        )
        result = run_pipeline(
            args.train_db, args.valtest_db, starting_index,
            args.external_root, args.cache_root, args.output_dir,
            workers=args.workers, limit=args.limit, resume=not args.no_resume,
            recovery_options={"exhaustive": args.exhaustive},
            conservative_missing=not args.no_conservative,
        )
    else:
        result = audit_recovery_files(
            args.train_recovery, args.reference_recovery, args.output_dir,
            not args.no_conservative,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


__all__ = [
    "FrameSelector", "HeapRecoveryTask", "HeapSelectors", "audit_recovery_files",
    "build_energy_index", "build_valtest_index", "collect_heap_selectors",
    "ensure_lmdb_cache", "iter_heap_recovery", "recover_heaps", "run_pipeline",
]


if __name__ == "__main__":
    _main()
