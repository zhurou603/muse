"""File hashes, immutable data manifests, and checkpoint lineage helpers."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: str | Path, role: str, with_hash: bool = True) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    record = {
        "path": str(path), "role": str(role), "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if with_hash:
        record["sha256"] = sha256_file(path)
    return record


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_data_manifest(path: str | Path, files: Iterable[Mapping[str, Any]],
                        config: Mapping[str, Any]) -> dict[str, Any]:
    records = [dict(item) for item in files]
    paths = [item["path"] for item in records]
    if len(paths) != len(set(paths)):
        raise ValueError("数据清单存在重复路径")
    if any("sha256" not in item for item in records):
        raise ValueError("正式 S5 数据清单必须包含每个文件的 SHA256")
    payload = {
        "schema_version": "muse_encoder.s5_data_manifest.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "label_policy": {
            "a_train": "official train.jsonl only",
            "aux": "official permitted auxiliary labels only",
            "external_corpora": "unlabeled inputs only",
            "validation_test_labels_used": False,
            "validation_test_inputs_use": "decontamination fingerprints only",
        },
        "files": records,
        "config": dict(config),
    }
    atomic_json(path, payload)
    return payload


def verify_data_manifest(path: str | Path, full_hash: bool = False,
                         small_file_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "muse_encoder.s5_data_manifest.v1":
        raise ValueError("S5 数据清单 schema 不匹配")
    if payload["label_policy"].get("validation_test_labels_used") is not False:
        raise ValueError("数据清单合规标记异常")
    checked = []
    for record in payload["files"]:
        file_path = Path(record["path"])
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        size = file_path.stat().st_size
        if int(record["bytes"]) != size:
            raise ValueError(f"{file_path}: 文件大小与数据清单不一致")
        # Target/index/manifest files are always rehashed.  Multi-GB immutable
        # corpora are fully hashed at preparation; full_hash repeats everything.
        if full_hash or size <= small_file_bytes or record.get("always_rehash", False):
            actual = sha256_file(file_path)
            if actual != record["sha256"]:
                raise ValueError(f"{file_path}: SHA256 与数据清单不一致")
            checked.append(str(file_path))
    return {"manifest": str(path), "files": len(payload["files"]),
            "startup_rehashed": len(checked), "full_hash": bool(full_hash)}


def checkpoint_record(path: str | Path, parent: str | None = None,
                      extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    return {
        "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path),
        "parent": parent, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **dict(extra or {}),
    }
