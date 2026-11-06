from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np

from .aux import AUX_SOURCES, NamespacedRecord, load_hit_ids
from .corpora import PRETRAIN_ROOT
from .trainer import sha256_file

AUX_INDEX_ROOT = PRETRAIN_ROOT / "index" / "aux"
LOCATION_DTYPE = np.dtype([("file", "<i2"), ("line", "<i8"), ("offset", "<i8")])


def build_aux_index(name: str, output: Path | None = None) -> Path:
    spec = AUX_SOURCES[name]
    output = output or AUX_INDEX_ROOT / f"{name}.locations.npy"
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp.npy")
    locations = np.lib.format.open_memmap(tmp, mode="w+", dtype=LOCATION_DTYPE,
                                          shape=(spec.expected_kept,))
    hits = load_hit_ids(spec.hits_path)
    raw = excluded = kept = 0
    for file_index, path in enumerate(spec.paths):
        with path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.strip():
                    raise ValueError(f"{path}:{raw + 1}: 空行")
                record_id = json.loads(line)["id"]
                raw += 1
                if record_id in hits:
                    excluded += 1
                    continue
                if kept >= spec.expected_kept:
                    raise ValueError(f"{name}: 保留记录超过预期 {spec.expected_kept}")
                locations[kept] = (file_index, raw - 1, offset)
                kept += 1
    locations.flush()
    del locations
    counts = (raw, excluded, kept)
    expected = (spec.expected_raw, spec.expected_excluded, spec.expected_kept)
    if counts != expected:
        raise ValueError(f"{name}: 实际计数 {counts} != 预期 {expected}")
    os.replace(tmp, output)
    metadata = {
        "source": name,
        "raw": raw,
        "excluded": excluded,
        "kept": kept,
        "hits_path": str(spec.hits_path),
        "hits_sha256": sha256_file(spec.hits_path),
        "location_dtype": LOCATION_DTYPE.descr,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


class AuxIndex:
    def __init__(self, name: str, index_path: Path | None = None,
                 verify_hits: bool = True):
        self.spec = AUX_SOURCES[name]
        self.index_path = index_path or AUX_INDEX_ROOT / f"{name}.locations.npy"
        if not self.index_path.exists():
            build_aux_index(name, self.index_path)
        self.locations = np.load(self.index_path, mmap_mode="r")
        if len(self.locations) != self.spec.expected_kept:
            raise ValueError(f"{name}: 索引条数不符")
        self.hits = load_hit_ids(self.spec.hits_path) if verify_hits else None
        # 句柄按线程隔离：prefetch>1 的并发 get 共享句柄会让 seek/readline 交错。
        self._local = threading.local()
        self._handles_lock = threading.Lock()
        self._all_handles: list = []

    def __len__(self):
        return len(self.locations)

    def _handle(self, file_index: int):
        handles = getattr(self._local, "handles", None)
        if handles is None:
            handles = self._local.handles = {}
        handle = handles.get(file_index)
        if handle is None:
            handle = self.spec.paths[file_index].open("rb")
            handles[file_index] = handle
            with self._handles_lock:
                self._all_handles.append(handle)
        return handle

    def get(self, row: int) -> NamespacedRecord:
        if row < 0 or row >= len(self):
            raise IndexError(row)
        file_index, line_number, offset = self.locations[row]
        handle = self._handle(int(file_index))
        handle.seek(int(offset))
        payload = json.loads(handle.readline())
        record_id = payload["id"]
        if self.hits is not None and record_id in self.hits:
            raise RuntimeError(f"{self.spec.name}: 索引含排除 ID {record_id}")
        return NamespacedRecord(self.spec.name, record_id, payload)

    def close(self):
        with self._handles_lock:
            for handle in self._all_handles:
                try:
                    handle.close()
                except OSError:
                    pass
            self._all_handles.clear()
        if hasattr(self._local, "handles"):
            self._local.handles.clear()

    def __del__(self):
        self.close()
