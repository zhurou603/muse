from __future__ import annotations

import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .corpora import (
    CORPORA,
    PRETRAIN_ROOT,
    CorpusSpec,
    TextRecord,
    crop_seed,
    iter_corpus,
    random_smiles_views,
)

INDEX_ROOT = PRETRAIN_ROOT / "index"


class IndexedTextCorpus:
    def __init__(self, spec: CorpusSpec, index_path: Path | None = None):
        self.spec = spec
        self.index_path = index_path or INDEX_ROOT / f"{spec.name}.offsets.npy"
        if not self.index_path.exists():
            self.build_index()
        self.offsets = np.load(self.index_path, mmap_mode="r")
        if len(self.offsets) != spec.records + 1:
            raise ValueError(f"{spec.name}: 索引记录数 {len(self.offsets) - 1} != {spec.records}")
        self.data = np.memmap(spec.path, dtype=np.uint8, mode="r")
        if self.offsets[-1] != len(self.data):
            raise ValueError(f"{spec.name}: 索引末尾与文件字节数不一致")

    def build_index(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".tmp.npy")
        offsets = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.int64,
                                            shape=(self.spec.records + 1,))
        count = 0
        if self.spec.kind in ("tsv", "augmented_tsv"):
            with self.spec.path.open("rb") as stream:
                offsets[0] = 0
                for line in stream:
                    if not line.strip():
                        raise ValueError(f"{self.spec.path}:{count + 1}: 空行")
                    count += 1
                    if count > self.spec.records:
                        raise ValueError(f"{self.spec.name}: 实际记录数超过 manifest")
                    offsets[count] = stream.tell()
        elif self.spec.kind == "fasta":
            with self.spec.path.open("rb") as stream:
                while True:
                    position = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    if line.startswith(b">"):
                        if count >= self.spec.records:
                            raise ValueError(f"{self.spec.name}: 实际记录数超过 manifest")
                        offsets[count] = position
                        count += 1
                offsets[count] = stream.tell()
        else:
            raise ValueError(f"未知语料格式: {self.spec.kind}")
        if count != self.spec.records:
            raise ValueError(f"{self.spec.name}: 实际记录数 {count} != manifest {self.spec.records}")
        offsets.flush()
        del offsets
        os.replace(tmp, self.index_path)

    def _bytes(self, row: int) -> bytes:
        if row < 0 or row >= self.spec.records:
            raise IndexError(row)
        return self.data[self.offsets[row]:self.offsets[row + 1]].tobytes()

    def get(self, row: int, view: int = 0) -> TextRecord:
        block = self._bytes(row).decode("utf-8")
        if self.spec.kind == "tsv":
            fields = block.rstrip("\n").split("\t")
            if len(fields) < 2:
                raise ValueError(f"{self.spec.path}: 非法行 {row}")
            record = TextRecord(self.spec.name, fields[0], fields[1], row)
            if view:
                views = random_smiles_views(record.text, 5, crop_seed(42, record))
                record = TextRecord(record.source, record.key, views[view], row, view)
            return record
        if self.spec.kind == "augmented_tsv":
            fields = block.rstrip("\n").split("\t")
            if len(fields) != 4:
                raise ValueError(f"{self.spec.path}: 非法增强行 {row}")
            key, source_row, view_id, smiles = fields
            return TextRecord(self.spec.name, key, smiles, int(source_row), int(view_id))
        lines = block.splitlines()
        if not lines or not lines[0].startswith(">"):
            raise ValueError(f"{self.spec.path}: 非法 FASTA 行 {row}")
        key = lines[0][1:].split()[0]
        return TextRecord(self.spec.name, key, "".join(lines[1:]), row)


def open_corpus(name: str) -> IndexedTextCorpus:
    return IndexedTextCorpus(CORPORA[name])


def _augment_worker(item):
    row, key, smiles, seed = item
    record = TextRecord("pi1m", key, smiles, row)
    try:
        views = random_smiles_views(smiles, 5, crop_seed(seed, record))
        failure = None
    except Exception as exc:
        views = [smiles] * 5
        failure = f"{type(exc).__name__}: {str(exc)[:180]}"
    return row, key, views, failure


def build_pi1m_view_cache(output: Path | None = None, seed: int = 42,
                          workers: int = 32, limit: int | None = None):
    output = output or INDEX_ROOT / "pi1m_views.tsv"
    output.parent.mkdir(parents=True, exist_ok=True)
    records = list(iter_corpus("pi1m", limit)) if limit is not None else iter_corpus("pi1m")
    tmp = output.with_suffix(".tmp.tsv")
    failures_path = output.with_suffix(".failures.jsonl")
    failures = []
    count = 0
    with Pool(workers) as pool, tmp.open("w", encoding="utf-8") as stream:
        items = ((r.row, r.key, r.text, seed) for r in records)
        for row, key, views, failure in pool.imap(_augment_worker, items, chunksize=64):
            if failure:
                failures.append({"row": row, "id": key, "reason": failure,
                                 "fallback": "original_smiles_all_views"})
            for view, smiles in enumerate(views):
                stream.write(f"{key}\t{row}\t{view}\t{smiles}\n")
                count += 1
    os.replace(tmp, output)
    expected = (limit if limit is not None else CORPORA["pi1m"].records) * 5
    if count != expected:
        raise RuntimeError(f"PI1M 增强行数 {count} != {expected}")
    failures_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n"
                                    for record in failures), encoding="utf-8")
    metadata = {"source": "pi1m", "source_records": count // 5,
                "views_per_record": 5, "seed": seed, "records": count,
                "augmentation_failures": len(failures),
                "failures_path": str(failures_path)}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return CorpusSpec("pi1m_views", output, "augmented_tsv", count)
