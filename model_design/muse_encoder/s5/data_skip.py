"""Explicit, persistent per-record data skipping for S5 training."""
from __future__ import annotations

import datetime
import json
import os
import re
import socket
from contextlib import nullcontext
from pathlib import Path

import numpy as np


def is_skippable_data_error(exc: BaseException) -> bool:
    if isinstance(exc, (OSError, MemoryError)):
        return False
    if exc.__class__.__name__ == "OutOfMemoryError":
        return False
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        if any(marker in message for marker in ("nccl", "cuda", "hip", "processgroup", "brokenprocesspool")):
            return False
    return isinstance(exc, Exception)


def load_skipped_rows(run_dir: str | Path) -> dict[str, set[int]]:
    path = Path(run_dir) / "data_skips.jsonl"
    rows: dict[str, set[int]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record.get("event") != "training_input_skipped":
                    continue
                source = str(record["source"])
                row = int(record["row"])
                if row < 0:
                    raise ValueError("negative row")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: 非法跳过记录") from exc
            rows.setdefault(source, set()).add(row)
    return rows


class DataSkipLogger:
    def __init__(self, run_dir: str | Path | None, rank: int):
        self.path = Path(run_dir) / "data_skips.jsonl" if run_dir is not None else None
        self.rank = int(rank)

    def log(self, dataset, pointer, exc: BaseException, micro_step: int,
            stage: str) -> dict:
        error = str(exc)
        cause = exc.__cause__ or exc.__context__
        match = re.search(r"muse__[A-Za-z0-9_]+", error)
        record = {
            "schema_version": "muse_encoder.s5_data_skip.v1",
            "event": "training_input_skipped",
            "timestamp": datetime.datetime.now().astimezone().isoformat(),
            "host": socket.gethostname(),
            "rank": self.rank,
            "worker_pid": os.getpid(),
            "micro_step": int(micro_step),
            "source": dataset.name,
            "row": int(pointer.row),
            "view": int(pointer.view),
            "stage": stage,
            "record_id": match.group(0) if match else None,
            "error_type": type(exc).__name__,
            "error": error[:4000],
            "cause_type": type(cause).__name__ if cause is not None else None,
            "cause": str(cause)[:4000] if cause is not None else None,
        }
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                written = os.write(descriptor, payload)
                if written != len(payload):
                    raise OSError(f"跳过日志写入不完整: {written}/{len(payload)}")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return record


def fetch_valid_batch(dataset, sampler, rng, pointers, align: bool,
                      skip_logger: DataSkipLogger, micro_step: int,
                      dataset_lock=None, max_replacements: int = 256, cfg=None):
    records = []
    metadata = []
    accepted = []
    align_pairs = []
    skipped = []
    lock = dataset_lock if dataset_lock is not None else nullcontext()
    from muse_encoder.pretrain.batching import collate_metadata
    if cfg is None:
        cfg = getattr(dataset, "cfg", None)
    if cfg is None and hasattr(dataset, "base"):
        cfg = getattr(dataset.base, "cfg", None)
    if cfg is None:
        raise RuntimeError(f"{dataset.name}: 无法确定单条 aux 校验配置")

    def fetch(pointer):
        stage = "dataset.get"
        replacements = 0
        while True:
            try:
                with lock:
                    record, item = dataset.get(pointer)
                stage = "dataset.metadata"
                collate_metadata([record], [item], cfg)
                view_record = view_item = None
                if align and dataset.kind == "aux_molecule" and record.coord_mask is not None \
                        and np.any(record.coord_mask):
                    stage = "dataset.get_2d_view"
                    with lock:
                        view_record, view_item = dataset.get_2d_view(pointer)
                return pointer, record, item, view_record, view_item
            except Exception as exc:
                if not is_skippable_data_error(exc):
                    raise
                replacements += 1
                if replacements > max_replacements:
                    raise RuntimeError(
                        f"{dataset.name}: micro_step={micro_step} "
                        f"单样本替换超过 {max_replacements} 次") from exc
                skipped.append(skip_logger.log(dataset, pointer, exc, micro_step, stage))
                sampler.exclude = frozenset(set(sampler.exclude) | {int(pointer.row)})
                pointer = sampler.sample(rng)
                stage = "dataset.get"

    views = []
    for original in pointers:
        pointer, record, item, view_record, view_item = fetch(original)
        accepted.append(pointer)
        records.append(record)
        metadata.append(item)
        if view_record is not None:
            views.append((len(accepted) - 1, view_record, view_item))
    original_count = len(records)
    for offset, (index, view_record, view_item) in enumerate(views):
        align_pairs.append([index, original_count + offset])
        records.append(view_record)
        metadata.append(view_item)
    return records, metadata, accepted, align_pairs, skipped
