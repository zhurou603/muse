"""Persistent preprocessing caches and parallel frozen-Encoder evaluation."""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import pickle
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from muse_encoder.featurize import RecordFeaturizer
from muse_encoder.inference import (
    _can_window,
    _encode_long_record,
    _expected_tokens,
    _token_output,
)
from muse_encoder.pretrain import dashboard as D
from muse_encoder.pretrain import omc25
from muse_encoder import tokenizers as tokenizer_module


CACHE_SCHEMA = "muse.parallel_dashboard.cache.v1"
_ENCODER = None
_ENCODER_CHECKPOINT = None
_ENCODING_DEVICE = None


def _fingerprint_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("对象数组不能用于缓存指纹")
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": array.dtype.str,
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {str(key): _fingerprint_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_fingerprint_value(item) for item in value]
    if type(value).__module__.startswith("rdkit."):
        from rdkit import Chem
        if isinstance(value, Chem.Mol):
            return {"__rdkit_mol_sha256__": hashlib.sha256(value.ToBinary()).hexdigest()}
    raise TypeError(f"不支持缓存指纹类型: {type(value).__name__}")


def _digest(payload: Any) -> str:
    raw = json.dumps(_fingerprint_value(payload), ensure_ascii=False,
                     sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _module_digest(module) -> str:
    return D.sha256_file(Path(module.__file__))


@lru_cache(maxsize=1)
def _normalization_version() -> dict[str, str]:
    return {
        "dashboard": _module_digest(D),
        "omc25": _module_digest(omc25),
    }


@lru_cache(maxsize=1)
def _featurization_version() -> dict[str, str]:
    return {
        "dashboard": _module_digest(D),
        "inference": _module_digest(__import__("muse_encoder.inference", fromlist=["__name__"])),
        "featurize": _module_digest(__import__("muse_encoder.featurize", fromlist=["__name__"])),
        "tokenizers": _module_digest(tokenizer_module),
    }


@lru_cache(maxsize=1)
def _tokenizer_vocab() -> dict[str, dict[str, int]]:
    return {name: tokenizer.vocab for name, tokenizer in D.load_tokenizers().items()}


def _external_signature(records: Sequence[Mapping[str, Any]], external_base: Path) -> list:
    paths = sorted({ref["source_path"] for record in records
                    for ref in record.get("external_inputs") or []})
    signatures = []
    for source_path in paths:
        path = external_base / source_path
        stat = path.stat()
        signatures.append({
            "source_path": source_path,
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    return signatures


def _normalization_key(records: Sequence[Mapping[str, Any]], external_base: Path,
                       data_version: str) -> str:
    return _digest({
        "schema": CACHE_SCHEMA,
        "records": records,
        "external": _external_signature(records, external_base),
        "data_version": data_version,
        "code": _normalization_version(),
    })


def _prepared_key(records: Sequence[Mapping[str, Any]], normalization_key: str, cfg) -> str:
    return _digest({
        "schema": CACHE_SCHEMA,
        "normalization_key": normalization_key,
        "model_config": cfg.to_dict(),
        "vocab": _tokenizer_vocab(),
        "code": _featurization_version(),
    })


def _cache_path(cache_root: Path, tier: str, namespace: str, key: str) -> Path:
    return cache_root / tier / D._safe_name(namespace) / f"{key}.pkl"


def _read_cache(path: Path, key: str, ids: list[str]):
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if (payload.get("schema") != CACHE_SCHEMA or payload.get("key") != key
            or payload.get("ids") != ids):
        raise ValueError(f"缓存握手失败: {path}")
    return payload["value"]


def _write_cache(path: Path, key: str, ids: list[str], value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            pickle.dump({
                "schema": CACHE_SCHEMA,
                "key": key,
                "ids": ids,
                "value": value,
            }, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalize_batch(records, external_base: Path, external_cache: Path):
    resolver = D.ExternalStructureResolver(external_base, external_cache)
    normalized = []
    failed = []
    try:
        resolver.prepare(records)
    except Exception as exc:
        return normalized, [{"id": record["id"], "error": str(exc)[:500]} for record in records]
    for record in records:
        try:
            normalized.append(D.normalize_record(record, resolver))
        except Exception as exc:
            failed.append({"id": record["id"], "error": str(exc)[:500]})
    return normalized, failed


def _normalization_job(job):
    (index, records, external_base, external_cache, cache_path, key) = job
    ids = [record["id"] for record in records]
    cached = _read_cache(cache_path, key, ids)
    if cached is not None:
        return index, cached, [], 1
    normalized, failed = _normalize_batch(records, external_base, external_cache)
    if not failed:
        _write_cache(cache_path, key, ids, normalized)
    return index, normalized, failed, 0


def _load_encoder(checkpoint_path: str, device: str):
    global _ENCODER, _ENCODER_CHECKPOINT
    identity = (checkpoint_path, device)
    if _ENCODER_CHECKPOINT != identity:
        for attempt in range(6):
            try:
                _ENCODER, _, _ = D.load_checkpoint(Path(checkpoint_path), device)
                break
            except torch.OutOfMemoryError:
                if attempt == 5:
                    raise
                delay = min(30, 5 * 2**attempt)
                print(f"[fast-eval] {device}: 等待前序 worker 释放显存，{delay}s 后重试 "
                      f"({attempt + 1}/5)", flush=True)
                time.sleep(delay)
        _ENCODER_CHECKPOINT = identity
    return _ENCODER, _ENCODER.cfg


@torch.no_grad()
def _encode_items(encoder, cfg, raw, items, batch_size: int, token_level: bool, device: str):
    ok_ids = []
    embeddings = []
    token_chunks = []
    failed = []

    def encode_chunk(chunk_raw, chunk_items):
        if not chunk_raw:
            return
        pending_ids = []
        pending_embeddings = []
        pending_tokens = []
        try:
            for item in chunk_items:
                if item["error"] is not None:
                    raise ValueError(item["error"])
            prepared = [item["record"] for item in chunk_items]
            regular = []
            for raw_record, record in zip(chunk_raw, prepared):
                if _can_window(record, cfg):
                    embedding, tokens = _encode_long_record(
                        encoder, None, cfg, record, batch_size, token_level, device)
                    expected = _expected_tokens(raw_record, record)
                    if token_level and len(tokens) != expected:
                        raise RuntimeError(f"{raw_record['id']}: 长序列 token 数不符")
                    pending_ids.append(raw_record["id"])
                    pending_embeddings.append(np.asarray(embedding, dtype=np.float32))
                    if token_level:
                        pending_tokens.append(tokens)
                else:
                    regular.append((raw_record, record))
            if regular:
                output = D._encode_prepared_chunk(
                    encoder, cfg, [record for _, record in regular], device)
                graph = output["embedding"].float().cpu().numpy()
                tokens = output["tokens"].float().cpu().numpy()
                for index, (raw_record, record) in enumerate(regular):
                    token_out = _token_output(record, tokens[index]) if token_level else None
                    if token_level and len(token_out) != _expected_tokens(raw_record, record):
                        raise RuntimeError(f"{raw_record['id']}: token 数不符")
                    pending_ids.append(raw_record["id"])
                    pending_embeddings.append(np.asarray(graph[index], dtype=np.float32))
                    if token_level:
                        pending_tokens.append(token_out)
        except Exception as exc:
            if len(chunk_raw) == 1:
                failed.append({"id": str(chunk_raw[0]["id"]), "error": str(exc)[:500]})
                return
            middle = len(chunk_raw) // 2
            encode_chunk(chunk_raw[:middle], chunk_items[:middle])
            encode_chunk(chunk_raw[middle:], chunk_items[middle:])
        else:
            ok_ids.extend(pending_ids)
            embeddings.extend(pending_embeddings)
            token_chunks.extend(pending_tokens)

    encode_chunk(raw, items)
    embedding_array = (np.stack(embeddings).astype(np.float32) if embeddings
                       else np.empty((0, cfg.embed_dim), dtype=np.float32))
    token_array = None
    offsets = None
    if token_level:
        token_array = (np.concatenate(token_chunks, axis=0).astype(np.float32) if token_chunks
                       else np.empty((0, cfg.embed_dim), dtype=np.float32))
        offsets = np.zeros(len(token_chunks) + 1, dtype=np.int64)
        for index, chunk in enumerate(token_chunks):
            offsets[index + 1] = offsets[index] + len(chunk)
    if not np.isfinite(embedding_array).all():
        raise ValueError("表征含 NaN/Inf")
    if token_array is not None and not np.isfinite(token_array).all():
        raise ValueError("逐 token 表征含 NaN/Inf")
    return D.EncodedSplit(ok_ids, embedding_array, token_array, offsets, failed)


def _encoding_job(job):
    (index, raw, checkpoint_path, device, cache_path, key, batch_size, token_level) = job
    if _ENCODING_DEVICE is not None:
        device = _ENCODING_DEVICE
    started = time.monotonic()
    ids = [record["id"] for record in raw]
    items = _read_cache(cache_path, key, ids)
    cache_hit = int(items is not None)
    if items is None:
        _, cfg = _load_encoder(checkpoint_path, device)
        featurizer = RecordFeaturizer(cfg, D.load_tokenizers())
        items = []
        for record in raw:
            try:
                prepared = D._prepare_featurized(featurizer, record)
                items.append({"record": prepared, "error": None})
            except Exception as exc:
                items.append({"record": None, "error": str(exc)[:500]})
        if all(item["error"] is None for item in items):
            _write_cache(cache_path, key, ids, items)
    encoder, cfg = _load_encoder(checkpoint_path, device)
    encoded = _encode_items(encoder, cfg, raw, items, batch_size, token_level, device)
    return index, encoded, cache_hit, time.monotonic() - started


def _init_encoding_worker(device_queue):
    global _ENCODING_DEVICE
    _ENCODING_DEVICE = device_queue.get(timeout=30)


def _parallel_map(function, jobs, processes: int, *, devices=None):
    jobs = list(jobs)
    total = len(jobs)
    interval = max(1, total // 20)
    started = time.monotonic()

    def report(completed):
        if completed == total or completed % interval == 0:
            print(f"[fast-eval] {function.__name__}: completed={completed}/{total} "
                  f"elapsed={time.monotonic() - started:.1f}s", flush=True)

    if devices is not None:
        devices = tuple(devices)
        if len(devices) != processes or len(set(devices)) != processes:
            raise ValueError("Encoding workers require one distinct device each")
    if not jobs:
        return []
    if processes == 1:
        results = []
        for completed, job in enumerate(jobs, start=1):
            results.append(function(job))
            report(completed)
        return results
    results = []
    context = mp.get_context("spawn")
    device_queue = None
    options = {}
    if devices is not None:
        device_queue = context.Queue()
        for device in devices:
            device_queue.put(device)
        options = {"initializer": _init_encoding_worker, "initargs": (device_queue,)}
    try:
        with context.Pool(processes=processes, **options) as pool:
            workers = list(pool._pool)
            iterator = pool.imap_unordered(function, jobs, chunksize=1)
            while len(results) < total:
                if devices is not None and any(not worker.is_alive() for worker in workers):
                    raise RuntimeError("Encoding worker exited; refusing silent worker replacement")
                try:
                    result = iterator.next(timeout=1)
                except mp.TimeoutError:
                    continue
                results.append(result)
                report(len(results))
    finally:
        if device_queue is not None:
            device_queue.close()
            device_queue.join_thread()
    return results


def _merge_encoded(parts, embed_dim: int, normalization_errors):
    parts.sort(key=lambda part: part[0])
    ids = []
    embeddings = []
    tokens = []
    token_counts = []
    failed = []
    prepared_hits = 0
    for _, encoded, cache_hit, _elapsed in parts:
        ids.extend(encoded.ids)
        embeddings.append(encoded.embeddings)
        failed.extend(encoded.failed)
        prepared_hits += cache_hit
        if encoded.tokens is not None:
            tokens.append(encoded.tokens)
            token_counts.extend(np.diff(encoded.offsets).tolist())
    embedding_array = (np.concatenate(embeddings).astype(np.float32) if embeddings
                       else np.empty((0, embed_dim), dtype=np.float32))
    token_array = None
    offsets = None
    if tokens:
        token_array = np.concatenate(tokens).astype(np.float32)
        offsets = np.zeros(len(token_counts) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(token_counts)
    failed.extend(normalization_errors)
    return D.EncodedSplit(ids, embedding_array, token_array, offsets, failed), prepared_hits


def _validate_encoded_cache(encoded, ids, embed_dim, token_level, *, require_complete=False):
    success = encoded.ids
    failures = [item["id"] for item in encoded.failed]
    if (len(ids) != len(set(ids)) or len(success) != len(set(success))
            or len(failures) != len(set(failures)) or set(success) & set(failures)
            or set(success) | set(failures) != set(ids)):
        raise ValueError("特征缓存ID重复、缺失或成功失败重叠")
    if any(not isinstance(item.get("error"), str) or not item["error"] for item in encoded.failed):
        raise ValueError("特征缓存失败原因缺失")
    if require_complete and failures:
        raise ValueError("完整特征缓存包含失败记录")
    if (not isinstance(encoded.embeddings, np.ndarray)
            or encoded.embeddings.shape != (len(success), embed_dim)
            or not np.isfinite(encoded.embeddings).all()):
        raise ValueError("特征缓存embedding形状或数值无效")
    if not token_level:
        if encoded.tokens is not None or encoded.offsets is not None:
            raise ValueError("特征缓存token模式不匹配")
        return
    if not success and encoded.tokens is None and encoded.offsets is None:
        return
    if (not isinstance(encoded.tokens, np.ndarray) or encoded.tokens.ndim != 2
            or encoded.tokens.shape[1] != embed_dim or not np.isfinite(encoded.tokens).all()
            or not isinstance(encoded.offsets, np.ndarray)
            or encoded.offsets.shape != (len(success) + 1,)
            or not np.issubdtype(encoded.offsets.dtype, np.integer)
            or encoded.offsets[0] != 0 or encoded.offsets[-1] != len(encoded.tokens)
            or (np.diff(encoded.offsets) < 0).any()):
        raise ValueError("特征缓存token/offsets不对齐或数值无效")


def encode_records_parallel(
        records: Sequence[Mapping[str, Any]], *, checkpoint_path: Path,
        checkpoint_sha: str, cfg, cache_root: Path, external_base: Path,
        external_cache: Path, namespace: str, data_version: str,
        devices: Sequence[str], cpu_workers: int, batch_size: int,
        token_level: bool, probe_config=D.PROBE_CONFIG):
    if not records:
        return D.EncodedSplit([], np.empty((0, cfg.embed_dim), dtype=np.float32),
                              None, None, []), {"central_hit": True}
    if not devices:
        raise ValueError("至少需要一个评估设备")
    if cpu_workers < 1 or batch_size < 1:
        raise ValueError("cpu_workers 和 batch_size 必须为正数")
    cache_root = Path(cache_root)
    checkpoint_path = Path(checkpoint_path).resolve()
    raw_batches = [list(records[start:start + batch_size])
                   for start in range(0, len(records), batch_size)]
    normalization_keys = [_normalization_key(batch, external_base, data_version)
                          for batch in raw_batches]
    central_payload = {
        "schema": CACHE_SCHEMA,
        "checkpoint_sha256": checkpoint_sha,
        "model_config": cfg.to_dict(),
        "normalization_keys": normalization_keys,
        "featurization_code": _featurization_version(),
        "vocab": _tokenizer_vocab(),
        "token_level": token_level,
        "batch_size": batch_size,
    }
    central_key = _digest(central_payload)
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("输入ID重复，不能按ID复用特征")
    central_path = _cache_path(cache_root, "embeddings", namespace, central_key)
    cached = _read_cache(central_path, central_key, ids)
    if cached is None:
        legacy_key = _digest({**central_payload, "probe_config": probe_config.to_json()})
        legacy_path = _cache_path(cache_root, "embeddings", namespace, legacy_key)
        cached = _read_cache(legacy_path, legacy_key, ids)
        if cached is not None:
            _validate_encoded_cache(cached, ids, cfg.embed_dim, token_level, require_complete=True)
            _write_cache(central_path, central_key, ids, cached)
    if cached is not None:
        _validate_encoded_cache(cached, ids, cfg.embed_dim, token_level, require_complete=True)
        print(f"[fast-eval] {namespace}: central embedding cache hit", flush=True)
        return cached, {"central_hit": True, "normalization_hits": len(raw_batches),
                        "prepared_hits": len(raw_batches)}

    partial_path = _cache_path(cache_root, "embedding_progress_v1", namespace, central_key)
    partial = _read_cache(partial_path, central_key, ids)
    reused_ids = set()
    if partial is not None:
        _validate_encoded_cache(partial, ids, cfg.embed_dim, token_level)
        if not partial.ids or not partial.failed:
            raise ValueError("部分特征缓存必须同时包含成功和失败记录")
        reused_ids = set(partial.ids)
        remaining = [record for record in records if record["id"] not in reused_ids]
        raw_batches = [remaining[start:start + batch_size]
                       for start in range(0, len(remaining), batch_size)]
        normalization_keys = [_normalization_key(batch, external_base, data_version)
                              for batch in raw_batches]
        print(f"[fast-eval] {namespace}: reuse={len(reused_ids)} retry={len(remaining)}", flush=True)

    jobs = [
        (index, batch, external_base, external_cache,
         _cache_path(cache_root, "normalized", namespace, normalization_keys[index]),
         normalization_keys[index])
        for index, batch in enumerate(raw_batches)
    ]
    normalized_parts = _parallel_map(_normalization_job, jobs, min(cpu_workers, len(jobs)))
    normalized_parts.sort(key=lambda part: part[0])
    normalized = []
    normalization_errors = []
    normalization_hits = 0
    for _, batch, failed, cache_hit in normalized_parts:
        normalized.extend(batch)
        normalization_errors.extend(failed)
        normalization_hits += cache_hit
    print(f"[fast-eval] {namespace}: normalized={len(normalized)}/{len(records)} "
          f"cache_batches={normalization_hits}/{len(raw_batches)}", flush=True)

    encoding_jobs = []
    for index, start in enumerate(range(0, len(normalized), batch_size)):
        batch = normalized[start:start + batch_size]
        normalization_key = _normalization_key(batch, external_base, data_version)
        prepared_key = _prepared_key(batch, normalization_key, cfg)
        encoding_jobs.append((
            index, batch, str(checkpoint_path), devices[index % len(devices)],
            _cache_path(cache_root, "prepared", namespace, prepared_key), prepared_key,
            batch_size, token_level,
        ))
    if encoding_jobs:
        worker_count = min(len(devices), len(encoding_jobs))
        parts = _parallel_map(_encoding_job, encoding_jobs, worker_count,
                              devices=devices[:worker_count])
        encoded, prepared_hits = _merge_encoded(parts, cfg.embed_dim, normalization_errors)
    else:
        encoded = D.EncodedSplit([], np.empty((0, cfg.embed_dim), dtype=np.float32),
                                 None, None, normalization_errors)
        prepared_hits = 0
    pending_ids = [record["id"] for record in records if record["id"] not in reused_ids]
    _validate_encoded_cache(encoded, pending_ids, cfg.embed_dim, token_level)
    if partial is not None:
        retained = D.EncodedSplit(partial.ids, partial.embeddings, partial.tokens, partial.offsets, [])
        encoded, _ = _merge_encoded([(0, retained, 0, 0), (1, encoded, 0, 0)], cfg.embed_dim, [])
        rows = {record_id: index for index, record_id in enumerate(encoded.ids)}
        ordered_ids = [record_id for record_id in ids if record_id in rows]
        if ordered_ids != encoded.ids:
            order = [rows[record_id] for record_id in ordered_ids]
            encoded.embeddings = encoded.embeddings[order]
            if encoded.tokens is not None:
                chunks = [encoded.tokens[encoded.offsets[index]:encoded.offsets[index + 1]]
                          for index in order]
                encoded.tokens = np.concatenate(chunks, axis=0)
                encoded.offsets = np.zeros(len(chunks) + 1, dtype=np.int64)
                encoded.offsets[1:] = np.cumsum([len(chunk) for chunk in chunks])
            encoded.ids = ordered_ids
    _validate_encoded_cache(encoded, ids, cfg.embed_dim, token_level)
    if not encoded.failed:
        _write_cache(central_path, central_key, ids, encoded)
    elif encoded.ids:
        _write_cache(partial_path, central_key, ids, encoded)
    stats = {
        "central_hit": False,
        "partial_hit": partial is not None,
        "reused_embedding_rows": len(reused_ids),
        "retried_records": len(pending_ids) if partial is not None else 0,
        "normalization_hits": normalization_hits,
        "normalization_batches": len(raw_batches),
        "prepared_hits": prepared_hits,
        "prepared_batches": len(encoding_jobs),
    }
    print(f"[fast-eval] {namespace}: encoded={len(encoded.ids)} failed={len(encoded.failed)} "
          f"prepared_cache_batches={prepared_hits}/{len(encoding_jobs)}", flush=True)
    return encoded, stats
