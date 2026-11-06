"""SMILES 语料可解析性审计：用训练同款的严格图解析（RDKit 2026）全量扫描。

产出 data/pretrain/audit/smiles_unparseable.<name>.rows.txt（0-based 行号，逐行），
供 SourceSampler 的排除位图使用。分块断点续跑：每 100 万行一个 chunk，
结果先写 .partial 再合并；重跑自动跳过已有 chunk。

用法：
  python -m muse_encoder.pretrain.audit_smiles pubchem --workers 96
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from multiprocessing import Pool, TimeoutError as PoolTimeoutError
from pathlib import Path

from .corpora import CORPORA, PRETRAIN_ROOT

CHUNK = 1_000_000
AUDIT_DIR = PRETRAIN_ROOT / "audit"


def _scan_chunk(args) -> tuple[int, list[int]]:
    name, start, end = args
    from .indexed import open_corpus  # 迟导入：worker 进程内初始化
    corpus = open_corpus(name)
    from ..structures import graph_from_smiles
    bad = []
    for row in range(start, end):
        smiles = corpus.get(row).text
        try:
            graph_from_smiles(smiles)
        except Exception:
            bad.append(row)
    return start, bad


def _run_isolated(name: str, start: int, end: int, timeout: float):
    """独立进程扫 [start,end)；超时即 terminate。返回 (bad_rows 或 None, timed_out)。"""
    pool = Pool(1)
    try:
        result = pool.map_async(_scan_chunk, [(name, start, end)])
        return result.get(timeout=timeout)[0][1], False
    except PoolTimeoutError:
        return None, True
    finally:
        pool.terminate()
        pool.join()


def _rescue_chunk(name: str, start: int, end: int, bad: list[int]) -> None:
    """二分救援停滞块：解析时间随规模线性下降，毒化行最终隔离到单行。

    单行 10s 仍解析不了的分子对训练数据加载同样是毒药，按不可解析计入。
    """
    timeout = max(10.0, (end - start) / 2000.0)
    result, timed_out = _run_isolated(name, start, end, timeout)
    if not timed_out:
        bad.extend(result)
        return
    if end - start == 1:
        bad.append(start)
        return
    mid = (start + end) // 2
    _rescue_chunk(name, start, mid, bad)
    _rescue_chunk(name, mid, end, bad)


def audit_corpus(name: str, workers: int = 96, stall_timeout: float = 900.0) -> dict:
    if name not in ("pubchem", "chembl"):
        raise ValueError(f"暂只支持 pubchem/chembl: {name}")
    total = CORPORA[name].records
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    final_path = AUDIT_DIR / f"smiles_unparseable.{name}.rows.txt"
    partial = AUDIT_DIR / f".smiles_unparseable.{name}.partial.jsonl"
    done = set()
    if partial.exists():
        with partial.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done.add(json.loads(line)["chunk_start"])
    chunks = [(name, s, min(s + CHUNK, total)) for s in range(0, total, CHUNK)]
    todo = [c for c in chunks if c[1] not in done]
    print(f"{name}: {total} 行, {len(chunks)} 块, 已完成 {len(done)}, 待办 {len(todo)}",
          flush=True)
    with Pool(workers) as pool:
        iterator = pool.imap_unordered(_scan_chunk, todo, chunksize=1)
        stalled = False
        while True:
            try:
                start, bad = iterator.next(timeout=stall_timeout)
            except PoolTimeoutError:
                stalled = True
                break
            except StopIteration:
                break
            with partial.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"chunk_start": start, "bad_rows": bad}) + "\n")
            done.add(start)
            print(f"{name}: chunk {start} 完成，坏行 {len(bad)}；累计块 {len(done)}/{len(chunks)}",
                  flush=True)
        if stalled:
            # 毒化行让整块卡死：终止池，对缺失块做二分救援，结果同样落 partial。
            print(f"{name}: {stall_timeout}s 无进展，终止池并救援缺失块", flush=True)
            pool.terminate()
            pool.join()
            for chunk_name, start, end in todo:
                if start in done:
                    continue
                bad_rows: list[int] = []
                _rescue_chunk(name, start, end, bad_rows)
                with partial.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"chunk_start": start,
                                             "bad_rows": sorted(bad_rows)}) + "\n")
                done.add(start)
                print(f"{name}: chunk {start} 救援完成，坏行 {len(bad_rows)}", flush=True)
    rows = set()
    with partial.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.update(json.loads(line)["bad_rows"])
    fd, tmp = tempfile.mkstemp(dir=AUDIT_DIR, prefix=".audit", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in sorted(rows):
            handle.write(f"{row}\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, final_path)
    report = {"corpus": name, "records": total, "unparseable": len(rows),
              "rate": len(rows) / total, "output": str(final_path)}
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=("pubchem", "chembl"))
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--stall-timeout", type=float, default=900.0,
                        help="无任何块完成的最长等待秒数，超时终止并二分救援缺失块")
    args = parser.parse_args()
    audit_corpus(args.name, args.workers, args.stall_timeout)


if __name__ == "__main__":
    main()
