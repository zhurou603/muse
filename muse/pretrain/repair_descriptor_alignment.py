"""修复 PubChem 描述符行对齐：描述符行数必须 == 语料行数（缺失行补中位数）。

旧流水线跳过解析失败行导致错位（train 侧按语料行号取描述符）。
本脚本：按 key 对齐语料与 keys.txt，找出被跳过的行号，插入列中位数行，
重写 keys.txt 为全量语料 key，重算 pubchem.norm.json。

用法：python -m muse.pretrain.repair_descriptor_alignment --name pubchem
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from muse.pretrain import deleak as D

OUT = D.PRETRAIN_DIR / "descriptors"
CORPUS = D.PRETRAIN_DIR / "corpora_clean"
N_DESC = 208
BLOCK = 1_000_000


def _atomic_text(path: Path):
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    return os.fdopen(fd, "w", encoding="utf-8", newline=""), tmp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    name = args.name
    desc_path = OUT / f"{name}.desc.f32.npy"
    keys_path = OUT / f"{name}.keys.txt"
    norm_path = OUT / f"{name}.norm.json"
    corpus_path = CORPUS / f"{name}.tsv"

    src = np.load(desc_path, mmap_mode="r")
    keys = keys_path.read_text(encoding="utf-8").split("\n")
    if keys and keys[-1] == "":
        keys.pop()
    n_src, n_keys = src.shape[0], len(keys)
    assert n_src == n_keys, (n_src, n_keys)
    n_bad_reported = json.loads(norm_path.read_text(encoding="utf-8"))["n_bad"]
    print(f"{name}: 描述符 {n_src} 行, keys {n_keys}, 报告坏行 {n_bad_reported}", flush=True)

    # pass 1：按 key 走齐，定位被跳过的语料行
    skip_rows = []
    ki = 0
    corpus_rows = 0
    with corpus_path.open(encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            key = line.split("\t", 1)[0]
            if ki < n_keys and key == keys[ki]:
                ki += 1
            else:
                skip_rows.append(row)
            corpus_rows = row + 1
    if ki != n_keys or len(skip_rows) != n_bad_reported:
        raise SystemExit(f"对齐失败：ki={ki} n_keys={n_keys} skip={len(skip_rows)} "
                         f"bad_reported={n_bad_reported}")
    print(f"语料 {corpus_rows} 行，跳过 {len(skip_rows)} 行", flush=True)

    # pass 2：列中位数（从现有有效行计算，一次一列）
    t0 = time.monotonic()
    medians = np.empty(N_DESC, dtype=np.float32)
    for j in range(N_DESC):
        medians[j] = float(np.median(np.asarray(src[:, j], dtype=np.float64)))
    print(f"中位数完成 {time.monotonic()-t0:.0f}s", flush=True)

    # pass 3：段式批量复制（跳过点只有 ~108 个，段间两侧连续）
    fd, tmp_desc = tempfile.mkstemp(prefix=".repair.", suffix=".npy", dir=OUT)
    os.close(fd)
    os.unlink(tmp_desc)
    out = np.lib.format.open_memmap(tmp_desc, mode="w+", dtype=np.float32,
                                    shape=(corpus_rows, N_DESC))
    sums = np.zeros(N_DESC, dtype=np.float64)
    sumsqs = np.zeros(N_DESC, dtype=np.float64)

    def accumulate(block):
        nonlocal sums, sumsqs
        block64 = np.asarray(block, dtype=np.float64)
        sums = sums + block64.sum(axis=0)
        sumsqs = sumsqs + np.square(block64).sum(axis=0)

    src_start = out_start = 0
    for skip in skip_rows:
        length = skip - out_start
        if length:
            out[out_start:skip] = src[src_start:src_start + length]
            accumulate(out[out_start:skip])
        out[skip] = medians
        accumulate(out[skip:skip + 1])
        src_start += length
        out_start = skip + 1
    if out_start < corpus_rows:
        out[out_start:corpus_rows] = src[src_start:n_src]
        accumulate(out[out_start:corpus_rows])
    if src_start + (corpus_rows - out_start) != n_src:
        raise RuntimeError(f"段式修复后行数不符: src 用至 {src_start + (corpus_rows - out_start)} != {n_src}")
    out.flush()
    del out
    os.replace(tmp_desc, desc_path)
    print(f"段式修复完成 {time.monotonic()-t0:.0f}s", flush=True)

    # keys.txt = 全量语料 key（与语料逐行对齐）
    keys_handle, tmp_keys = _atomic_text(keys_path)
    try:
        with corpus_path.open(encoding="utf-8") as handle:
            for line in handle:
                keys_handle.write(line.split("\t", 1)[0] + "\n")
        keys_handle.flush()
        os.fsync(keys_handle.fileno())
        os.replace(tmp_keys, keys_path)
    except BaseException:
        keys_handle.close()
        os.unlink(tmp_keys)
        raise

    mean = sums / corpus_rows
    var = np.maximum(sumsqs / corpus_rows - np.square(mean), 0.0)
    norm = {"mean": mean.tolist(), "std": np.sqrt(var).tolist(),
            "n": int(corpus_rows), "n_bad": 0,
            "note": "行保留修复版：解析失败行=列中位数，行数==语料行数",
            "elapsed_seconds": round(time.monotonic() - t0, 1)}
    D.write_json(norm_path, norm)
    print(json.dumps({k: v for k, v in norm.items() if k not in ("mean", "std")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
