"""MTR 描述符预计算（方案 §5.2(4)）：SMILES 语料 → RDKit 200+ 维描述符。

输入：corpora_clean/<corpus>.tsv（key<TAB>canonical_smiles）或官方 aux SMILES。
输出：data/pretrain/descriptors/<corpus>.desc.f32.npy（[N, D] float32，NaN→列中位数）
      + <corpus>.keys.txt（与行对齐的 key）
      + <corpus>.norm.json（训练集列均值/方差，供 z-score 归一化）
      + descriptor_names.txt（一次写出，全部语料共用同一描述符集）

大规模语料（如 PubChem 40M，单跑数小时）不再把结果堆在内存里：
结果增量写入 <name>.desc.f32.partial.npy（memmap）与 <name>.keys.txt.partial，
<name>.progress.json 原子记录输入行/输出行/keys 字节偏移；进程被杀后
重新执行同一命令即可从最近的 checkpoint 续跑，已完成行不会重算。
全部完成后才原子生成最终 .npy（按实际行数裁剪）、keys 与 norm。

用法：
  python -m muse.pretrain.compute_descriptors --input corpora_clean/pubchem.tsv --name pubchem
"""

import argparse
import itertools
import json
import os
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from muse.pretrain import deleak as D

OUT = D.PRETRAIN_DIR / "descriptors"

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

DESC_LIST = [(name, fn) for name, fn in Descriptors._descList]
NAMES_PATH = OUT / "descriptor_names.txt"
N_DESC = len(DESC_LIST)

FLUSH_ROWS = 200_000
CHECKPOINT_LINES = 1_000_000
BLOCK_ROWS = 1_000_000


def _worker(line):
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 2:
        return None
    key, smi = parts[0], parts[1]
    mol = Chem.MolFromSmiles(smi.replace("*", "C"))  # PSMILES 端基按 v0 约定替换
    if mol is None:
        return key, None  # 行保留：NaN 行，后续按列中位数修补，保证与语料逐行对齐
    vals = []
    for _name, fn in DESC_LIST:
        try:
            vals.append(float(fn(mol)))
        except Exception:
            vals.append(float("nan"))
    return key, vals


def _paths(name):
    return {
        "desc": OUT / f"{name}.desc.f32.npy",
        "keys": OUT / f"{name}.keys.txt",
        "norm": OUT / f"{name}.norm.json",
        "partial_desc": OUT / f"{name}.desc.f32.partial.npy",
        "partial_keys": OUT / f"{name}.keys.txt.partial",
        "progress": OUT / f"{name}.progress.json",
    }


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp) if os.path.exists(tmp) else None
        raise


def _count_lines(path):
    count = 0
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            count += block.count(b"\n")
    return count


def _flush(mm, keys_handle, keys_buf, rows_buf, output_rows):
    """把缓冲的行写入 memmap/keys 并落盘，返回新的 output_rows。"""
    if not rows_buf:
        return output_rows
    with np.errstate(over="ignore"):
        arr = np.asarray(rows_buf, dtype=np.float32)  # 溢出的 inf 由中位数修补
    if arr.shape[1] != N_DESC:
        raise RuntimeError(f"描述符列数 {arr.shape[1]} != {N_DESC}")
    mm[output_rows:output_rows + arr.shape[0]] = arr
    mm.flush()
    keys_handle.write("".join(f"{key}\n" for key in keys_buf))
    keys_handle.flush()
    os.fsync(keys_handle.fileno())
    keys_buf.clear()
    rows_buf.clear()
    return output_rows + arr.shape[0]


def _write_progress(path, progress):
    _atomic_json(path, progress)


def process(input_path, name, workers, chunk, total_lines=None, limit=None,
            stop_after_lines=None, checkpoint_lines=CHECKPOINT_LINES,
            block_rows=BLOCK_ROWS):
    """流式计算描述符；支持被杀后续跑。stop_after_lines 仅用于续跑测试。"""
    paths = _paths(name)
    if paths["desc"].exists() and paths["keys"].exists() and paths["norm"].exists():
        print(f"{name}: 最终产物已存在，跳过", flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if not NAMES_PATH.exists():
        NAMES_PATH.write_text("\n".join(n for n, _ in DESC_LIST) + "\n",
                              encoding="utf-8")
    src = Path(input_path)
    progress_path = paths["progress"]
    t0 = time.monotonic()

    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("done"):
            raise RuntimeError(f"{name}: progress 标记完成但最终产物缺失，请人工核对")
        alloc_rows = int(progress["alloc_rows"])
        output_rows = int(progress["output_rows"])
        skip_lines = int(progress["next_input_line"])
        n_bad = int(progress["n_bad"])
        keys_bytes = int(progress["keys_bytes"])
        with open(paths["partial_keys"], "r+b") as handle:
            handle.truncate(keys_bytes)
        print(f"{name}: 从 checkpoint 续跑，跳过 {skip_lines} 行，"
              f"已有 {output_rows} 行输出", flush=True)
    else:
        counted = total_lines if total_lines is not None else _count_lines(src)
        alloc_rows = min(counted, limit) if limit else counted
        if alloc_rows < 1:
            raise ValueError("输入为空")
        np.lib.format.open_memmap(paths["partial_desc"], mode="w+",
                                  dtype=np.float32, shape=(alloc_rows, N_DESC))
        open(paths["partial_keys"], "w", encoding="utf-8").close()
        progress = {}
        output_rows = skip_lines = n_bad = keys_bytes = 0
        print(f"{name}: 全新运行，预分配 {alloc_rows} 行 × {N_DESC} 维", flush=True)

    mm = np.lib.format.open_memmap(paths["partial_desc"], mode="r+")
    if mm.shape != (alloc_rows, N_DESC):
        raise RuntimeError(f"partial memmap shape {mm.shape} != {(alloc_rows, N_DESC)}")

    keys_buf, rows_buf = [], []
    seen = skip_lines
    last_checkpoint = seen
    with open(paths["partial_keys"], "a", encoding="utf-8", newline="") as keys_handle, \
            src.open(encoding="utf-8") as source, Pool(workers) as pool:
        lines = itertools.islice(source, skip_lines, None)
        for res in pool.imap(_worker, lines, chunksize=chunk):
            seen += 1
            if limit is not None and not progress and seen > alloc_rows:
                break
            if res is None:
                raise ValueError(f"第 {seen} 行缺少 key<TAB>smiles 结构，无法保持行对齐")
            key, vals = res
            if vals is None:
                n_bad += 1
                vals = [float("nan")] * N_DESC
            keys_buf.append(key)
            rows_buf.append(vals)
            if len(rows_buf) >= FLUSH_ROWS:
                output_rows = _flush(mm, keys_handle, keys_buf, rows_buf, output_rows)
            if seen - last_checkpoint >= checkpoint_lines:
                output_rows = _flush(mm, keys_handle, keys_buf, rows_buf, output_rows)
                _write_progress(progress_path, {
                    "done": False, "next_input_line": seen,
                    "output_rows": output_rows, "keys_bytes": keys_handle.tell(),
                    "n_bad": n_bad, "alloc_rows": alloc_rows,
                })
                last_checkpoint = seen
                el = time.monotonic() - t0
                done = seen - skip_lines
                print(f"{name}: {seen / 1e6:.0f}M, bad={n_bad}, "
                      f"{done / el / 1000:.0f}k/s", flush=True)
            if stop_after_lines is not None and seen - skip_lines >= stop_after_lines:
                output_rows = _flush(mm, keys_handle, keys_buf, rows_buf, output_rows)
                _write_progress(progress_path, {
                    "done": False, "next_input_line": seen,
                    "output_rows": output_rows, "keys_bytes": keys_handle.tell(),
                    "n_bad": n_bad, "alloc_rows": alloc_rows,
                })
                print(f"{name}: stop_after_lines={stop_after_lines} 触发，"
                      f"落盘后退出（未生成最终产物）", flush=True)
                return
        output_rows = _flush(mm, keys_handle, keys_buf, rows_buf, output_rows)

    n_rows = output_rows
    if n_rows < 1:
        raise RuntimeError(f"{name}: 没有任何有效输出")
    if n_rows > alloc_rows:
        raise RuntimeError(f"{name}: 输出行数 {n_rows} 超过预分配 {alloc_rows}")

    # NaN/Inf → 列中位数（全列无效则 0）。先扫描受影响列与行号，再逐列修补，
    # 避免一次性把全量数组读入内存。块尾部必须裁到 n_rows：partial memmap
    # 预分配 alloc_rows >= n_rows，越界部分从未写入。
    bad_by_col = {}
    for start in range(0, n_rows, block_rows):
        end = min(start + block_rows, n_rows)
        block = np.asarray(mm[start:end])
        bad_rows, bad_cols = np.nonzero(~np.isfinite(block))
        for col in np.unique(bad_cols):
            rows_here = bad_rows[bad_cols == col] + start
            bucket = bad_by_col.setdefault(int(col), [])
            bucket.append(rows_here)
    for col, chunks in bad_by_col.items():
        rows = np.concatenate(chunks)
        column = np.asarray(mm[:n_rows, col], dtype=np.float64)
        finite = np.isfinite(column)
        median = float(np.median(column[finite])) if finite.any() else 0.0
        mm[rows, col] = np.float32(median)
    if bad_by_col:
        mm.flush()

    # 归一化统计在修补后的最终数据上计算（float64 累加）。
    sums = np.zeros(N_DESC, dtype=np.float64)
    sumsqs = np.zeros(N_DESC, dtype=np.float64)
    for start in range(0, n_rows, block_rows):
        end = min(start + block_rows, n_rows)
        block = np.asarray(mm[start:end], dtype=np.float64)
        sums += block.sum(axis=0)
        sumsqs += np.square(block).sum(axis=0)
    mean = sums / n_rows
    var = np.maximum(sumsqs / n_rows - np.square(mean), 0.0)

    # 原子生成裁剪后的最终 .npy，再切换 keys/norm。
    fd, tmp_desc = tempfile.mkstemp(prefix=f".{paths['desc'].name}.", suffix=".tmp",
                                    dir=paths["desc"].parent)
    os.close(fd)
    os.unlink(tmp_desc)
    final = np.lib.format.open_memmap(tmp_desc, mode="w+", dtype=np.float32,
                                      shape=(n_rows, N_DESC))
    for start in range(0, n_rows, block_rows):
        end = min(start + block_rows, n_rows)
        final[start:end] = mm[start:end]
    final.flush()
    del final
    os.replace(tmp_desc, paths["desc"])
    os.replace(paths["partial_keys"], paths["keys"])
    paths["partial_desc"].unlink()

    norm = {"mean": mean.tolist(), "std": np.sqrt(var).tolist(),
            "n": int(n_rows), "n_bad": int(n_bad),
            "input_lines": int(seen),
            "elapsed_seconds": round(time.monotonic() - t0, 1)}
    D.write_json(paths["norm"], norm)
    _write_progress(progress_path, {
        "done": True, "next_input_line": seen, "output_rows": n_rows,
        "keys_bytes": 0, "n_bad": n_bad, "alloc_rows": alloc_rows,
    })
    print(json.dumps({k: v for k, v in norm.items()
                      if k not in ("mean", "std")}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--total-lines", type=int, default=None,
                    help="输入行数（已知则跳过预扫描）")
    ap.add_argument("--checkpoint-lines", type=int, default=CHECKPOINT_LINES)
    ap.add_argument("--block-rows", type=int, default=BLOCK_ROWS,
                    help="收尾阶段行块大小（测试用小值覆盖多块路径）")
    ap.add_argument("--stop-after-lines", type=int, default=None,
                    help="处理 N 行后落盘退出（仅用于续跑测试）")
    args = ap.parse_args()
    process(args.input, args.name, args.workers, args.chunk,
            total_lines=args.total_lines, limit=args.limit,
            stop_after_lines=args.stop_after_lines,
            checkpoint_lines=args.checkpoint_lines, block_rows=args.block_rows)


if __name__ == "__main__":
    main()
