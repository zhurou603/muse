"""分子语料去泄漏：PubChem / ChEMBL / PI1M。

流程：并行计算 (canonical SMILES, InChIKey) → 命中 A 榜 val/test 指纹者剔除并记录
→ 可选固定种子抽样到目标规模 → 写出清洗后语料 + 剔除日志 + 统计 JSON。

用法：
  python -m muse.pretrain.dedup_molecules pubchem [--sample 40000000]
  python -m muse.pretrain.dedup_molecules chembl
  python -m muse.pretrain.dedup_molecules pi1m
"""

import argparse
import gzip
import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from muse.pretrain import deleak as D

CORPORA = D.PRETRAIN_DIR / "corpora"
CLEAN = D.PRETRAIN_DIR / "corpora_clean"
AUDIT = D.PRETRAIN_DIR / "audit"

# 并行 worker 用的全局指纹（fork 继承，只读）
_G = {}


def _init_worker(iks, cans):
    _G["iks"] = iks
    _G["cans"] = cans


def _mol_worker(item):
    """item=(line_no, key, smiles) → (line_no, key, can, ik, leak)。"""
    line_no, key, smi = item
    try:
        can, ik = D.mol_fingerprint(smi)
    except Exception:
        return line_no, key, None, None, "rdkit_error"
    if can is None:
        return line_no, key, None, None, "unparseable"
    leak = (ik is not None and ik in _G["iks"]) or can in _G["cans"]
    return line_no, key, can, ik, leak


def iter_pubchem(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                yield i, parts[0], parts[1]


def iter_chembl(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        cols = header.rstrip("\n").split("\t")
        try:
            ci, cs = cols.index("chembl_id"), cols.index("canonical_smiles")
        except ValueError:
            raise SystemExit(f"chemreps 列名异常: {cols}")
        for i, line in enumerate(f, start=2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) > max(ci, cs):
                yield i, parts[ci], parts[cs]


def iter_pi1m(path):
    import csv
    with Path(path).open(encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        lower = [c.strip().lower() for c in header]
        # PI1M_v2: 第一列常为 PSMILES 或 smiles
        si = next((i for i, c in enumerate(lower) if "smiles" in c), 0)
        for i, row in enumerate(reader, start=2):
            if len(row) > si and row[si].strip():
                yield i, row[si].strip(), row[si].strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", choices=["pubchem", "chembl", "pi1m"])
    ap.add_argument("--sample", type=int, default=None,
                    help="清洗后固定种子抽样条数（默认全保留）")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None, help="调试：只处理前 N 条")
    args = ap.parse_args()

    src = {"pubchem": CORPORA / "pubchem_cid_smiles.gz",
           "chembl": CORPORA / "chembl_37_chemreps.txt.gz",
           "pi1m": CORPORA / "PI1M_v2.csv"}[args.corpus]
    if not src.exists():
        raise SystemExit(f"语料文件不存在: {src}")
    it = {"pubchem": iter_pubchem, "chembl": iter_chembl,
          "pi1m": iter_pi1m}[args.corpus](src)

    iks, cans = D.load_molecule_fingerprints()
    print(f"指纹: {len(iks)} inchikey, {len(cans)} canonical", flush=True)

    CLEAN.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    out_path = CLEAN / f"{args.corpus}.tsv"
    rej_path = AUDIT / f"{args.corpus}.removed.jsonl"
    stat_path = AUDIT / f"{args.corpus}.stats.json"

    t0 = time.monotonic()
    n_in = n_kept = n_leak = n_bad = 0
    rng = random.Random(42)
    reservoir = []           # 固定种子蓄水池抽样（仅在 --sample 时使用）
    with Pool(args.workers, initializer=_init_worker,
              initargs=(iks, cans)) as pool, \
            out_path.open("w", encoding="utf-8") as out, \
            rej_path.open("w", encoding="utf-8") as rej:
        for line_no, key, can, ik, leak in pool.imap(
                _mol_worker, it, chunksize=args.chunk):
            n_in += 1
            if args.limit and n_in > args.limit:
                break
            if leak in ("unparseable", "rdkit_error"):
                n_bad += 1
                rej.write(json.dumps({"line": line_no, "key": key,
                                      "reason": leak}) + "\n")
            elif leak:
                n_leak += 1
                rej.write(json.dumps({"line": line_no, "key": key,
                                      "reason": "valtest_fingerprint_hit",
                                      "inchikey": ik}) + "\n")
            else:
                n_kept += 1
                if args.sample:
                    if len(reservoir) < args.sample:
                        reservoir.append((key, can))
                    else:
                        j = rng.randrange(n_kept)
                        if j < args.sample:
                            reservoir[j] = (key, can)
                else:
                    out.write(f"{key}\t{can}\n")
            if n_in % 2_000_000 == 0:
                el = time.monotonic() - t0
                print(f"{args.corpus}: {n_in/1e6:.0f}M in, kept={n_kept/1e6:.2f}M, "
                      f"leak={n_leak}, bad={n_bad}, {n_in/el/1000:.0f}k/s", flush=True)
    n_sampled = n_kept
    if args.sample:
        n_sampled = len(reservoir)
        rng.shuffle(reservoir)
        with out_path.open("w", encoding="utf-8") as out:
            for key, can in reservoir:
                out.write(f"{key}\t{can}\n")
    stats = {"corpus": args.corpus, "source_file": src.name,
             "input_records": n_in, "kept": n_kept, "sampled_out": n_sampled,
             "leak_removed": n_leak, "unparseable": n_bad,
             "elapsed_seconds": round(time.monotonic() - t0, 1)}
    D.write_json(stat_path, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
