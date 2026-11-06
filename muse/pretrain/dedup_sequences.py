"""序列语料去泄漏：UniRef50 / PDB seqres / UniParc 短肽。

判定：完全匹配 A 榜 val/test 序列，或估计 10-mer 覆盖率 ≥0.5（保守阈值，
见 deleak.py）→ 剔除并记录。输出清洗后 FASTA + 剔除日志 + 统计 JSON。
UniParc 额外执行长度/标准氨基酸过滤和固定种子蓄水池抽样。

用法：
  python -m muse.pretrain.dedup_sequences uniref50 [--workers 64]
  python -m muse.pretrain.dedup_sequences pdb
  python -m muse.pretrain.dedup_sequences uniparc_le50 --max-residues 50 \
      --canonical-only --sample 8000000 --seed 42
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

_G = {}


def _init_worker(exact, kmers):
    _G["exact"] = exact
    _G["kmers"] = kmers


def _seq_worker(item):
    rid, seq = item
    seq = seq.strip().upper()
    if not seq:
        return rid, seq, "empty"
    if seq in _G["exact"]:
        return rid, seq, "exact_match"
    n = len(seq)
    if n < D.KMER:
        return rid, seq, "short_peptide_hit" if seq in _G["kmers"] else None
    hits = total = 0
    for i in range(0, n - D.KMER + 1, D.KMER_STRIDE):
        total += 1
        if seq[i:i + D.KMER] in _G["kmers"]:
            hits += 1
    frac = hits / total if total else 0.0
    if frac >= D.KMER_HIT_FRAC:
        return rid, seq, f"kmer_cover_{frac:.2f}"
    return rid, seq, None


def iter_fasta(path):
    rid, buf = None, []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(">"):
                if rid is not None:
                    yield rid, "".join(buf)
                rid = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if rid is not None:
        yield rid, "".join(buf)


def iter_parquet_dir(path):
    """目录下全部 parquet 分卷；自动识别 id/sequence 列。"""
    import pyarrow.parquet as pq
    files = sorted(Path(path).glob("*.parquet"))
    if not files:
        raise SystemExit(f"目录无 parquet 文件: {path}")
    cols = pq.read_schema(files[0]).names
    seq_col = next((c for c in cols if "seq" in c.lower()), None)
    if seq_col is None:
        raise SystemExit(f"找不到序列列: {cols}")
    id_col = next((c for c in cols if c.lower() in
                   ("id", "uniref_id", "accession", "name", "repid")), None)
    print(f"parquet 列: {cols} -> id={id_col} seq={seq_col}", flush=True)
    for fp in files:
        t = pq.read_table(fp)
        ids = t[id_col].to_pylist() if id_col else [f"{fp.stem}:{i}"
                                                    for i in range(t.num_rows)]
        for rid, seq in zip(ids, t[seq_col].to_pylist()):
            if seq:
                yield str(rid), str(seq)
        print(f"  {fp.name}: 累计读取", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", choices=["uniref50", "uniref50_hf", "pdb", "uniparc_le50"])
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", type=int, default=None,
                    help="去泄漏和过滤后的蓄水池抽样条数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-residues", type=int, default=None)
    ap.add_argument("--canonical-only", action="store_true")
    args = ap.parse_args()
    if args.sample is not None and args.sample < 1:
        ap.error("--sample 必须为正数")
    if args.max_residues is not None and args.max_residues < 1:
        ap.error("--max-residues 必须为正数")

    src = {"uniref50": CORPORA / "uniref50.fasta.gz",
           "uniref50_hf": CORPORA / "uniref50_hf",
           "pdb": CORPORA / "pdb_seqres.txt.gz",
           "uniparc_le50": CORPORA / "uniparc_le50_2026_03.fasta.gz"}[args.corpus]
    if not src.exists():
        raise SystemExit(f"语料文件不存在: {src}")
    record_iter = iter_parquet_dir(src) if src.is_dir() else iter_fasta(src)
    out_name = {"uniref50_hf": "uniref50", "uniparc_le50": "uniparc_peptides"}.get(
        args.corpus, args.corpus)

    exact, kmers = D.load_sequence_fingerprints()
    print(f"指纹: {len(exact)} 序列, {len(kmers)} {D.KMER}-mers", flush=True)

    CLEAN.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    out_path = CLEAN / f"{out_name}.fasta"
    rej_path = AUDIT / f"{out_name}.removed.jsonl"
    stat_path = AUDIT / f"{out_name}.stats.json"

    t0 = time.monotonic()
    n_in = n_kept = n_leak = n_filtered = 0
    reasons = {}
    filter_reasons = {}
    canonical = frozenset("ACDEFGHIKLMNPQRSTVWY")
    rng = random.Random(args.seed)
    reservoir = []
    with Pool(args.workers, initializer=_init_worker,
              initargs=(exact, kmers)) as pool, \
            out_path.open("w", encoding="utf-8") as out, \
            rej_path.open("w", encoding="utf-8") as rej:
        for rid, seq, verdict in pool.imap(
                _seq_worker, record_iter, chunksize=args.chunk):
            n_in += 1
            if args.limit and n_in > args.limit:
                break
            filter_reason = None
            if args.max_residues is not None and len(seq) > args.max_residues:
                filter_reason = "too_long"
            elif args.canonical_only and not set(seq) <= canonical:
                filter_reason = "noncanonical"
            if verdict:
                n_leak += 1
                reasons[verdict.split("_cover")[0]] = \
                    reasons.get(verdict.split("_cover")[0], 0) + 1
                rej.write(json.dumps({"id": rid, "reason": verdict,
                                      "length": len(seq)}) + "\n")
            elif filter_reason:
                n_filtered += 1
                filter_reasons[filter_reason] = filter_reasons.get(filter_reason, 0) + 1
                rej.write(json.dumps({"id": rid, "reason": filter_reason,
                                      "length": len(seq)}) + "\n")
            else:
                n_kept += 1
                if args.sample:
                    if len(reservoir) < args.sample:
                        reservoir.append((rid, seq))
                    else:
                        index = rng.randrange(n_kept)
                        if index < args.sample:
                            reservoir[index] = (rid, seq)
                else:
                    out.write(f">{rid}\n{seq}\n")
            if n_in % 1_000_000 == 0:
                el = time.monotonic() - t0
                print(f"{args.corpus}: {n_in/1e6:.0f}M in, kept={n_kept/1e6:.2f}M, "
                      f"leak={n_leak}, filtered={n_filtered}, "
                      f"{n_in/el/1000:.0f}k/s", flush=True)
    n_sampled = n_kept
    if args.sample:
        n_sampled = len(reservoir)
        rng.shuffle(reservoir)
        with out_path.open("w", encoding="utf-8") as out:
            for rid, seq in reservoir:
                out.write(f">{rid}\n{seq}\n")
    stats = {"corpus": args.corpus, "source_file": src.name,
             "input_records": n_in, "kept": n_kept, "sampled_out": n_sampled,
             "removed": n_leak, "filtered": n_filtered,
             "removal_reasons": reasons, "filter_reasons": filter_reasons,
             "max_residues": args.max_residues,
             "canonical_only": args.canonical_only,
             "sample_seed": args.seed if args.sample else None,
             "kmer": D.KMER, "kmer_stride": D.KMER_STRIDE,
             "kmer_hit_frac_threshold": D.KMER_HIT_FRAC,
             "elapsed_seconds": round(time.monotonic() - t0, 1)}
    D.write_json(stat_path, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
