from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .corpora import PRETRAIN_ROOT, iter_corpus

CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def build_natural_index(source: str, max_residues: int = 50, limit: int | None = None):
    output = PRETRAIN_ROOT / "index" / f"natural_peptides.{source}.rows.npy"
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp.npy")
    chunks = []
    current = []
    stats = {"source": source, "input": 0, "kept": 0, "too_long": 0,
             "noncanonical": 0, "max_residues": max_residues}
    for record in iter_corpus(source, limit):
        stats["input"] += 1
        if len(record.text) > max_residues:
            stats["too_long"] += 1
            continue
        if not record.text or not set(record.text) <= CANONICAL_AA:
            stats["noncanonical"] += 1
            continue
        current.append(record.row)
        stats["kept"] += 1
        if len(current) >= 1_000_000:
            chunks.append(np.asarray(current, dtype=np.int64))
            current = []
    if current:
        chunks.append(np.asarray(current, dtype=np.int64))
    rows = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)
    np.save(tmp, rows)
    os.replace(tmp, output)
    stats["output"] = str(output)
    stats["index_records"] = len(rows)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-residues", type=int, default=50)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    summary = [build_natural_index(source, args.max_residues, args.limit)
               for source in ("uniref50", "pdb")]
    output = PRETRAIN_ROOT / "index" / "natural_peptides.stats.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
