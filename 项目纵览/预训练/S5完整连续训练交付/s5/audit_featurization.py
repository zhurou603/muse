"""Deterministic degraded/truncation audit for external chemical corpora."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from muse_encoder.config import MuseConfig
from muse_encoder.featurize import RecordFeaturizer
from muse_encoder.pretrain.corpora import CORPORA
from muse_encoder.pretrain.indexed import INDEX_ROOT, CorpusSpec, IndexedTextCorpus, open_corpus
from muse_encoder.tokenizers import load_tokenizers

from .provenance import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("pi1m", "pubchem", "chembl"), required=True)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-degraded-fraction", type=float, default=0.05)
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("limit 必须为正数")
    tokenizers = load_tokenizers()
    featurizer = RecordFeaturizer(MuseConfig(), tokenizers)
    if args.source == "pi1m":
        records = CORPORA["pi1m"].records * 5
        corpus = IndexedTextCorpus(CorpusSpec("pi1m_views", INDEX_ROOT / "pi1m_views.tsv",
                                              "augmented_tsv", records))
    else:
        records = CORPORA[args.source].records
        corpus = open_corpus(args.source)
    rows = np.unique(np.linspace(0, records - 1, min(args.limit, records), dtype=np.int64))
    stats = {"source": args.source, "population": records, "sampled": 0, "degraded": 0,
             "truncated": 0, "errors": 0, "tokens": 0, "max_tokens": 0,
             "selection": "unique evenly spaced row/view indices"}
    with args.output.with_suffix(".failures.jsonl").open("w", encoding="utf-8") as failures:
        for row in rows:
            text = corpus.get(int(row))
            try:
                if args.source == "pi1m":
                    record = featurizer.featurize_polymer(psmiles=text.text, sequence=True, strict=False)
                else:
                    record = featurizer.featurize_molecule(text.text, sequence=True)
                stats["sampled"] += 1
                degraded = bool(record.meta.get("degraded"))
                truncated = bool(record.meta.get("truncated"))
                stats["degraded"] += int(degraded)
                stats["truncated"] += int(truncated)
                stats["tokens"] += int(record.n_real_tokens)
                stats["max_tokens"] = max(stats["max_tokens"], int(record.n_real_tokens))
                if degraded or truncated:
                    failures.write(json.dumps({"row": int(row), "key": text.key,
                                               "degraded": degraded, "truncated": truncated}) + "\n")
            except Exception as exc:
                stats["errors"] += 1
                failures.write(json.dumps({"row": int(row), "key": text.key,
                                           "error": f"{type(exc).__name__}: {str(exc)[:500]}"}) + "\n")
    denominator = max(stats["sampled"], 1)
    stats["degraded_fraction"] = stats["degraded"] / denominator
    stats["truncated_fraction"] = stats["truncated"] / denominator
    stats["error_fraction"] = stats["errors"] / max(len(rows), 1)
    stats["mean_tokens"] = stats["tokens"] / denominator
    stats["passed"] = stats["degraded_fraction"] <= args.max_degraded_fraction \
        and stats["truncated"] == 0 and stats["errors"] == 0
    atomic_json(args.output, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if not stats["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
