from __future__ import annotations

import argparse
import json
from itertools import chain, islice
from pathlib import Path

from .corpora import CORPORA, PRETRAIN_ROOT, build_chemical_vocab, iter_corpus, vocab_coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PRETRAIN_ROOT / "vocab")
    parser.add_argument("--sample-records", type=int, default=5_000_000)
    parser.add_argument("--max-size", type=int, default=4096)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--coverage-limit", type=int, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    pubchem_n = min(4_500_000, args.sample_records)
    chembl_n = args.sample_records - pubchem_n
    chemical = chain(iter_corpus("pubchem", pubchem_n), iter_corpus("chembl", chembl_n))
    smiles = build_chemical_vocab(chemical, False, args.output / "smiles.json",
                                  args.min_freq, args.max_size)
    psmiles = build_chemical_vocab(iter_corpus("pi1m"), True, args.output / "psmiles.json",
                                   args.min_freq, args.max_size)

    chemical_eval = chain(iter_corpus("pubchem", args.coverage_limit),
                          iter_corpus("chembl", args.coverage_limit))
    coverage = {
        "smiles": vocab_coverage(chemical_eval, smiles["vocab"]),
        "psmiles": vocab_coverage(iter_corpus("pi1m", args.coverage_limit),
                                  psmiles["vocab"], polymer=True),
        "sample_records": args.sample_records,
        "corpora": {name: spec.records for name, spec in CORPORA.items()},
    }
    (args.output / "coverage.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(coverage, ensure_ascii=False, indent=2), flush=True)
    for name in ("smiles", "psmiles"):
        if coverage[name]["record_coverage"] < 0.995:
            raise SystemExit(f"{name} 词表记录覆盖率低于 99.5%")


if __name__ == "__main__":
    main()
