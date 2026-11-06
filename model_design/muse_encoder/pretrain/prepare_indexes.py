from __future__ import annotations

import argparse
import json

from .corpora import CORPORA
from .indexed import IndexedTextCorpus, build_pi1m_view_cache, open_corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--skip-views", action="store_true")
    args = parser.parse_args()
    summary = {}
    for name in CORPORA:
        corpus = open_corpus(name)
        first = corpus.get(0)
        last = corpus.get(corpus.spec.records - 1)
        summary[name] = {"records": corpus.spec.records,
                         "first": first.key, "last": last.key}
        print(name, summary[name], flush=True)
    if not args.skip_views:
        spec = build_pi1m_view_cache(workers=args.workers)
        corpus = IndexedTextCorpus(spec)
        summary[spec.name] = {"records": spec.records,
                              "first": corpus.get(0).key,
                              "last": corpus.get(spec.records - 1).key}
        print(spec.name, summary[spec.name], flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
