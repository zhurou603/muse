from __future__ import annotations

import argparse
import json

from .aux import AUX_SOURCES
from .aux_index import AUX_INDEX_ROOT, AuxIndex, build_aux_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", choices=sorted(AUX_SOURCES),
                        default=sorted(AUX_SOURCES))
    args = parser.parse_args()
    summary = {}
    for name in args.sources:
        path = build_aux_index(name)
        index = AuxIndex(name, path)
        first = index.get(0)
        last = index.get(len(index) - 1)
        summary[name] = {"kept": len(index), "first": first.id, "last": last.id}
        index.close()
        print(name, summary[name], flush=True)
    AUX_INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    output = AUX_INDEX_ROOT / "summary.json"
    previous = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    previous.update(summary)
    output.write_text(json.dumps(previous, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
