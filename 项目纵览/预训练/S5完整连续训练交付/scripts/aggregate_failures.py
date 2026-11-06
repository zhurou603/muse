"""Aggregate historical fallback/failure JSONL without changing source records."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = collections.Counter()
    examples = {}
    total = 0
    for path in args.paths:
        files = list(path.rglob("*.jsonl")) if path.is_dir() else [path]
        for file in files:
            with file.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    total += 1
                    reason = row.get("reason") or row.get("error") or row.get("detail") \
                        or row.get("status") or "unclassified"
                    reason = str(reason).splitlines()[0][:300]
                    source = row.get("source") or row.get("dataset") or row.get("subset") or "unknown"
                    key = (str(source), reason)
                    counts[key] += 1
                    examples.setdefault(key, {"file": str(file), "line": line_number,
                                              "id": row.get("id")})
    payload = {
        "total_records": total,
        "groups": [{"source": source, "reason": reason, "count": count,
                    "example": examples[(source, reason)]}
                   for (source, reason), count in counts.most_common()],
        "note": "这是记录级审计聚合，不等于 90/90 预测缺失；需另行计算 encoder coverage。",
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"total_records": total, "groups": len(counts), "output": str(args.output)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
