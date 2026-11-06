from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from s4_common import domain_source_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config = json.loads((args.run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    counts = Counter()
    fetch_seconds = []
    stream_files = sorted(args.run_dir.glob("data_stream.rank*.jsonl"))
    for path in stream_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            counts.update(row["source_counts"])
            fetch_seconds.append(float(row["fetch_seconds"]))
    if not counts:
        raise ValueError("没有数据流日志")
    forbidden = [name for name in counts if name == "omc25" or name.startswith("OMC25/")]
    if forbidden:
        raise ValueError(f"晶体训练源被读取: {forbidden}")
    total = sum(counts.values())
    expected = config["source_weights"]
    realized = {name: count / total for name, count in sorted(counts.items())}
    train_rows = []
    train_path = args.run_dir / "train.jsonl"
    if train_path.exists():
        train_rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    for row in train_rows:
        if not all(math.isfinite(value) for value in row["losses"].values()):
            raise ValueError(f"非有限训练 loss: step={row['step']}")
    result = {
        "schema_version": "muse.s4.training_log_audit.v1",
        "run_dir": str(args.run_dir),
        "mode": config["mode"],
        "total_samples": total,
        "source_counts": dict(sorted(counts.items())),
        "configured_weights": expected,
        "realized_weights": realized,
        "domain_counts": domain_source_weights(dict(counts)),
        "max_absolute_sampling_deviation": max(
            abs(realized.get(name, 0.0) - weight) for name, weight in expected.items()),
        "stream_files": [str(path) for path in stream_files],
        "micro_batches": len(fetch_seconds),
        "fetch_seconds_mean": sum(fetch_seconds) / len(fetch_seconds),
        "fetch_seconds_max": max(fetch_seconds),
        "train_log_rows": len(train_rows),
        "last_train_log": train_rows[-1] if train_rows else None,
        "crystal_sources_read": [],
        "all_losses_finite": True if train_rows else None,
    }
    output = args.out or args.run_dir / "log_audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "total_samples", "domain_counts", "max_absolute_sampling_deviation",
        "micro_batches", "train_log_rows", "all_losses_finite")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
