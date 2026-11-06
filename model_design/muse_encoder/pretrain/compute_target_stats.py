from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aux import AUX_SOURCES
from .corpora import PRETRAIN_ROOT
from .targets import compute_target_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", choices=sorted(AUX_SOURCES),
                        default=sorted(AUX_SOURCES))
    parser.add_argument("--output", type=Path,
                        default=PRETRAIN_ROOT / "targets" / "aux_targets.json")
    args = parser.parse_args()
    shadow_path = PRETRAIN_ROOT / "manifest" / "shadow_properties.json"
    shadow = set(json.loads(shadow_path.read_text(encoding="utf-8"))["properties"])
    payload = compute_target_stats(args.sources, args.output, shadow)
    print(json.dumps({
        "sources": args.sources,
        "tasks": len(payload["specs"]),
        "shadow_tasks": sum(spec["shadow"] for spec in payload["specs"].values()),
        "output": str(args.output),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
