from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from muse_encoder.pretrain.dashboard import load_checkpoint
from muse_encoder.pretrain.trainer import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖导出 checkpoint: {args.output}")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = dict(payload["config"])
    model_config = dict(config["model"])
    state = {key: value for key, value in payload["model"].items()
             if not key.startswith("heads.aux_heads.")}
    removed = sorted(key for key in payload["model"] if key.startswith("heads.aux_heads."))
    model_config["aux_specs"] = {}
    config["model"] = model_config
    exported = {
        "model": state,
        "config": config,
        "state": {
            "step": payload.get("state", {}).get("step"),
            "source_checkpoint": str(args.checkpoint.resolve()),
            "source_sha256": sha256_file(args.checkpoint),
            "export": "inference-only; all auxiliary heads and optimizer/scheduler state removed",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(exported, temporary)
    os.replace(temporary, args.output)
    encoder, cfg, info = load_checkpoint(args.output, "cpu")
    if cfg.aux_specs or any(key.startswith("heads.aux_heads.") for key in encoder.state_dict()):
        raise RuntimeError("导出后仍包含辅助头")
    result = {
        "checkpoint": str(args.output.resolve()),
        "sha256": info["sha256"],
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_sha256": exported["state"]["source_sha256"],
        "step": exported["state"]["step"],
        "removed_auxiliary_parameters": len(removed),
        "optimizer_removed": True,
        "scheduler_removed": True,
        "cpu_strict_load": True,
    }
    result_path = args.output.with_suffix(".export.json")
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
