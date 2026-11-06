"""Sequential asynchronous evaluation watcher; never writes the training run."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .provenance import atomic_json
from .runtime import load_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-run", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    evaluation = config["evaluation"]
    history_path = args.evaluation_root / "watcher.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {
        "sentinel_sha256": [], "full_sha256": [], "last_full_time": 0.0,
    }
    while True:
        latest_path = args.train_run / "latest.json"
        if latest_path.exists():
            try:
                latest = json.loads(latest_path.read_text(encoding="utf-8"))
                step = int(latest["step"])
                sha = latest["sha256"]
                now = time.time()
                sentinel_due = step >= int(evaluation.get("sentinel_interval_steps", 100)) \
                    and step % int(evaluation.get("sentinel_interval_steps", 100)) == 0
                full_due = step >= int(evaluation.get("full_min_steps", 200)) and (
                    now - float(history.get("last_full_time", 0))
                    >= int(evaluation.get("full_interval_seconds", 86400)))
                mode = "full" if full_due else ("sentinel" if sentinel_due else None)
                seen_key = f"{mode}_sha256" if mode else None
                if mode and sha not in history.setdefault(seen_key, []):
                    command = [
                        sys.executable, "-m", "muse_encoder.s5.evaluate_candidate",
                        "--config", str(args.config), "--checkpoint", str(latest_path),
                        "--evaluation-root", str(args.evaluation_root), "--mode", mode,
                        "--device", args.device, "--state", str(args.state),
                    ]
                    completed = subprocess.run(command, check=False)
                    history[seen_key].append(sha)
                    history.setdefault("events", []).append({
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "mode": mode,
                        "step": step, "sha256": sha, "returncode": completed.returncode,
                    })
                    if mode == "full":
                        history["last_full_time"] = now
                    atomic_json(history_path, history)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                # A malformed/latest transient must not kill the training process.
                print(f"[evaluation-watcher] {type(exc).__name__}: {exc}", flush=True)
        if args.once:
            return
        time.sleep(max(1, args.poll_seconds))


if __name__ == "__main__":
    main()
