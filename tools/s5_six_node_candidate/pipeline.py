"""Chain a running 47-rank smoke to resume acceptance and formal training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PYTHON = "/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_state(path, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pid": os.getpid(), **fields}
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def wait_for_smoke(root, pid):
    path = root / "control/cluster_state.json"
    while True:
        state = read_json(path)
        if state.get("state") != "running":
            if state.get("state") == "completed" and state.get("mode") == "smoke-start":
                return Path(state["session"])
            raise RuntimeError(f"Initial smoke did not complete: {state}")
        if state.get("pid") != pid:
            raise RuntimeError(f"Initial smoke orchestrator changed: {state}")
        try:
            os.kill(pid, 0)
        except ProcessLookupError as exc:
            raise RuntimeError("Initial smoke orchestrator disappeared without a terminal state") from exc
        time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--stop-after", type=int, required=True)
    parser.add_argument("--source-git-commit", required=True)
    args = parser.parse_args()
    root = args.candidate.resolve()
    tools = root / "tools/s5_six_node_candidate"
    cluster = tools / "cluster.py"
    state_path = root / "control/pipeline_state.json"
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    base = read_json(root / "contract.json")["source_state"]["step"]
    own_pid = os.getpid()
    try:
        if (root / "runs/main").exists():
            raise RuntimeError("Formal main already exists; refusing a duplicate start")
        write_state(state_path, state="waiting_for_smoke", wait_pid=args.wait_pid,
                    source_git_commit=args.source_git_commit, pipeline_sha256=source_hash)
        wait_for_smoke(root, args.wait_pid)
        write_state(state_path, state="smoke_resume", source_git_commit=args.source_git_commit,
                    pipeline_sha256=source_hash)
        subprocess.run([PYTHON, "-B", str(cluster), "run", "--candidate", str(root),
                        "--mode", "smoke-resume", "--stop-after", str(base + 21),
                        "--max-attempts", "1"], cwd=root, check=True)
        subprocess.run([PYTHON, "-B", str(tools / "check_acceptance.py"),
                        "--candidate", str(root)], cwd=root, check=True)
        accepted = read_json(root / "acceptance.json")
        if accepted["status"] != "smoke_and_resume_pass" or accepted["ranks"] != 47:
            raise RuntimeError(f"Acceptance does not allow formal training: {accepted}")
        write_state(state_path, state="formal_training", acceptance=accepted,
                    source_git_commit=args.source_git_commit, pipeline_sha256=source_hash)
        subprocess.run([PYTHON, "-B", str(cluster), "run", "--candidate", str(root),
                        "--mode", "train-start", "--stop-after", str(args.stop_after),
                        "--max-attempts", "20"], cwd=root, check=True)
        write_state(state_path, state="completed", source_git_commit=args.source_git_commit,
                    pipeline_sha256=source_hash)
    except BaseException as exc:
        write_state(state_path, state="failed", error=f"{type(exc).__name__}: {exc}",
                    source_git_commit=args.source_git_commit, pipeline_sha256=source_hash)
        raise


if __name__ == "__main__":
    main()
