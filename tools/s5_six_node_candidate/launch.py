"""Prepare one pinned launch session, then start that SAME session on all six nodes."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

from prepare_candidate import digest, require, write_json

NODES = [
    ("VM-2-17-tencentos", "172.17.2.17"),
    ("VM-2-184-tencentos", "172.17.2.184"),
    ("VM-2-46-tencentos", "172.17.2.46"),
    ("VM-1-84-tencentos", "172.17.1.84"),
    ("VM-0-171-tencentos", "172.17.0.171"),
    ("VM-0-178-tencentos", "172.17.0.178"),
]
PYTHON = "/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python"


def check_candidate(root):
    prepared = json.loads((root / "prepared.json").read_text())
    for relative, checksum in prepared["files"].items():
        require(digest(root / relative) == checksum, f"Candidate file changed: {relative}")
    contract = json.loads((root / "contract.json").read_text())
    require((root / "data").resolve() == Path(contract["repo_root"]) / "data", "Data link changed")
    return contract


def prepare(args):
    root = args.candidate.resolve()
    contract = check_candidate(root)
    base_step = contract["source_state"]["step"]
    smoke = args.mode.startswith("smoke-")
    start = args.mode.endswith("-start")
    run = root / "runs" / ("smoke" if smoke else "main")
    if not smoke:
        accepted = json.loads((root / "acceptance.json").read_text())
        require(accepted["contract_sha256"] == digest(root / "contract.json") and
                accepted["status"] == "smoke_and_resume_pass", "Smoke acceptance is missing/stale")
        require(args.stop_after is not None, "Training requires explicit absolute --stop-after")
    if start:
        require(not run.exists(), "Start requires a fresh run directory; use resume for an existing run")
        checkpoint = root / "handoff/source.pt"
        checkpoint_hash = contract["checkpoint_sha256"]
        step = base_step
    else:
        pointer = json.loads((run / "latest.json").read_text())
        checkpoint = Path(pointer["path"]).resolve()
        require(checkpoint.parent == run / "checkpoints", "Latest points outside this run")
        checkpoint_hash = pointer["sha256"]
        step = pointer["step"]
        if checkpoint.name == "final.pt":
            checkpoint = checkpoint.with_name(f"candidate_end_{step:08d}.pt")
            checkpoint_hash = digest(checkpoint)
        if smoke:
            require(step == base_step + 20, "Smoke resume must follow the completed 20-step smoke")
    require(digest(checkpoint) == checkpoint_hash, "Pinned checkpoint hash mismatch")
    stop = base_step + (20 if start else 21) if smoke else args.stop_after
    require(stop > step, "Stop step must be greater than checkpoint step")
    if start:
        run.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(root / "handoff/data_skips.jsonl", run / "data_skips.jsonl")
    session = root / "sessions" / f"{args.mode}_{time.time_ns()}.json"
    session.parent.mkdir(exist_ok=True)
    write_json(session, {"candidate": str(root), "mode": args.mode, "run_dir": str(run),
                         "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                         "resume_step": step, "stop_after": stop,
                         "contract_sha256": digest(root / "contract.json"),
                         "master_addr": NODES[0][1], "master_port": 29688})
    print(f"SESSION: {session}")
    print(f"48 ranks; 8 x 1 per rank; nominal batch=384; absolute steps {step}->{stop}")
    print("After coordinating all 48 cards, run on EACH node:")
    print(f"{PYTHON} -B {Path(__file__).resolve()} start --session {session}")


def start(args):
    session_path = args.session.resolve()
    session = json.loads(session_path.read_text())
    root = Path(session["candidate"])
    contract = check_candidate(root)
    require(digest(root / "contract.json") == session["contract_sha256"], "Session contract changed")
    require(digest(session["checkpoint"]) == session["checkpoint_sha256"], "Session checkpoint changed")
    hostname = socket.gethostname().split(".")[0]
    hosts = [value[0] for value in NODES]
    require(hostname in hosts, f"Unexpected host: {hostname}")
    rank = hosts.index(hostname)
    addresses = subprocess.check_output(["ip", "-4", "-o", "addr", "show", "dev", "eth0"], text=True)
    require(f"inet {NODES[rank][1]}/" in addresses, "Host/IP mapping mismatch")
    lock_path = Path("/tmp") / f"muse_s5_six_node_{os.getuid()}.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if rank == 0:
            with socket.socket() as probe:
                probe.bind((session["master_addr"], session["master_port"]))
        env = os.environ.copy()
        env.update({"REPO_ROOT": contract["repo_root"],
                    "PYTHONPATH": str(root / "model_design"), "PYTHONDONTWRITEBYTECODE": "1",
                    "OMP_NUM_THREADS": "1", "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
                    "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
                    "PYTORCH_ALLOC_CONF": "expandable_segments:True", "S5_PROCESS_PREFETCH": "1",
                    "S5_NCCL_TIMEOUT_SECONDS": "900", "NCCL_SOCKET_IFNAME": "=eth0",
                    "GLOO_SOCKET_IFNAME": "eth0", "NCCL_IB_GID_INDEX": "3",
                    "NCCL_IB_DISABLE": "0", "NCCL_NET": "IB", "NCCL_DEBUG": "INFO",
                    "NCCL_DEBUG_SUBSYS": "INIT,NET",
                    "NCCL_IB_HCA": "=mlx5_bond_0:1,mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1"})
        for name in ("ROCR_VISIBLE_DEVICES", "NCCL_DEBUG_FILE", "PYTHONHOME"):
            env.pop(name, None)
        command = [PYTHON, "-B", "-u", "-m", "torch.distributed.run", "--nnodes=6",
                   "--nproc_per_node=8", f"--node_rank={rank}", "--max_restarts=0",
                   "--rdzv_conf=timeout=900",
                   f"--master_addr={session['master_addr']}", f"--master_port={session['master_port']}",
                   "-m", "muse_encoder.s5.train", "--config", str(root / "config.yaml"),
                   "--run-dir", session["run_dir"], "--resume", session["checkpoint"],
                   "--resume-sha256", session["checkpoint_sha256"],
                   "--expansion-contract", str(root / "contract.json"),
                   "--stop-after", str(session["stop_after"]),
                   "--micro-batch", "8", "--accumulation-steps", "1", "--prefetch", "1"]
        log_dir = root / "logs" / session_path.stem
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / f"{hostname}_{time.time_ns()}.log"
        print(f"NODE {rank}/6 {hostname}; log={log}", flush=True)
        print(" ".join(command), flush=True)
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, env=env, cwd=root, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1,
                                       start_new_session=True)
            try:
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line, end="", flush=True)
                code = process.wait()
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                raise
        require(code == 0, f"Node failed exit={code}; inspect {log}; no automatic restart")
        print("NODE COMPLETE; all six nodes and saved checkpoints still need acceptance.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--candidate", type=Path, required=True)
    prep.add_argument("--mode", choices=("smoke-start", "smoke-resume", "train-start", "train-resume"), required=True)
    prep.add_argument("--stop-after", type=int)
    launch = commands.add_parser("start")
    launch.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.action == "prepare" else start)(args)


if __name__ == "__main__":
    main()
