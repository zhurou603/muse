"""Detached six-node lifecycle: probe, communicate, train, stop and inspect."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
import uuid

from launch import NODES, PYTHON, check_candidate
from prepare_candidate import require


def ssh_options():
    values = ["-o", "ConnectTimeout=10",
                  "-o", f"BatchMode={'no' if os.environ.get('S5_SSH_PASSWORD_AUTH') else 'yes'}"]
    values += shlex.split(os.environ.get("S5_SSH_OPTIONS", ""))
    if os.environ.get("S5_KNOWN_HOSTS"):
        values += ["-o", f"UserKnownHostsFile={os.environ['S5_KNOWN_HOSTS']}",
                   "-o", "StrictHostKeyChecking=yes"]
    return values


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


class Cluster:
    def __init__(self, root):
        self.root = root.resolve()
        self.tools = self.root / "tools/s5_six_node_candidate"
        self.contract = check_candidate(self.root)
        self.local = socket.gethostname().split(".")[0]
        require(self.local in dict(NODES), f"Unexpected controller host: {self.local}")

    def ssh(self, ip, command, timeout=30):
        return subprocess.run(["ssh", *ssh_options(), ip, command],
                              capture_output=True, text=True, timeout=timeout)

    def probe(self, args):
        def one(item):
            host, ip = item
            command = [PYTHON, "-B", str(self.tools / "node_probe.py")]
            if host == self.local:
                result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            else:
                result = self.ssh(ip, shlex.join(command), timeout=135)
            if result.returncode:
                return {"host": host, "ok": False, "error": result.stderr[-2000:]}
            value = json.loads(result.stdout)
            value["ok"] = (len(value["devices"]) == 8 and not value["s5_training_processes"] and
                           all(row["free_mib"] >= args.require_free_mib for row in value["devices"]))
            return value
        with ThreadPoolExecutor(max_workers=6) as executor:
            rows = list(executor.map(one, NODES))
        report = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "required_free_mib": args.require_free_mib, "ok": all(r["ok"] for r in rows),
                  "nodes": rows}
        atomic_json(self.root / "preflight.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return 0 if report["ok"] else 2

    def comm(self, _args):
        label = time.strftime("%Y%m%d_%H%M%S")
        log_dir = self.root / "logs" / f"communication_{label}"
        log_dir.mkdir(parents=True, exist_ok=False)
        script = self.root.parent / "start_cluster_comm.sh"
        def one(item):
            host, ip = item
            command = ["bash", str(script), "--six-nodes"]
            if host != self.local:
                command = ["ssh", *ssh_options(), ip, shlex.join(command)]
            with (log_dir / f"{host}.log").open("w", encoding="utf-8") as stream:
                try:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, timeout=660)
                    return {"host": host, "returncode": result.returncode}
                except subprocess.TimeoutExpired:
                    return {"host": host, "returncode": 124}
        with ThreadPoolExecutor(max_workers=6) as executor:
            rows = list(executor.map(one, NODES))
        for row in rows:
            text = (log_dir / f"{row['host']}.log").read_text(errors="replace")
            row["roce_observed"] = "NET/IB" in text
            row["rank_pass_count"] = text.count("RANK PASS:")
        master = (log_dir / f"{NODES[0][0]}.log").read_text(errors="replace")
        ok = (all(row["returncode"] == 0 and row["roce_observed"] and row["rank_pass_count"] == 8
                   for row in rows) and "DDP COMMUNICATION PASS: 48 ranks" in master)
        report = {"ok": ok, "nodes": rows, "log_dir": str(log_dir),
                  "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        atomic_json(self.root / "communication_acceptance.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return 0 if ok else 2

    def start_nodes(self, session):
        nodes = []
        for host, ip in NODES:
            log = self.root / "logs" / "drivers" / f"{session.stem}.{host}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            if host == self.local:
                stream = log.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    [PYTHON, "-B", str(self.tools / "launch.py"), "start", "--session", str(session)],
                    stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    start_new_session=True)
                pid = process.pid
            else:
                command = (f"setsid {shlex.quote(PYTHON)} -B {shlex.quote(str(self.tools / 'launch.py'))} "
                           f"start --session {shlex.quote(str(session))} > {shlex.quote(str(log))} "
                           "2>&1 < /dev/null & echo $!")
                result = self.ssh(ip, command)
                if result.returncode or not result.stdout.strip().isdigit():
                    raise RuntimeError(f"Cannot start {host}: {result.stdout} {result.stderr}")
                pid = int(result.stdout.strip())
            nodes.append({"host": host, "ip": ip, "pid": pid, "session": session.stem, "log": str(log)})
        atomic_json(self.root / "control" / f"{session.stem}.drivers.json", nodes)
        return nodes

    def node_status(self, node):
        path = self.root / "control" / "drivers" / f"{node['session']}.{node['host']}.json"
        state = read_json(path, {"state": "unknown"})
        if state.get("session") != node["session"]:
            return {"state": "unknown", "path": str(path)}
        if state.get("state") in ("completed", "failed"):
            return state
        result = (subprocess.run(["ps", "-p", str(node["pid"]), "-o", "stat="],
                                 capture_output=True, text=True) if node["host"] == self.local
                  else self.ssh(node["ip"], f"ps -p {int(node['pid'])} -o stat="))
        if result.returncode or not result.stdout.strip() or result.stdout.strip().startswith("Z"):
            return {**state, "state": "failed", "returncode": None,
                    "error": "Driver exited without a complete status record"}
        return state

    def terminate_nodes(self, nodes):
        for node in nodes:
            if self.node_status(node).get("state") in ("completed", "failed"):
                continue
            if node["host"] == self.local:
                subprocess.run(["kill", "-TERM", str(node["pid"])], capture_output=True)
            else:
                self.ssh(node["ip"], f"kill -TERM {int(node['pid'])}")
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            states = [self.node_status(node) for node in nodes]
            if all(row.get("state") in ("completed", "failed") for row in states):
                return states
            time.sleep(5)
        raise RuntimeError("Drivers did not stop within 120 seconds; no PIDs were pattern-killed")

    def start_health(self, session):
        session_data = read_json(session)
        command = [PYTHON, "-B", "-m", "muse_encoder.s5.health",
                   "--run-dir", session_data["run_dir"], "--world-size", "48",
                   "--target-step", str(session_data["stop_after"]), "--session", session.stem,
                   "--poll-seconds", "15", "--stale-after", "60", "--no-progress-after", "900",
                   "--startup-grace", "1800", "--checkpoint-grace", "3600", "--gpu", "--dump-stacks"]
        for host, ip in NODES:
            command += ["--host", f"{host}={ip}"]
        path = self.root / "logs" / "health" / f"{session.stem}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "PYTHONPATH": str(self.root / "model_design")}
        stream = path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        return process

    def summary(self, session):
        data = read_json(session)
        path = Path(data["run_dir"]) / "health" / "sessions" / f"{session.stem}.summary.json"
        return read_json(path, {"status": "starting", "issues": []})

    def retry_allowed(self, session):
        text = ""
        for path in (self.root / "logs" / session.stem).glob("*.log"):
            text += path.read_text(errors="replace")[-100000:]
        fatal = ("Candidate file changed", "source drift", "Checkpoint/config mismatch",
                 "checkpoint SHA256", "Pinned checkpoint", "manifest", "out of memory",
                 "OutOfMemoryError", "非有限", "nonfinite", "RNG checkpoint")
        return not any(marker.lower() in text.lower() for marker in fatal)

    def run(self, args):
        require(args.mode == "train-resume" or not (self.root / "runs/main").exists(),
                "main already exists: explicit train-resume is required")
        control = self.root / "control"
        state_path = control / "cluster_state.json"
        previous = read_json(state_path, {})
        if previous.get("state") == "running" and previous.get("pid") != os.getpid():
            result = subprocess.run(["ps", "-p", str(previous["pid"]), "-o", "args="],
                                    capture_output=True, text=True)
            if "s5_six_node_candidate/cluster.py" in result.stdout:
                raise RuntimeError(f"Another orchestrator is active: {previous['pid']}")
        if args.acknowledge_stop:
            pending = self.root / "runs/main/control/stop.json"
            ack = read_json(self.root / "runs/main/control/stop_ack.json", {})
            request = read_json(pending, {})
            require(request.get("request_id") and request["request_id"] == ack.get("request_id"),
                    "Stop is not acknowledged with a complete 48-rank checkpoint")
            destination = self.root / "runs/main/control/stop_history" / f"{request['request_id']}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            pending.rename(destination)
        mode = args.mode
        failures = 0
        process = None
        pending_skip = None
        for attempt in range(1, args.max_attempts + 1):
            require(not (self.root / "runs/main/control/stop.json").exists(),
                    "A coordinated stop is pending; inspect it before another start")
            session = __import__("launch").prepare(argparse.Namespace(
                candidate=self.root, mode=mode, stop_after=args.stop_after))
            session_data = read_json(session)
            if pending_skip is not None:
                inherited = (self.root / "handoff/data_skips.jsonl").read_bytes()
                require(pending_skip.startswith(inherited), "Failed-attempt skip log is not an append-only extension")
                for line in pending_skip[len(inherited):].splitlines():
                    event = json.loads(line)
                    require(event["event"] == "training_input_skipped" and int(event["row"]) >= 0,
                            "Invalid failed-attempt skip record")
                (self.root / "runs/main/data_skips.jsonl").write_bytes(pending_skip)
                pending_skip = None
            atomic_json(state_path, {"state": "running", "pid": os.getpid(), "session": str(session),
                                     "attempt": attempt, "mode": mode,
                                     "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            print(f"CLUSTER ATTEMPT {attempt}: {session}", flush=True)
            nodes = self.start_nodes(session)
            process = self.start_health(session)
            stalled_since = None
            try:
                while True:
                    states = [self.node_status(node) for node in nodes]
                    if any(row.get("state") == "failed" for row in states):
                        self.terminate_nodes(nodes)
                        break
                    summary = self.summary(session)
                    if summary.get("healthy") and not summary.get("completed"):
                        marker = self.root / "control" / "first_healthy.json"
                        train = summary.get("train") or {}
                        if not marker.exists():
                            atomic_json(marker, {"session": str(session), "time": summary["time"],
                                                 "step": train.get("step")})
                        print(json.dumps({"healthy": True, "session": session.stem,
                                          "step": train.get("step")}), flush=True)
                    if summary.get("status") == "stalled":
                        stalled_since = stalled_since or time.monotonic()
                    else:
                        stalled_since = None
                    if stalled_since is not None and time.monotonic() - stalled_since >= 120:
                        self.terminate_nodes(nodes)
                        break
                    if all(row.get("state") == "completed" for row in states):
                        if args.mode.startswith("smoke"):
                            atomic_json(state_path, {"state": "completed", "pid": os.getpid(),
                                                     "session": str(session), "attempt": attempt})
                            return 0
                        latest = read_json(Path(session_data["run_dir"]) / "latest.json", {})
                        if latest.get("step", -1) >= args.stop_after:
                            atomic_json(state_path, {"state": "completed", "pid": os.getpid(),
                                                     "session": str(session), "latest": latest})
                            return 0
                        if (self.root / "runs/main/control/stop.json").exists():
                            atomic_json(state_path, {"state": "stopped", "pid": os.getpid(),
                                                     "session": str(session), "latest": latest})
                            return 0
                        break
                    time.sleep(15)
            finally:
                process.terminate()
            if args.mode.startswith("smoke") or not self.retry_allowed(session):
                atomic_json(state_path, {"state": "failed", "session": str(session), "attempt": attempt,
                                         "reason": "smoke failure or deterministic error; manual review required"})
                return 2
            failures += 1
            mode = "train-resume"
            if not (self.root / "runs/main/latest.json").exists():
                failed_run = self.root / "runs/main"
                pending_skip = (failed_run / "data_skips.jsonl").read_bytes()
                archive = self.root / "runs/history" / f"main_no_checkpoint_{time.time_ns()}"
                archive.parent.mkdir(parents=True, exist_ok=True)
                failed_run.rename(archive)
                mode = "train-start"
                print(f"No complete new checkpoint; failed run preserved at {archive}; "
                      "retrying the SAME pinned source checkpoint", flush=True)
            delay = min(300, failures * 30)
            print(f"Attempt failed; bounded resume {failures}/{args.max_attempts - 1} after {delay}s",
                  flush=True)
            time.sleep(delay)
        atomic_json(state_path, {"state": "failed", "reason": "maximum attempts exhausted"})
        return 2

    def stop(self, args):
        state = read_json(self.root / "control/cluster_state.json", {})
        require(state.get("session"), "No recorded session")
        session = Path(state["session"])
        data = read_json(session)
        run_dir = Path(data["run_dir"])
        request = {"action": "stop", "request_id": str(uuid.uuid4()),
                   "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "reason": args.reason}
        atomic_json(run_dir / "control/stop.json", request)
        nodes = read_json(self.root / "control" / f"{session.stem}.drivers.json", [])
        deadline = time.monotonic() + args.grace_seconds
        while time.monotonic() < deadline:
            states = [self.node_status(node) for node in nodes]
            if all(row.get("state") in ("completed", "failed") for row in states):
                latest = read_json(run_dir / "latest.json", {})
                checkpoint = Path(latest["path"])
                rng_count = len(list(checkpoint.parent.glob(checkpoint.stem + ".rng_rank*.pt")))
                require(rng_count == 48, f"Stopped without 48 RNG files: {rng_count}")
                atomic_json(run_dir / "control/stop_ack.json", {**request, "latest": latest,
                                                               "rng_sidecars": rng_count})
                print(json.dumps({"stopped": True, "latest": latest, "rng_sidecars": rng_count}, indent=2))
                return 0
            time.sleep(10)
        print("Graceful stop deadline reached; terminating only recorded drivers", flush=True)
        self.terminate_nodes(nodes)
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("probe", "comm", "run", "stop", "status"):
        command = sub.add_parser(name)
        command.add_argument("--candidate", type=Path, required=True)
        if name == "probe":
            command.add_argument("--require-free-mib", type=int, default=55000)
        if name == "run":
            command.add_argument("--mode", choices=("smoke-start", "smoke-resume", "train-start", "train-resume"), required=True)
            command.add_argument("--stop-after", type=int)
            command.add_argument("--max-attempts", type=int, default=20)
            command.add_argument("--acknowledge-stop", action="store_true")
        if name == "stop":
            command.add_argument("--grace-seconds", type=int, default=900)
            command.add_argument("--reason", default="operator requested coordinated stop")
    args = parser.parse_args()
    cluster = Cluster(args.candidate)
    if args.action == "status":
        state = read_json(cluster.root / "control/cluster_state.json", {})
        result = {"cluster": state}
        if state.get("session"):
            result["health"] = cluster.summary(Path(state["session"]))
        print(json.dumps(result, indent=2), flush=True)
        return
    if args.action == "run":
        require(args.stop_after is not None, "Explicit absolute --stop-after is required")
        require(args.max_attempts >= 1, "max-attempts must be positive")
    return_code = {"probe": cluster.probe, "comm": cluster.comm,
                   "run": cluster.run, "stop": cluster.stop}[args.action](args)
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
