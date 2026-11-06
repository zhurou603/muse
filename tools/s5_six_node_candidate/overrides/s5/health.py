"""Per-rank training heartbeats, progress audit and fail-closed stall alerts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from .provenance import atomic_json


def enable_stack_dump_signal() -> None:
    import faulthandler

    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)


class RankEventLogger:
    def __init__(self, run_dir: Path, rank: int, world_size: int, device: str,
                 heartbeat_seconds: float = 15.0):
        if heartbeat_seconds <= 0:
            raise ValueError("心跳间隔必须为正数")
        self.path = Path(run_dir) / "health" / f"rank_{int(rank):02d}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = device
        self.heartbeat_seconds = float(heartbeat_seconds)
        self.session = os.environ.get("S5_SESSION_ID") or f"{time.time_ns()}-{os.getpid()}"
        self.started_at = time.time()
        self.phase = "initializing"
        self.micro_step = -1
        self.optimizer_step = 0
        self.last_progress_at = None
        self.active = {}
        self._sequence = 0
        self._status = "starting"
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._stopped = False

    def _write(self, event: str, **fields: Any) -> None:
        with self._lock:
            active = [
                {"stage": value[0], "micro_step": value[1],
                 "optimizer_step": value[2], "elapsed_seconds": time.monotonic() - value[3]}
                for value in self.active.values()
            ]
            record = {
                "schema": "muse_encoder.s5_rank_health.v1",
                "timestamp": time.time(),
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "event": event,
                "session": self.session,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "rank": self.rank,
                "world_size": self.world_size,
                "device": self.device,
                "status": self._status,
                "phase": self.phase,
                "micro_step": self.micro_step,
                "optimizer_step": self.optimizer_step,
                "last_progress_at": self.last_progress_at,
                "active_stages": active,
                **fields,
            }
        with self._write_lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()

    def start(self) -> None:
        self._write("start")
        self._thread = threading.Thread(
            target=self._heartbeat_loop, name=f"s5-health-rank{self.rank}", daemon=True)
        self._thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.heartbeat_seconds):
            self._write("heartbeat")

    @contextmanager
    def stage(self, phase: str, micro_step: int, optimizer_step: int):
        with self._lock:
            self._sequence += 1
            token = self._sequence
            self.active[token] = (phase, int(micro_step), int(optimizer_step), time.monotonic())
            self.phase = phase
            self.micro_step = int(micro_step)
            self.optimizer_step = int(optimizer_step)
        started = time.monotonic()
        self._write("stage_start", stage=phase)
        try:
            yield
        except Exception as exc:
            self._write("stage_end", stage=phase, status="error",
                        error_type=type(exc).__name__, error=str(exc)[:500],
                        duration_seconds=time.monotonic() - started)
            raise
        else:
            self._write("stage_end", stage=phase, duration_seconds=time.monotonic() - started)
        finally:
            with self._lock:
                self.active.pop(token, None)

    def progress(self, micro_step: int, optimizer_step: int, **metrics: Any) -> None:
        with self._lock:
            self.micro_step = int(micro_step)
            self.optimizer_step = int(optimizer_step)
            self.last_progress_at = time.time()
            self._status = "running"
        self._write("progress", **metrics)

    def event(self, event: str, **fields: Any) -> None:
        self._write(event, **fields)

    def error(self, exc: Exception) -> None:
        with self._lock:
            self._status = "error"
            self.phase = "error"
        self._write("error", error_type=type(exc).__name__, error=str(exc)[:500])

    def stop(self, status: str = "stopped", reason: str = "") -> None:
        if self._stopped:
            return
        self._stopped = True
        self._status = status
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.heartbeat_seconds * 2))
        self._write("stopped", reason=reason)


def _jsonl_tail(path: Path, max_bytes: int = 262144) -> tuple[list[dict[str, Any]], bool]:
    if not path.exists():
        return [], False
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - max_bytes))
        data = stream.read()
    if size > max_bytes:
        data = data[data.find(b"\n") + 1:]
    rows = []
    malformed = False
    for line in data.splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            malformed = True
    return rows, malformed


def _hcu_snapshot(remote_host: Optional[str] = None) -> list[dict[str, Any]]:
    command = ["hy-smi"] if remote_host is None else [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", remote_host, "hy-smi"]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=25)
    rows = []
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) < 7 or not fields[0].isdigit():
            continue
        rows.append({
            "device": int(fields[0]), "temperature": fields[1], "power": fields[2],
            "vram_percent": float(fields[5].rstrip("%")),
            "hcu_percent": float(fields[6].rstrip("%")),
        })
    return rows


def collect_health(run_dir: Path, world_size: int, now: Optional[float] = None,
                   stale_after: float = 90.0, no_progress_after: float = 600.0,
                   startup_grace: float = 1800.0, checkpoint_grace: float = 1800.0,
                   target_step: Optional[int] = None,
                   startup_timestamp: Optional[float] = None,
                   expected_session: Optional[str] = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    run_dir = Path(run_dir)
    startup_timestamp = run_dir.stat().st_mtime if startup_timestamp is None else startup_timestamp
    issues = []
    ranks = {}
    progress_ranks = 0
    for rank in range(world_size):
        events, malformed = _jsonl_tail(run_dir / "health" / f"rank_{rank:02d}.jsonl")
        if not events:
            if now - startup_timestamp <= startup_grace:
                ranks[str(rank)] = {"status": "starting"}
            else:
                issues.append({"rank": rank, "reason": "missing_rank_events"})
                ranks[str(rank)] = {"status": "missing"}
            continue
        latest = events[-1]
        if expected_session is not None and latest.get("session") != expected_session:
            ranks[str(rank)] = {"status": "starting", "expected_session": expected_session}
            continue
        age = max(0.0, now - float(latest["timestamp"]))
        session = latest.get("session")
        session_events = [event for event in events
                          if session is None or event.get("session") == session]
        started_at = float(latest.get("started_at") or session_events[0]["timestamp"])
        progress = [event for event in session_events if event.get("event") == "progress"]
        last_progress = float(progress[-1]["timestamp"]) if progress else None
        progress_age = now - last_progress if last_progress is not None else now - started_at
        phase = latest.get("phase", "unknown")
        active = latest.get("active_stages") or []
        checkpointing = phase == "checkpoint" or any(
            stage.get("stage") == "checkpoint" for stage in active)
        progress_limit = max(no_progress_after, checkpoint_grace) if checkpointing else no_progress_after
        if last_progress is None:
            progress_limit = max(progress_limit, startup_grace)
        reason = None
        if latest.get("status") == "error":
            reason = "rank_error"
        elif latest.get("status") == "stopped" and (
                target_step is None or int(latest.get("optimizer_step", -1)) < int(target_step)):
            reason = "stopped_before_target"
        elif malformed:
            reason = "malformed_event_tail"
        elif age > stale_after:
            reason = "heartbeat_stale"
        elif progress_age > progress_limit:
            reason = "no_progress"
        if reason:
            issues.append({"rank": rank, "reason": reason, "event_age_seconds": age,
                           "progress_age_seconds": progress_age,
                           "pid": latest.get("pid"), "host": latest.get("host"),
                           "event_timestamp": float(latest["timestamp"])})
        if last_progress is not None:
            progress_ranks += 1
        ranks[str(rank)] = {
            "status": reason or latest.get("status", "running" if last_progress is not None else "starting"),
            "event_age_seconds": age, "progress_age_seconds": progress_age,
            "phase": phase, "micro_step": latest.get("micro_step"),
            "optimizer_step": latest.get("optimizer_step"), "pid": latest.get("pid"),
            "host": latest.get("host"), "session": latest.get("session"),
            "active_stages": active,
        }

    course = {}
    malformed_course = False
    for rank in range(world_size):
        rows, malformed = _jsonl_tail(run_dir / f"course_rank{rank}.jsonl")
        malformed_course |= malformed
        course[rank] = {int(row["micro_step"]): row.get("source") for row in rows}
    common = set.intersection(*(set(rows) for rows in course.values())) if len(course) == world_size else set()
    source_consistent = all(len({course[rank][step] for rank in range(world_size)}) == 1
                            for step in common) if common else None
    if source_consistent is False or malformed_course:
        issues.append({"reason": "course_source_mismatch" if source_consistent is False
                       else "malformed_course_tail"})
    if progress_ranks and source_consistent is not True:
        issues.append({"reason": "course_not_complete_across_ranks"})

    latest = None
    latest_path = run_dir / "latest.json"
    if latest_path.exists():
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        checkpoint = Path(latest["path"])
        if not checkpoint.is_absolute():
            checkpoint = Path.cwd() / checkpoint
        rng_count = len(list(checkpoint.parent.glob(checkpoint.stem + ".rng_rank*.pt")))
        latest = {**latest, "path": str(checkpoint), "rng_sidecars": rng_count}
        if rng_count != world_size:
            issues.append({"reason": "checkpoint_rng_incomplete", "rng_sidecars": rng_count})

    train_record = None
    rows, malformed_train = _jsonl_tail(run_dir / "train.jsonl")
    if rows:
        train_record = rows[-1]
        values = list((train_record.get("losses") or {}).values())
        if "grad_norm" in train_record:
            values.append(train_record["grad_norm"])
        if not values or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                             for value in values) or malformed_train:
            issues.append({"reason": "nonfinite_or_malformed_training_metrics"})

    if target_step is not None and latest is not None and int(latest["step"]) >= int(target_step):
        completed = True
    else:
        completed = False
    if target_step is not None and latest is not None and int(latest["step"]) >= int(target_step):
        completed = all(row.get("status") == "completed" and
                        int(row.get("optimizer_step", -1)) + 1 >= int(target_step)
                        for row in ranks.values()) and len(ranks) == world_size
    healthy = not issues and progress_ranks == world_size and source_consistent is True
    return {
        "schema": "muse_encoder.s5_training_health.v1",
        "timestamp": now, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
        "run_dir": str(run_dir),
        "status": "stalled" if issues else ("completed" if completed else ("healthy" if healthy else "starting")),
        "healthy": healthy, "completed": completed, "world_size": world_size,
        "progress_ranks": progress_ranks, "source_consistent": source_consistent,
        "ranks": ranks, "issues": issues, "latest": latest, "train": train_record,
    }


def _verify_checkpoint(summary: dict[str, Any], state: dict[str, Any]) -> None:
    latest = summary.get("latest")
    if not latest:
        return
    sha = latest.get("sha256")
    if sha == state.get("verified_checkpoint_sha256"):
        return
    digest = hashlib.sha256()
    with Path(latest["path"]).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != sha:
        raise RuntimeError("训练 latest checkpoint SHA256 不一致")
    state["verified_checkpoint_sha256"] = sha


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def ssh_options():
    import shlex
    values = shlex.split(os.environ.get("S5_SSH_OPTIONS", ""))
    if os.environ.get("S5_KNOWN_HOSTS"):
        values += ["-o", f"UserKnownHostsFile={os.environ['S5_KNOWN_HOSTS']}",
                   "-o", "StrictHostKeyChecking=yes"]
    return values


def _dump_stack(issue: dict[str, Any], remote_host: str = None,
                host_addresses: dict[str, str] | None = None) -> dict[str, Any]:
    pid = int(issue["pid"])
    host = issue.get("host")
    if host in (None, socket.gethostname()):
        try:
            os.kill(pid, signal.SIGUSR1)
        except OSError as exc:
            return {"pid": pid, "host": host, "signal": "SIGUSR1",
                    "requested": False, "error": str(exc)[:500]}
        return {"pid": pid, "host": host, "signal": "SIGUSR1", "requested": True}
    address = (host_addresses or {}).get(host, remote_host)
    if not address:
        return {"pid": pid, "host": host, "requested": False, "error": "No host address configured"}
    completed = subprocess.run([
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", *ssh_options(), address,
        f"kill -USR1 {pid}"], capture_output=True, text=True, timeout=20)
    return {"pid": pid, "host": host, "signal": "SIGUSR1",
            "requested": completed.returncode == 0, "error": completed.stderr[:500]}


def monitor(args: argparse.Namespace) -> int:
    label = args.session or "default"
    state_path = args.run_dir / "health" / "sessions" / f"{label}.monitor_state.json"
    summary_path = args.run_dir / "health" / "sessions" / f"{label}.summary.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    state.setdefault("monitor_started_at", time.time())
    host_addresses = dict(value.split("=", 1) for value in args.host)
    while True:
        summary = collect_health(
            args.run_dir, args.world_size, stale_after=args.stale_after,
            no_progress_after=args.no_progress_after, startup_grace=args.startup_grace,
            checkpoint_grace=args.checkpoint_grace, target_step=args.target_step,
            startup_timestamp=state["monitor_started_at"], expected_session=args.session)
        summary["session"] = args.session
        try:
            _verify_checkpoint(summary, state)
        except Exception as exc:
            summary["healthy"] = False
            summary["status"] = "stalled"
            summary["issues"].append({"reason": "checkpoint_hash_invalid", "error": str(exc)[:500]})
        if args.gpu:
            from concurrent.futures import ThreadPoolExecutor
            def gpu(item):
                host, address = item
                try:
                    return host, {"devices": _hcu_snapshot(
                        None if host in (socket.gethostname(), socket.gethostname().split(".")[0]) else address)}
                except Exception as exc:
                    return host, {"error": f"{type(exc).__name__}: {exc}"[:500]}
            with ThreadPoolExecutor(max_workers=len(host_addresses)) as executor:
                summary["hcu"] = dict(executor.map(gpu, host_addresses.items()))
        signature = json.dumps({"status": summary["status"], "issues": summary["issues"]},
                               sort_keys=True, ensure_ascii=False)
        if signature != state.get("alert_signature"):
            alert = {"time": summary["time"], "session": args.session,
                     "status": summary["status"], "issues": summary["issues"],
                     "latest": summary["latest"]}
            if args.dump_stacks and summary["status"] == "stalled":
                alert["stack_dumps"] = [
                    _dump_stack(issue, args.remote_host, host_addresses)
                    for issue in summary["issues"]
                    if issue.get("pid") and float(issue.get("event_timestamp") or 0)
                    >= state["monitor_started_at"]
                ]
            _append_jsonl(args.run_dir / "health" / "alerts.jsonl", alert)
            state["alert_signature"] = signature
            print(json.dumps({"status": summary["status"], "healthy": summary["healthy"],
                              "progress_ranks": summary["progress_ranks"],
                              "issues": summary["issues"]}, ensure_ascii=False), flush=True)
        atomic_json(state_path, state)
        atomic_json(summary_path, summary)
        if args.session is None:
            atomic_json(args.run_dir / "health" / "summary.json", summary)
        if args.once:
            return 0 if summary["healthy"] or summary["status"] == "starting" else 2
        time.sleep(max(1.0, args.poll_seconds))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--target-step", type=int, default=None)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--stale-after", type=float, default=90.0)
    parser.add_argument("--no-progress-after", type=float, default=600.0)
    parser.add_argument("--startup-grace", type=float, default=1800.0)
    parser.add_argument("--checkpoint-grace", type=float, default=1800.0)
    parser.add_argument("--remote-host", default="172.17.0.178")
    parser.add_argument("--host", action="append", default=[],
                        help="hostname=IP; repeat for each training node")
    parser.add_argument("--session")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--dump-stacks", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if min(args.world_size, args.stale_after, args.no_progress_after,
           args.startup_grace, args.checkpoint_grace) <= 0:
        parser.error("world size 与健康阈值必须为正数")
    if not args.host:
        args.host = [f"{socket.gethostname()}=127.0.0.1", f"remote={args.remote_host}"]
    raise SystemExit(monitor(args))


if __name__ == "__main__":
    main()
