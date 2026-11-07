"""串行运行完整 A-test；每次成功后等待并选择新的完整训练 checkpoint。"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAINING_RUN = ROOT / "tools/s5_six_node_step17500_47rank_r3_20261106/runs/main"
DEFAULT_STATE_DIR = ROOT / "model_design/runs/s5-continuous-four-domain-v1/continuous_a_test"
LAUNCHER = Path("model_design/launches/s5-continuous-20261104/run_full_a_test_single.sh")
FAST_EVAL = Path("model_design/launches/eval16-20261027/fast_eval.py")
REPAIRS = ROOT / "data/pretrain/manifest/a_test_encoding_repairs.v3.json"
EXPECTED_REPAIRS_SHA256 = "b58153d47e0fce27c49187bb2c6681094fa47a7bacffeac89bf74f51f4c106b8"
CODE_PATHS = ("muse", "model_design/muse_encoder", "model_design/launches/eval16-20261027",
              str(LAUNCHER))


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary.exists():
                temporary.unlink()


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def pid_status(pid):
    if not isinstance(pid, int) or pid <= 0:
        return None
    result = subprocess.run(["ps", "-p", str(pid), "-o", "stat=,args="],
                            capture_output=True, text=True, timeout=10, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    status, _, command = result.stdout.strip().partition(" ")
    if status.startswith("Z"):
        return None
    return command


def select_checkpoint(training_run, world_size=47):
    training_run = Path(training_run).resolve()
    pointer = read_json(training_run / "latest.json")
    step = pointer.get("step")
    if not isinstance(step, int) or step < 0 or pointer.get("micro_step") != step:
        raise RuntimeError("latest.json 的 step/micro_step 无效")
    checkpoint = Path(pointer["path"]).resolve()
    expected = training_run / "checkpoints" / f"step_{step:08d}.pt"
    if checkpoint != expected or not checkpoint.is_file():
        raise RuntimeError(f"latest checkpoint 路径或文件无效: {checkpoint}")
    claimed_sha = pointer.get("sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", claimed_sha):
        raise RuntimeError("latest checkpoint 缺少有效 SHA256")
    journal_match = None
    with (training_run / "checkpoints/checkpoints.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            if entry.get("step") == step and Path(entry["path"]).resolve() == checkpoint:
                journal_match = entry
    if journal_match is None or journal_match.get("sha256") != claimed_sha:
        raise RuntimeError("latest checkpoint 与 checkpoints.jsonl 血缘不一致")
    missing_rng = [rank for rank in range(world_size)
                   if not checkpoint.with_name(f"{checkpoint.stem}.rng_rank{rank}.pt").is_file()]
    if missing_rng:
        raise RuntimeError(f"checkpoint RNG 不完整: {missing_rng}")
    before = checkpoint.stat()
    if before.st_size <= 0:
        raise RuntimeError("checkpoint 为空")
    actual_sha = sha256(checkpoint)
    after = checkpoint.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("checkpoint 校验期间发生变化")
    if actual_sha != claimed_sha:
        raise RuntimeError("checkpoint 实际 SHA256 与 latest.json 不一致")
    return {"step": step, "path": str(checkpoint), "sha256": actual_sha,
            "world_size": world_size, "pointer": pointer,
            "journal": journal_match}


def validate_result(run_dir, checkpoint_sha, repairs_sha):
    run_dir = Path(run_dir)
    scores = read_json(run_dir / "scores.json")
    check = read_json(run_dir / "check.json")
    manifest = read_json(run_dir / "predictions/manifest.json")
    summary = read_json(run_dir / "a_test_summary.json")
    if scores.get("schema_version") != "muse.scores.v1":
        raise RuntimeError("scores.json schema 不正确")
    evaluation = scores.get("evaluation", {})
    if evaluation.get("board") != "A" or evaluation.get("split") != "test":
        raise RuntimeError("不是完整 A-test test split")
    if scores.get("num_domains") != 4 or scores.get("num_datasets") != 13:
        raise RuntimeError("A-test 领域或数据集数量不完整")
    if scores.get("num_tasks") != 90 or scores.get("scored_tasks") != 90:
        raise RuntimeError("A-test 未完成 90/90")
    if scores.get("complete") is not True:
        raise RuntimeError("A-test complete != true")
    if scores.get("missing_predictions") or scores.get("invalid_predictions"):
        raise RuntimeError("A-test 存在缺失或无效预测")
    if check != {"consistent": True, "problems": []}:
        raise RuntimeError("官方 check 未通过")
    if manifest.get("board") != "A" or manifest.get("split") != "test" or manifest.get("heads"):
        raise RuntimeError("预测 manifest 不符合单种子 A-test 协议")
    if summary.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("评估结果的 checkpoint 血缘不一致")
    if summary.get("input_repairs_sha256") != repairs_sha:
        raise RuntimeError("评估结果的编码修复清单血缘不一致")
    failed_files = sorted(run_dir.rglob("*.failed.json"))
    nonempty = []
    for path in failed_files:
        if read_json(path):
            nonempty.append(str(path))
    if len(failed_files) < 90 or nonempty:
        raise RuntimeError(f"编码失败审计不完整或仍有失败: {len(failed_files)} files, {nonempty}")
    return {"final_score": scores["final_score"], "complete": True, "scored_tasks": 90,
            "domains": scores.get("domains", {}), "check": check,
            "encoding_failed_files": len(failed_files), "nonempty_failed_files": nonempty,
            "scores": str(run_dir / "scores.json"),
            "summary": str(run_dir / "a_test_summary.json")}


class ContinuousATest:
    def __init__(self, training_run=DEFAULT_TRAINING_RUN, state_dir=DEFAULT_STATE_DIR,
                 poll_seconds=60, code_commit=None, repairs=REPAIRS,
                 expected_repairs_sha256=EXPECTED_REPAIRS_SHA256):
        self.training_run = Path(training_run).resolve()
        self.state_dir = Path(state_dir).resolve()
        self.poll_seconds = poll_seconds
        self.code_commit = code_commit
        self.repairs_source = Path(repairs).resolve()
        self.expected_repairs_sha256 = expected_repairs_sha256
        self.source_dir = self.state_dir / "source"
        self.repairs_path = self.state_dir / "a_test_encoding_repairs.v3.json"
        self.state_path = self.state_dir / "state.json"
        self.events_path = self.state_dir / "events.jsonl"
        self.stop_path = self.state_dir / "STOP"
        self.python = Path("/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python")
        if not self.python.is_file():
            self.python = Path(sys.executable)
        self.stop_requested = False
        self.state = {}
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        if self.state_path.is_file():
            self.state = read_json(self.state_path)
        self.events = []
        if self.events_path.is_file():
            with self.events_path.open(encoding="utf-8") as handle:
                self.events = [json.loads(line) for line in handle if line.strip()]

    def event(self, event, **values):
        entry = {"timestamp": now(), "event": event, "daemon_pid": os.getpid(), **values}
        append_jsonl(self.events_path, entry)
        self.events.append(entry)

    def save_state(self, status, **values):
        self.state = {**self.state, "schema": "muse.continuous_a_test.v1",
                      "status": status, "time": now(), "daemon_pid": os.getpid(), **values}
        write_json(self.state_path, self.state)

    def prepare_source(self):
        metadata_path = self.source_dir / "source_snapshot.json"
        if self.code_commit is None and metadata_path.is_file():
            self.code_commit = read_json(metadata_path).get("git_commit")
        if self.code_commit is None:
            self.code_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if not re.fullmatch(r"[0-9a-f]{40}", self.code_commit):
            raise RuntimeError("A-test 源码 commit 无效")
        if metadata_path.is_file():
            metadata = read_json(metadata_path)
            if metadata.get("git_commit") != self.code_commit:
                raise RuntimeError("已有源码快照与请求的 commit 不一致，请使用新的 state-dir")
            for relative in (LAUNCHER, FAST_EVAL):
                if sha256(self.source_dir / relative) != metadata["files"][str(relative)]:
                    raise RuntimeError(f"冻结源码已变化: {relative}")
            if sha256(self.repairs_path) != metadata["repairs_sha256"]:
                raise RuntimeError("冻结编码修复清单已变化")
            return metadata
        if self.source_dir.exists():
            raise RuntimeError("源码快照目录已存在但元数据缺失，拒绝覆盖")
        diff = subprocess.run(["git", "diff", "--quiet", self.code_commit, "--", *CODE_PATHS],
                              cwd=ROOT, check=False)
        if diff.returncode != 0:
            raise RuntimeError("待冻结的 A-test 源码与指定 commit 不一致")
        repairs_sha = sha256(self.repairs_source)
        if repairs_sha != self.expected_repairs_sha256:
            raise RuntimeError("编码修复清单与已验收 A-test v3 协议不一致")
        temporary = self.state_dir / f"source.tmp.{os.getpid()}"
        archive = self.state_dir / f"source.{os.getpid()}.tar"
        temporary.mkdir()
        try:
            with archive.open("wb") as handle:
                subprocess.run(["git", "archive", "--format=tar", self.code_commit],
                               cwd=ROOT, stdout=handle, check=True)
            with tarfile.open(archive) as bundle:
                for member in bundle.getmembers():
                    target = Path(member.name)
                    if target.is_absolute() or ".." in target.parts or member.islnk():
                        raise RuntimeError(f"源码快照含不安全路径: {member.name}")
                    if member.issym() and (Path(member.linkname).is_absolute()
                                           or ".." in Path(member.linkname).parts):
                        raise RuntimeError(f"源码快照含不安全链接: {member.name}")
                bundle.extractall(temporary)
            (temporary / "data").mkdir(exist_ok=True)
            (temporary / "data/muse-a").symlink_to(ROOT / "data/muse-a", target_is_directory=True)
            (temporary / "data/pretrain").symlink_to(ROOT / "data/pretrain", target_is_directory=True)
            for relative in (LAUNCHER, FAST_EVAL):
                if not (temporary / relative).is_file():
                    raise RuntimeError(f"源码快照缺少 {relative}")
            shutil.copyfile(self.repairs_source, self.repairs_path)
            metadata = {"git_commit": self.code_commit,
                        "files": {str(path): sha256(temporary / path)
                                  for path in (LAUNCHER, FAST_EVAL)},
                        "repairs_sha256": repairs_sha,
                        "data_links": {"data/muse-a": str(ROOT / "data/muse-a"),
                                       "data/pretrain": str(ROOT / "data/pretrain")},
                        "created_at": now()}
            write_json(temporary / "source_snapshot.json", metadata)
            os.replace(temporary, self.source_dir)
            self.event("source_prepared", **metadata)
            return metadata
        finally:
            if archive.exists():
                archive.unlink()
            if temporary.exists():
                shutil.rmtree(temporary)

    def check_device(self):
        code = ("import json,torch; "
                "free,total=torch.cuda.mem_get_info(7); "
                "print(json.dumps({'physical_device':7,'free_mib':free//2**20,"
                "'total_mib':total//2**20}))")
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("HIP_VISIBLE_DEVICES", None)
        result = subprocess.run([str(self.python), "-B", "-c", code], env=env,
                                capture_output=True, text=True, timeout=90, check=True)
        inventory = json.loads(result.stdout.strip().splitlines()[-1])
        return inventory["free_mib"] >= 55000, inventory

    def attempted_hashes(self):
        kinds = {"evaluation_launching", "evaluation_succeeded", "evaluation_failed", "stopped"}
        return {entry.get("checkpoint_sha256") for entry in self.events
                if entry.get("event") in kinds and entry.get("checkpoint_sha256")}

    def wait(self, pid, process, run_dir, sleep):
        while process.poll() is None if process is not None else pid_status(pid) is not None:
            if self.stop_requested or self.stop_path.is_file():
                self.stop_requested = True
                command = pid_status(pid)
                if command is not None and str(run_dir) in command:
                    os.killpg(pid, signal.SIGTERM)
                return None
            sleep(2)
        return process.returncode if process is not None else None

    def execute(self, record, evaluator, sleep):
        run_dir = self.state_dir / "runs" / f"step_{record['step']:08d}"
        run_dir.mkdir(parents=True, exist_ok=False)
        frozen = run_dir / "checkpoint.pt"
        os.link(record["path"], frozen)
        if sha256(frozen) != record["sha256"]:
            raise RuntimeError("冻结 checkpoint 副本的 SHA256 不一致")
        cache = ROOT / "data/pretrain/cache/continuous_a_test" / f"step_{record['step']:08d}"
        source_metadata = read_json(self.source_dir / "source_snapshot.json")
        launch = {"status": "prepared", "time": now(), "checkpoint": str(frozen),
                  "source_checkpoint": record["path"], "checkpoint_step": record["step"],
                  "checkpoint_sha256": record["sha256"], "world_size": 47,
                  "code_git_commit": source_metadata["git_commit"],
                  "launch_script_sha256": source_metadata["files"][str(LAUNCHER)],
                  "fast_eval_sha256": source_metadata["files"][str(FAST_EVAL)],
                  "input_repairs": str(self.repairs_path),
                  "input_repairs_sha256": source_metadata["repairs_sha256"],
                  "cache_root": str(cache), "physical_device": 7,
                  "num_tasks": 90, "full_test_split": True, "train_cap": 50000,
                  "batch_size": 64, "cpu_workers": 14, "single_seed": True,
                  "run_dir": str(run_dir), "uploaded": False, "model_selected": False}
        write_json(run_dir / "launch_record.json", launch)
        self.event("evaluation_launching", checkpoint_step=record["step"],
                   checkpoint_sha256=record["sha256"], run_dir=str(run_dir))
        if evaluator is None:
            command = ["bash", str(self.source_dir / LAUNCHER), str(frozen), str(run_dir),
                       f"s5-continuous-step{record['step']}-a-test-v3", str(self.repairs_path)]
            env = dict(os.environ, CACHE_ROOT=str(cache))
            log_path = run_dir / "evaluation.log"
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=self.source_dir, env=env,
                                           stdin=subprocess.DEVNULL, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            pid = process.pid
        else:
            process, pid = None, None
            log_path = run_dir / "evaluation.log"
        launch.update({"status": "running", "pid": pid, "log": str(log_path)})
        write_json(run_dir / "launch_record.json", launch)
        self.save_state("running", current_run=str(run_dir), current_pid=pid,
                        checkpoint_step=record["step"], checkpoint_sha256=record["sha256"])
        if evaluator is not None:
            returncode = evaluator(record, run_dir, self.repairs_path)
            if self.stop_requested or self.stop_path.is_file():
                returncode = None
        else:
            returncode = self.wait(pid, process, run_dir, sleep)
        if self.stop_requested or returncode is None:
            launch["status"] = "stopped"
            write_json(run_dir / "launch_record.json", launch)
            self.event("stopped", checkpoint_step=record["step"],
                       checkpoint_sha256=record["sha256"], run_dir=str(run_dir))
            self.save_state("stopped", reason="stop requested", current_pid=None)
            return
        if returncode != 0:
            launch.update({"status": "failed", "returncode": returncode})
            write_json(run_dir / "launch_record.json", launch)
            self.event("evaluation_failed", checkpoint_step=record["step"],
                       checkpoint_sha256=record["sha256"], returncode=returncode,
                       run_dir=str(run_dir))
            self.save_state("waiting_new_checkpoint", last_failed_step=record["step"],
                            last_failed_sha256=record["sha256"], current_pid=None)
            return
        try:
            result = validate_result(run_dir, record["sha256"], source_metadata["repairs_sha256"])
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            launch.update({"status": "failed", "validation_error": str(error)})
            write_json(run_dir / "launch_record.json", launch)
            raise
        write_json(run_dir / "continuous_result.json", result)
        launch.update({"status": "completed", "score_ready": True, "completed_at": now()})
        write_json(run_dir / "launch_record.json", launch)
        self.event("evaluation_succeeded", checkpoint_step=record["step"],
                   checkpoint_sha256=record["sha256"], run_dir=str(run_dir), **result)
        self.save_state("waiting_new_checkpoint", last_completed_step=record["step"],
                        last_completed_sha256=record["sha256"], last_result=result,
                        current_pid=None)

    def recover_running(self, sleep):
        if self.state.get("status") != "running":
            return False
        pid = self.state.get("current_pid")
        run_dir = Path(self.state.get("current_run", ""))
        command = pid_status(pid)
        if command is not None and (str(run_dir) not in command or str(LAUNCHER) not in command):
            raise RuntimeError("运行状态中的 PID 不属于对应 A-test，拒绝接管")
        if command is not None:
            self.wait(pid, None, run_dir, sleep)
        if self.stop_requested:
            self.save_state("stopped", reason="stop requested", current_pid=None)
            return True
        launch_path = run_dir / "launch_record.json"
        launch = read_json(launch_path)
        step = self.state["checkpoint_step"]
        checkpoint_sha = self.state["checkpoint_sha256"]
        try:
            result = validate_result(run_dir, checkpoint_sha, launch["input_repairs_sha256"])
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            launch.update({"status": "failed", "recovery_error": str(error)})
            write_json(launch_path, launch)
            self.event("evaluation_failed", checkpoint_step=step, checkpoint_sha256=checkpoint_sha,
                       run_dir=str(run_dir), recovery_error=str(error))
            self.save_state("waiting_new_checkpoint", last_failed_step=step,
                            last_failed_sha256=checkpoint_sha, current_pid=None)
            return True
        write_json(run_dir / "continuous_result.json", result)
        launch.update({"status": "completed", "score_ready": True, "completed_at": now()})
        write_json(launch_path, launch)
        self.event("evaluation_succeeded", checkpoint_step=step, checkpoint_sha256=checkpoint_sha,
                   run_dir=str(run_dir), recovered=True, **result)
        self.save_state("waiting_new_checkpoint", last_completed_step=step,
                        last_completed_sha256=checkpoint_sha, last_result=result, current_pid=None)
        return True

    def run(self, max_rounds=0, evaluator=None, device_check=None, prepare=True, sleep=time.sleep):
        if self.stop_path.exists():
            self.stop_path.unlink()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if prepare:
            self.prepare_source()
        device_check = device_check or self.check_device
        self.event("daemon_started", training_run=str(self.training_run))
        if self.recover_running(sleep):
            if max_rounds == 1 or self.stop_requested:
                return
        self.save_state("waiting_new_checkpoint", training_run=str(self.training_run),
                        source_dir=str(self.source_dir), poll_seconds=self.poll_seconds,
                        current_pid=None)
        rounds = 0
        while not self.stop_requested and not self.stop_path.is_file():
            pointer = read_json(self.training_run / "latest.json")
            if pointer.get("sha256") in self.attempted_hashes():
                self.save_state("waiting_new_checkpoint", current_pid=None,
                                latest_seen_step=pointer.get("step"),
                                latest_seen_sha256=pointer.get("sha256"))
                sleep(self.poll_seconds)
                continue
            record = select_checkpoint(self.training_run)
            free, inventory = device_check()
            if not free:
                self.save_state("waiting_device", current_pid=None,
                                reserved_device=7, device=inventory)
                sleep(self.poll_seconds)
                continue
            try:
                self.execute(record, evaluator, sleep)
            except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
                self.event("evaluation_failed", checkpoint_step=record["step"],
                           checkpoint_sha256=record["sha256"], error=str(error))
                self.save_state("waiting_new_checkpoint", last_failed_step=record["step"],
                                last_failed_sha256=record["sha256"],
                                last_error=str(error), current_pid=None)
            rounds += 1
            if max_rounds and rounds >= max_rounds:
                return


def acquire_lock(state_dir):
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    handle = (state_dir / "daemon.lock").open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError("已有持续 A-test 守护程序持有锁，拒绝重复启动")
    handle.write(str(os.getpid()) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--training-run", type=Path, default=DEFAULT_TRAINING_RUN)
    run.add_argument("--poll-seconds", type=float, default=60)
    run.add_argument("--code-commit")
    run.add_argument("--once", action="store_true")
    commands.add_parser("status")
    commands.add_parser("stop")
    args = parser.parse_args()
    state_dir = args.state_dir.resolve()
    if args.command == "status":
        state = read_json(state_dir / "state.json")
        pid = state.get("daemon_pid")
        state["daemon_alive"] = pid_status(pid) is not None
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return
    if args.command == "stop":
        write_json(state_dir / "STOP", {"time": now(), "pid": os.getpid()})
        print(str(state_dir / "STOP"))
        return
    lock = acquire_lock(state_dir)
    daemon = ContinuousATest(args.training_run, state_dir, args.poll_seconds, args.code_commit)
    signal.signal(signal.SIGTERM, lambda *_: setattr(daemon, "stop_requested", True))
    signal.signal(signal.SIGINT, lambda *_: setattr(daemon, "stop_requested", True))
    try:
        daemon.run(max_rounds=1 if args.once else 0)
    except Exception as error:
        daemon.event("daemon_failed", error=str(error))
        daemon.save_state("failed", error=str(error), current_pid=None)
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
