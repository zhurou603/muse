import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE_PATH = (Path(__file__).resolve().parents[1] /
               "launches/s5-continuous-20261104/continuous_a_test.py")
SPEC = importlib.util.spec_from_file_location("continuous_a_test", MODULE_PATH)
continuous_a_test = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(continuous_a_test)


class ContinuousATestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.training = self.root / "training"
        self.checkpoints = self.training / "checkpoints"
        self.checkpoints.mkdir(parents=True)
        self.state = self.root / "state"

    def tearDown(self):
        self.temporary.cleanup()

    def make_checkpoint(self, step=1):
        checkpoint = self.checkpoints / f"step_{step:07d}.pt"
        payload = f"checkpoint-{step}".encode()
        checkpoint.write_bytes(payload)
        for rank in range(47):
            checkpoint.with_name(f"{checkpoint.stem}.rng_rank{rank}.pt").write_bytes(b"rng")
        digest = continuous_a_test.sha256(checkpoint)
        pointer = {"path": str(checkpoint), "step": step, "micro_step": step,
                   "sha256": digest}
        continuous_a_test.write_json(self.training / "latest.json", pointer)
        continuous_a_test.append_jsonl(
            self.checkpoints / "checkpoints.jsonl",
            {"path": str(checkpoint), "step": step, "micro_step": step, "sha256": digest})
        return pointer

    def make_daemon(self):
        daemon = continuous_a_test.ContinuousATest(
            training_run=self.training, state_dir=self.state, poll_seconds=1)
        daemon.source_dir.mkdir(parents=True)
        daemon.repairs_path.write_bytes(b"repairs")
        daemon.expected_repairs_sha256 = continuous_a_test.sha256(daemon.repairs_path)
        continuous_a_test.write_json(daemon.source_dir / "source_snapshot.json", {
            "git_commit": "a" * 40,
            "files": {str(continuous_a_test.LAUNCHER): "launcher",
                      str(continuous_a_test.FAST_EVAL): "evaluator"},
            "repairs_sha256": daemon.expected_repairs_sha256})
        return daemon

    def write_valid_result(self, record, run_dir, repairs_path):
        scores = {"schema_version": "muse.scores.v1",
                  "evaluation": {"board": "A", "split": "test"},
                  "num_domains": 4, "num_datasets": 13, "num_tasks": 90,
                  "scored_tasks": 90, "complete": True, "missing_predictions": [],
                  "invalid_predictions": [], "final_score": 60.0,
                  "domains": {"small_molecule": 50.0}}
        check = {"consistent": True, "problems": []}
        continuous_a_test.write_json(run_dir / "scores.json", scores)
        continuous_a_test.write_json(run_dir / "check.json", check)
        continuous_a_test.write_json(run_dir / "predictions/manifest.json",
                                     {"board": "A", "split": "test"})
        continuous_a_test.write_json(run_dir / "a_test_summary.json", {
            "checkpoint_sha256": record["sha256"],
            "input_repairs_sha256": continuous_a_test.sha256(repairs_path),
            "final_score": 60.0})
        for index in range(90):
            continuous_a_test.write_json(
                run_dir / f"embeddings/task{index}/test.failed.json", [])
        return 0


class CheckpointSelectionTests(ContinuousATestCase):
    def test_selects_hash_and_rng_complete_latest_checkpoint(self):
        pointer = self.make_checkpoint(36600)
        record = continuous_a_test.select_checkpoint(self.training)
        self.assertEqual(record["step"], 36600)
        self.assertEqual(record["sha256"], pointer["sha256"])
        self.assertEqual(record["world_size"], 47)

    def test_rejects_changed_checkpoint_payload(self):
        self.make_checkpoint()
        Path(json.loads((self.training / "latest.json").read_text())["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "实际 SHA256"):
            continuous_a_test.select_checkpoint(self.training)

    def test_rejects_incomplete_rng(self):
        pointer = self.make_checkpoint()
        checkpoint = Path(pointer["path"])
        checkpoint.with_name(f"{checkpoint.stem}.rng_rank46.pt").unlink()
        with self.assertRaisesRegex(RuntimeError, "RNG 不完整"):
            continuous_a_test.select_checkpoint(self.training)


class ResultValidationTests(ContinuousATestCase):
    def test_accepts_complete_scores_and_empty_encoding_failures(self):
        pointer = self.make_checkpoint()
        run_dir = self.root / "result"
        run_dir.mkdir()
        repairs = self.root / "repairs.json"
        repairs.write_bytes(b"repairs")
        self.write_valid_result(pointer, run_dir, repairs)
        result = continuous_a_test.validate_result(
            run_dir, pointer["sha256"], continuous_a_test.sha256(repairs))
        self.assertEqual(result["final_score"], 60.0)
        self.assertEqual(result["encoding_failed_files"], 90)

    def test_rejects_nonempty_encoding_failure(self):
        pointer = self.make_checkpoint()
        run_dir = self.root / "result"
        run_dir.mkdir()
        repairs = self.root / "repairs.json"
        repairs.write_bytes(b"repairs")
        self.write_valid_result(pointer, run_dir, repairs)
        continuous_a_test.write_json(run_dir / "embeddings/task0/test.failed.json",
                                     [{"id": "bad"}])
        with self.assertRaisesRegex(RuntimeError, "编码失败"):
            continuous_a_test.validate_result(
                run_dir, pointer["sha256"], continuous_a_test.sha256(repairs))


class LoopTests(ContinuousATestCase):
    def test_waits_for_new_checkpoint_and_runs_serially(self):
        self.make_checkpoint(1)
        daemon = self.make_daemon()
        calls = []
        sleeps = []

        def evaluator(record, run_dir, repairs_path):
            calls.append(record["step"])
            return self.write_valid_result(record, run_dir, repairs_path)

        def sleep(seconds):
            sleeps.append(seconds)
            self.make_checkpoint(2)

        daemon.run(max_rounds=2, evaluator=evaluator,
                   device_check=lambda: (True, {"free_mib": 60000}),
                   prepare=False, sleep=sleep)
        self.assertEqual(calls, [1, 2])
        self.assertEqual(sleeps, [1])
        state = continuous_a_test.read_json(daemon.state_path)
        self.assertEqual(state["last_completed_step"], 2)
        self.assertEqual(len([event for event in daemon.events
                              if event["event"] == "evaluation_succeeded"]), 2)

    def test_failed_checkpoint_is_not_retried(self):
        self.make_checkpoint(1)
        daemon = self.make_daemon()
        calls = []
        sleeps = []

        def evaluator(record, run_dir, repairs_path):
            calls.append(record["step"])
            return 1

        def sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) == 2:
                daemon.stop_requested = True

        daemon.run(evaluator=evaluator, device_check=lambda: (True, {}),
                   prepare=False, sleep=sleep)
        self.assertEqual(calls, [1])
        self.assertEqual(daemon.state["status"], "waiting_new_checkpoint")
        self.assertEqual(daemon.state["last_failed_step"], 1)

    def test_single_instance_lock(self):
        lock = continuous_a_test.acquire_lock(self.state)
        try:
            with self.assertRaisesRegex(RuntimeError, "拒绝重复启动"):
                continuous_a_test.acquire_lock(self.state)
        finally:
            lock.close()


if __name__ == "__main__":
    unittest.main()
