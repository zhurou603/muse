from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from muse_encoder.s5.health import RankEventLogger, collect_health


class TrainingHealthTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _course(self, sources=("qmugs", "qmugs")):
        for rank, source in enumerate(sources):
            (self.root / f"course_rank{rank}.jsonl").write_text(
                json.dumps({"micro_step": 0, "source": source}) + "\n", encoding="utf-8")

    def _progress(self, rank, timestamp):
        path = self.root / "health" / f"rank_{rank:02d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "timestamp": timestamp, "event": "progress", "status": "running",
            "phase": "compute", "rank": rank, "micro_step": 0, "optimizer_step": 0,
        }) + "\n", encoding="utf-8")

    def test_logger_emits_stages_heartbeat_and_progress(self):
        logger = RankEventLogger(self.root, 0, 1, "cuda:0", heartbeat_seconds=0.02)
        logger.start()
        with logger.stage("fetch", 0, 0):
            pass
        time.sleep(0.04)
        logger.progress(0, 0, batch_samples=8)
        logger.stop("completed", "test_finished")
        rows = [json.loads(line) for line in logger.path.read_text().splitlines()]
        events = {row["event"] for row in rows}
        self.assertTrue({"start", "stage_start", "stage_end", "heartbeat", "progress", "stopped"} <= events)
        self.assertEqual(rows[-1]["status"], "completed")
        self.assertNotIn("labels", rows[-1])
        self.assertNotIn("targets", rows[-1])

    def test_healthy_requires_all_rank_progress_and_consistent_course(self):
        now = time.time()
        self._progress(0, now)
        self._progress(1, now)
        self._course()
        result = collect_health(self.root, 2, now=now)
        self.assertTrue(result["healthy"])
        self.assertEqual(result["status"], "healthy")
        self.assertEqual(result["progress_ranks"], 2)

    def test_stale_heartbeat_is_stalled(self):
        now = time.time()
        self._progress(0, now - 1000)
        self._progress(1, now)
        self._course()
        result = collect_health(self.root, 2, now=now, stale_after=10)
        self.assertFalse(result["healthy"])
        self.assertEqual(result["status"], "stalled")
        self.assertTrue(any(row["reason"] == "heartbeat_stale" for row in result["issues"]))

    def test_checkpoint_grace_does_not_hide_other_stalls(self):
        now = time.time()
        for rank in range(2):
            path = self.root / "health" / f"rank_{rank:02d}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join([
                json.dumps({"timestamp": now - 100, "event": "progress", "status": "running"}),
                json.dumps({"timestamp": now, "event": "heartbeat", "status": "running",
                            "phase": "checkpoint", "active_stages": []}),
            ]) + "\n", encoding="utf-8")
        self._course()
        checkpoint = collect_health(self.root, 2, now=now, no_progress_after=10,
                                    checkpoint_grace=1800)
        self.assertTrue(checkpoint["healthy"])
        for rank in range(2):
            path = self.root / "health" / f"rank_{rank:02d}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[-1]["phase"] = "fetch"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        stalled = collect_health(self.root, 2, now=now, no_progress_after=10,
                                 checkpoint_grace=1800)
        self.assertTrue(any(row["reason"] == "no_progress" for row in stalled["issues"]))

    def test_course_mismatch_and_nonfinite_metrics_are_rejected(self):
        now = time.time()
        self._progress(0, now)
        self._progress(1, now)
        self._course(("qmugs", "qm9"))
        result = collect_health(self.root, 2, now=now)
        self.assertTrue(any(row["reason"] == "course_source_mismatch" for row in result["issues"]))
        self._course()
        (self.root / "train.jsonl").write_text(
            '{"step":20,"losses":{"total":NaN},"grad_norm":1.0}\n', encoding="utf-8")
        result = collect_health(self.root, 2, now=now)
        self.assertTrue(any(row["reason"] == "nonfinite_or_malformed_training_metrics"
                            for row in result["issues"]))

    def test_stack_dump_failure_does_not_stop_monitor(self):
        from unittest.mock import patch
        from muse_encoder.s5.health import _dump_stack

        with patch("muse_encoder.s5.health.os.kill", side_effect=ProcessLookupError("gone")):
            result = _dump_stack({"pid": 12345, "host": None}, "172.17.0.178")
        self.assertFalse(result["requested"])
        self.assertIn("gone", result["error"])

    def test_stack_dump_signal_survives_in_child_process(self):
        import subprocess
        import sys

        code = (
            "import os, signal; "
            "from muse_encoder.s5.health import enable_stack_dump_signal; "
            "enable_stack_dump_signal(); os.kill(os.getpid(), signal.SIGUSR1); "
            "print('survived')"
        )
        completed = subprocess.run([sys.executable, "-c", code],
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("survived", completed.stdout)
        self.assertIn("Current thread", completed.stderr)

    def test_old_session_progress_does_not_mask_new_session_startup(self):
        now = time.time()
        for rank in range(2):
            path = self.root / "health" / f"rank_{rank:02d}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join([
                json.dumps({"timestamp": now - 1000, "event": "progress",
                            "session": "old", "status": "running"}),
                json.dumps({"timestamp": now, "event": "heartbeat",
                            "session": "new", "status": "starting", "phase": "initializing"}),
            ]) + "\n", encoding="utf-8")
        result = collect_health(self.root, 2, now=now)
        self.assertEqual(result["status"], "starting")
        self.assertFalse(result["issues"])


if __name__ == "__main__":
    unittest.main()
