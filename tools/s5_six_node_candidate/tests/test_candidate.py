from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
from cluster import Cluster, read_json
from muse_encoder.s5.health import RankEventLogger, collect_health
from muse_encoder.s5.runtime import S5Trainer
from muse_encoder.pretrain.trainer import TrainState


class ControlledStopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def trainer(self, rank=0):
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        return S5Trainer(model, optimizer, scheduler, self.root / "run", 1, 20,
                         torch.device("cpu"), rank=rank, bf16=False)

    def test_no_control_keeps_original_behavior(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("S5_CONTROL_FILE", None)
            self.assertFalse(self.trainer().should_stop())

    def test_nonzero_rank_obeys_rank0_collective(self):
        trainer = self.trainer(rank=1)
        request = {"action": "stop", "request_id": "one"}
        with patch.dict(os.environ, {"S5_CONTROL_FILE": str(self.root / "missing")}), \
             patch("muse_encoder.s5.runtime.dist.is_initialized", return_value=True), \
             patch("muse_encoder.s5.runtime.dist.broadcast_object_list",
                   side_effect=lambda value, src: value.__setitem__(0, request)) as broadcast:
            self.assertTrue(trainer.should_stop())
        broadcast.assert_called_once()

    def test_invalid_request_fails_instead_of_silent_resume(self):
        path = self.root / "stop.json"
        path.write_text("{}")
        with patch.dict(os.environ, {"S5_CONTROL_FILE": str(path)}):
            with self.assertRaisesRegex(ValueError, "Invalid"):
                self.trainer().should_stop()

    def test_stop_at_optimizer_boundary_saves_final(self):
        trainer = self.trainer()
        trainer.should_stop = lambda: trainer.state.step >= 2
        def stream(start):
            for step in range(start, 20):
                batch = {"token_ids": torch.ones(2, 3, dtype=torch.long)}
                target = torch.ones(2, 1)
                yield batch, target
        class LossModel(torch.nn.Module):
            def __init__(self, linear):
                super().__init__()
                self.linear = linear
            def forward(self, batch, target):
                prediction = self.linear(batch["token_ids"].float())
                return {"total": (prediction - target).square().mean()}
        trainer.model = LossModel(trainer.model)
        trainer.fit(stream, {"schema": "test"})
        self.assertEqual((trainer.state.step, trainer.state.micro_step), (2, 2))
        latest = read_json(self.root / "run/latest.json")
        self.assertEqual(latest["step"], 2)
        self.assertTrue(Path(latest["path"]).is_file())
        self.assertFalse(list((self.root / "run/checkpoints").glob("*.tmp-rank*")))


class HealthSessionTests(unittest.TestCase):
    def test_old_session_cannot_mask_new_start_or_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = 1000.0
            for rank in range(2):
                logger = RankEventLogger(root, rank, 2, "cuda:0")
                logger._write("progress", timestamp=now - 1000, session="old",
                              status="running", micro_step=10, optimizer_step=10)
                logger._write("error", timestamp=now, session="new", status="error",
                              micro_step=10, optimizer_step=10)
            summary = collect_health(root, 2, now=now, expected_session="different")
            self.assertEqual(summary["status"], "starting")
            self.assertFalse(summary["issues"])
            summary = collect_health(root, 2, now=now, expected_session="new")
            self.assertEqual(summary["status"], "stalled")
            self.assertEqual(len(summary["issues"]), 2)

    def test_all_completed_ranks_and_checkpoint_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoints/final.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"test")
            (root / "latest.json").write_text(json.dumps({
                "path": str(checkpoint), "step": 22, "sha256": "test"}))
            for rank in range(2):
                checkpoint.with_suffix(f".rng_rank{rank}.pt").write_bytes(b"rng")
                logger = RankEventLogger(root, rank, 2, "cuda:0")
                logger._write("progress", timestamp=1000, session="s", status="running",
                              micro_step=21, optimizer_step=21)
                logger._write("stopped", timestamp=1001, session="s", status="completed",
                              micro_step=21, optimizer_step=21)
                (root / f"course_rank{rank}.jsonl").write_text(
                    json.dumps({"micro_step": 21, "source": "qm9"}) + "\n")
            summary = collect_health(root, 2, now=1002, target_step=22, expected_session="s")
            self.assertTrue(summary["completed"])
            self.assertEqual(summary["status"], "completed")


class OrchestratorTests(unittest.TestCase):
    def test_retry_policy_distinguishes_transient_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs/session"
            logs.mkdir(parents=True)
            cluster = Cluster.__new__(Cluster)
            cluster.root = root
            (logs / "node.log").write_text("NCCL timeout after connection was lost")
            self.assertTrue(cluster.retry_allowed(Path("session")))
            (logs / "node.log").write_text("HIP out of memory")
            self.assertFalse(cluster.retry_allowed(Path("session")))
            (logs / "node.log").write_text("Candidate file changed: train.py")
            self.assertFalse(cluster.retry_allowed(Path("session")))


if __name__ == "__main__":
    unittest.main()
