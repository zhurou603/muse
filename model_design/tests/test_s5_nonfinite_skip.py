from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from muse_encoder.s5.runtime import S5Trainer, warmup_stable_scheduler


class FiniteLossNonfiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value * 0.0

    @staticmethod
    def backward(ctx, grad_output):
        return torch.full_like(grad_output, float("inf"))


class LossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, batch, targets):
        value = self.linear(batch["token_ids"].float()).sum()
        return {"total": FiniteLossNonfiniteGradient.apply(value)}


class NonfiniteSkipTests(unittest.TestCase):
    def make(self, root, total=1):
        model = LossModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        scheduler = warmup_stable_scheduler(optimizer, 50)
        trainer = S5Trainer(model, optimizer, scheduler, root, 1, total,
                            torch.device("cpu"), rank=0, log_interval=1,
                            checkpoint_interval=100, bf16=False)
        trainer.course_log_path = root / "course_rank0.jsonl"
        trainer.course_log_path.write_text(json.dumps({
            "micro_step": 0, "source": "qm9", "domain": "small_molecule",
            "rows": [101, 202], "views": [0, 0]}) + "\n")
        return trainer

    def stream(self, start):
        for _ in range(start, 2):
            yield {"token_ids": torch.ones(2, 2, dtype=torch.long)}, {}

    def environment(self, enabled=True, consecutive=3):
        return patch.dict(os.environ, {
            "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "",
            "S5_ALLOW_NONFINITE_SKIP": "1" if enabled else "0",
            "S5_MAX_NONFINITE_SKIPS": "20",
            "S5_MAX_CONSECUTIVE_NONFINITE_SKIPS": str(consecutive),
        }, clear=False)

    def test_default_remains_fatal(self):
        with tempfile.TemporaryDirectory() as tmp, self.environment(enabled=False):
            trainer = self.make(Path(tmp))
            with self.assertRaisesRegex(FloatingPointError, "非有限梯度"):
                trainer.fit(self.stream, {"schema": "test"})
            self.assertEqual(trainer.state.step, 0)
            self.assertFalse((Path(tmp) / "optimization_skips.jsonl").exists())

    def test_explicit_skip_is_audited_without_parameter_update(self):
        with tempfile.TemporaryDirectory() as tmp, self.environment():
            root = Path(tmp)
            trainer = self.make(root)
            before = {key: value.clone() for key, value in trainer.model.state_dict().items()}
            trainer.fit(self.stream, {"schema": "test"})
            self.assertEqual((trainer.state.step, trainer.state.micro_step), (1, 1))
            self.assertEqual(trainer.scheduler.last_epoch, 1)
            for key, value in trainer.model.state_dict().items():
                self.assertTrue(torch.equal(value, before[key]))
            rows = [json.loads(x) for x in (root / "optimization_skips.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source"], "qm9")
            self.assertEqual(rows[0]["rows"], [101, 202])
            self.assertEqual(rows[0]["grad_norm"], "inf")
            self.assertFalse(rows[0]["optimizer_update"])
            train = json.loads((root / "train.jsonl").read_text())
            self.assertTrue(train["skipped"])
            self.assertNotIn("grad_norm", train)
            payload = torch.load(root / "checkpoints/final.pt", map_location="cpu", weights_only=False)
            self.assertEqual(payload["config"]["nonfinite_skip_count"], 1)
            restored = self.make(root)
            restored.resume(root / "checkpoints/final.pt")
            self.assertEqual(restored.nonfinite_skip_count, 1)

    def test_consecutive_budget_stops_instead_of_hiding_divergence(self):
        with tempfile.TemporaryDirectory() as tmp, self.environment(consecutive=1):
            trainer = self.make(Path(tmp), total=1)
            trainer.fit(self.stream, {"schema": "test"})
            trainer.total_steps = 2
            with self.assertRaisesRegex(FloatingPointError, "连续非有限梯度"):
                trainer.fit(self.stream, {"schema": "test"})
            self.assertEqual(trainer.state.step, 1)
            self.assertEqual(trainer.nonfinite_skip_count, 1)


if __name__ == "__main__":
    unittest.main()
