from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from muse_encoder.pretrain.trainer import TrainState
from s5.runtime import (
    S5Trainer,
    S5Wrapper,
    build_optimizer,
    sha256_file,
    strict_initialize,
    warmup_stable_scheduler,
)


class DummyEncoder(nn.Module):
    def __init__(self, include_new=True):
        super().__init__()
        self.trunk = nn.Linear(3, 3)
        self.heads = nn.Module()
        self.heads.aux_heads = nn.ModuleDict({"old": nn.Linear(3, 1)})
        if include_new:
            self.heads.aux_heads["new"] = nn.Linear(3, 1)


class RuntimeTest(unittest.TestCase):
    def test_strict_init_and_parameter_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = DummyEncoder(include_new=False)
            path = Path(tmp) / "parent.pt"
            torch.save({"model": parent.state_dict()}, path)
            model = DummyEncoder(include_new=True)
            record = strict_initialize(model, path)
            self.assertEqual(record["new_aux_prefixes"], ["heads.aux_heads.new"])
            wrapped = S5Wrapper(model, {})
            optimizer = build_optimizer(wrapped, record, 1e-5, 1e-4, 0.01)
            self.assertEqual(len(optimizer.param_groups), 2)
            self.assertEqual(optimizer.param_groups[0]["lr"], 1e-5)
            self.assertEqual(optimizer.param_groups[1]["lr"], 1e-4)

    def test_unexpected_parameter_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = DummyEncoder(include_new=True)
            payload = parent.state_dict()
            payload["unknown.weight"] = torch.zeros(1)
            path = Path(tmp) / "parent.pt"
            torch.save({"model": payload}, path)
            with self.assertRaises(RuntimeError):
                strict_initialize(DummyEncoder(), path)

    def test_source_loss_weights_do_not_leak(self):
        class LossEncoder(nn.Module):
            def forward(self):
                return None
            def pretraining_losses(self, batch, targets, weights):
                self.seen = (targets, weights)
                return {"total": torch.tensor(float(weights["force"]))}
        encoder = LossEncoder()
        wrapped = S5Wrapper(encoder, {"force": 30.0, "energy": 10.0})
        output = wrapped({}, {"_s5_loss_weights": {"force": 0.0, "energy": 1.0}, "x": 1})
        self.assertEqual(float(output["total"]), 0.0)
        targets, weights = encoder.seen
        self.assertNotIn("_s5_loss_weights", targets)
        self.assertEqual(weights["force"], 0.0)
        self.assertEqual(weights["energy"], 1.0)

    def test_checkpoint_latest_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            encoder = DummyEncoder()
            wrapped = S5Wrapper(encoder, {})
            optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-5)
            scheduler = warmup_stable_scheduler(optimizer, 2)
            trainer = S5Trainer(wrapped, optimizer, scheduler, run_dir, 1, 100,
                                torch.device("cpu"), checkpoint_interval=10, bf16=False)
            trainer.state = TrainState(step=7, micro_step=7, samples=56)
            saved = trainer.save("step_00000007", {"schema": "test"}, {"s5_init": {"parent": "self"}})
            latest = json.loads((run_dir / "latest.json").read_text())
            self.assertEqual(latest["step"], 7)
            self.assertEqual(latest["sha256"], sha256_file(saved))
            self.assertTrue(saved.with_suffix(".rng_rank0.pt").is_file())

            restored_encoder = DummyEncoder()
            restored = S5Wrapper(restored_encoder, {})
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-5)
            restored_scheduler = warmup_stable_scheduler(restored_optimizer, 2)
            restored_trainer = S5Trainer(restored, restored_optimizer, restored_scheduler,
                                         run_dir, 1, 100, torch.device("cpu"), bf16=False)
            state = restored_trainer.resume(run_dir / "latest.json")
            self.assertEqual((state.step, state.micro_step, state.samples), (7, 7, 56))
            for key, value in encoder.state_dict().items():
                self.assertTrue(torch.equal(value, restored_encoder.state_dict()[key]))
            resumed_from = json.loads((run_dir / "resumed_from.json").read_text())
            self.assertEqual(resumed_from["sha256"], latest["sha256"])

    def test_warmup_then_constant(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        scheduler = warmup_stable_scheduler(optimizer, 50)
        for _ in range(50):
            optimizer.step()
            scheduler.step()
        value = scheduler.get_last_lr()[0]
        for _ in range(1000):
            scheduler.step()
        self.assertAlmostEqual(value, 1e-5)
        self.assertAlmostEqual(scheduler.get_last_lr()[0], value)


if __name__ == "__main__":
    unittest.main()
