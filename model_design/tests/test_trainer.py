from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from muse_encoder.pretrain.trainer import (
    PretrainWrapper,
    Trainer,
    seed_everything,
    warmup_cosine_scheduler,
)


class ToyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.dropout = torch.nn.Dropout(0.25)

    def pretraining_losses(self, batch, targets, weights):
        prediction = self.linear(self.dropout(batch["x"])).squeeze(-1)
        loss = torch.nn.functional.mse_loss(prediction, targets["y"])
        return {"total": loss, "mse": loss}


def make_trainer(run_dir, total_steps):
    seed_everything(123)
    encoder = ToyEncoder()
    model = PretrainWrapper(encoder, {})
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = warmup_cosine_scheduler(optimizer, total_steps, 2)
    return Trainer(model, optimizer, scheduler, Path(run_dir), 2, total_steps,
                   torch.device("cpu"), log_interval=1, checkpoint_interval=3,
                   bf16=False)


def stream_factory(start):
    for index in range(start, 12):
        generator = torch.Generator().manual_seed(10_000 + index)
        x = torch.randn(4, 2, generator=generator)
        y = torch.randn(4, generator=generator)
        yield {"token_ids": torch.zeros(4, 2, dtype=torch.long), "x": x}, {"y": y}


class TrainerTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            full = make_trainer(Path(tmp) / "full", 6)
            full.fit(stream_factory, {"test": True})
            full_path = Path(tmp) / "full" / "checkpoints" / "final.pt"
            full_model = torch.load(full_path, weights_only=False)["model"]

            first = make_trainer(Path(tmp) / "part", 6)

            def first_half(start):
                stream = stream_factory(start)
                for _ in range(6 - start):
                    yield next(stream)

            first.fit(first_half, {"test": True})
            part_path = Path(tmp) / "part" / "checkpoints" / "final.pt"

            resumed = make_trainer(Path(tmp) / "resume", 6)
            resumed.resume(part_path)
            self.assertEqual(resumed.state.step, 3)
            self.assertEqual(resumed.state.micro_step, 6)
            resumed.fit(stream_factory, {"test": True})
            resumed_path = Path(tmp) / "resume" / "checkpoints" / "final.pt"
            resumed_model = torch.load(resumed_path, weights_only=False)["model"]

            self.assertEqual(set(full_model), set(resumed_model))
            for key in full_model:
                torch.testing.assert_close(full_model[key], resumed_model[key], rtol=0, atol=0)
            self.assertTrue(part_path.with_suffix(".rng_rank0.pt").exists())
            records = [json.loads(line) for line in
                       (Path(tmp) / "resume" / "checkpoints" / "checkpoints.jsonl").read_text().splitlines()]
            self.assertEqual(records[-1]["step"], 6)
            self.assertEqual(len(records[-1]["sha256"]), 64)

    def test_stream_must_end_on_accumulation_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = make_trainer(tmp, 2)

            def short_stream(start):
                yield from stream_factory(start)
                yield from stream_factory(11)

            original = trainer.total_steps
            trainer.total_steps = 20
            with self.assertRaisesRegex(RuntimeError, "累积边界"):
                trainer.fit(short_stream, {})
            trainer.total_steps = original


if __name__ == "__main__":
    unittest.main()
