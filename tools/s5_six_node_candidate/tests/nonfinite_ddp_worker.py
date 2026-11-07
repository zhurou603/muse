"""Real two-process test of one finite-loss/nonfinite-gradient DDP step."""
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

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


def main():
    root = Path(sys.argv[1])
    rank = int(os.environ["RANK"])
    dist.init_process_group("gloo")
    model = DistributedDataParallel(LossModel())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = warmup_stable_scheduler(optimizer, 50)
    trainer = S5Trainer(model, optimizer, scheduler, root, 1, 1,
                        torch.device("cpu"), rank=rank, log_interval=1,
                        checkpoint_interval=100, bf16=False)
    trainer.course_log_path = root / f"course_rank{rank}.jsonl"
    trainer.course_log_path.write_text(json.dumps({
        "micro_step": 0, "source": "qm9", "domain": "small_molecule",
        "rows": [rank], "views": [0]}) + "\n")
    def stream(start):
        yield {"token_ids": torch.ones(2, 2, dtype=torch.long)}, {}
    trainer.fit(stream, {"schema": "two_process_nonfinite_skip"})
    assert trainer.state.step == trainer.state.micro_step == 1
    assert trainer.scheduler.last_epoch == 1
    dist.barrier()
    if rank == 0:
        rows = [json.loads(x) for x in (root / "optimization_skips.jsonl").read_text().splitlines()]
        assert len(rows) == 2 and {row["rank"] for row in rows} == {0, 1}
        assert all(row["grad_norm"] == "inf" and not row["optimizer_update"] for row in rows)
        train = json.loads((root / "train.jsonl").read_text())
        assert train["skipped"] and "grad_norm" not in train
        assert (root / "checkpoints/final.rng_rank0.pt").exists()
        assert (root / "checkpoints/final.rng_rank1.pt").exists()
        print("TWO-PROCESS NONFINITE SKIP PASS; both ranks skipped without an optimizer update", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
