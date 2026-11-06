"""Two-process Gloo verification for coordinated optimizer-boundary stopping."""
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from muse_encoder.s5.runtime import S5Trainer


class LossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, batch, target):
        prediction = self.linear(batch["token_ids"].float())
        return {"total": (prediction - target).square().mean()}


def main():
    root = Path(sys.argv[1])
    rank = int(os.environ["RANK"])
    dist.init_process_group("gloo")
    model = DistributedDataParallel(LossModel())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    trainer = S5Trainer(model, optimizer, scheduler, root, 1, 20,
                        torch.device("cpu"), rank=rank, bf16=False)
    control = root / "control/stop.json"
    os.environ["S5_CONTROL_FILE"] = str(control)

    def stream(start):
        for _ in range(start, 20):
            if rank == 0 and trainer.state.step == 2:
                control.parent.mkdir(exist_ok=True)
                control.write_text(json.dumps({"action": "stop", "request_id": "gloo-boundary"}))
            yield {"token_ids": torch.ones(2, 3, dtype=torch.long)}, torch.ones(2, 1)

    trainer.fit(stream, {"schema": "cpu_ddp_stop_test"})
    assert trainer.state.step == trainer.state.micro_step == 2
    dist.barrier()
    if rank == 0:
        latest = json.loads((root / "latest.json").read_text())
        assert latest["step"] == 2
        assert Path(latest["path"]).is_file()
        print("TWO-PROCESS COORDINATED STOP PASS: final step=2, both ranks synchronized", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
