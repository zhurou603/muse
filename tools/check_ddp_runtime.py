"""Check collectives and DDP gradients without production models or files."""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
import socket

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-world-size", type=int, default=8)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if world_size != args.expected_world_size:
        raise RuntimeError(f"Expected world size {args.expected_world_size}, got {world_size}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != local_size:
        raise RuntimeError("Visible accelerator count must equal local worker count")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    print(json.dumps({"host": socket.gethostname(), "rank": rank,
                      "local_rank": local_rank, "pid": os.getpid(),
                      "device": str(device), "status": "initializing"}), flush=True)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    try:
        # DDP must broadcast rank zero's weights to all other workers.
        layer = torch.nn.Linear(16, 1, bias=False).to(device)
        with torch.no_grad():
            layer.weight.fill_(rank + 0.25)
        model = DistributedDataParallel(layer, device_ids=[local_rank])
        torch.testing.assert_close(layer.weight, torch.full_like(layer.weight, 0.25),
                                   rtol=0, atol=0)
        for step in range(3):
            values = torch.full((4096,), float(rank + 1 + step), device=device)
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            expected_sum = world_size * (world_size + 1) / 2 + world_size * step
            torch.testing.assert_close(values, torch.full_like(values, expected_sum),
                                       rtol=0, atol=0)
            model.zero_grad(set_to_none=True)
            inputs = torch.full((2, 16), float(rank + 1 + step), device=device)
            model(inputs).mean().backward()
            if layer.weight.grad is None:
                raise RuntimeError("DDP produced no weight gradient")
            expected_gradient = (world_size + 1) / 2 + step
            torch.testing.assert_close(
                layer.weight.grad, torch.full_like(layer.weight, expected_gradient),
                rtol=1e-6, atol=1e-6,
            )
            torch.cuda.synchronize(device)
        dist.barrier()
        print(f"RANK PASS: host={socket.gethostname()} rank={rank} device={device}", flush=True)
        if rank == 0:
            print(f"DDP COMMUNICATION PASS: {world_size} ranks; broadcast, all-reduce, "
                  "and averaged gradients checked for 3 rounds.", flush=True)
            print("No optimizer updates or files written. Not a throughput or full-model test.",
                  flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
