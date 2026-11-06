"""Small per-device runtime checks; no model, optimizer, files or collectives."""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys

import torch


def check_device(index: int) -> dict:
    device = torch.device(f"cuda:{index}")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.ones((32, 32), dtype=dtype, device=device, requires_grad=True)
        identity = torch.eye(32, dtype=dtype, device=device)
        output = x @ identity
        torch.testing.assert_close(output, torch.ones_like(output), rtol=0, atol=0)
        output.float().sum().backward()
        if x.grad is None:
            raise RuntimeError(f"Missing gradient: {device}, {dtype}")
        torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)
        torch.cuda.synchronize(device)
        del output, identity, x
    torch.cuda.empty_cache()
    return {
        "logical_device": index,
        "name": props.name,
        "total_memory_gib": round(props.total_memory / 2**30, 2),
        "fp32_forward_backward": "PASS",
        "bf16_forward_backward": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-devices", type=int, default=8)
    args = parser.parse_args()
    if args.expected_devices < 1:
        raise ValueError("expected-devices must be positive")
    print(json.dumps({
        "host": socket.gethostname(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "visibility": {key: os.environ.get(key) for key in (
            "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"
        )},
    }, indent=2), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("No usable accelerator detected by PyTorch")
    count = torch.cuda.device_count()
    if count != args.expected_devices:
        raise RuntimeError(f"Expected {args.expected_devices} devices, found {count}")
    for index in range(count):
        print(f"CHECK logical device {index}...", flush=True)
        print(json.dumps(check_device(index)), flush=True)
    print("NODE COMPUTE PASS: all visible devices passed FP32/BF16 forward/backward.", flush=True)
    print("No model loaded or optimizer updates. DDP/network/data pipeline NOT tested.", flush=True)


if __name__ == "__main__":
    main()
