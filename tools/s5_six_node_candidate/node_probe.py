"""Read-only node identity, process and per-device memory inventory."""
import json
import os
import platform
import socket
import subprocess
import sys

import torch


def main():
    rows = subprocess.check_output(
        ["ps", "-C", "python", "-C", "python3", "-C", "torchrun", "-o", "pid=,args="],
        text=True,
    ).splitlines()
    conflicts = []
    for row in rows:
        pid_text, _, command = row.strip().partition(" ")
        if pid_text.isdigit() and int(pid_text) != os.getpid() and "muse_encoder.s5.train" in command:
            conflicts.append({"pid": int(pid_text), "command": command[:500]})
    devices = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        devices.append({"logical_device": index, "free_mib": free // 2**20,
                        "total_mib": total // 2**20})
    print(json.dumps({
        "host": socket.gethostname(), "python": sys.executable,
        "python_version": platform.python_version(), "torch": torch.__version__,
        "hip": torch.version.hip, "s5_training_processes": conflicts,
        "devices": devices,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
