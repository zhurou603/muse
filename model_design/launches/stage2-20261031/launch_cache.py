"""Launch the 16 frozen-Encoder train/validation cache shards."""
import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path("/yinghepool/tech-dcu/zhaoyao")
STAGE = ROOT / "model_design/launches/stage2-20261031"
SHARDS = ROOT / "model_design/launches/eval16-20261027/shards.json"
FAST_EVAL = ROOT / "model_design/launches/eval16-20261027/fast_eval.py"
CHECKPOINT = ROOT / "model_design/runs/s3-residual-pilot-02/checkpoints/final.pt"
OUTPUT = ROOT / "model_design/runs/s3-residual-pilot-02/stage2_validation_cache"
PYTHON = "/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", type=int, choices=(0, 1), required=True)
    parser.add_argument("--only", default=None)
    args = parser.parse_args()
    shards = json.loads(SHARDS.read_text(encoding="utf-8"))["shards"]
    selected = shards[args.node * 8:(args.node + 1) * 8]
    if args.only:
        wanted = set(args.only.split(","))
        selected = [shard for shard in selected if shard["name"] in wanted]
    env = {
        **os.environ,
        "PYTHONPATH": f"{ROOT}/model_design:{ROOT}",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for shard in selected:
        gpu = int(shard["name"][1:]) % 8
        output = OUTPUT / shard["name"]
        output.mkdir(parents=True, exist_ok=True)
        command = [
            PYTHON, str(FAST_EVAL),
            "--checkpoint", str(CHECKPOINT),
            "--output-dir", str(output),
            "--devices", str(gpu),
            "--device", f"cuda:{gpu}",
            "--cpu-workers", "8",
            "--batch-size", "64",
            "--subsets", ",".join(shard["subsets"]),
            "--train-cap", "200000",
            "--second-split", "validation",
            "--single-seed",
            "--name", "stage2-frozen-baseline",
        ]
        log = (output / "cache.log").open("a")
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=env,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
        (output / "cache.pid").write_text(str(process.pid), encoding="utf-8")
        print(f"{shard['name']} node{args.node} cuda:{gpu} pid={process.pid}", flush=True)


if __name__ == "__main__":
    main()
