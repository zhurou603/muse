"""按 shards.json 在本机启动 8 个评估分片（每片一张卡，setsid 脱离会话）。"""
import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path("/yinghepool/tech-dcu/zhaoyao")
STAGE = ROOT / "model_design/launches/eval16-20261027"
RUNS = ROOT / os.environ.get("EVAL_RUNS", "model_design/runs/formal-s1a/eval_step4000_shards")
PY = "/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python"
CKPT = ROOT / os.environ.get("EVAL_CKPT", "model_design/runs/formal-s1a/checkpoints/step_00004000.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=int, choices=[0, 1], required=True)
    parser.add_argument("--only", default=None, help="逗号分隔的分片名，调试用")
    args = parser.parse_args()
    shards = json.loads((STAGE / "shards.json").read_text())["shards"]
    selected = shards[args.node * 8:(args.node + 1) * 8]
    if args.only:
        wanted = set(args.only.split(","))
        selected = [s for s in selected if s["name"] in wanted]
    for index, shard in enumerate(selected):
        gpu = shards.index(shard) % 8
        out = RUNS / shard["name"]
        out.mkdir(parents=True, exist_ok=True)
        log = (out / "shard.log").open("a")
        cmd = [PY, str(STAGE / "shard_eval.py"), "--checkpoint", str(CKPT),
               "--output-dir", str(out), "--device", f"cuda:{gpu}",
               "--subsets", ",".join(shard["subsets"]), "--train-cap", os.environ.get("EVAL_OMC_CAP" if "OMC25/omc25" in shard["subsets"] else "EVAL_TRAIN_CAP", "200000")]
        env = {**os.environ, "PYTHONPATH": f"{ROOT}/model_design:{ROOT}",
               "PYTHONDONTWRITEBYTECODE": "1"}
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                   start_new_session=True, stdin=subprocess.DEVNULL)
        print(f"{shard['name']} gpu=cuda:{gpu} pid={process.pid} subsets={len(shard['subsets'])}", flush=True)


if __name__ == "__main__":
    main()
