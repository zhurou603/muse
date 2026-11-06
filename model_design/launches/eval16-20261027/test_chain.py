"""validation 分片完成一片就接力启动对应 test 分片；16 片齐后产出 A 榜官方提交文件。

train 表征/外部结构缓存硬链接复用，只新编码 test；探针按原配方重训后预测 test。
最终产物：submission_step4000/scores.json（官方 score.py score）+ check.json（官方核对）。
"""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/yinghepool/tech-dcu/zhaoyao")
STAGE = ROOT / "model_design/launches/eval16-20261027"
VAL = ROOT / os.environ.get("EVAL_VAL_ROOT", "model_design/runs/formal-s1a/eval_step4000_shards")
TEST = ROOT / os.environ.get("EVAL_TEST_ROOT", "model_design/runs/formal-s1a/eval_step4000_test_shards")
SUBM = ROOT / os.environ.get("EVAL_SUBM", "model_design/runs/formal-s1a/submission_step4000")
PY = "/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python"
CKPT = ROOT / os.environ.get("EVAL_CKPT", "model_design/runs/formal-s1a/checkpoints/step_00004000.pt")
DATA = ROOT / "data/muse-a/MUSE-A/data"
SCORER = ROOT / "data/muse-a/eval/MUSE-eval/score.py"
ENV = {**os.environ, "PYTHONPATH": f"{ROOT}/model_design:{ROOT}", "PYTHONDONTWRITEBYTECODE": "1"}


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def launch_test_shard(shard):
    name = shard["name"]
    index = int(name[1:])
    node, gpu = index // 8, index % 8
    out = TEST / name
    out.mkdir(parents=True, exist_ok=True)
    for ds_dir in (VAL / name / "embeddings").iterdir():
        for sub_dir in ds_dir.iterdir():
            target = out / "embeddings" / ds_dir.name / sub_dir.name
            target.mkdir(parents=True, exist_ok=True)
            for f in sub_dir.glob("train.*"):
                link = target / f.name
                if not link.exists():
                    os.link(f, link)
    if (VAL / name / "external_cache").is_dir() and not (out / "external_cache").exists():
        os.symlink(VAL / name / "external_cache", out / "external_cache")
    cmd = [PY, str(STAGE / "shard_eval.py"), "--checkpoint", str(CKPT),
           "--output-dir", str(out), "--device", f"cuda:{gpu}",
           "--subsets", ",".join(shard["subsets"]), "--second-split", "test", "--train-cap", os.environ.get("EVAL_OMC_CAP" if "OMC25/omc25" in shard["subsets"] else "EVAL_TRAIN_CAP", "200000")]
    if node == 0:
        logf = (out / "shard.log").open("a")
        subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=ENV,
                         start_new_session=True, stdin=subprocess.DEVNULL)
    else:
        remote = (f"cd {ROOT} && setsid nohup env PYTHONPATH={ROOT}/model_design:{ROOT} "
                  f"PYTHONDONTWRITEBYTECODE=1 {' '.join(cmd)} > {out}/shard.log 2>&1 < /dev/null &")
        try:
            subprocess.run(["ssh", "-o", "BatchMode=yes", "172.17.0.178", remote],
                           timeout=90, check=False)
        except subprocess.TimeoutExpired:
            log(f"{name}: ssh 启动未返回（远端可能已启动，下轮按 shard.log 收养）")
    (out / "launched.txt").write_text(str(time.time()), encoding="utf-8")
    log(f"launched {name} node{node} cuda:{gpu}")


def finalize():
    SUBM.mkdir(parents=True, exist_ok=True)
    subprocess.run([PY, str(STAGE / "merge_and_score.py"), "--shards-root", str(TEST),
                    "--output-dir", str(SUBM), "--split", "test"], env=ENV, check=True)
    subprocess.run([PY, str(SCORER), "score", "--predictions", str(SUBM / "predictions"),
                    "--data-root", str(DATA), "--board", "A", "--split", "test",
                    "--out", str(SUBM / "scores.json")], env=ENV, check=True)
    with (SUBM / "check.json").open("w") as stream:
        subprocess.run([PY, str(SCORER), "check", "--scores", str(SUBM / "scores.json"),
                        "--data-root", str(DATA), "--board", "A"], env=ENV, check=True, stdout=stream)
    scores = json.loads((SUBM / "scores.json").read_text())
    (SUBM / "SUBMISSION-READY").write_text(json.dumps({
        "final_score": scores["final_score"], "scored_tasks": scores["scored_tasks"],
        "checkpoint": str(CKPT)}, indent=2), encoding="utf-8")
    log(f"SUBMISSION-READY final_score={scores['final_score']}")


def main():
    shards = json.loads((STAGE / "shards.json").read_text())["shards"]
    log("test chain watcher started")
    while True:
        for shard in shards:
            name = shard["name"]
            if (VAL / name / "results.json").exists() and not (TEST / name / "launched.txt").exists():
                if (TEST / name / "results.json").exists():
                    (TEST / name / "launched.txt").write_text("completed", encoding="utf-8")
                elif (TEST / name / "shard.log").exists():
                    (TEST / name / "launched.txt").write_text("adopted", encoding="utf-8")
                    log(f"{name}: 检测到已有 shard.log，按已启动收养")
                else:
                    try:
                        launch_test_shard(shard)
                    except Exception as exc:
                        log(f"{name}: 启动失败（下轮重试）: {str(exc)[:200]}")
        done = sum((TEST / s["name"] / "results.json").exists() for s in shards)
        log(f"test shards done {done}/16")
        if done == 16:
            finalize()
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
