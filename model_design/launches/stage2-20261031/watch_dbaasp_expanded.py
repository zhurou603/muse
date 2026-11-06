"""Evaluate expanded DBAASP checkpoints as they become available."""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/yinghepool/tech-dcu/zhaoyao")
RUN = ROOT / "model_design/runs/stage2-dbaasp-expanded-300"
PYTHON = "/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python"
ENV = {
    **os.environ,
    "PYTHONPATH": f"{ROOT}/model_design:{ROOT}",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def run_logged(command, log_path):
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, env=ENV, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def main():
    for step, filename in ((100, "step_00000100.pt"), (200, "step_00000200.pt"),
                           (300, "final.pt")):
        checkpoint = RUN / "checkpoints" / filename
        deadline = time.monotonic() + 4 * 3600
        while not checkpoint.is_file():
            if time.monotonic() > deadline:
                raise TimeoutError(f"等待 checkpoint 超时: {checkpoint}")
            time.sleep(30)
        evaluation = RUN / f"eval{step}"
        rescue = RUN / f"rescue{step}"
        run_logged([
            PYTHON, str(ROOT / "model_design/launches/eval16-20261027/fast_eval.py"),
            "--checkpoint", str(checkpoint), "--output-dir", str(evaluation),
            "--devices", "2", "--device", "cuda:2", "--cpu-workers", "8",
            "--batch-size", "64", "--subsets", "dbaasp_v4/target_activity",
            "--second-split", "validation", "--single-seed",
            "--name", f"stage2-dbaasp-expanded{step}",
        ], RUN / f"eval{step}.log")
        run_logged([
            PYTHON, str(ROOT / "model_design/launches/stage2-20261031/rescue_heads.py"),
            "--cache-root", str(evaluation), "--output-dir", str(rescue),
            "--device", "cuda:2", "--task-id", "dbaasp_v4/target_activity/target_activity",
        ], RUN / f"rescue{step}.log")
        result_path = rescue / "dbaasp_v4__target_activity__target_activity/result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        status = {
            "step": step, "checkpoint": str(checkpoint),
            "mae": result["validation_metrics"]["value"],
            "score": 100 * result["validation_metrics"]["score"],
            "test_used": False,
        }
        with (RUN / "evaluation_progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(status) + "\n")
        print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
