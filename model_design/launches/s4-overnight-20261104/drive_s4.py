from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from aggregate_s4 import aggregate
from s4_common import PARENT, RUN_NAME, RUN_ROOT

ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "model_design/launches/s4-overnight-20261104"
CONFIG = ROOT / "model_design/configs/s4_overnight_polymer_consolidation.yaml"
FAST_EVAL = ROOT / "model_design/launches/eval16-20261027/fast_eval.py"
RESCUE = ROOT / "model_design/launches/stage2-20261031/rescue_heads.py"
PYTHON = Path("/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python")
DBAASP_TASK = "dbaasp_v4/target_activity/target_activity"
PARENT_DBAASP_RESCUE = 34.14737327267912
SUBSETS = (
    "Polymer-Genome-2016/DFT-HSE06,Polymer-Genome-2016/DFT-rPW86,"
    "TransPolymer/EPS,TransPolymer/Eea,TransPolymer/Egb,TransPolymer/Egc,"
    "TransPolymer/Ei,TransPolymer/Nc,TransPolymer/Xc,TransPolymer/PE_I,"
    "TransPolymer/PE_II,TransPolymer/OPV,Aldeghi-Coley-Copolymer/vipea,"
    "PolyGraphMT/ALPHA_DFT,PolyGraphMT/BANDGAP_DFT,PolyGraphMT/BULK_MD,"
    "PolyGraphMT/CP_GC,PolyGraphMT/DC_MD,PolyGraphMT/DIF_MD,PolyGraphMT/ETOTAL_DFT,"
    "PolyGraphMT/HOMO_DFT,PolyGraphMT/LUMO_DFT,PolyGraphMT/MU_DFT,PolyGraphMT/PE_MD,"
    "PolyGraphMT/POISSON_MD,PolyGraphMT/RG_MD,PolyGraphMT/RHO_MD,PolyGraphMT/RI_MD,"
    "PolyGraphMT/SHEAR_MD,PolyGraphMT/TC_MD,PolyGraphMT/TD_MD,PolyGraphMT/VISC_MD,"
    "PolyGraphMT/YOUNG_MD,dbaasp_v4/target_activity"
)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def event(name, **payload):
    row = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": name, **payload}
    with (RUN_ROOT / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def environment():
    env = os.environ.copy()
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "PYTHONPATH": ":".join([
            str(ROOT / "model_design"), str(ROOT / "model_design/launches/stage2-20261031"),
            str(STAGE), str(ROOT / "model_design/launches/eval16-20261027"), str(ROOT)]),
    })
    return env


def run_parallel(specs):
    processes = []
    handles = []
    try:
        for spec in specs:
            log_path = spec["log"]
            log_path.parent.mkdir(parents=True, exist_ok=True)
            if log_path.exists():
                raise FileExistsError(f"拒绝覆盖日志: {log_path}")
            handle = log_path.open("w", encoding="utf-8")
            handles.append(handle)
            env = environment()
            extra_env = spec.get("env", {})
            command = spec["command"]
            if spec.get("remote"):
                remote_env = environment()
                remote_env.update(extra_env)
                assignments = [f"{key}={shlex.quote(str(value))}"
                               for key, value in remote_env.items()
                               if key in ("PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS",
                                          "PYTORCH_ALLOC_CONF", "PYTHONPATH",
                                          "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")]
                command = ["ssh", "-o", "BatchMode=yes", "172.17.0.178",
                           f"cd {ROOT} && env {' '.join(assignments)} {shlex.join(command)}"]
            else:
                env.update(extra_env)
            processes.append(subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT))
        codes = [process.wait() for process in processes]
    finally:
        for handle in handles:
            handle.close()
    if any(codes):
        raise RuntimeError(f"并行任务失败: {codes}")


def training_spec(mode, step):
    run = RUN_ROOT / mode
    command = [
        str(PYTHON), "-B", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        "--nproc-per-node=4", str(STAGE / "run_s4.py"), "--config", str(CONFIG),
        "--run-dir", str(run), "--mode", mode, "--stop-after", str(step),
        "--micro-batch", "16", "--accumulation-steps", "1", "--prefetch", "1",
    ]
    if step > 50:
        command.extend(["--resume", str(run / "checkpoints/final.pt")])
    return {
        "command": command,
        "log": RUN_ROOT / "logs" / f"train_step{step}_{mode}.log",
        "env": {"CUDA_VISIBLE_DEVICES": "0,1,2,3" if mode == "treatment" else "4,5,6,7",
                "HIP_VISIBLE_DEVICES": "0,1,2,3" if mode == "treatment" else "4,5,6,7"},
        "remote": True,
    }


def evaluation_spec(mode, step):
    output = RUN_ROOT / "evaluation" / f"step{step}" / f"gate_{mode}"
    command = [
        str(PYTHON), "-B", "-u", str(FAST_EVAL), "--checkpoint",
        str(RUN_ROOT / mode / "checkpoints/final.pt"), "--output-dir", str(output),
        "--cache-root", str(ROOT / "data/pretrain/cache/frozen_eval"),
        "--external-cache", str(ROOT / "data/pretrain/cache/omc25"),
        "--device", "cuda:0" if mode == "treatment" else "cuda:4",
        "--devices", "1,2,3" if mode == "treatment" else "5,6,7",
        "--cpu-workers", "12", "--batch-size", "64", "--train-cap", "50000",
        "--subsets", SUBSETS, "--second-split", "validation", "--single-seed",
        "--name", f"s4-{mode}-gate{step}",
    ]
    return {"command": command, "log": RUN_ROOT / "logs" / f"eval_step{step}_{mode}.log",
            "remote": True}


def rescue_spec(step):
    run = RUN_ROOT / "evaluation" / f"step{step}" / "gate_treatment"
    return {
        "command": [str(PYTHON), "-B", "-u", str(RESCUE), "--cache-root", str(run),
                    "--output-dir", str(run / "rescue_dbaasp"), "--task-id", DBAASP_TASK,
                    "--device", "cuda:0"],
        "log": RUN_ROOT / "logs" / f"rescue_dbaasp_step{step}.log",
        "remote": True,
    }


def rescued_score(step):
    path = RUN_ROOT / "evaluation" / f"step{step}" / "gate_treatment/rescue_dbaasp" / \
        "summary_cuda_0.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != 1 or rows[0]["task_id"] != DBAASP_TASK or rows[0]["test_used"]:
        raise RuntimeError("DBAASP rescue 结果无效")
    return float(rows[0]["best"]["metrics"]["score"]) * 100.0


def gate_decision(step, parent, treatment, control, rescue):
    gap_parent = treatment["gap_mean"] > parent["gap_mean"]
    gap_control = treatment["gap_mean"] > control["gap_mean"]
    polymer_parent = treatment["polymer_dataset_sum"] > parent["polymer_dataset_sum"]
    polymer_control = treatment["polymer_dataset_sum"] > control["polymer_dataset_sum"]
    dbaasp_default_delta = treatment["dbaasp_default"] - parent["dbaasp_default"]
    dbaasp_rescue_delta = rescue - PARENT_DBAASP_RESCUE
    dbaasp_retained = dbaasp_default_delta >= -5.0 and dbaasp_rescue_delta >= -5.0
    checks = {
        "gap_above_parent": gap_parent,
        "gap_above_replay_control": gap_control,
        "polymer_above_parent": polymer_parent,
        "polymer_above_replay_control": polymer_control,
        "dbaasp_retained": dbaasp_retained,
    }
    if step == 50:
        passed = gap_parent and gap_control and dbaasp_retained
    elif step == 100:
        improvement = treatment["datasets"]["Polymer-Genome-2016"] >= \
            parent["datasets"]["Polymer-Genome-2016"] + 3.0 \
            or treatment["gap_mean"] >= parent["gap_mean"] + 5.0
        checks["polymer_improvement_threshold"] = improvement
        passed = all(checks.values())
    else:
        raise ValueError(f"没有 step{step} 的提前门控")
    return {
        "step": step,
        "passed": passed,
        "checks": checks,
        "dbaasp_default_delta": dbaasp_default_delta,
        "dbaasp_rescue": rescue,
        "dbaasp_rescue_delta": dbaasp_rescue_delta,
        "parent": parent,
        "treatment": treatment,
        "control": control,
        "test_used": False,
    }


def main():
    parent_path = RUN_ROOT / "evaluation/step0/gate_parent/official_scores.json"
    parent = None
    history = []
    for step in (50, 100):
        event("training_started", step=step, modes=["treatment", "control"], effective_batch=64,
              execution="remote178", treatment_gpus="0-3", control_gpus="4-7")
        run_parallel([training_spec(mode, step) for mode in ("treatment", "control")])
        event("training_completed", step=step, modes=["treatment", "control"])
        event("gate_evaluation_started", step=step, split="validation", scope="polymer_and_dbaasp")
        run_parallel([evaluation_spec(mode, step) for mode in ("treatment", "control")])
        run_parallel([rescue_spec(step)])
        if parent is None:
            parent = aggregate([parent_path], "gate")
            write_json(RUN_ROOT / "evaluation/step0/parent_gate.json", parent)
        scores = {}
        for mode in ("treatment", "control"):
            output = RUN_ROOT / "evaluation" / f"step{step}"
            scores[mode] = aggregate([output / f"gate_{mode}/official_scores.json"], "gate")
            write_json(output / f"{mode}_gate.json", scores[mode])
        decision = gate_decision(step, parent, scores["treatment"], scores["control"], rescued_score(step))
        write_json(RUN_ROOT / "evaluation" / f"step{step}/gate_decision.json", decision)
        history.append(decision)
        write_json(RUN_ROOT / "gate_history.json", {"run_name": RUN_NAME, "history": history})
        event("gate_completed", step=step, passed=decision["passed"], checks=decision["checks"],
              gap_mean=scores["treatment"]["gap_mean"], dbaasp_rescue=decision["dbaasp_rescue"])
        if not decision["passed"]:
            write_json(RUN_ROOT / "stopped.json", {
                "step": step, "reason": "gate_failed", "decision": decision})
            print(json.dumps({"status": "stopped", "step": step, "checks": decision["checks"]},
                             ensure_ascii=False), flush=True)
            return
    write_json(RUN_ROOT / "gate100_passed.json", {"status": "passed", "step": 100})
    print(json.dumps({"status": "gate100_passed", "next": "resume to 200/300 and full validation"}),
          flush=True)


if __name__ == "__main__":
    main()
