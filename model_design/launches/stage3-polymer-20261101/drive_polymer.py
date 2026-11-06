"""Drive polymer training, evaluation, hard gates, and checkpoint selection."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/yinghepool/tech-dcu/zhaoyao")
STAGE = ROOT / "model_design/launches/stage3-polymer-20261101"
RUN = ROOT / "model_design/runs/stage3-polymer-electronic-v1"
PYTHON = "/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python"
REMOTE = "172.17.0.178"
ENV = (
    "PYTHONDONTWRITEBYTECODE=1 "
    "PYTHONPATH=model_design:model_design/launches/stage2-20261031:"
    "model_design/launches/stage3-polymer-20261101:."
)
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
    "PolyGraphMT/YOUNG_MD"
)


def remote(command):
    return ["ssh", "-o", "BatchMode=yes", REMOTE, f"cd {ROOT} && {command}"]


def parallel(commands):
    processes = [subprocess.Popen(remote(command)) for command in commands]
    codes = [process.wait() for process in processes]
    if any(codes):
        raise RuntimeError(f"远端任务失败: {codes}")


def training_command(mode, stop_after, resume=False):
    run = RUN / mode
    gpus = "0" if mode == "treatment" else "4"
    checkpoint = f"--resume {run / 'checkpoints/final.pt'}" if resume else ""
    return (
        f"env CUDA_VISIBLE_DEVICES={gpus} {ENV} {PYTHON} {STAGE}/run_polymer.py "
        f"--run-dir {run} --mode {mode} --stop-after {stop_after} "
        f"--micro-batch 16 --accumulation-steps 4 {checkpoint}"
    )


def evaluation_commands(step):
    commands = []
    for mode in ("treatment", "control"):
        output = RUN / "evaluation" / f"step{step}" / mode
        gpus = "0,1,2,3" if mode == "treatment" else "4,5,6,7"
        device = "0" if mode == "treatment" else "4"
        checkpoint = RUN / mode / "checkpoints/final.pt"
        command = (
            f"env {ENV} {PYTHON} {ROOT}/model_design/launches/eval16-20261027/fast_eval.py "
            f"--checkpoint {checkpoint} --output-dir {output} --devices {gpus} "
            f"--device cuda:{device} --cpu-workers 16 --batch-size 64 --subsets {SUBSETS} "
            f"--second-split validation --single-seed --name stage3-{mode}{step} "
            f"> {RUN / mode / f'eval{step}.log'} 2>&1"
        )
        commands.append(command)
    return commands


def aggregate(step):
    output = RUN / "evaluation" / f"step{step}"
    results = {}
    for mode in ("treatment", "control"):
        out = output / f"{mode}_scores.json"
        subprocess.run([
            PYTHON, str(STAGE / "aggregate_polymer.py"), "--scores",
            str(output / mode / "official_scores.json"), "--out", str(out),
        ], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        results[mode] = json.loads(out.read_text(encoding="utf-8"))
    return results


def update_comparison(parent, history):
    payload = {
        "schema_version": "muse.stage3.polymer_pilot_comparison.v1",
        "parent": parent, "steps": history, "test_used": False,
    }
    (RUN / "pilot_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    forgetting = []
    for entry in history:
        treatment = entry["treatment"]
        forgetting.append({
            "step": entry["step"],
            "dataset_delta": {
                name: treatment["datasets"][name] - parent["datasets"][name]
                for name in parent["datasets"]
            },
            "J_delta": treatment["J_polymer"] - parent["J_polymer"],
        })
    (RUN / "polymer_forgetting.json").write_text(
        json.dumps(forgetting, ensure_ascii=False, indent=2), encoding="utf-8")


def stop(reason):
    (RUN / "stopped_reason.txt").write_text(reason + "\n", encoding="utf-8")
    print(reason, flush=True)


def select_best(parent, history):
    candidates = []
    for entry in history:
        treatment = entry["treatment"]
        control = entry["control"]
        if treatment["gap_mean"] <= parent["gap_mean"] \
                or treatment["gap_mean"] <= control["gap_mean"]:
            continue
        candidates.append(entry)
    if not candidates:
        best = {
            "step": 0, "checkpoint": str(ROOT / "model_design/runs/s3-residual-pilot-02/checkpoints/final.pt"),
            "score": parent, "reason": "no treatment candidate passed the basic gate",
        }
    else:
        entry = max(candidates, key=lambda item: (
            item["treatment"]["J_polymer"],
            item["treatment"]["datasets"]["Polymer-Genome-2016"],
            -item["step"],
        ))
        checkpoint_name = "final.pt" if entry["step"] == 300 else f"step_{entry['step']:08d}.pt"
        best = {
            "step": entry["step"],
            "checkpoint": str(RUN / "treatment/checkpoints" / checkpoint_name),
            "score": entry["treatment"], "reason": "maximum J_polymer; PG and fewer steps break ties",
        }
    checkpoint = Path(best["checkpoint"])
    best["sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (RUN / "best_checkpoint.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    parent = json.loads((RUN / "parent_scores.json").read_text(encoding="utf-8"))
    history = []
    for step in (50, 100, 200, 300):
        print(f"[stage3] train step={step} {time.strftime('%F %T')}", flush=True)
        parallel([training_command(mode, step, resume=step > 50) for mode in ("treatment", "control")])
        shutil.copyfile(RUN / "treatment/config.yaml", RUN / "config.yaml")
        shutil.copyfile(RUN / "treatment/data_manifest.jsonl", RUN / "data_manifest.jsonl")
        shutil.copyfile(RUN / "treatment/train_log.jsonl", RUN / "train_log.jsonl")
        shutil.copyfile(RUN / "treatment/checkpoints.jsonl", RUN / "checkpoints.jsonl")
        print(f"[stage3] evaluate step={step}", flush=True)
        parallel(evaluation_commands(step))
        results = aggregate(step)
        entry = {"step": step, **results}
        history.append(entry)
        update_comparison(parent, history)
        treatment, control = results["treatment"], results["control"]
        basic = treatment["gap_mean"] > parent["gap_mean"] \
            and treatment["gap_mean"] > control["gap_mean"]
        if step == 50 and not (basic and treatment["J_polymer"] >= parent["J_polymer"]):
            stop(f"stopped_at_50: basic_gate_failed basic={basic} "
                 f"J={treatment['J_polymer']:.6f} parent_J={parent['J_polymer']:.6f}")
            break
        if step == 100 and not (
                basic and treatment["gap_mean"] >= parent["gap_mean"] + 5.0
                and treatment["datasets"]["Polymer-Genome-2016"] > parent["datasets"]["Polymer-Genome-2016"]
                and treatment["J_polymer"] >= parent["J_polymer"]):
            stop(f"stopped_at_100: improvement_gate_failed gap_mean={treatment['gap_mean']:.6f} "
                 f"required={parent['gap_mean'] + 5.0:.6f} J={treatment['J_polymer']:.6f}")
            break
    else:
        stop("completed_300")
    select_best(parent, history)


if __name__ == "__main__":
    main()
