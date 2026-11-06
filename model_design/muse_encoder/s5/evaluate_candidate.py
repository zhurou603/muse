"""Read-only frozen-checkpoint evaluation and fail-closed best_fixed gating."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from .provenance import atomic_json, checkpoint_record, sha256_file
from .runtime import load_yaml, resolve_resume_path


def summarize_result(result: Mapping[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    tasks = result.get("tasks") or {}
    missing = sum(int(row.get("missing_predictions") or 0) for row in tasks.values())
    total = sum(int(row.get("n_validation_total") or 0) for row in tasks.values())
    properties = (result.get("shadow") or {}).get("properties") or {}
    shadow_scores = []
    for value in properties.values():
        metrics = value.get("metrics") or {}
        score = value.get("normalized_score", metrics.get("score"))
        if value.get("status") == "ok" and score is not None:
            shadow_scores.append(float(score))
    return {
        "J_encoder_fixed": float(summary["final_score"]),
        "domains": {name: float(value["score"]) for name, value in (summary.get("domains") or {}).items()},
        "complete": bool(summary.get("complete")),
        "is_full_a_board": bool(summary.get("is_full_a_board")),
        "smoke": bool(result.get("smoke")),
        "scored_tasks": int(summary.get("scored_tasks", 0)),
        "num_tasks": int(summary.get("num_tasks", 0)),
        "fallback_predictions": missing,
        "validation_observations": total,
        "encoder_missing_fraction": missing / max(total, 1),
        "shadow_status": (result.get("shadow") or {}).get("status"),
        "shadow_properties": len(properties),
        "shadow_ok": len(shadow_scores),
        "shadow_mean_score": sum(shadow_scores) / len(shadow_scores) if shadow_scores else None,
        "fixed_probe_protocol": result.get("fixed_probe_protocol"),
    }


def validate_full(metrics: Mapping[str, Any], eval_config: Mapping[str, Any]) -> list[str]:
    reasons = []
    if not metrics["complete"] or not metrics["is_full_a_board"] or metrics["smoke"]:
        reasons.append("不是完整 90 项 validation")
    if (metrics["scored_tasks"], metrics["num_tasks"]) != (90, 90):
        reasons.append(f"任务覆盖 {metrics['scored_tasks']}/{metrics['num_tasks']} != 90/90")
    if metrics["fixed_probe_protocol"] != "muse_encoder.s5_fixed_robust_probe.v1":
        reasons.append("固定 robust probe 协议标记缺失")
    expected_shadow = int(eval_config.get("shadow_properties", 8))
    if metrics["shadow_status"] != "ok" or metrics["shadow_ok"] != expected_shadow:
        reasons.append(f"shadow-B {metrics['shadow_ok']}/{expected_shadow} 不完整")
    if metrics["encoder_missing_fraction"] > float(eval_config.get("max_encoder_missing_fraction", 0.001)):
        reasons.append("Encoder 缺失/回退比例超过门槛")
    return reasons


def validate_bootstrap(metrics: Mapping[str, Any], state: Mapping[str, Any],
                       eval_config: Mapping[str, Any]) -> list[str]:
    """Bootstrap exactly once; an existing best checkpoint is never overwritten."""
    reasons = validate_full(metrics, eval_config)
    if state.get("best_fixed"):
        reasons.append("best_fixed 已存在；禁止用 --bootstrap 覆盖，请使用新 state 或普通门控")
    return reasons


def gate(metrics: Mapping[str, Any], best: Mapping[str, Any],
         eval_config: Mapping[str, Any]) -> list[str]:
    reasons = validate_full(metrics, eval_config)
    minimum = float(best["J_encoder_fixed"]) + float(eval_config.get("min_j_delta", 0.02))
    if float(metrics["J_encoder_fixed"]) < minimum:
        reasons.append(f"J_encoder_fixed {metrics['J_encoder_fixed']:.6f} < {minimum:.6f}")
    thresholds = eval_config.get("domain_forget_thresholds", {})
    for domain, old_score in (best.get("domains") or {}).items():
        drop = float(old_score) - float(metrics["domains"].get(domain, -1e9))
        allowed = float(thresholds.get(domain, 0.20))
        if drop > allowed:
            reasons.append(f"{domain} 遗忘 {drop:.6f} > {allowed:.6f}")
    if best.get("shadow_mean_score") is not None and metrics.get("shadow_mean_score") is not None:
        drop = float(best["shadow_mean_score"]) - float(metrics["shadow_mean_score"])
        allowed = float(eval_config.get("shadow_forget_threshold", 0.01))
        if drop > allowed:
            reasons.append(f"shadow-B 遗忘 {drop:.6f} > {allowed:.6f}")
    return reasons


def freeze_checkpoint(source: Path, candidate_dir: Path) -> dict[str, Any]:
    source = resolve_resume_path(source)
    candidate_dir.mkdir(parents=True, exist_ok=False)
    frozen = candidate_dir / "encoder.pt"
    temporary = candidate_dir / "encoder.pt.tmp"
    shutil.copyfile(source, temporary)
    os.replace(temporary, frozen)
    source_hash, frozen_hash = sha256_file(source), sha256_file(frozen)
    if source_hash != frozen_hash:
        raise RuntimeError("冻结 checkpoint SHA256 不一致")
    return checkpoint_record(frozen, extra={"source": str(source.resolve()),
                                            "source_sha256": source_hash})


def resume_frozen_checkpoint(source: Path, candidate_dir: Path,
                             evaluation_root: Path) -> tuple[dict[str, Any], str]:
    candidate_dir = candidate_dir.resolve()
    if candidate_dir.parent != evaluation_root.resolve():
        raise ValueError("续跑目录必须直接位于 evaluation-root 下")
    frozen_path = candidate_dir / "encoder.pt"
    state_path = candidate_dir / "dashboard" / "run_state.json"
    manifest_path = candidate_dir / "dashboard" / "predictions" / "manifest.json"
    for path in (frozen_path, state_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"续跑所需文件不存在: {path}")
    source = resolve_resume_path(source)
    source_hash, frozen_hash = sha256_file(source), sha256_file(frozen_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    recorded_hash = (state.get("checkpoint") or {}).get("sha256")
    if recorded_hash is None or not source_hash == frozen_hash == recorded_hash:
        raise RuntimeError("续跑父 checkpoint、冻结副本与 run_state SHA256 不一致")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    name = manifest.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("续跑 predictions manifest 缺少 name")
    return checkpoint_record(frozen_path, extra={"source": str(source.resolve()),
                                                "source_sha256": source_hash}), name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path,
                        help="续跑同一 evaluation-root 下已有的完整候选目录，复用已校验缓存")
    parser.add_argument("--mode", choices=("sentinel", "full"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encoding-devices", default=None)
    parser.add_argument("--cpu-workers", type=int, default=14)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--bootstrap", action="store_true",
                        help="仅对父模型完整结果显式建立 best_fixed 基线")
    args = parser.parse_args()
    config = load_yaml(args.config)
    eval_config = config["evaluation"]
    stamp = time.strftime("%Y%m%dT%H%M%S")
    if args.resume_from is None:
        candidate_dir = args.evaluation_root / f"{stamp}_{args.mode}"
        frozen = freeze_checkpoint(args.checkpoint, candidate_dir)
        name = f"s5-{args.mode}-{stamp}"
    else:
        candidate_dir = args.resume_from
        frozen, name = resume_frozen_checkpoint(
            args.checkpoint, candidate_dir, args.evaluation_root)
    command = [
        sys.executable, "-m", "muse_encoder.s5.robust_dashboard",
        "--checkpoint", str(candidate_dir / "encoder.pt"),
        "--output-dir", str(candidate_dir / "dashboard"),
        "--device", args.device,
        "--batch-size", str(eval_config.get("batch_size", 128)),
        "--team", str(eval_config.get("team", "zhaoyao")),
        "--name", name,
    ]
    if eval_config.get("official_graph_repairs"):
        command += ["--official-graph-repairs", str(eval_config["official_graph_repairs"])]
    if args.encoding_devices:
        command += ["--devices", args.encoding_devices, "--cpu-workers", str(args.cpu_workers)]
    if eval_config.get("train_cap") is not None:
        command += ["--train-cap", str(eval_config["train_cap"])]
    if args.mode == "sentinel":
        command += ["--limit-per-split", str(eval_config["sentinel_limit"]),
                    "--shadow-limit", str(eval_config["sentinel_shadow_limit"])]
        if eval_config.get("sentinel_subsets"):
            command += ["--subsets", ",".join(eval_config["sentinel_subsets"])]
    log_path = candidate_dir / "evaluation.log"
    with log_path.open("a" if args.resume_from is not None else "w", encoding="utf-8") as log:
        if args.resume_from is not None:
            log.write(f"\n[resume] {time.strftime('%Y-%m-%dT%H:%M:%S%z')} checkpoint={frozen['sha256']}\n")
            log.flush()
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"评估失败，见 {log_path}")
    result_path = candidate_dir / "dashboard" / "results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metrics = summarize_result(result)
    metrics.update({"mode": args.mode, "frozen_checkpoint": frozen,
                    "results": str(result_path), "evaluated_at": stamp})

    if args.mode == "sentinel":
        metrics["accepted"] = False
        metrics["note"] = "sentinel/smoke 只监控，不接受 best_fixed，不是完整总分"
        atomic_json(candidate_dir / "decision.json", metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    full_errors = validate_full(metrics, eval_config)
    state = json.loads(args.state.read_text(encoding="utf-8")) if args.state.exists() else {}
    if args.bootstrap:
        reasons = validate_bootstrap(metrics, state, eval_config)
    elif not state.get("best_fixed"):
        reasons = full_errors + ["缺少显式父模型 best_fixed 基线"]
    else:
        reasons = gate(metrics, state["best_fixed"], eval_config)
    metrics["accepted"] = not reasons
    metrics["rejection_reasons"] = reasons
    if not reasons:
        state["best_fixed"] = metrics
        state.setdefault("history", []).append({
            "accepted": True, "J_encoder_fixed": metrics["J_encoder_fixed"],
            "checkpoint_sha256": frozen["sha256"], "results": str(result_path),
        })
        atomic_json(args.state, state)
    else:
        state.setdefault("rejected", []).append({
            "checkpoint_sha256": frozen["sha256"], "results": str(result_path),
            "reasons": reasons,
        })
        # Rejections are audit history only; best_fixed remains byte-for-byte intact.
        atomic_json(args.state.with_suffix(".rejected.json"), state.get("rejected", []))
    atomic_json(candidate_dir / "decision.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if reasons:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
