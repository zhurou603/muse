"""Package complete S5 A-test results, actual training configuration and loss snapshots."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import time
import zipfile


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def csv_bytes(rows):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["step", "micro_step", "samples_rank0", "lr", "elapsed_seconds",
                     "total", "energy", "force", "stress", "grad_norm", "batch_samples",
                     "skipped", "skip_reason"])
    for row in rows:
        losses = row.get("losses", {})
        writer.writerow([row.get(k) for k in ("step", "micro_step", "samples", "lr", "elapsed_seconds")]
                        + [losses.get(k) for k in ("total", "energy", "force", "stress")]
                        + [row.get("grad_norm"), row.get("batch_samples"),
                           row.get("skipped", False), row.get("skip_reason")])
    return output.getvalue().encode("utf-8")


def svg_loss(rows, skipped_steps):
    def values(key, gradient=False):
        result = []
        for row in rows:
            value = row.get("grad_norm") if gradient else row.get("losses", {}).get(key)
            if isinstance(value, (int, float)):
                result.append((row["step"], float(value)))
        return result

    def panel(points, y, title, color, skip_steps):
        width, height = 1060, 230
        left, right = 82, 1020
        xmin = min(x for x, _ in points)
        xmax = max(x for x, _ in points)
        ymin = min(v for _, v in points)
        ymax = max(v for _, v in points)
        yrange = ymax - ymin or 1.0
        ymin -= yrange * 0.05
        ymax += yrange * 0.05
        def xy(point):
            x, value = point
            px = left + (x - xmin) / max(1, xmax - xmin) * (right - left)
            py = y + height - (value - ymin) / (ymax - ymin) * height
            return px, py
        path = " ".join(("M" if i == 0 else "L") + f"{xy(p)[0]:.2f},{xy(p)[1]:.2f}"
                         for i, p in enumerate(points))
        parts = [f'<text x="{left}" y="{y - 16}" font-size="18" font-weight="bold">{html.escape(title)}</text>']
        for index in range(5):
            value = ymin + (ymax - ymin) * index / 4
            py = y + height - height * index / 4
            parts.append(f'<line x1="{left}" y1="{py:.2f}" x2="{right}" y2="{py:.2f}" stroke="#ddd"/>')
            parts.append(f'<text x="{left - 8}" y="{py + 4:.2f}" text-anchor="end" font-size="12">{value:.2f}</text>')
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.6"/>')
        for step in skip_steps:
            if xmin <= step <= xmax:
                px = left + (step - xmin) / max(1, xmax - xmin) * (right - left)
                parts.append(f'<line x1="{px:.2f}" y1="{y}" x2="{px:.2f}" y2="{y + height}" stroke="#c44" stroke-dasharray="5,4"/>')
        for step in sorted({xmin, (xmin + xmax) // 2, xmax}):
            px = left + (step - xmin) / max(1, xmax - xmin) * (right - left)
            parts.append(f'<text x="{px:.2f}" y="{y + height + 24}" text-anchor="middle" font-size="13">step {step}</text>')
        return "\n".join(parts)

    total = values("total")
    gradient = [(step, value) for step, value in values("total", gradient=True)]
    body = [panel(total, 55, "S5 total loss (rank0 log; mixed curriculum; unsmoothed)", "#1769aa", skipped_steps),
            panel(gradient, 390, "Gradient norm before clipping", "#237a45", skipped_steps),
            '<text x="82" y="656" font-size="13" fill="#555">Red dashed lines: nonfinite optimizer steps skipped without a parameter update. No validation/test labels are used for encoder training.</text>']
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="680" viewBox="0 0 1100 680">\n'
            '<rect width="100%" height="100%" fill="white"/>\n'
            '<g font-family="sans-serif" fill="#222">\n' + "\n".join(body) + "\n</g>\n</svg>\n").encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source_run = repo / "model_design/runs/s5-continuous-four-domain-v1"
    candidate = repo / "tools/s5_six_node_step17500_47rank_r3_20261106"
    run = candidate / "runs/main"
    require = lambda condition, message: (_ for _ in ()).throw(ValueError(message)) if not condition else None
    files = {}

    def add_bytes(path, data, category, source):
        require(path not in files, f"Duplicate archive path: {path}")
        files[path] = {"data": data, "category": category, "source": source,
                       "sha256": digest_bytes(data), "bytes": len(data)}

    def add_file(path, target, category):
        path = Path(path)
        require(path.is_file(), f"Missing input: {path}")
        add_bytes(target, path.read_bytes(), category, str(path))

    evaluation_rows = []
    for score_path in sorted(source_run.glob("official_a_test_*/scores.json")):
        eval_run = score_path.parent
        score = json.loads(score_path.read_text())
        summary_path = eval_run / "a_test_summary.json"
        check_path = eval_run / "check.json"
        require(summary_path.is_file() and check_path.is_file(), f"Incomplete evaluation metadata: {eval_run}")
        summary = json.loads(summary_path.read_text())
        check = json.loads(check_path.read_text())
        require(score["complete"] and score["scored_tasks"] == score["num_tasks"] == 90,
                f"Not a complete 90-task evaluation: {eval_run}")
        require(score["missing_predictions"] == [] and score["invalid_predictions"] == [],
                f"Invalid/missing predictions: {eval_run}")
        require(check == {"consistent": True, "problems": []}, f"Official check failed: {eval_run}")
        require(summary["check"] == check, f"Summary/check mismatch: {eval_run}")
        checkpoint = Path(summary["checkpoint"])
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        require(checkpoint_sha == summary["checkpoint_sha256"], f"Checkpoint hash mismatch: {checkpoint}")
        failed = list(eval_run.glob("**/*.failed.json"))
        nonempty = [p for p in failed if json.loads(p.read_text())]
        step = int(checkpoint.stem.split("_")[-1])
        entry = {
            "step": step, "run": eval_run.name, "score": score["final_score"],
            "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
            "complete": True, "scored_tasks": 90, "official_consistent": True,
            "input_repairs": summary.get("input_repairs"),
            "input_repairs_sha256": summary.get("input_repairs_sha256"),
            "train_cap": summary.get("train_cap", 50000), "single_seed": summary.get("single_seed", True),
            "domains": {name: value["score"] for name, value in score["domains"].items()},
            "failed_files": len(failed), "nonempty_failed_files": len(nonempty),
            "encoding_failure_records": sum(len(json.loads(p.read_text())) for p in failed),
            "predictions_manifest_sha256": score["submission"]["predictions_manifest_sha256"],
        }
        evaluation_rows.append(entry)
        for name in ("scores.json", "check.json", "a_test_summary.json"):
            add_file(eval_run / name, f"evaluations/{eval_run.name}/{name}", "complete_a_test")
        for name in ("comparisons.json", "comparison_to_step10800.json", "launch_record.json"):
            path = eval_run / name
            if path.is_file():
                add_file(path, f"evaluations/{eval_run.name}/{name}", "evaluation_comparison")

    require(len(evaluation_rows) == 4, f"Expected four complete S5 A-tests, found {len(evaluation_rows)}")
    evaluation_rows.sort(key=lambda row: row["step"])
    for previous, current in zip(evaluation_rows, evaluation_rows[1:]):
        current["delta_vs_previous"] = current["score"] - previous["score"]

    train_bytes = (run / "train.jsonl").read_bytes()
    train_rows = [json.loads(line) for line in train_bytes.splitlines() if line.strip()]
    require(train_rows, "No training loss records")
    for row in train_rows:
        require(all(isinstance(value, (int, float)) for value in row["losses"].values()), "Invalid loss")
    skip_bytes = (run / "optimization_skips.jsonl").read_bytes()
    skip_rows = [json.loads(line) for line in skip_bytes.splitlines() if line.strip()]
    skip_steps = sorted(set(row["step"] for row in skip_rows))
    require(all(row["optimizer_update"] is False for row in skip_rows), "Skipped step applied an optimizer update")

    add_bytes("loss/train_47rank.jsonl", train_bytes, "loss_snapshot", str(run / "train.jsonl"))
    add_file(source_run / "main/train.jsonl", "loss/train_15rank_source.jsonl", "source_loss_history")
    add_bytes("loss/train_47rank.csv", csv_bytes(train_rows), "loss_csv", "same frozen train.jsonl snapshot")
    add_bytes("loss/train_47rank.svg", svg_loss(train_rows, skip_steps), "loss_svg", "same frozen train.jsonl snapshot")
    add_bytes("loss/optimization_skips.jsonl", skip_bytes, "bounded_nonfinite_skip_audit",
              str(run / "optimization_skips.jsonl"))
    for name in ("throughput_100step_comparison.json", "throughput_400step_comparison.json"):
        add_file(candidate / "control" / name, f"loss/{name}", "measured_throughput")
    add_file(run / "data_skips.jsonl", "loss/data_skips.jsonl", "persistent_input_exclusions")

    for source, target in (
        (repo / "model_design/configs/s5_continuous.yaml", "config/s5_continuous.yaml"),
        (candidate / "config.yaml", "config/candidate_config.yaml"),
        (run / "config.resolved.yaml", "config/config.resolved.yaml"),
        (candidate / "contract.json", "config/expansion_contract.json"),
        (candidate / "prepared.json", "config/source_snapshot_prepared.json"),
        (repo / "tools/s5_six_node_candidate/source_hashes.json", "config/source_hashes.json"),
    ):
        add_file(source, target, "training_configuration")

    pointer_path = run / "latest.json"
    pointer_bytes = pointer_path.read_bytes()
    pointer = json.loads(pointer_bytes)
    checkpoint = Path(pointer["path"])
    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint_sha = digest_bytes(checkpoint_bytes)
    require(checkpoint_sha == pointer["sha256"], "Latest checkpoint hash mismatch")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["state"]
    require(state["step"] == pointer["step"], "Checkpoint step mismatch")
    require(payload["scheduler"]["last_epoch"] == state["step"], "Scheduler step mismatch")
    require(payload["optimizer"]["state"], "Missing optimizer state")
    require(all(torch.isfinite(value).all() for value in payload["model"].values()), "Nonfinite model parameter")
    rng_count = len(list(checkpoint.parent.glob(checkpoint.stem + ".rng_rank*.pt")))
    require(rng_count == 47, "Incomplete checkpoint RNG")
    require(payload["config"].get("nonfinite_skip_count", 0) == len(skip_steps),
            "Checkpoint skip count differs from audit")
    add_bytes("checkpoints/latest.json", pointer_bytes, "checkpoint_pointer", str(pointer_path))
    add_bytes("checkpoints/current_checkpoint_metadata.json", json_bytes({
        "path": str(checkpoint), "sha256": checkpoint_sha, "state": state,
        "metadata": payload["metadata"], "rng_sidecars": rng_count,
        "model_weights_included": False,
    }), "checkpoint_metadata", str(checkpoint))
    add_bytes("config/actual_runtime_from_checkpoint.json", json_bytes(payload["config"]),
              "actual_training_runtime", str(checkpoint))
    add_file(candidate / "control/resume_after_nonfinite_30700.json",
             "checkpoints/nonfinite_recovery_30700.json", "checkpoint_acceptance")
    add_file(candidate / "control/formal_checkpoint_17600_verified.json",
             "checkpoints/first_formal_checkpoint_17600.json", "checkpoint_acceptance")

    cluster_state = json.loads((candidate / "control/cluster_state.json").read_text())
    health_path = run / "health/sessions" / f"{Path(cluster_state['session']).stem}.summary.json"
    add_file(health_path, "health/current_training_summary.json", "training_health")
    health = json.loads(health_path.read_text())

    scoreboard = io.StringIO()
    writer = csv.writer(scoreboard)
    writer.writerow(["step", "run", "final_score", "delta_vs_previous", "complete", "official_consistent",
                     "small_molecule", "polymer", "organic_crystal", "biomacromolecule",
                     "checkpoint_sha256", "input_repairs_sha256"])
    for row in evaluation_rows:
        writer.writerow([row["step"], row["run"], row["score"], row.get("delta_vs_previous"),
                         row["complete"], row["official_consistent"],
                         *[row["domains"][name] for name in ("small_molecule", "polymer", "organic_crystal", "biomacromolecule")],
                         row["checkpoint_sha256"], row["input_repairs_sha256"]])
    add_bytes("summary/a_test_scoreboard.csv", scoreboard.getvalue().encode("utf-8"),
              "a_test_scoreboard", "complete official A-test scores")

    best = max(evaluation_rows, key=lambda row: row["score"])
    summary = {
        "complete_evaluations": len(evaluation_rows),
        "all_90_tasks_and_officially_consistent": True,
        "highest_local_full_a_test": {"step": best["step"], "run": best["run"], "score": best["score"]},
        "training_checkpoint_snapshot": pointer,
        "training_health_time": health["time"], "training_healthy": health["healthy"],
        "training_progress_ranks": health["progress_ranks"],
        "loss_records": len(train_rows),
        "loss_step_range": [min(row["step"] for row in train_rows), max(row["step"] for row in train_rows)],
        "loss_snapshot_sha256": digest_bytes(train_bytes),
        "nonfinite_skip_steps": skip_steps,
        "nonfinite_skip_records": len(skip_rows),
        "nonfinite_skip_optimizer_updates": False,
        "model_weights_included": False, "raw_predictions_included": False,
        "official_submission_overwritten": False, "uploaded": False, "model_selection_performed": False,
        "validation_test_labels_used_for_encoder_training": False,
    }
    add_bytes("summary/package_summary.json", json_bytes(summary), "package_summary", "generated from snapshots")
    lines = [
        "# S5 训练与完整 A-test 结果包", "",
        "本包只读收集当前 S5 结果，不上传、不覆盖正式提交，不替代模型质量门控。", "",
        "## 四次完整 A-test", "",
        "均为90/90、complete=true、missing/invalid为空、官方check一致；单种子、train_cap=50000。", "",
        "| checkpoint step | run | score | 与前一个列出的checkpoint分差 |", "|---:|---|---:|---:|",
    ]
    for row in evaluation_rows:
        delta = "—" if "delta_vs_previous" not in row else f"{row['delta_vs_previous']:+.12f}"
        lines.append(f"| {row['step']} | `{row['run']}` | {row['score']:.12f} | {delta} |")
    lines += ["", f"当前本地最高完整A-test为step {best['step']}：**{best['score']:.12f}**。这不是未经质量门控的best_fixed或自动选模。", "",
              "## 内容", "",
              "- `evaluations/`：四次完整A-test的scores、check、summary及已有比较/启动记录。",
              "- `config/`：原始YAML、解析后YAML、实际checkpoint运行配置、扩容contract和源码hash。",
              "- `loss/`：47rank与15rank原始JSONL、47rank CSV和SVG矢量曲线、异常输入与非有限步审计、实测吞吐。",
              "- `checkpoints/`：只读checkpoint指针、元数据、hash和恢复验收，不包含模型权重。",
              "- `summary/`：完整成绩CSV与包级摘要。",
              "- `manifest.json`、`SHA256SUMS`：逐文件hash与来源。", "",
              "## 当前训练快照与边界", "",
              f"- 最新保存点：step {pointer['step']}；SHA256 `{pointer['sha256']}`。",
              f"- 健康摘要时间：{health['time']}；47rank healthy={health['healthy']}，课程一致={health['source_consistent']}。",
              f"- loss记录{len(train_rows)}条，step范围 {summary['loss_step_range'][0]}–{summary['loss_step_range'][1]}；混合四域课程，不保证单调下降。",
              "- loss是rank0日志，不是重新计算的全rank均值；samples是rank0展开行计数，不是全局唯一数据量。",
              "- SVG未平滑、未删除离群点；红色竖线为非有限优化步跳过，并非普通训练loss。",
              f"- 非有限优化步：{skip_steps}；共{len(skip_rows)}条rank级记录，未执行这些步的参数更新。",
              "- 原始训练采样有放回，不能用step直接换算epoch；token/s未记录。",
              "- 原始预测文件、原始数据和模型权重未打包；scores保留预测manifest hash，可回源核验。",
              "- 未读取validation/test标签训练或归一化Encoder，不加载第三方预训练权重。",
              "- 正式提交目录未被覆盖，未上传赛事平台。", ""]
    add_bytes("README.md", "\n".join(lines).encode("utf-8"), "readme", "generated package guide")

    manifest = {
        "schema": "s5.training_a_test_package.v1", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo": str(repo), "git_commit": __import__("subprocess").check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "summary": summary,
        "files": [{"path": path, "category": item["category"], "source": item["source"],
                   "sha256": item["sha256"], "bytes": item["bytes"]}
                  for path, item in sorted(files.items())],
    }
    add_bytes("manifest.json", json_bytes(manifest), "manifest", "generated file inventory")
    checksums = "".join(f"{item['sha256']}  {path}\n" for path, item in sorted(files.items()))
    add_bytes("SHA256SUMS", checksums.encode("utf-8"), "checksums", "hash of every package payload")
    prefix = "s5_training_a_test_results"
    output = args.output.resolve()
    require(not output.exists(), f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".tmp")
    require(not temporary.exists(), f"Temporary output already exists: {temporary}")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path, item in sorted(files.items()):
                info = zipfile.ZipInfo(f"{prefix}/{path}", time.localtime()[:6])
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, item["data"])
        with zipfile.ZipFile(temporary) as archive:
            require(archive.testzip() is None, "ZIP CRC validation failed")
            for path, item in files.items():
                require(digest_bytes(archive.read(f"{prefix}/{path}")) == item["sha256"],
                        f"ZIP payload hash mismatch: {path}")
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    result = {"output": str(output), "sha256": digest_bytes(output.read_bytes()),
              "bytes": output.stat().st_size, "files": len(files),
              "complete_a_tests": len(evaluation_rows),
              "highest_local_score": best["score"], "checkpoint_step": pointer["step"],
              "loss_records": len(train_rows), "nonfinite_skip_steps": skip_steps,
              "zip_crc_verified": True, "all_payload_hashes_verified": True,
              "model_weights_included": False, "raw_predictions_included": False,
              "training_stopped_for_packaging": False,
              "official_submission_overwritten": False, "uploaded": False}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
