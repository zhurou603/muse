"""Freeze an isolated candidate from audited production code and one S5 checkpoint (CPU only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path, text=False):
    if text:
        return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/yinghepool/tech-dcu/zhaoyao"))
    parser.add_argument("--source-run", type=Path, default=Path(
        "/yinghepool/tech-dcu/zhaoyao/model_design/runs/s5-continuous-four-domain-v1/main"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-log", type=Path,
                        help="已审核的追加式排除清单；必须包含源run的全部历史记录")
    args = parser.parse_args()
    require(not args.output.exists(), "Output already exists; use a new candidate directory")
    repo = args.repo.resolve()
    os.environ["REPO_ROOT"] = str(repo)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    import torch
    import yaml

    import subprocess
    pin = json.loads((ROOT / "source_hashes.json").read_text())
    require(pin.get("schema_version") == "s5.source_pin.v2", "Run pin_source.py on a reviewed commit")
    actual_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    source_commit = pin["source_git_commit"]
    ancestry = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                               source_commit, actual_commit])
    require(ancestry.returncode == 0, "Reviewed source commit is not an ancestor of production")
    subprocess.run(["git", "-C", str(repo), "diff", "--quiet", source_commit, actual_commit, "--",
                    "model_design/muse_encoder", "model_design/configs/s5_continuous.yaml", "muse"], check=True)
    subprocess.run(["git", "-C", str(repo), "diff", "--quiet", actual_commit, "--",
                    "model_design/muse_encoder", "model_design/configs/s5_continuous.yaml", "muse"], check=True)
    expected = pin["files"]
    for relative, checksum in expected.items():
        require(digest(repo / relative, text=True) == checksum,
                f"Production source drift: {relative}; obtain updated source before patching")
    package = repo / "model_design/muse_encoder"
    actual_py = {p.relative_to(repo).as_posix()
                 for base in (package, repo / "muse") for p in base.rglob("*.py")}
    require(actual_py == {p for p in expected if p.endswith(".py")}, "Python source inventory drift")
    cp = args.checkpoint.resolve()
    pointer = None
    if cp.suffix == ".json":
        pointer = json.loads(cp.read_text())
        cp = Path(pointer["path"]).resolve()
    require(cp.parent == args.source_run.resolve() / "checkpoints", "Unexpected source checkpoint directory")
    cp_hash = digest(cp)
    if pointer is not None:
        require(cp_hash == pointer["sha256"], "Checkpoint pointer hash mismatch")
    print(f"CPU CHECKPOINT CHECK: {cp}; reading full optimizer state...", flush=True)
    payload = torch.load(cp, map_location="cpu", weights_only=False)
    cfg_path = repo / "model_design/configs/s5_continuous.yaml"
    config = yaml.safe_load(os.path.expandvars(cfg_path.read_text(encoding="utf-8")))
    require(payload["config"]["config"] == config, "Current YAML differs from checkpoint configuration")
    contract = {
        "schema_version": "s5.expansion.15to47.v1", "repo_root": str(repo),
        "source_git_commit": source_commit,
        "source_git_tree": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", f"{source_commit}^{{tree}}"], text=True).strip(),
        "source_run": str(args.source_run.resolve()), "source_checkpoint": str(cp),
        "checkpoint_sha256": cp_hash, "source_world_size": 15, "target_world_size": 47,
        "source_state": payload["state"], "micro_batch": 8, "accumulation_steps": 1,
        "prefetch_per_rank": 1, "nominal_global_batch": 376,
        "data_manifest_sha256": digest(config["data"]["data_manifest"]),
        "learning_rate_policy": "restore optimizer and scheduler; no LR scaling or warmup restart",
        "sampling_policy": "retain stateless rank/micro-step seeds; no exact old trajectory guarantee",
        "rng_policy": "reseed ALL ranks on 15->47 only; strictly restore sidecars on 47->47",
        "samples_counter": "retain rank0-local historical expanded rows, not global coverage",
        "reserved_devices": {"VM-0-171-tencentos": [7], "purpose": "independent A-test"},
    }
    sys.path.insert(0, str(ROOT / "overrides"))
    from s5.expansion import resume_plan
    resume_plan(payload, config, contract, "preparation", cp_hash, 47, 0, 8, 1)
    require(payload["config"]["world_size"] == 15, "Initial preparation requires a 15-rank checkpoint")
    if pointer is not None:
        require(pointer["step"] == payload["state"]["step"] and
                pointer["micro_step"] == payload["state"]["micro_step"], "Pointer state mismatch")
    print(json.dumps({"state": payload["state"], "old_batch": 120, "new_batch": 376,
                      "optimizer_groups": [{k: g[k] for k in ("lr", "weight_decay", "group_name")}
                                           for g in payload["optimizer"]["param_groups"]]}, indent=2), flush=True)
    del payload

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    candidate_package = out / "model_design/muse_encoder"
    shutil.copytree(package, candidate_package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(repo / "muse", out / "muse", ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", "results", "scoreboard.csv"))
    for relative, checksum in expected.items():
        if relative.endswith(".py"):
            require(digest(out / relative, text=True) == checksum, f"Source changed while copying: {relative}")
    for override in (ROOT / "overrides").rglob("*.py"):
        target = candidate_package / override.relative_to(ROOT / "overrides")
        target.write_text(override.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    tools_out = out / "tools/s5_six_node_candidate"
    tools_out.mkdir(parents=True)
    for source in ROOT.rglob("*.py"):
        target = tools_out / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        shutil.copymode(source, target)
    (out / "data").symlink_to(repo / "data", target_is_directory=True)
    (out / "config.yaml").write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
    (out / "handoff").mkdir()
    shutil.copyfile(cp, out / "handoff/source.pt")
    require(digest(out / "handoff/source.pt") == cp_hash, "Checkpoint changed while copying")
    base_skip_path = args.source_run / "data_skips.jsonl"
    skip_path = (args.skip_log or base_skip_path).resolve()
    if skip_path.exists():
        if skip_path != base_skip_path:
            require(skip_path.read_bytes().startswith(base_skip_path.read_bytes()),
                    "追加排除清单未完整包含源run历史，拒绝丢失坏样本排除")
        skip_hash = digest(skip_path)
        shutil.copyfile(skip_path, out / "handoff/data_skips.jsonl")
        require(digest(out / "handoff/data_skips.jsonl") == skip_hash == digest(skip_path),
                "Skip log changed during copy; retry after source training is stopped")
        for line in (out / "handoff/data_skips.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("event") == "training_input_skipped":
                    require(isinstance(record["source"], str) and int(record["row"]) >= 0,
                            "Invalid source skip record")
        contract["inherited_skip_sha256"] = skip_hash
        contract["inherited_skip_source"] = str(skip_path)
    else:
        (out / "handoff/data_skips.jsonl").write_text("", encoding="utf-8")
        contract["inherited_skip_sha256"] = digest(out / "handoff/data_skips.jsonl")
    write_json(out / "contract.json", contract)
    files = {p.relative_to(out).as_posix(): digest(p)
             for base in (out / "model_design", out / "tools", out / "muse")
             for p in base.rglob("*.py")}
    for name in ("config.yaml", "contract.json", "handoff/data_skips.jsonl"):
        files[name] = digest(out / name)
    write_json(out / "prepared.json", {"files": files, "source_hashes": expected,
                                       "status": "prepared_not_runtime_verified"})
    print(f"PREPARED: {out}; source checkpoint/production unchanged; no GPU or training.")
    print("This pins a checkpoint, not a live latest pointer. Coordinate the source stop before final handoff.")


if __name__ == "__main__":
    main()
