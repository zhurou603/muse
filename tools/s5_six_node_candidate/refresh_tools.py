"""Refresh only a never-launched candidate after a reviewed tool-only commit."""
import argparse
from pathlib import Path
import shutil
import subprocess

from prepare_candidate import digest, require, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    root = args.candidate.resolve()
    require(not (root / "runs").exists(), "Training/smoke exists: prepare a new candidate instead")
    require(not (root / "control/drivers").exists(), "A node has launched: prepare a new candidate")
    source = Path(__file__).resolve().parent
    target = root / "tools/s5_six_node_candidate"
    prepared_path = root / "prepared.json"
    prepared = __import__("json").loads(prepared_path.read_text())
    changed = {}
    for path in source.rglob("*.py"):
        relative = path.relative_to(source)
        destination = target / relative
        key = destination.relative_to(root).as_posix()
        if not destination.exists() or digest(path) != digest(destination):
            changed[key] = {"before": digest(destination) if destination.exists() else None,
                            "after": digest(path)}
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    for path in (root / "model_design").rglob("*.py"):
        override = source / "overrides" / path.relative_to(root / "model_design/muse_encoder")
        if override.exists() and digest(path) != digest(override):
            key = path.relative_to(root).as_posix()
            changed[key] = {"before": digest(path), "after": digest(override)}
            path.write_bytes(override.read_bytes())
    for key, value in changed.items():
        prepared["files"][key] = value["after"]
    for base in (root / "model_design", root / "tools"):
        for path in base.rglob("*.py"):
            key = path.relative_to(root).as_posix()
            checksum = digest(path)
            if prepared["files"].get(key) != checksum and key not in changed:
                changed[key] = {"before": prepared["files"].get(key), "after": checksum}
            prepared["files"][key] = checksum
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    prepared["tool_git_commit"] = commit
    write_json(prepared_path, prepared)
    write_json(root / f"prelaunch_revision_{commit}.json", {"changed": changed,
                "reason": "reviewed prelaunch changes only; no training or smoke session has started"})
    print(f"Refreshed {len(changed)} files at {commit}; checkpoint/config/contract unchanged")


if __name__ == "__main__":
    main()
