"""Deliberately pin a reviewed, committed production tree before prepare."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from prepare_candidate import require, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    relative = ["model_design/muse_encoder", "model_design/configs/s5_continuous.yaml", "muse"]
    subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", *relative], check=True)
    subprocess.run(["git", "-C", str(repo), "diff", "--cached", "--quiet", "HEAD", "--", *relative], check=True)
    paths = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "model_design/muse_encoder/*.py",
         "model_design/configs/s5_continuous.yaml", "muse/**/*.py"], text=True).splitlines()
    root = Path(__file__).resolve().parent
    values = {}
    for name in sorted(paths):
        payload = subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:{name}"])
        require(payload == (repo / name).read_bytes(), f"Working tree differs: {name}")
        values[name] = hashlib.sha256(payload.decode("utf-8").encode("utf-8")).hexdigest()
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    write_json(root / "source_hashes.json", {
        "schema_version": "s5.source_pin.v2", "source_git_commit": commit, "files": values,
    })
    print(f"PINNED {len(values)} committed source files at {commit}")


if __name__ == "__main__":
    main()
