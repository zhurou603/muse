"""Safely extract a .tar.zst archive without links, path traversal or overwrites."""
from __future__ import annotations

import argparse
import os
import subprocess
import tarfile
from pathlib import Path, PurePosixPath


def safe_name(name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"非法归档路径: {name!r}")
    return Path(*pure.parts)


def extract(archive: Path, output_root: Path):
    label = archive.name[:-len(".tar.zst")] if archive.name.endswith(".tar.zst") else archive.stem
    destination = output_root / label
    if destination.exists():
        raise FileExistsError(f"拒绝覆盖解包目录: {destination}")
    temporary = output_root / (label + ".extracting")
    if temporary.exists():
        raise FileExistsError(f"存在未完成解包目录，请人工检查: {temporary}")
    temporary.mkdir(parents=True)
    process = subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE)
    seen = set()
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as tar:
            for member in tar:
                relative = safe_name(member.name)
                if member.issym() or member.islnk() or member.isdev() or not (member.isfile() or member.isdir()):
                    raise ValueError(f"拒绝链接/设备/非常规成员: {member.name}")
                key = str(relative)
                if key in seen and not member.isdir():
                    raise ValueError(f"重复归档成员: {key}")
                seen.add(key)
                target = temporary / relative
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise ValueError(f"无法读取成员: {member.name}")
                with target.open("xb") as output:
                    remaining = member.size
                    while remaining:
                        block = source.read(min(8 * 1024 * 1024, remaining))
                        if not block:
                            raise EOFError(f"成员提前结束: {member.name}")
                        output.write(block)
                        remaining -= len(block)
        if process.wait() != 0:
            raise RuntimeError("zstd 解压失败")
        os.replace(temporary, destination)
    except Exception:
        process.kill()
        process.wait()
        raise
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    extract(args.archive, args.output_root)
