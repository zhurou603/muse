"""Verify every packaged failure, write checksums, and create the delivery ZIP."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_files(root: Path):
    return sorted(path for path in root.rglob("*") if path.is_file()
                  and path.name not in {"CHECKSUMS.sha256", "delivery.json"}
                  and path.suffix != ".log" and "__pycache__" not in path.parts)


def main() -> None:
    root = Path(__file__).resolve().parent
    verification = root / "verification.json"
    subprocess.run([sys.executable, str(root / "reproduce.py"),
                    "--output", str(verification)], cwd=root, check=True)
    result = json.loads(verification.read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if result["selected"] != result["reproduced"] or result["not_reproduced"]:
        raise RuntimeError("仍有未复现用例")
    if result["selected"] != manifest["total_failures"]:
        raise RuntimeError("验证数量与 manifest 不一致")
    for path in (root / "records").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if "labels" in record or "targets" in record:
                raise RuntimeError(f"{path}: 包含禁止交付的标签")
    files = payload_files(root)
    checksums = {str(path.relative_to(root)): sha256_file(path) for path in files}
    (root / "CHECKSUMS.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items()),
        encoding="utf-8",
    )
    archive = root.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(f"拒绝覆盖已有交付 ZIP: {archive}")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in [*files, root / "CHECKSUMS.sha256"]:
            output.write(path, Path(root.name) / path.relative_to(root))
    with zipfile.ZipFile(archive) as delivered:
        if delivered.testzip() is not None:
            raise RuntimeError("ZIP CRC 校验失败")
        for name, digest in checksums.items():
            if hashlib.sha256(delivered.read(str(Path(root.name) / name))).hexdigest() != digest:
                raise RuntimeError(f"ZIP 内容校验失败: {name}")
    summary = {
        "root": str(root), "archive": str(archive),
        "archive_sha256": sha256_file(archive),
        "files": len(files), "verified_failures": result["reproduced"],
        "archive_crc_and_payload_hashes_verified": True,
    }
    (root / "delivery.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
