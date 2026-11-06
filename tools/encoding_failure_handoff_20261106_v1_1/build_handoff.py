"""Build a self-contained, label-free reproduction package for encoding failures."""
from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import sys
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import rdBase

from muse_encoder.pretrain import dashboard as D
from muse_encoder.pretrain.molecule3d import (
    _resolve_external_file,
    external_base,
    pick_geom_conformer,
)
from muse_encoder.s5.provenance import sha256_file


def canonical_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def cached_file_hash(path: Path, cache: dict) -> str:
    if path not in cache:
        cache[path] = sha256_file(path)
    return cache[path]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_sdf_block(path: Path, offset: int) -> bytes:
    with path.open("rb") as stream:
        stream.seek(offset)
        lines = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"SDF offset={offset} 缺少结束分隔符")
            lines.append(line)
            if line.strip() == b"$$$$":
                return b"".join(lines)


def build_sdf_offsets(path: Path) -> dict[str, int]:
    offsets = {}
    expect_name = True
    with path.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if expect_name:
                record_id = line.decode("utf-8").strip()
                if not record_id:
                    raise ValueError(f"{path}: 空 SDF record_id")
                offsets[record_id] = offset
                expect_name = False
            if line.strip() == b"$$$$":
                expect_name = True
    return offsets


def main() -> None:
    root = Path(__file__).resolve().parent
    source_run = D.PROJECT_ROOT / "model_design/runs/s5-continuous-four-domain-v1"
    failure_root = source_run / "evaluation/encoding_failure_debug_step4100/embeddings"
    if not failure_root.is_dir():
        raise FileNotFoundError(failure_root)
    board = D.load_board(D.DATA_ROOT, "A")
    failures = []
    counts = Counter()
    structure_cache = {}
    source_hash_cache = {}
    tar_cache = {}
    object_cache = {}
    sdf_index_cache = {}

    for failure_path in sorted(failure_root.glob("**/*.failed.json")):
        rows = json.loads(failure_path.read_text(encoding="utf-8"))
        if not rows:
            continue
        relative = failure_path.relative_to(failure_root)
        dataset, subset = relative.parts[:2]
        split = relative.parts[2].split(".", 1)[0]
        source_file = D._split_file(D.DATA_ROOT, board.subsets[(dataset, subset)], split)
        wanted = {row["id"] for row in rows}
        records = {}
        with source_file.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["id"] in wanted:
                    records[record["id"]] = record
        if set(records) != wanted:
            raise RuntimeError(f"{dataset}/{subset}/{split}: 原始输入 ID 不完整")

        record_output = root / "records" / f"{dataset}__{subset}__{split}.jsonl"
        record_output.parent.mkdir(parents=True, exist_ok=True)
        with record_output.open("w", encoding="utf-8") as stream:
            for row in rows:
                original = records[row["id"]]
                public = dict(original)
                public.pop("labels", None)
                public.pop("targets", None)
                stream.write(json.dumps(public, ensure_ascii=False, separators=(",", ":")) + "\n")
                failure = {
                    "dataset": dataset,
                    "subset": subset,
                    "split": split,
                    "id": row["id"],
                    "error": row["error"],
                    "records_file": str(record_output.relative_to(root)),
                    "original_record_sha256": canonical_hash(original),
                    "source_jsonl": str(source_file),
                    "source_jsonl_sha256": cached_file_hash(source_file, source_hash_cache),
                }
                if dataset == "QM9":
                    refs = [ref for ref in original.get("external_inputs", [])
                            if ref.get("format") == "sdf"]
                    if len(refs) != 1:
                        raise ValueError(f"{row['id']}: QM9 必须恰好一个 SDF 引用")
                    ref = refs[0]
                    source_path = _resolve_external_file(
                        external_base("qm9"), ref["source_path"])
                    if source_path not in sdf_index_cache:
                        sdf_index_cache[source_path] = build_sdf_offsets(source_path)
                    record_id = ref["selector"]["record_id"]
                    offset = sdf_index_cache[source_path][record_id]
                    key = (dataset, row["id"])
                    if key not in structure_cache:
                        target = root / "structures" / "qm9" / f"{hashlib.sha256(row['id'].encode()).hexdigest()}.sdf"
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(read_sdf_block(source_path, offset))
                        structure_cache[key] = target
                    failure["structure_file"] = str(structure_cache[key].relative_to(root))
                    failure["official_source_path"] = str(source_path)
                elif dataset == "GEOM-Drugs":
                    refs = [ref for ref in original.get("external_inputs", [])
                            if ref.get("format") == "rdkit_pickle"]
                    if len(refs) != 1:
                        raise ValueError(f"{row['id']}: GEOM 必须恰好一个 rdkit_pickle 引用")
                    ref = refs[0]
                    selector = ref["selector"]
                    source_path = _resolve_external_file(
                        external_base("geom_drugs"), ref["source_path"])
                    key = (dataset, row["id"])
                    if key not in structure_cache:
                        if source_path not in tar_cache:
                            tar_cache[source_path] = tarfile.open(source_path, "r:*")
                        archive = tar_cache[source_path]
                        object_key = (source_path, selector["archive_member"])
                        if object_key not in object_cache:
                            member = archive.extractfile(selector["archive_member"])
                            if member is None:
                                raise KeyError(f"tar 缺少 {selector['archive_member']}")
                            object_cache[object_key] = pickle.loads(member.read())
                        obj = object_cache[object_key]
                        mol = pick_geom_conformer(
                            obj, selector["conformer_index"], selector["geom_id"])
                        target = root / "structures" / "geom_drugs" / f"{hashlib.sha256(row['id'].encode()).hexdigest()}.rdkit.bin"
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(mol.ToBinary())
                        structure_cache[key] = target
                    failure["structure_file"] = str(structure_cache[key].relative_to(root))
                    failure["official_source_path"] = str(source_path)
                failures.append(failure)
                counts[f"{dataset}/{subset}/{split}"] += 1

    for archive in tar_cache.values():
        archive.close()
    failures.sort(key=lambda row: (row["dataset"], row["subset"], row["split"], row["id"]))
    with (root / "failures.jsonl").open("w", encoding="utf-8") as stream:
        for row in failures:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    if len(failures) != 1846 or sum(counts.values()) != 1846:
        raise RuntimeError(f"失败总数 {len(failures)} != 1846")

    vendor = root / "vendor"
    vendor.mkdir(exist_ok=True)
    shutil.copyfile(D.PROJECT_ROOT / "model_design/muse_encoder/structures.py", vendor / "structures.py")
    code_paths = [
        "model_design/launches/eval16-20261027/fast_eval.py",
        "model_design/muse_encoder/pretrain/parallel_dashboard.py",
        "model_design/muse_encoder/pretrain/dashboard.py",
        "model_design/muse_encoder/featurize.py",
        "model_design/muse_encoder/structures.py",
    ]
    manifest = {
        "schema_version": "muse.encoding_failure_handoff.v1",
        "created_by": "build_handoff.py",
        "total_failures": len(failures),
        "counts": dict(sorted(counts.items())),
        "labels_removed": True,
        "validation_test_labels_used_for_training": False,
        "source_run": str(source_run),
        "source_failure_root": str(failure_root),
        "checkpoint_sha256": "0ca37432a780467b483ec528524f2b49470f15f93bf96a6fa4c5ba5d31c63dff",
        "source_code_sha256": {
            path: sha256_file(D.PROJECT_ROOT / path) for path in code_paths
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "reproduce_command": "python reproduce.py --output verification.json",
    }
    write_json(root / "manifest.json", manifest)
    print(json.dumps({"root": str(root), "total_failures": len(failures),
                      "counts": manifest["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
