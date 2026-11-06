"""Build auditable QM9 input repairs from the matching official SDF record."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from muse_encoder.featurize import RecordFeaturizer
from muse_encoder.pretrain import dashboard as D


def make_repair_entry(record: Mapping[str, Any], mol, mol_block: str,
                      featurizer: RecordFeaturizer, source_path: str,
                      source_sha256: str, parse_mode: str,
                      original_errors: Sequence[str], artifact_root: Path,
                      source_format: str = "sdf") -> dict[str, Any]:
    import numpy as np

    if parse_mode not in ("strict", "unsanitized_official_graph", "official_rdkit_conformer"):
        raise ValueError("未知官方图解析模式")
    if source_format not in ("sdf", "rdkit_pickle"):
        raise ValueError("未知官方结构格式")
    refs = [ref for ref in record.get("external_inputs", [])
            if ref.get("role") in ("molecular_structure", "molecular_conformer")
            and ref.get("format") == source_format]
    if len(refs) != 1 or refs[0].get("source_path") != source_path:
        raise ValueError("官方图修复与输入结构引用不匹配")
    selector = refs[0].get("selector") or {}
    if source_format == "sdf" and selector.get("record_id") != record["id"]:
        raise ValueError("官方图结构 record_id 不匹配")
    if source_format == "rdkit_pickle" and not {
            "archive_member", "conformer_index", "geom_id"} <= set(selector):
        raise ValueError("GEOM 修复缺少完整构象选择器")
    if source_format == "sdf" and (
            not mol_block.startswith(record["id"] + "\n") or "M  END" not in mol_block):
        raise ValueError("官方图结构块与 record_id 不匹配")
    if "M  END" not in mol_block:
        raise ValueError("官方图修复缺少结构块")
    public = {
        "record_id": record["id"],
        "source": "qm9" if source_format == "sdf" else "geom_drugs",
        "source_path": source_path,
        "source_sha256": source_sha256,
        "source_format": source_format,
        "selector": selector,
        "input_fingerprint": D._official_graph_input_fingerprint(record),
        "original_smiles": record["inputs"]["smiles"],
        "original_errors": list(original_errors),
        "parse_mode": parse_mode,
        "representation": "official_graph",
    }
    prepared = featurizer.featurize_official_molecule(
        mol, record_id=record["id"], repair=public)
    inputs = record.get("inputs") or {}
    if "charge" in inputs or "multiplicity" in inputs:
        prepared.scalar_cond = np.asarray(prepared.scalar_cond, dtype=np.float32).copy()
        if "charge" in inputs:
            prepared.scalar_cond[0] = float(inputs["charge"]) / 10.0
        if "multiplicity" in inputs:
            prepared.scalar_cond[1] = float(inputs["multiplicity"]) / 10.0
    if Path(record["id"]).name != record["id"]:
        raise ValueError("修复 record_id 不能作为路径跳转")
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_root / f"{record['id']}.npz"
    if artifact_path.exists():
        raise FileExistsError(f"拒绝覆盖已有修复数组: {artifact_path}")
    arrays = {}
    for name in (
        "token_ids", "atom_features", "bond_mat", "spd_mat", "coords",
        "coord_mask", "pair_disp", "token_kind", "segment_ids", "pool_weights",
        "scalar_cond",
    ):
        value = getattr(prepared, name)
        if value is not None:
            arrays[name] = np.asarray(value)
    metadata = dict(prepared.meta)
    for key, value in list(metadata.items()):
        if isinstance(value, np.ndarray):
            metadata[key] = value.tolist()
    temporary = artifact_path.with_suffix(".npz.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez(
                stream, **arrays, domain=prepared.domain, modality=prepared.modality,
                use_atom_features=prepared.use_atom_features,
                n_real_tokens=prepared.n_real_tokens,
                meta_json=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            )
        temporary.replace(artifact_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        **public,
        "structure_block_sha256": hashlib.sha256(mol_block.encode("utf-8")).hexdigest(),
        "model_input_sha256": featurizer.official_graph_fingerprint(prepared),
        "atom_count": prepared.n_real_tokens,
        "bond_count": len(prepared.meta["bond_pairs"]),
        "artifact": artifact_path.name,
        "artifact_sha256": D.sha256_file(artifact_path),
        "mol_block": mol_block,
    }


def _structure_blocks(path: Path, selected: set[str]) -> dict[str, str]:
    found = {}
    with path.open("rb") as stream:
        block = []
        for line in stream:
            block.append(line)
            if line.strip() != b"$$$$":
                continue
            record_id = block[0].decode("utf-8").strip()
            if record_id in selected:
                ends = [i for i, value in enumerate(block) if value.strip() == b"M  END"]
                if len(ends) != 1:
                    raise ValueError(f"{record_id}: 官方 SDF 结构块缺少唯一 M  END")
                found[record_id] = b"".join(block[:ends[0] + 1]).decode("utf-8")
            block = []
    missing = selected - set(found)
    if missing:
        raise KeyError(f"官方 SDF 缺少记录: {sorted(missing)[:5]}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-dashboard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖已有修复清单: {args.output}")
    _model, cfg, _info = D.load_checkpoint(args.checkpoint, "cpu")
    del _model
    featurizer = RecordFeaturizer(cfg, D.load_tokenizers())
    metadata_path = args.baseline_dashboard / "embeddings/QM9/qm9/validation.encoding.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    failed = {}
    for row in metadata.get("failed", []):
        failed.setdefault(row["id"], []).append(row["error"])
    board = D.load_board(D.DATA_ROOT, "A")
    subset = board.subsets[("QM9", "qm9")]
    records = [record for record in D._read_jsonl(D._split_file(D.DATA_ROOT, subset, "validation"))
               if record["id"] in failed]
    if {record["id"] for record in records} != set(failed):
        raise RuntimeError("失败记录与官方 validation 输入 ID 不一致")
    source_paths = {ref["source_path"] for record in records
                    for ref in record.get("external_inputs", [])
                    if ref.get("role") == "molecular_structure" and ref.get("format") == "sdf"}
    if len(source_paths) != 1:
        raise ValueError("QM9 修复必须只有一个官方 SDF 来源")
    source_path = source_paths.pop()
    source_file = D._AUX_EXTERNAL_BASES["qm9"] / source_path
    if not source_file.resolve().is_relative_to(D._AUX_EXTERNAL_BASES["qm9"].resolve()):
        raise ValueError("官方 SDF source_path 超出官方目录")
    source_sha256 = D.sha256_file(source_file)
    blocks = _structure_blocks(source_file, set(failed))
    from rdkit import Chem
    artifact_dir = args.output.stem + ".data"
    artifact_root = args.output.parent / artifact_dir
    entries = []
    for record in sorted(records, key=lambda value: value["id"]):
        block = blocks[record["id"]]
        try:
            strict_mol = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False)
        except Exception:
            strict_mol = None
        parse_mode = "strict" if strict_mol is not None else "unsanitized_official_graph"
        model_mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
        if model_mol is None:
            raise ValueError(f"{record['id']}: 官方 SDF 无法解析")
        entries.append(make_repair_entry(
            record, model_mol, block, featurizer, source_path, source_sha256, parse_mode,
            failed[record["id"]], artifact_root=artifact_root,
        ))
    payload = {
        "schema": "muse_encoder.official_graph_repairs.v1",
        "source": "qm9",
        "source_path": source_path,
        "source_sha256": source_sha256,
        "source_validation_encoding_sha256": D.sha256_file(metadata_path),
        "generator_sha256": D.sha256_file(Path(__file__)),
        "contains_labels": False,
        "artifact_dir": artifact_dir,
        "repairs": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    D._write_json(args.output, payload)
    try:
        D.configure_official_graph_repairs(args.output)
    finally:
        D.configure_official_graph_repairs(None)
    print(json.dumps({
        "output": str(args.output), "output_sha256": D.sha256_file(args.output),
        "repairs": len(entries),
        "strict": sum(entry["parse_mode"] == "strict" for entry in entries),
        "unsanitized_official_graph": sum(
            entry["parse_mode"] == "unsanitized_official_graph" for entry in entries),
        "contains_labels": False,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
