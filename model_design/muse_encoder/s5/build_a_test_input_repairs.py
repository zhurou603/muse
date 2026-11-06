"""Build auditable representations for the packaged A-test encoding failures."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger

from muse_encoder.featurize import RecordFeaturizer
from muse_encoder.pretrain import dashboard as D
from muse_encoder.pretrain.molecule3d import _resolve_external_file, external_base
from muse_encoder.s5.official_graph_repairs import make_repair_entry
from muse_encoder.structures import graph_from_smiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖已有修复清单: {args.output}")
    root = args.handoff_root.resolve()
    handoff = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not handoff.get("labels_removed"):
        raise ValueError("只接受已剥离标签的失败交付包")
    failures = [json.loads(line) for line in (root / "failures.jsonl").read_text().splitlines()
                if line.strip()]
    records = {}
    for path in (root / "records").glob("*.jsonl"):
        for record in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
            if "labels" in record or "targets" in record:
                raise ValueError(f"{path}: 输入修复生成器拒绝读取标签")
            if record["id"] in records:
                raise ValueError(f"重复输入 ID: {record['id']}")
            records[record["id"]] = record
    if len(failures) != handoff["total_failures"] \
            or {row["id"] for row in failures} != set(records) or len(failures) != len(records):
        raise ValueError("失败清单与无标签输入不一致")
    _model, cfg, _info = D.load_checkpoint(args.checkpoint, "cpu")
    del _model
    featurizer = RecordFeaturizer(cfg, D.load_tokenizers())
    artifact_dir = args.output.stem + ".data"
    artifact_root = args.output.parent / artifact_dir
    source_hashes = {}
    entries = []
    for failure in failures:
        record = records[failure["id"]]
        source_jsonl = Path(failure["source_jsonl"]).resolve()
        if not source_jsonl.is_relative_to(D.DATA_ROOT.resolve()):
            raise ValueError("失败记录来源超出官方数据目录")
        if source_jsonl not in source_hashes:
            source_hashes[source_jsonl] = D.sha256_file(source_jsonl)
        if source_hashes[source_jsonl] != failure["source_jsonl_sha256"]:
            raise RuntimeError(f"原始输入来源 hash 已改变: {source_jsonl}")
        provenance = {
            "dataset": failure["dataset"], "subset": failure["subset"],
            "split": failure["split"], "source_jsonl": str(source_jsonl),
            "source_jsonl_sha256": failure["source_jsonl_sha256"],
        }
        if failure["dataset"] in ("QM9", "GEOM-Drugs"):
            source = "qm9" if failure["dataset"] == "QM9" else "geom_drugs"
            refs = [ref for ref in record.get("external_inputs", [])
                    if ref.get("role") in ("molecular_structure", "molecular_conformer")]
            if len(refs) != 1:
                raise ValueError(f"{record['id']}: 需要唯一官方分子结构")
            ref = refs[0]
            source_file = _resolve_external_file(external_base(source), ref["source_path"])
            if source_file not in source_hashes:
                source_hashes[source_file] = D.sha256_file(source_file)
            structure_path = root / failure["structure_file"]
            if failure["dataset"] == "QM9":
                block = structure_path.read_text(encoding="utf-8")
                mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
                strict = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False)
                parse_mode = "strict" if strict is not None else "unsanitized_official_graph"
            else:
                mol = Chem.Mol(structure_path.read_bytes())
                block = Chem.MolToMolBlock(mol, kekulize=False)
                parse_mode = "official_rdkit_conformer"
            if mol is None:
                raise ValueError(f"{record['id']}: 官方结构无法读取")
            entry = make_repair_entry(
                record, mol, block, featurizer, ref["source_path"],
                source_hashes[source_file], parse_mode, [failure["error"]],
                artifact_root=artifact_root, source_format=ref["format"],
            )
        else:
            if failure["dataset"] == "dbaasp_v4":
                representation = "raw_peptide_sequence"
            elif failure["dataset"] == "PolyGraphMT" \
                    and str(record["inputs"].get("psmiles")).strip().lower() == "nan":
                representation = "missing_polymer_sequence"
            else:
                smiles = record["inputs"].get("ligand_smiles", record["inputs"].get("smiles"))
                try:
                    graph_from_smiles(smiles, allow_unsanitized=True)
                except (ValueError, RuntimeError):
                    representation = "smiles_sequence"
                else:
                    representation = "unsanitized_topology"
            entries.append({
                "record_id": record["id"], "representation": representation,
                "input_fingerprint": D._input_repair_fingerprint(record),
                "original_errors": [failure["error"]], **provenance,
            })
            continue
        entry.update(provenance)
        entries.append(entry)
        if len(entries) % 250 == 0:
            print(f"[input-repairs] built={len(entries)}/{len(failures)}", flush=True)
    payload = {
        "schema": "muse_encoder.input_repairs.v1",
        "contains_labels": False,
        "validation_test_labels_used_for_training": False,
        "handoff_manifest_sha256": D.sha256_file(root / "manifest.json"),
        "checkpoint_sha256": D.sha256_file(args.checkpoint),
        "artifact_dir": artifact_dir,
        "generator_sha256": D.sha256_file(Path(__file__)),
        "code_sha256": {
            "dashboard": D.sha256_file(Path(D.__file__)),
            "featurize": D.sha256_file(D.PROJECT_ROOT / "model_design/muse_encoder/featurize.py"),
            "structures": D.sha256_file(D.PROJECT_ROOT / "model_design/muse_encoder/structures.py"),
        },
        "representation_policy": {
            "official_graph": "same-record official structure; no forced SMILES connectivity alignment",
            "smiles_sequence": "raw chemical tokens with no 3D coordinates or inferred bond graph",
            "unsanitized_topology": "explicit RDKit sanitize=False topology with no unverified 3D coordinates",
            "raw_peptide_sequence": "unsupported notation is retained as unknown residue tokens",
            "missing_polymer_sequence": "one explicit unknown token; the missing structure is not reconstructed",
        },
        "repairs": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    D._write_json(args.output, payload)
    try:
        D.configure_official_graph_repairs(args.output)
    finally:
        D.configure_official_graph_repairs(None)
    print(json.dumps({
        "output": str(args.output), "sha256": D.sha256_file(args.output),
        "total": len(entries), "representations": dict(Counter(
            entry.get("representation", "official_graph") for entry in entries)),
        "contains_labels": False,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    RDLogger.DisableLog("rdApp.*")
    main()
