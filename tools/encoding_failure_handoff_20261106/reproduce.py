"""Reproduce the packaged encoding failures without model weights or A-test labels."""
from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from pathlib import Path


def load_structures(root: Path):
    import importlib.util
    import sys
    path = root / "vendor" / "structures.py"
    spec = importlib.util.spec_from_file_location("handoff_structures", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def raw_smiles(record):
    inputs = record.get("inputs") or {}
    for key in ("smiles", "ligand_smiles", "psmiles", "csmiles"):
        if key in inputs:
            return inputs[key]
    raise ValueError(f"{record['id']}: 找不到 SMILES 输入")


def peptide_check(sequence: str) -> None:
    allowed = set("ACDEFGHIKLMNPQRSTVWYBZUXO")
    allowed_lower = set("acdefghiklmnpqrstvwy")
    chars = [char for char in sequence.strip() if not char.isspace()]
    unsupported = sorted({
        char for char in chars
        if char not in allowed and char not in allowed_lower and char != "x"
    })
    if unsupported:
        raise ValueError(f"肽序列含当前残基/原子路由不支持的修饰记号: {unsupported}")


def reproduce_failure(failure, record, root: Path, structures):
    from rdkit import Chem

    dataset = failure["dataset"]
    if dataset == "dbaasp_v4":
        peptide_check(record["inputs"]["sequence"])
        return
    if dataset == "QM9":
        block = (root / failure["structure_file"]).read_bytes()
        supplier = Chem.ForwardSDMolSupplier(
            io.BytesIO(block), removeHs=False, sanitize=True)
        mol = next(supplier, None)
        if mol is None:
            raise ValueError(f"SDF 严格解析失败: {failure['id']}")
        structures.align_coordinates(raw_smiles(record), mol)
        return
    if dataset == "GEOM-Drugs":
        mol = Chem.Mol((root / failure["structure_file"]).read_bytes())
        structures.align_coordinates(raw_smiles(record), mol)
        return
    smiles = raw_smiles(record)
    if not isinstance(smiles, str):
        raise ValueError(f"RDKit 无法解析: {str(smiles)[:80]}")
    structures.graph_from_smiles(smiles)


def matches(expected: str, observed: str) -> bool:
    prefixes = ("RDKit 无法解析", "SDF 严格解析失败", "SMILES/SDF", "肽序列含")
    for prefix in prefixes:
        if expected.startswith(prefix):
            return observed.startswith(prefix)
    return expected == observed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--id", default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    failures = [json.loads(line) for line in (root / "failures.jsonl").read_text().splitlines()
                if line.strip()]
    if args.dataset:
        failures = [row for row in failures if row["dataset"] == args.dataset]
    if args.id:
        failures = [row for row in failures if row["id"] == args.id]
    if not failures:
        raise ValueError("过滤后没有失败用例")
    records = {}
    for row in failures:
        path = root / row["records_file"]
        if path not in records:
            records[path] = {
                record["id"]: record
                for record in map(json.loads, path.read_text(encoding="utf-8").splitlines())
            }
    structures = load_structures(root)
    results = []
    for failure in failures:
        record = records[root / failure["records_file"]][failure["id"]]
        if "labels" in record or "targets" in record:
            raise RuntimeError("交付输入不得包含标签")
        observed = None
        reproduced = False
        try:
            reproduce_failure(failure, record, root, structures)
        except Exception as exc:
            observed = str(exc)
            reproduced = matches(failure["error"], observed)
        results.append({
            "dataset": failure["dataset"], "subset": failure["subset"],
            "split": failure["split"], "id": failure["id"],
            "expected": failure["error"], "observed": observed,
            "reproduced": reproduced,
        })
    counts = Counter((row["dataset"], row["reproduced"]) for row in results)
    summary = {
        "schema_version": "muse.encoding_failure_reproduction.v1",
        "selected": len(results),
        "reproduced": sum(row["reproduced"] for row in results),
        "not_reproduced": [row for row in results if not row["reproduced"]],
        "by_dataset": {
            dataset: {"total": sum(count for (name, _ok), count in counts.items() if name == dataset),
                      "reproduced": counts.get((dataset, True), 0)}
            for dataset in sorted({row["dataset"] for row in results})
        },
        "source_total_failures": manifest["total_failures"],
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if summary["not_reproduced"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
