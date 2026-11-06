from __future__ import annotations

import argparse
import csv
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from muse.pretrain.deleak import load_molecule_fingerprints, mol_fingerprint
from ..peptide import annotate_peptide, fragment_recovery_rate
from .corpora import PRETRAIN_ROOT


def _annotation_worker(item):
    row, smiles = item
    from rdkit import Chem

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return row, "rdkit_parse", None
        group, sidechain, fragment = annotate_peptide(mol)
        recovery = fragment_recovery_rate(mol, fragment)
        return row, recovery, (group, sidechain, fragment)
    except Exception as exc:
        return row, f"{type(exc).__name__}: {str(exc)[:180]}", None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=PRETRAIN_ROOT / "corpora" / "CycPeptMPDB_Peptide_All.csv")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    clean_dir = PRETRAIN_ROOT / "corpora_clean"
    audit_dir = PRETRAIN_ROOT / "audit"
    clean_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    inchikeys, canonical_fingerprints = load_molecule_fingerprints()
    seen = set()
    kept = []
    removed = []
    with args.input.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for index, record in enumerate(reader):
            if args.limit is not None and index >= args.limit:
                break
            record_id = record["CycPeptMPDB_ID"]
            canonical, inchikey = mol_fingerprint(record["SMILES"])
            base = {"id": record_id}
            if canonical is None:
                removed.append({**base, "reason": "rdkit_parse"})
            elif (inchikey is not None and inchikey in inchikeys) or canonical in canonical_fingerprints:
                removed.append({**base, "reason": "valtest_fingerprint_hit", "inchikey": inchikey})
            elif canonical in seen:
                removed.append({**base, "reason": "duplicate_structure"})
            else:
                seen.add(canonical)
                kept.append((record_id, canonical, record.get("HELM", ""), record.get("Sequence", "")))
    with (audit_dir / "cycpept.removed.jsonl").open("w", encoding="utf-8") as stream:
        for record in removed:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    clean_path = clean_dir / "cycpept.tsv"
    with clean_path.open("w", encoding="utf-8") as stream:
        for record_id, canonical, _helm, _sequence in kept:
            stream.write(f"{record_id}\t{canonical}\n")
    with (clean_dir / "cycpept.meta.jsonl").open("w", encoding="utf-8") as stream:
        for record_id, _canonical, helm, sequence in kept:
            stream.write(json.dumps({"id": record_id, "helm": helm, "monomers": sequence},
                                    ensure_ascii=False) + "\n")

    supported_rows = []
    arrays = [[], [], []]
    offsets = [0]
    unsupported = []
    recoveries = []
    with Pool(args.workers) as pool:
        for row, status, annotation in pool.imap(
                _annotation_worker, [(i, item[1]) for i, item in enumerate(kept)], chunksize=8):
            if annotation is None:
                unsupported.append({"id": kept[row][0], "row": row, "reason": status})
                continue
            group, sidechain, fragment = annotation
            supported_rows.append(row)
            recoveries.append(float(status))
            for target, values in zip(arrays, (group, sidechain, fragment)):
                target.append(values)
            offsets.append(offsets[-1] + len(group))
    output = clean_dir / "cycpept.annotations.npz"
    tmp = clean_dir / "cycpept.annotations.tmp.npz"
    np.savez(tmp,
             supported_rows=np.asarray(supported_rows, dtype=np.int64),
             offsets=np.asarray(offsets, dtype=np.int64),
             group_ids=np.concatenate(arrays[0]).astype(np.int16),
             sidechain_flags=np.concatenate(arrays[1]).astype(bool),
             fragment_ids=np.concatenate(arrays[2]).astype(np.int16))
    os.replace(tmp, output)
    with (audit_dir / "cycpept.unsupported.jsonl").open("w", encoding="utf-8") as stream:
        for record in unsupported:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stats = {
        "input_records": len(kept) + len(removed),
        "unique_clean": len(kept),
        "removed": len(removed),
        "annotation_supported": len(supported_rows),
        "annotation_unsupported": len(unsupported),
        "annotation_support_rate": len(supported_rows) / max(1, len(kept)),
        "fragment_recovery_min": min(recoveries, default=None),
        "fragment_recovery_mean": float(np.mean(recoveries)) if recoveries else None,
        "rdkit_version": __import__("rdkit").__version__,
        "labels_used": [],
    }
    (audit_dir / "cycpept.stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
