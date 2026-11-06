"""Build decontaminated protein replay indexes and train-only DBAASP targets."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for path in (str(ROOT), str(ROOT / "model_design")):
    if path not in sys.path:
        sys.path.insert(0, path)

from dbaasp import DESCRIPTOR_NAMES, peptide_descriptors, robust_stats
from muse_encoder.pretrain.a_train import compute_a_train_target_stats, iter_train_records, source_spec
from muse_encoder.pretrain.corpora import CORPORA, PRETRAIN_ROOT, iter_corpus
from muse_encoder.pretrain.dashboard import DATA_ROOT, _peptide_sequence_stats, load_board, sha256_file
from muse_encoder.pretrain.indexed import open_corpus

OUTPUT = ROOT / "data/pretrain/stage2/dbaasp"
SEQUENCE_KEYS = ("sequence", "protein_sequence", "sequence_1", "sequence_2", "seq1", "seq2")


def normalized_sequence(value: str) -> str:
    return "".join(str(value).split()).upper()


def validation_test_sequences() -> set[str]:
    board = load_board()
    excluded = set()
    for subset in board.subsets.values():
        for split in ("validation", "test"):
            path = DATA_ROOT / subset.directory / f"{split}.jsonl"
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    inputs = record.get("inputs") or {}
                    for key in SEQUENCE_KEYS:
                        value = inputs.get(key)
                        if isinstance(value, str) and value.strip():
                            excluded.add(normalized_sequence(value))
    return excluded


def _save_rows(path: Path, rows):
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, np.asarray(rows, dtype=np.int64))
    temporary.replace(path)


def build_replay_indexes(output: Path, excluded: set[str]):
    report = {"excluded_validation_test_sequences": len(excluded), "sources": {}}
    old_short = np.load(PRETRAIN_ROOT / "index/natural_peptides.uniref50.rows.npy")
    corpus = open_corpus("uniref50")
    kept = []
    for offset, row in enumerate(old_short):
        if offset % 200_000 == 0:
            print(f"[prepare] uniref50 short {offset}/{len(old_short)}", flush=True)
        text = corpus.get(int(row)).text
        if normalized_sequence(text) not in excluded:
            kept.append(int(row))
    _save_rows(output / "natural_uniref50.rows.npy", kept)
    report["sources"]["natural_uniref50"] = {
        "input_rows": len(old_short), "kept_rows": len(kept),
        "source_file": str(CORPORA["uniref50"].path),
        "source_sha256": sha256_file(CORPORA["uniref50"].path),
        "rows_sha256": sha256_file(output / "natural_uniref50.rows.npy"),
    }

    old_pdb_short = set(np.load(PRETRAIN_ROOT / "index/natural_peptides.pdb.rows.npy").tolist())
    long_rows, short_rows = [], []
    for index, record in enumerate(iter_corpus("pdb")):
        if index % 100_000 == 0:
            print(f"[prepare] pdb replay {index}/{CORPORA['pdb'].records}", flush=True)
        if normalized_sequence(record.text) in excluded:
            continue
        long_rows.append(record.row)
        if record.row in old_pdb_short:
            short_rows.append(record.row)
    _save_rows(output / "pdb.rows.npy", long_rows)
    _save_rows(output / "natural_pdb.rows.npy", short_rows)
    for name, rows, original in (
        ("pdb", long_rows, CORPORA["pdb"].records),
        ("natural_pdb", short_rows, len(old_pdb_short)),
    ):
        report["sources"][name] = {
            "input_rows": original, "kept_rows": len(rows),
            "source_file": str(CORPORA["pdb"].path),
            "source_sha256": sha256_file(CORPORA["pdb"].path),
            "rows_sha256": sha256_file(output / f"{name}.rows.npy"),
        }
    return report


def build_targets(output: Path):
    source = source_spec("dbaasp_v4/target_activity")
    base_path = output / "targets.base.json"
    payload = compute_a_train_target_stats([source], base_path)
    raw_values, descriptors, unsupported = [], [], []
    for row, record in enumerate(iter_train_records(source)):
        observations = record["labels"]["target_activity"]
        if len(observations) != 1:
            raise ValueError(f"{record['id']}: DBAASP pilot 预期每条一个 MIC 观测")
        try:
            _peptide_sequence_stats(record["inputs"]["sequence"])
        except ValueError:
            unsupported.append(row)
        raw_values.append(float(observations[0]["value"]))
        descriptors.append(peptide_descriptors(record["inputs"]["sequence"]))
    unsupported_path = output / "unsupported_train.rows.txt"
    unsupported_path.write_text("".join(f"{row}\n" for row in unsupported), encoding="utf-8")
    values = np.asarray(raw_values, dtype=np.float32)
    median, scale = robust_stats(values)
    log_median, log_scale = robust_stats(np.log1p(values))
    descriptor_median, descriptor_scale = robust_stats(np.stack(descriptors))
    key = next(iter(payload["specs"]))
    payload["stats"][key]["mean"] = [float(median)]
    payload["stats"][key]["std"] = [float(scale)]
    payload["label_policy"]["regression"] = "single official train observation; median/MAD scaling"
    payload["stage2"] = {
        "log_mic": {"median": float(log_median), "scale": float(log_scale)},
        "descriptors": {
            "names": list(DESCRIPTOR_NAMES),
            "median": descriptor_median.tolist(),
            "scale": descriptor_scale.tolist(),
        },
    }
    path = output / "targets.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    return {
        "train_records": len(values), "target_key": key,
        "median": float(median), "mad_scale": float(scale),
        "unsupported_train_records": len(unsupported),
        "unsupported_train_rows_sha256": sha256_file(unsupported_path),
        "targets_sha256": sha256_file(path),
    }


def main():
    output = OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    excluded = validation_test_sequences()
    exclusion_hash = hashlib.sha256("\n".join(sorted(excluded)).encode("utf-8")).hexdigest()
    print(f"[prepare] validation/test exclusion sequences={len(excluded)}", flush=True)
    report = {
        "schema_version": "muse.stage2.dbaasp_data.v1",
        "data_manifest_sha256": sha256_file(DATA_ROOT / "manifest.json"),
        "exclusion_sha256": exclusion_hash,
        "test_labels_used": False,
        **build_replay_indexes(output, excluded),
        "targets": build_targets(output),
    }
    (output / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
