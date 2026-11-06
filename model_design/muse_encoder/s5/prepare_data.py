"""Prepare S5 robust train targets and a hashed, train/aux-only data manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from muse_encoder.pretrain.a_train import _TrainJsonlStore, _safe_source, load_source_specs
from muse_encoder.pretrain.aux import AUX_SOURCES
from muse_encoder.pretrain.aux_index import AUX_INDEX_ROOT
from muse_encoder.pretrain.corpora import CORPORA, DESCRIPTOR_ROOT, PRETRAIN_ROOT
from muse_encoder.pretrain.indexed import INDEX_ROOT

from .curriculum import Curriculum
from .data import expand_slots
from .provenance import file_record, write_data_manifest
from .robust_targets import compute_s5_target_stats
from .runtime import load_yaml


def required_sources(config):
    registered = load_source_specs(Path(config["data"]["data_root"]))
    resolved = dict(config)
    resolved["curriculum"] = {"slots": expand_slots(config, registered)}
    curriculum = Curriculum.from_config(resolved)
    names = {option for slot in curriculum.slots for option in slot.options}
    names.update(curriculum.dbaasp_sources.values())
    return registered, curriculum, names


def write_a_train_failure_rows(spec, failure_ids_path, output_path):
    failure_ids = {
        json.loads(line)["id"] for line in Path(failure_ids_path).read_text().splitlines() if line.strip()
    }
    rows = []
    with spec.train_path.open(encoding="utf-8") as stream:
        for row, line in enumerate(stream):
            if json.loads(line)["id"] in failure_ids:
                rows.append(row)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")
    temporary.replace(output_path)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--recompute-targets", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    paths = config["data"]
    registered, curriculum, names = required_sources(config)
    official = set()
    for name in names:
        if name in registered:
            official.add(name)
        elif name.rsplit("/", 1)[0] in registered:
            official.add(name.rsplit("/", 1)[0])
    generic_sources = sorted(official - {"dbaasp_v4/target_activity"})
    target_path = Path(paths["a_train_targets"])
    if args.recompute_targets or not target_path.exists():
        compute_s5_target_stats(generic_sources, target_path, Path(paths["data_root"]))

    # Build every random-access train index in this single preparation process;
    # DDP workers must never race to create the same offsets/metadata pair.
    train_indexes = {}
    for name in sorted(official):
        spec = registered[name]
        index_path = PRETRAIN_ROOT / "index" / "a_train" / f"{_safe_source(name)}.offsets.npy"
        _TrainJsonlStore(spec, index_path)
        train_indexes[name] = index_path

    files = []
    def add(path, role):
        record = file_record(path, role, with_hash=True)
        record["always_rehash"] = role in {
            "official_manifest", "a_train_targets", "dbaasp_targets", "aux_targets",
            "descriptor_normalizer", "replay_rows", "external_provenance",
        }
        files.append(record)

    add(Path(paths["data_root"]) / "manifest.json", "official_manifest")
    for name in sorted(official):
        add(registered[name].train_path, "official_train_jsonl")
        add(train_indexes[name], "a_train_index")
        add(train_indexes[name].with_suffix(".json"), "a_train_index_metadata")
        if name == "QM9/qm9":
            failure_ids = PRETRAIN_ROOT / "index" / "molecule3d" / "qm9" / "qm9.read_failures.jsonl"
            failure_rows = PRETRAIN_ROOT / "index" / "a_train" / "QM9__qm9.registered_3d_failures.rows.txt"
            write_a_train_failure_rows(registered[name], failure_ids, failure_rows)
            add(failure_rows, "a_train_registered_3d_failures")
    add(target_path, "a_train_targets")
    add(paths["dbaasp_targets"], "dbaasp_targets")
    add(paths["aux_targets"], "aux_targets")
    add(paths["mtr_norm"], "descriptor_normalizer")
    replay_dir = Path(paths["dbaasp_replay_dir"])
    for name in ("natural_uniref50", "natural_pdb", "pdb"):
        add(replay_dir / f"{name}.rows.npy", "replay_rows")
    if "natural_uniref50" in names:
        add(INDEX_ROOT / "natural_peptides.uniref50.rows.npy", "natural_peptide_index")
    if "natural_pdb" in names:
        add(INDEX_ROOT / "natural_peptides.pdb.rows.npy", "natural_peptide_index")
    for name in sorted(names & set(CORPORA)):
        add(CORPORA[name].path, "unlabeled_corpus")
        add(INDEX_ROOT / f"{name}.offsets.npy", "corpus_index")
        if name in {"pubchem", "chembl", "pi1m"}:
            add(DESCRIPTOR_ROOT / f"{name}.desc.f32.npy", "descriptors")
            add(DESCRIPTOR_ROOT / f"{name}.keys.txt", "descriptor_keys")
    if "pi1m" in names:
        add(INDEX_ROOT / "pi1m_views.tsv", "pi1m_five_views")
        add(INDEX_ROOT / "pi1m_views.offsets.npy", "pi1m_view_index")
    for name in sorted(names & set(AUX_SOURCES)):
        spec = AUX_SOURCES[name]
        for path in spec.paths:
            add(path, "official_aux_jsonl")
        add(spec.hits_path, "aux_exclusion_hits")
        add(AUX_INDEX_ROOT / f"{name}.locations.npy", "aux_index")
    for name in paths.get("molecule3d_sources", []):
        failures = PRETRAIN_ROOT / "index" / "molecule3d" / name / f"{name}.read_failures.jsonl"
        if failures.exists():
            add(failures, "molecule3d_registered_failures")
    for item in paths.get("provenance_files", []):
        add(item["path"], item["role"])

    # Same file can be referenced by two roles; retain the first role and hash.
    unique = {}
    for record in files:
        unique.setdefault(record["path"], record)
    manifest_path = Path(paths["data_manifest"])
    summary = {
        "official_sources_for_generic_targets": generic_sources,
        "dbaasp_kept_on_s2_recipe": True,
        "required_source_count": len(names),
        "curriculum_slots": len(curriculum.slots),
    }
    payload = write_data_manifest(manifest_path, unique.values(), {
        "training_config": str(args.config.resolve()), "summary": summary,
        "external_labels_used": False, "test_labels_used": False,
    })
    print(json.dumps({"manifest": str(manifest_path), "files": len(payload["files"]),
                      **summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
