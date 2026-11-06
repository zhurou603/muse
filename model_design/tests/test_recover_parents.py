import io
import json
import sqlite3
import sys
import tarfile
import tempfile
import unittest
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muse_encoder.pretrain.crystal_audit import StartingCrystal, StartingCrystalIndex
from muse_encoder.pretrain.recover_parents import (
    build_energy_index,
    build_valtest_index,
    collect_heap_selectors,
    iter_heap_recovery,
    run_pipeline,
)


ETHANOL_KEY = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
HEAP_A = "AAAA-1-gener-abcdef"
HEAP_B = "BBBB-1-gener-fedcba"


def molecule_atoms(smiles):
    from ase import Atoms
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(molecule, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(molecule)
    return Atoms(
        numbers=[atom.GetAtomicNum() for atom in molecule.GetAtoms()],
        positions=molecule.GetConformer().GetPositions(), pbc=False,
    )


def ase_payload(atoms, sid):
    return {
        "numbers": {"__ndarray__": [[len(atoms)], "int64", atoms.numbers.tolist()]},
        "positions": {"__ndarray__": [[len(atoms), 3], "float64", atoms.positions.reshape(-1).tolist()]},
        "cell": {"__ndarray__": [[3, 3], "float64", [0] * 9]},
        "pbc": {"__ndarray__": [[3], "bool", [False] * 3]},
        "unique_id": sid,
        "data": {"sid": sid},
    }


def make_lmdb(path, rows):
    import lmdb

    environment = lmdb.open(str(path), subdir=False, map_size=8 * 1024 * 1024)
    with environment.begin(write=True) as transaction:
        for row_id, (sid, atoms) in enumerate(rows, start=1):
            payload = json.dumps(ase_payload(atoms, sid), separators=(",", ":")).encode()
            transaction.put(str(row_id).encode(), zlib.compress(payload))
    environment.close()


def make_record(record_id, heap, frame, row_id):
    sid = f"{heap}-{frame}"
    return {
        "id": record_id,
        "inputs": {"chemical_formula": "C2H6O", "atom_count": 9},
        "external_inputs": [{
            "id": record_id, "role": "periodic_structure", "format": "ase_lmdb",
            "selector": {"archive_member": "train/data0000.aselmdb",
                         "lmdb_row_id": row_id, "sid": sid},
            "source_path": "assets/part_00.tar",
        }],
        "labels": {"energy": [{"value": -float(row_id), "unit": "eV"}]},
    }


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    return path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def metadata(heap):
    csd, z_value, step, xtal_id = heap.split("-", 3)
    return StartingCrystal(
        starting_heap=heap, csd_refcode=csd, z_value=int(z_value),
        genarris_step=step, xtal_id=xtal_id, split="train", nframes=2,
        mol_composition="C2 H6 O", xtal_composition="C2 H6 O",
        mol_natoms=9, xtal_natoms=9, spacegroup=1, csv_row=2,
    )


class RecoverParentsPipelineTest(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)
        self.external_root = self.root / "external"
        archive = self.external_root / "assets/part_00.tar"
        archive.parent.mkdir(parents=True)
        source_lmdb = self.root / "source.aselmdb"
        ethanol = molecule_atoms("CCO")
        dimethyl_ether = molecule_atoms("COC")
        make_lmdb(source_lmdb, [
            (f"{HEAP_A}-0", ethanol),
            (f"{HEAP_A}-1", ethanol),
            (f"{HEAP_B}-0", ethanol),
            (f"{HEAP_B}-1", dimethyl_ether),
        ])
        with tarfile.open(archive, "w") as tf:
            tf.add(source_lmdb, arcname="train/data0000.aselmdb", recursive=False)
        self.archive = archive
        self.original_archive = archive.read_bytes()
        self.train_jsonl = write_jsonl(self.root / "train.jsonl", [
            make_record("A0", HEAP_A, 0, 1),
            make_record("B0", HEAP_B, 0, 3),
            make_record("A1", HEAP_A, 1, 2),
            make_record("B1", HEAP_B, 1, 4),
        ])
        self.validation_jsonl = write_jsonl(self.root / "validation.jsonl", [
            make_record("reference-A1", HEAP_A, 1, 2),
        ])
        self.test_jsonl = write_jsonl(self.root / "test.jsonl", [])
        self.train_db = self.root / "train.sqlite3"
        self.valtest_db = self.root / "valtest.sqlite3"
        build_energy_index([self.train_jsonl], self.train_db, expected_counts=[4])
        build_valtest_index(self.validation_jsonl, self.test_jsonl, self.valtest_db, 1, 0)
        self.index = StartingCrystalIndex([metadata(HEAP_A), metadata(HEAP_B)])

    def tearDown(self):
        self._temporary_directory.cleanup()

    def run_pipeline(self, output, workers=1, limit=None):
        return run_pipeline(
            self.train_db, self.valtest_db, self.index,
            self.external_root, self.root / "cache", output,
            workers=workers, limit=limit,
            recovery_options={"use_vdw": (False,)},
        )

    def test_first_last_muse_split_hit_mismatch_resume_and_limit(self):
        selectors = collect_heap_selectors(self.train_db)
        self.assertEqual([item.starting_heap for item in selectors], [HEAP_A, HEAP_B])
        self.assertEqual((selectors[0].first.record_id, selectors[0].last.record_id), ("A0", "A1"))
        self.assertEqual((selectors[1].first.record_id, selectors[1].last.record_id), ("B0", "B1"))

        output = self.root / "output"
        result = self.run_pipeline(output, workers=2)

        self.assertEqual(result["train_recovery"]["processed"], 2)
        self.assertEqual(result["train_recovery"]["complete"], 1)
        self.assertEqual(result["train_recovery"]["missing"], 1)
        self.assertEqual(result["train_recovery"]["cross_frame_mismatch"], 1)
        self.assertEqual(result["reference_recovery"]["processed"], 1)
        train_records = {record["starting_heap"]: record for record in
                         iter_heap_recovery(output / "train.heap_recovery.jsonl")}
        self.assertEqual(train_records[HEAP_A]["parent_inchikey"], ETHANOL_KEY)
        self.assertEqual(train_records[HEAP_A]["missing"], [])
        self.assertEqual(
            [frame["record_id"] for frame in train_records[HEAP_A]["recovery"]["frames"]],
            ["A0", "A1"],
        )
        self.assertIsNone(train_records[HEAP_B]["parent_inchikey"])
        self.assertIsNone(train_records[HEAP_B]["parent_smiles"])
        self.assertIn("cross_frame_inchikey_mismatch", train_records[HEAP_B]["missing"])
        reference = list(iter_heap_recovery(output / "reference.heap_recovery.jsonl"))
        self.assertEqual(len(reference), 1)
        self.assertEqual(reference[0]["starting_heap"], HEAP_A)
        self.assertEqual(reference[0]["recovery"]["consensus"], "singleton_same_row_consistent")
        self.assertEqual(reference[0]["parent_inchikey"], ETHANOL_KEY)

        # CSV 把 HEAP_A 标为 train，但 reference 集合仍只能由 MUSE DB 决定。
        self.assertEqual(self.index[HEAP_A].split, "train")
        hits = read_jsonl(output / "triple_hits.jsonl")
        self.assertEqual([hit["starting_heap"] for hit in hits], [HEAP_A])
        suggestions = {row["starting_heap"]: row["reasons"] for row in
                       read_jsonl(output / "suggested_exclude_heaps.jsonl")}
        self.assertIn("triple_hit", suggestions[HEAP_A])
        self.assertIn("missing_training:cross_frame_inchikey_mismatch", suggestions[HEAP_B])
        self.assertFalse(result["audit"]["training_missing_ignored"])
        self.assertEqual(self.archive.read_bytes(), self.original_archive)

        first_bytes = (output / "train.heap_recovery.jsonl").read_bytes()
        resumed = self.run_pipeline(output, workers=1)
        self.assertEqual(resumed["train_recovery"]["processed"], 0)
        self.assertEqual(resumed["train_recovery"]["resumed"], 2)
        self.assertEqual(resumed["reference_recovery"]["resumed"], 1)
        self.assertEqual(resumed["train_recovery"]["cross_frame_mismatch"], 1)
        self.assertEqual((output / "train.heap_recovery.jsonl").read_bytes(), first_bytes)

        limited = self.run_pipeline(self.root / "limited", limit=1)
        self.assertEqual(limited["train_recovery"]["processed"], 1)
        limited_records = list(iter_heap_recovery(self.root / "limited/train.heap_recovery.jsonl"))
        self.assertEqual([record["starting_heap"] for record in limited_records], [HEAP_A])

    def test_valtest_builder_rejects_duplicate_ids_across_splits(self):
        with self.assertRaisesRegex(ValueError, "重复|冲突"):
            build_valtest_index(
                self.validation_jsonl, self.validation_jsonl,
                self.root / "duplicate.sqlite3",
            )

    def test_train_and_reference_db_must_be_distinct(self):
        with self.assertRaisesRegex(ValueError, "不同文件"):
            run_pipeline(
                self.train_db, self.train_db, self.index,
                self.external_root, self.root / "cache", self.root / "same-db",
            )


if __name__ == "__main__":
    unittest.main()
