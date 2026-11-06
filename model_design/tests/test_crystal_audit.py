import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muse_encoder.pretrain.crystal_audit import (
    StartingCrystal,
    StartingCrystalIndex,
    StructureSample,
    audit_crystal_triples,
    build_reference_triples,
    build_training_triples,
    build_triple_record,
    composition_counts,
    iter_starting_crystals,
    recover_parent_inchikey,
)
from muse_encoder.pretrain.omc25 import read_ase_lmdb_row


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_CSV = PROJECT_ROOT / "data/pretrain/corpora/omc25-starting-crystals.csv"
REAL_LMDB = PROJECT_ROOT / "data/pretrain/cache/omc25/part_00/train/data0000.aselmdb"
ETHANOL_INCHIKEY = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
OQECAP_INCHIKEY = "YDAJJIVFTHZYOQ-UHFFFAOYSA-N"


def ethanol_atoms():
    from ase import Atoms
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(molecule, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(molecule)
    atoms = Atoms(
        numbers=[atom.GetAtomicNum() for atom in molecule.GetAtoms()],
        positions=molecule.GetConformer().GetPositions(),
        pbc=False,
    )
    return atoms


def metadata(
    heap="TEST-1-gener-abcdef", split="train", spacegroup=1,
    mol_composition="C2 H6 O", xtal_composition="C2 H6 O",
):
    csd, z_value, step, xtal_id = heap.split("-", 3)
    return StartingCrystal(
        starting_heap=heap,
        csd_refcode=csd,
        z_value=int(z_value),
        genarris_step=step,
        xtal_id=xtal_id,
        split=split,
        nframes=1,
        mol_composition=mol_composition,
        xtal_composition=xtal_composition,
        mol_natoms=9 if mol_composition else None,
        xtal_natoms=9 if xtal_composition else None,
        spacegroup=spacegroup,
        csv_row=2,
    )


class TemporaryCrystalAuditTest(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)

    def tearDown(self):
        self._temporary_directory.cleanup()

    def write_csv(self, rows):
        path = self.root / "starting.csv"
        fields = [
            "", "csd_refcode", "z_value", "genarris_step", "xtal.id", "split",
            "nframes", "mol.composition", "xtal.composition", "mol.natoms",
            "xtal.natoms", "mol.mass", "xtal.mass", "xtal.spacegroup",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, row in enumerate(rows):
                writer.writerow({"": index, **row})
        return path

    def test_streaming_csv_and_exact_heap_mapping(self):
        path = self.write_csv([
            {
                "csd_refcode": "ABABAI", "z_value": 2, "genarris_step": "gener",
                "xtal.id": "d95bafcf2a4d27b", "split": "train", "nframes": 122,
                "mol.composition": "C22 H19 N O", "xtal.composition": "C44 H38 N2 O2",
                "mol.natoms": 43, "xtal.natoms": 86, "mol.mass": 313.4,
                "xtal.mass": 626.8, "xtal.spacegroup": 1,
            },
            {
                "csd_refcode": "EMPTY", "z_value": 1, "genarris_step": "press",
                "xtal.id": "xyz", "split": "val", "nframes": 2,
                "mol.composition": "C2 H6 O", "xtal.composition": "C2 H6 O",
                "mol.natoms": 9, "xtal.natoms": 9, "mol.mass": 46,
                "xtal.mass": 46, "xtal.spacegroup": "",
            },
        ])

        first = next(iter_starting_crystals(path, limit=1))
        index = StartingCrystalIndex.from_csv(path, expected_count=2)

        self.assertEqual(first.starting_heap, "ABABAI-2-gener-d95bafcf2a4d27b")
        self.assertEqual(first.metadata, ("ABABAI", 1, "C22 H19 N O"))
        self.assertEqual(first.mol_formula, "C22H19NO")
        self.assertEqual(first.xtal_formula, "C44H38N2O2")
        self.assertEqual(len(index), 2)
        self.assertIsNone(index["EMPTY-1-press-xyz"].spacegroup)
        self.assertIsNone(index.get("not-present"))
        self.assertEqual(composition_counts("C12 H8 F4"), {"C": 12, "H": 8, "F": 4})

    def test_ethanol_geometry_recovery_and_explicit_missing_fields(self):
        atoms = ethanol_atoms()

        result = recover_parent_inchikey(atoms, "C2 H6 O", exhaustive=True)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.inchikey, ETHANOL_INCHIKEY)
        self.assertEqual(result.matching_fragments, 1)
        self.assertEqual(result.successful_fragments, 1)
        self.assertTrue(all(attempt["status"] == "success" for attempt in result.attempts))

        sample = StructureSample("ethanol", "TEST-1-gener-abcdef", atoms)
        no_spacegroup = build_triple_record(sample, metadata(spacegroup=None))
        self.assertEqual(no_spacegroup["missing"], ["spacegroup"])
        self.assertEqual(no_spacegroup["parent_inchikey"], ETHANOL_INCHIKEY)
        no_composition = build_triple_record(
            sample, metadata(mol_composition=None)
        )
        self.assertEqual(no_composition["missing"], ["mol_composition", "parent_inchikey"])
        self.assertIsNone(no_composition["parent_inchikey"])
        self.assertIsNone(no_composition["recovery"])
        no_metadata = build_triple_record(sample, None)
        self.assertIn("starting_heap_metadata", no_metadata["missing"])
        self.assertIsNone(no_metadata["parent_inchikey"])

    def test_failed_fragment_selection_never_fabricates_inchikey(self):
        from ase import Atoms

        atoms = Atoms("C2", positions=[[0, 0, 0], [5, 0, 0]], pbc=False)
        result = recover_parent_inchikey(atoms, "C2")
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failure_reason, "no_composition_matching_fragment")
        self.assertIsNone(result.inchikey)
        self.assertEqual(result.matching_fragments, 0)

    def test_train_reference_interfaces_triple_hit_and_conservative_missing(self):
        atoms = ethanol_atoms()
        train_meta = metadata()
        reference_meta = metadata(heap="REF-1-gener-abcdef", split="val")
        index = StartingCrystalIndex([train_meta, reference_meta])
        train_sample = StructureSample("train", train_meta.starting_heap, atoms)
        reference_sample = StructureSample("reference", reference_meta.starting_heap, atoms)

        training = list(build_training_triples([train_sample, reference_sample], index))
        references = list(build_reference_triples([train_sample, reference_sample], index))
        explicit_reference = list(build_reference_triples(
            [train_sample, reference_sample], index, splits=None,
            heaps={train_meta.starting_heap},
        ))

        self.assertEqual([record["id"] for record in training], ["train"])
        self.assertEqual([record["id"] for record in references], ["reference"])
        self.assertEqual([record["id"] for record in explicit_reference], ["train"])
        audit = audit_crystal_triples(training, references)
        self.assertEqual(audit["summary"]["triple_hits"], 1)
        self.assertEqual(audit["triple_hits"][0]["id"], "train")
        self.assertEqual(audit["missing_training"], [])

        incomplete = dict(training[0])
        incomplete["parent_inchikey"] = None
        incomplete["missing"] = ["parent_inchikey"]
        fallback = audit_crystal_triples([incomplete], references, conservative_missing=True)
        self.assertEqual(fallback["summary"]["triple_hits"], 0)
        self.assertEqual(fallback["summary"]["missing_training"], 1)
        self.assertEqual(fallback["missing_training"][0]["missing"], ["parent_inchikey"])
        self.assertIn(
            "conservative_reduced_formula_spacegroup",
            fallback["conservative_hits"][0]["reasons"],
        )


@unittest.skipUnless(REAL_CSV.is_file() and REAL_LMDB.is_file(),
                     "真实 starting-crystals CSV 或 LMDB smoke member 不可用")
class RealCrystalRecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = StartingCrystalIndex.from_csv(REAL_CSV, expected_count=218_841)

    def test_real_row1_and_additional_samples(self):
        records = {}
        for row_id in (1, 2, 3, 100, 1000):
            with self.subTest(row_id=row_id):
                row = read_ase_lmdb_row(REAL_LMDB, row_id)
                sample = StructureSample.from_ase_row(row)
                metadata_row = self.index.get(sample.starting_heap)
                self.assertIsNotNone(metadata_row)
                record = build_triple_record(
                    sample, metadata_row,
                    {"exhaustive": True, "use_vdw": (False,)},
                )
                self.assertEqual(record["missing"], [], record.get("recovery"))
                self.assertEqual(record["recovery"]["status"], "success")
                self.assertIsNotNone(record["parent_inchikey"])
                records[row_id] = record

        self.assertEqual(records[1]["starting_heap"], "OQECAP-1-press-97b75093ceb3ffd")
        self.assertEqual(records[1]["spacegroup"], 1)
        self.assertEqual(records[1]["parent_inchikey"], OQECAP_INCHIKEY)
        for row_id in (2, 3, 100):
            self.assertEqual(records[row_id]["parent_inchikey"], OQECAP_INCHIKEY)
        self.assertEqual(records[1000]["starting_heap"], "FUPTOZ-2-press-b8a1abc1a6916c7")
        self.assertEqual(records[1000]["parent_composition"], "C6H2Cl8OS2")
        self.assertEqual(records[1000]["recovery"]["matching_fragments"], 2)
        self.assertEqual(records[1000]["recovery"]["successful_fragments"], 2)
        self.assertEqual(len(records[1000]["recovery"]["fragment_inchikeys"]), 1)


if __name__ == "__main__":
    unittest.main()
