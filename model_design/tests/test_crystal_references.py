from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from muse_encoder.pretrain.crystal_references import (
    RESIDUAL_MODE,
    ElementReferences,
    fit_element_references,
    formula_counts,
    write_references_and_stats,
)


def record(record_id, formula, energy, split="train"):
    count = sum(formula_counts(formula).values())
    return {
        "id": record_id,
        "inputs": {"chemical_formula": formula, "atom_count": count},
        "external_inputs": [{"selector": {"sid": f"{record_id}-1-gener-hash-0"}}],
        "labels": {"energy": [{"value": energy, "unit": "eV", "split": split}]},
    }


class CrystalReferencesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.records = [
            record("a", "C", -97.0),
            record("b", "H2", 1.0),
            record("c", "O2", -1.0),
            record("d", "CH4", -101.0),
            record("e", "C2H4", -200.5),
            record("f", "C2H4", -201.5),
            record("g", "C3H8", -304.75),
            record("h", "C3H8", -305.25),
        ]
        self.train = self.root / "train.jsonl"
        self.train.write_text("".join(json.dumps(row) + "\n" for row in self.records))

    def test_train_only_fit_reference_energy_and_stats_handshake(self):
        payload = fit_element_references(self.train)
        self.assertAlmostEqual(payload["element_energies_ev"][6], -100.0, places=7)
        self.assertAlmostEqual(payload["element_energies_ev"][1], -1.0, places=7)
        self.assertAlmostEqual(payload["element_energies_ev"][8], -2.0, places=7)
        self.assertAlmostEqual(payload["intercept_ev"], 3.0, places=7)
        self.assertGreater(payload["fit"]["residual_std_ev"], 0.0)

        references_path = self.root / "references.json"
        stats_path = self.root / "residual_stats.json"
        frame_stats = self.root / "frame_stats.json"
        frame_stats.write_text(json.dumps({"energy": {"mean": 0.0, "std": 1.0}}))
        write_references_and_stats(references_path, stats_path, payload, frame_stats)
        references = ElementReferences.from_json(references_path)
        self.assertEqual(references.sha256, hashlib.sha256(references_path.read_bytes()).hexdigest())
        self.assertAlmostEqual(references.reference_energy(np.array([6, 6, 1, 1, 1, 1])), -201.0)
        with self.assertRaisesRegex(ValueError, "未覆盖元素"):
            references.reference_energy(np.array([7]))
        stats = json.loads(stats_path.read_text())
        self.assertEqual(stats["energy"]["mode"], RESIDUAL_MODE)
        self.assertEqual(stats["energy"]["reference_sha256"], references.sha256)

    def test_validation_labels_are_rejected(self):
        invalid = list(self.records)
        invalid[0] = record("a", "C2H4", -200.5, split="validation")
        self.train.write_text("".join(json.dumps(row) + "\n" for row in invalid))
        with self.assertRaisesRegex(ValueError, "只允许 train split"):
            fit_element_references(self.train)

    def test_formula_parser_is_strict(self):
        with self.assertRaisesRegex(ValueError, "不支持或非法"):
            formula_counts("C2(H4)")


if __name__ == "__main__":
    unittest.main()
