from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from muse_encoder.pretrain import corpora as C


class CorpusTests(unittest.TestCase):
    def test_tsv_preserves_key_input_and_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pi1m.tsv"
            path.write_text("key-a\t*CC[IH]\nkey-b\t*CO*\n", encoding="utf-8")
            rows = list(C.iter_tsv(C.CorpusSpec("pi1m", path, "tsv", 2)))
        self.assertEqual(rows[0].key, "key-a")
        self.assertEqual(rows[0].text, "*CC[IH]")
        self.assertEqual(rows[0].row, 0)
        self.assertEqual(rows[1].row, 1)

    def test_fasta_streaming_and_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.fasta"
            path.write_text(">a description\nMK\nLM\n>b\nGG\n", encoding="utf-8")
            spec = C.CorpusSpec("x", path, "fasta", 2)
            self.assertEqual([r.text for r in C.iter_fasta(spec)], ["MKLM", "GG"])
            self.assertEqual([r.key for r in C.iter_fasta(spec, limit=1)], ["a"])

    def test_crop_is_deterministic_and_records_window(self):
        record = C.TextRecord("uniref50", "id", "ABCDEFGHIJ", 7)
        first = C.crop_sequence(record, 4, 42)
        second = C.crop_sequence(record, 4, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first.text), 4)
        self.assertEqual(record.text[first.crop_start:first.crop_start + 4], first.text)
        self.assertNotEqual(first.crop_seed, 0)
        self.assertEqual(C.crop_sequence(record, 20, 42), record)

    def test_random_smiles_views_are_seeded_and_preserve_polymer_sites(self):
        first = C.random_smiles_views("*CCO*", 5, 123)
        second = C.random_smiles_views("*CCO*", 5, 123)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(first[0], "*CCO*")
        for value in first:
            self.assertEqual(value.count("*"), 2)

    def test_random_views_keep_unsanitized_polymer_syntax(self):
        smiles = "*/C=C/c1cccc(S(=O)(=O)[Si](*)(=O)NCCCCCC)c1"
        views = C.random_smiles_views(smiles, 5, 42)
        self.assertEqual(len(views), 5)
        self.assertEqual(views[0], smiles)
        self.assertEqual(C.random_smiles_views(smiles, 5, 42), views)
        for value in views:
            self.assertEqual(value.count("*"), 2)
            self.assertIn("[Si]", value)

    def test_alignment_detects_key_and_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = Path(tmp) / "x.tsv"
            keys = Path(tmp) / "x.keys.txt"
            text.write_text("a\tCC\nb\tCO\n", encoding="utf-8")
            keys.write_text("a\nb\n", encoding="utf-8")
            self.assertEqual(C.validate_row_alignment(text, keys), 2)
            keys.write_text("a\nc\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "错位"):
                C.validate_row_alignment(text, keys)
            keys.write_text("a\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "行数"):
                C.validate_row_alignment(text, keys)

    def test_descriptor_global_norm_and_zero_variance_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, offset in (("a", 0.0), ("b", 2.0)):
                values = np.zeros((2, 208), dtype=np.float32)
                values[:, 0] = [offset, offset + 1.0]
                values[:, 1] = 7.0
                np.save(root / f"{name}.desc.f32.npy", values)
                (root / f"{name}.keys.txt").write_text("x\ny\n", encoding="utf-8")
            stores = [C.DescriptorStore("a", root), C.DescriptorStore("b", root)]
            norm = C.compute_global_norm(stores, chunk_size=1)
            self.assertEqual(norm["n"], 4)
            self.assertAlmostEqual(norm["mean"][0], 1.5)
            self.assertTrue(norm["mask"][0])
            self.assertFalse(norm["mask"][1])
            normalizer = C.DescriptorNormalizer(norm)
            transformed, mask = normalizer(stores[0].get(np.array([0, 1])))
            self.assertEqual(transformed.shape, (2, 208))
            self.assertTrue(np.isfinite(transformed).all())
            self.assertTrue((transformed[:, 1] == 0).all())
            self.assertFalse(mask[:, 1].any())

    def test_vocab_and_coverage(self):
        records = [C.TextRecord("x", "a", "CCO", 0), C.TextRecord("x", "b", "CCN", 1)]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "vocab.json"
            payload = C.build_chemical_vocab(records, False, output, min_freq=1)
            coverage = C.vocab_coverage(records, payload["vocab"])
            self.assertEqual(coverage["record_coverage"], 1.0)
            self.assertIn("C", payload["vocab"])
            self.assertEqual(json.loads(output.read_text())["name"], "smiles")


if __name__ == "__main__":
    unittest.main()
