from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from muse_encoder.pretrain.corpora import CorpusSpec
from muse_encoder.pretrain.indexed import IndexedTextCorpus, build_pi1m_view_cache


class IndexedCorpusTests(unittest.TestCase):
    def test_tsv_random_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.tsv"
            path.write_text("a\tCC\nb\t*CO[IH]\n", encoding="utf-8")
            corpus = IndexedTextCorpus(CorpusSpec("x", path, "tsv", 2),
                                       Path(tmp) / "x.offsets.npy")
            self.assertEqual(corpus.get(1).key, "b")
            self.assertEqual(corpus.get(1).text, "*CO[IH]")
            self.assertEqual(corpus.get(0).row, 0)
            with self.assertRaises(IndexError):
                corpus.get(2)

    def test_multiline_fasta_random_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.fasta"
            path.write_text(">a desc\nMK\nLM\n>b\nGG\n", encoding="utf-8")
            corpus = IndexedTextCorpus(CorpusSpec("x", path, "fasta", 2),
                                       Path(tmp) / "x.offsets.npy")
            self.assertEqual(corpus.get(0).text, "MKLM")
            self.assertEqual(corpus.get(1).key, "b")
            self.assertEqual(corpus.get(1).text, "GG")

    def test_pi1m_view_cache_keeps_source_row_and_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            clean = Path(tmp) / "pi1m.tsv"
            clean.write_text("a\t*CCO*\nb\t*CN*\n", encoding="utf-8")
            spec = CorpusSpec("pi1m", clean, "tsv", 2)
            with patch.dict("muse_encoder.pretrain.indexed.CORPORA", {"pi1m": spec}, clear=True):
                output_spec = build_pi1m_view_cache(Path(tmp) / "views.tsv", workers=2, limit=2)
                corpus = IndexedTextCorpus(output_spec, Path(tmp) / "views.offsets.npy")
                self.assertEqual(len(corpus.offsets), 11)
                for source_row in range(2):
                    for view in range(5):
                        record = corpus.get(source_row * 5 + view)
                        self.assertEqual(record.row, source_row)
                        self.assertEqual(record.view, view)
                        self.assertEqual(record.key, "ab"[source_row])
                        self.assertEqual(record.text.count("*"), 2)


if __name__ == "__main__":
    unittest.main()
