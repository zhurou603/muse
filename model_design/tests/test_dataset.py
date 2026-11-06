from __future__ import annotations

import unittest

import numpy as np

from muse_encoder.config import MuseConfig
from muse_encoder.pretrain.corpora import DescriptorNormalizer, PRETRAIN_ROOT
from muse_encoder.pretrain.dataset import PretrainDataset, source_spec
from muse_encoder.pretrain.sampling import SamplePointer
from muse_encoder.pretrain.targets import TargetNormalizer


def config():
    return MuseConfig(d_model=16, n_layers=1, n_heads=4, d_ffn=16,
                      num_experts=1, embed_dim=12, descriptor_dim=208,
                      max_seq_len=32, dropout=0.0)


def descriptor_normalizer():
    return DescriptorNormalizer({"mean": np.zeros(208), "std": np.ones(208),
                                 "mask": np.ones(208, dtype=bool)})


def qm9_normalizer():
    key = "qm9::u0::"
    return TargetNormalizer({
        "specs": {key: {"source": "qm9", "name": "u0", "unit": None,
                        "kind": "graph", "task_type": "regression", "dim": 1,
                        "shadow": False}},
        "stats": {key: {"mean": [-40.0], "std": [2.0], "mask": [True],
                        "n": [1], "n_bad": 0}},
    })


class DatasetTests(unittest.TestCase):
    def test_smiles_mtr_and_natural_peptide(self):
        cfg = config()
        chembl = PretrainDataset("chembl", cfg, descriptor_normalizer=descriptor_normalizer())
        record, item = chembl.get(SamplePointer("chembl", "small_molecule", 0))
        self.assertTrue(record.meta["sequence_frontend"])
        self.assertEqual(item["descriptor"][0].shape, (208,))
        self.assertEqual(item["descriptor"][1].shape, (208,))
        rows = np.load(PRETRAIN_ROOT / "index" / "natural_peptides.uniref50.rows.npy",
                       mmap_mode="r")
        natural = PretrainDataset("natural_uniref50", cfg)
        self.assertEqual(len(natural), len(rows))
        record, _ = natural.get(SamplePointer("natural_uniref50", "bio", 0))
        self.assertLessEqual(len(record.token_ids), cfg.max_seq_len)
        self.assertTrue((record.token_kind == 3).all())

    def test_cycpept_annotations_are_aligned(self):
        cfg = config()
        dataset = PretrainDataset("cycpept", cfg)
        self.assertEqual(len(dataset), 7326)
        record, _ = dataset.get(SamplePointer("cycpept", "bio", 0))
        self.assertEqual(record.meta["sample_type"], "peptide_atomic")
        for key in ("group_ids", "sidechain_flags", "fragment_ids"):
            self.assertEqual(len(record.meta[key]), len(record.token_ids))
        rows = np.load(PRETRAIN_ROOT / "index" / "natural_peptides.pdb.rows.npy",
                       mmap_mode="r")
        self.assertEqual(source_spec("natural_pdb").source.records, len(rows))

    def test_aux_molecule_2d_view_has_no_duplicate_labels(self):
        cfg = config()
        dataset = PretrainDataset("qm9", cfg)
        record, item = dataset.get_2d_view(SamplePointer("qm9", "small_molecule", 0))
        self.assertIsNone(record.coords)
        self.assertEqual(record.meta["sample_type"], "molecule")
        self.assertEqual(item, {})
        dataset.close()

    def test_aux_molecule_reads_official_3d_and_energy(self):
        cfg = config()
        dataset = PretrainDataset("qm9", cfg, normalizer=qm9_normalizer(),
                                  use_molecule3d=True)
        record, item = dataset.get(SamplePointer("qm9", "small_molecule", 0))
        self.assertIsNotNone(record.coords)
        self.assertTrue(record.coord_mask[1:].all())
        self.assertEqual(record.meta["structure_provenance"]["source"], "qm9")
        self.assertEqual(record.meta["structure_provenance"]["n_atoms"], record.n_real_tokens)
        self.assertEqual(item["energy"][0].shape, (1,))
        self.assertTrue(item["energy"][1].all())
        self.assertEqual(item["aux_records"], [])
        dataset.close()

    def test_molecule3d_gate_and_strict_failures(self):
        cfg = config()
        with self.assertRaisesRegex(ValueError, "无官方 3D 结构源"):
            PretrainDataset("chembl", cfg, use_molecule3d=True)
        plain = PretrainDataset("qm9", cfg, normalizer=qm9_normalizer())
        record, _ = plain.get(SamplePointer("qm9", "small_molecule", 0))
        self.assertIsNone(record.coords)
        plain.close()

        dataset = PretrainDataset("qm9", cfg, normalizer=qm9_normalizer(),
                                  use_molecule3d=True)
        aux_id = dataset._open_aux().get(0).id

        class _FailingStore:
            def get(self, row):
                raise ValueError("sdf_parse: synthetic")

            def close(self):
                pass

        dataset._structure = _FailingStore()
        for registered in (True, False):
            dataset._structure_failures = {aux_id} if registered else set()
            with self.assertRaisesRegex(ValueError, "禁止降级 2D"):
                dataset.get(SamplePointer("qm9", "small_molecule", 0))
        dataset.close()


if __name__ == "__main__":
    unittest.main()
