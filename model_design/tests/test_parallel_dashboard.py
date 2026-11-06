from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from muse_encoder.config import MuseConfig
from muse_encoder.featurize import RecordFeaturizer
from muse_encoder.model import MuseEncoder
from muse_encoder.pretrain import dashboard as D
from muse_encoder.pretrain.parallel_dashboard import encode_records_parallel
from muse_encoder.tokenizers import load_tokenizers


class ParallelDashboardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        tokenizers = load_tokenizers()
        self.cfg = MuseConfig(
            smiles_vocab_size=len(tokenizers["smiles"]),
            psmiles_vocab_size=len(tokenizers["psmiles"]),
            residue_vocab_size=len(tokenizers["residue"]),
            d_model=32, n_layers=2, n_heads=4, d_ffn=48, dropout=0.0,
            max_seq_len=32, num_experts=2, mole_every=1, embed_dim=24,
            contrastive_dim=8, descriptor_dim=8, crystal_composition_features=True,
        )
        torch.manual_seed(17)
        self.model = MuseEncoder(self.cfg)
        self.checkpoint = self.root / "toy.pt"
        self.sha = self._save_checkpoint()
        self.records = []
        for index in range(9):
            self.records.append({
                "id": f"crystal-{index}",
                "sample_type": "periodic_frame",
                "inputs": {
                    "lattice": np.eye(3).tolist(),
                    "frac_coords": [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25],
                                    [0.5 + index * 0.001, 0.5, 0.5]],
                    "elements": ["C", "O", "C"],
                },
                "external_inputs": [],
                "labels": {},
            })

    def tearDown(self):
        self.tmp.cleanup()

    def _save_checkpoint(self):
        torch.save({
            "config": self.cfg.to_dict(),
            "model": self.model.state_dict(),
            "state": {"step": 0},
        }, self.checkpoint)
        return D.sha256_file(self.checkpoint)

    def _encode(self, sha, workers=2):
        return encode_records_parallel(
            self.records,
            checkpoint_path=self.checkpoint,
            checkpoint_sha=sha,
            cfg=self.cfg,
            cache_root=self.root / "cache",
            external_base=self.root,
            external_cache=self.root / "external",
            namespace="ToyCrystal/toy/validation",
            data_version="toy-v1",
            devices=("cpu", "cpu") if workers == 2 else ("cpu",),
            cpu_workers=workers,
            batch_size=4,
            token_level=False,
        )

    def test_parallel_matches_reference_and_cache_is_checkpoint_scoped(self):
        reference = D.encode_records(
            self.model, RecordFeaturizer(self.cfg), self.cfg,
            self.records, batch_size=4, token_level=False, device="cpu",
        )
        encoded, stats = self._encode(self.sha)
        self.assertFalse(stats["central_hit"])
        self.assertEqual(encoded.ids, reference.ids)
        self.assertEqual(encoded.failed, [])
        np.testing.assert_array_equal(encoded.embeddings, reference.embeddings)

        cached, stats = self._encode(self.sha)
        self.assertTrue(stats["central_hit"])
        np.testing.assert_array_equal(cached.embeddings, encoded.embeddings)

        with torch.no_grad():
            self.model.readout.bias.add_(1.0)
        new_sha = self._save_checkpoint()
        updated, stats = self._encode(new_sha, workers=1)
        self.assertFalse(stats["central_hit"])
        self.assertEqual(stats["normalization_hits"], 3)
        self.assertEqual(stats["prepared_hits"], 3)
        self.assertFalse(np.array_equal(updated.embeddings, encoded.embeddings))

    def test_rdkit_normalized_records_match_reference(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")
        conformer = Chem.Conformer(mol.GetNumAtoms())
        for index in range(mol.GetNumAtoms()):
            conformer.SetAtomPosition(index, (float(index), 0.1 * index, 0.0))
        mol.AddConformer(conformer)
        mol.SetProp("_Name", "sdf-1")
        with Chem.SDWriter(str(self.root / "mols.sdf")) as writer:
            writer.write(mol)
        record = {
            "id": "sdf-record", "sample_type": "molecule",
            "inputs": {"smiles": "CCO"},
            "external_inputs": [{"role": "molecular_structure", "format": "sdf",
                                  "selector": {"record_id": "sdf-1"},
                                  "source_path": "mols.sdf"}],
            "labels": {},
        }
        resolver = D.ExternalStructureResolver(self.root, self.root / "external")
        resolver.prepare([record])
        normalized = D.normalize_record(record, resolver)
        reference = D.encode_records(
            self.model, RecordFeaturizer(self.cfg), self.cfg,
            [normalized], batch_size=1, token_level=False, device="cpu",
        )
        self.records = [record]
        encoded, stats = self._encode(self.sha, workers=1)
        self.assertFalse(stats["central_hit"])
        self.assertEqual(encoded.failed, [])
        np.testing.assert_array_equal(encoded.embeddings, reference.embeddings)

    def test_checkpoint_load_retries_transient_oom(self):
        from unittest.mock import patch
        from muse_encoder.pretrain import parallel_dashboard as P

        P._ENCODER = None
        P._ENCODER_CHECKPOINT = None
        with patch.object(P.D, "load_checkpoint", side_effect=[
                torch.OutOfMemoryError("temporary"), (self.model, self.cfg, {})]), \
                patch.object(P.time, "sleep") as sleep:
            encoder, cfg = P._load_encoder(str(self.checkpoint), "cuda:0")
        self.assertIs(encoder, self.model)
        self.assertIs(cfg, self.cfg)
        sleep.assert_called_once_with(5)
        P._ENCODER = None
        P._ENCODER_CHECKPOINT = None

    def test_encoding_worker_loads_the_repair_registry(self):
        from unittest.mock import patch
        from muse_encoder.pretrain import parallel_dashboard as P

        raw = [{"id": "repaired"}]
        items = [{"record": None, "error": None}]
        encoded = D.EncodedSplit(["repaired"], np.zeros((1, self.cfg.embed_dim)),
                                 None, None, [])
        job = (0, raw, str(self.checkpoint), "cpu", self.root / "prepared.pkl",
               "key", 1, False, "repairs.json")
        with patch.object(P, "_ENCODING_DEVICE", None), \
                patch.object(P, "_configure_graph_repairs") as configure, \
                patch.object(P, "_read_cache", return_value=items), \
                patch.object(P, "_load_encoder", return_value=(self.model, self.cfg)), \
                patch.object(P, "_encode_items", return_value=encoded):
            result = P._encoding_job(job)
        configure.assert_called_once_with("repairs.json")
        self.assertEqual(result[1].ids, ["repaired"])


if __name__ == "__main__":
    unittest.main()
