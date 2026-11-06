from __future__ import annotations

import unittest

import numpy as np
import torch
from rdkit import Chem

from muse_encoder.config import MuseConfig
from muse_encoder.featurize import RecordFeaturizer
from muse_encoder.model import MuseEncoder
from muse_encoder.pretrain.batching import BatchBuilder


class RealMixedIntegrationTests(unittest.TestCase):
    def test_four_domain_mixed_forward_backward(self):
        cfg = MuseConfig(
            d_model=32, n_layers=2, n_heads=4, d_ffn=48, num_experts=2,
            mole_every=1, embed_dim=24, contrastive_dim=8, descriptor_dim=208,
            max_seq_len=2048, dropout=0.0,
            aux_specs={"toy": {"kind": "graph", "dim": 1, "task_type": "regression"}},
        )
        fz = RecordFeaturizer(cfg)
        coords = np.asarray([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [2.0, 1.0, 0.2]])
        peptide = Chem.MolFromSequence("ACDE")
        records = [
            fz.featurize_molecule("CCO", coords=coords),
            fz.featurize_polymer(psmiles="*CCO*"),
            fz.featurize_crystal(np.eye(3) * 5.0, np.array([[0.0, 0.0, 0.0]]), ["Na"]),
            fz.featurize_sequence("ACDEFGHIKLMNPQRSTVWY"),
            fz.featurize_protein_ligand("ACDE", "CCO"),
            fz.featurize_peptide_atomic(smiles=Chem.MolToSmiles(peptide)),
        ]
        batch, targets = BatchBuilder(cfg).build(records, seed=41, denoise=True)
        batch_size, length = batch["token_ids"].shape
        # denoise 开启时，denoise 活跃样本的 pair_basis 清零（它由洁净坐标预计算，
        # 带噪输入再用会泄漏洁净几何）；pair_multiplicity 不受此门控。
        self.assertEqual(batch["pair_basis_mask"].tolist(),
                         [False, False, False, False, False, False])
        self.assertEqual(batch["pair_multiplicity_mask"].tolist(),
                         [False, False, True, False, False, False])
        clean_batch, _ = BatchBuilder(cfg).build(records, seed=41, denoise=False)
        # denoise 关闭（如 S3 能量/力阶段）：晶体保留完整 PBC pair basis。
        self.assertEqual(clean_batch["pair_basis_mask"].tolist(),
                         [False, False, True, False, False, False])
        targets.update({
            "energy": torch.randn(batch_size),
            "force": torch.randn(batch_size, length, 3),
            "force_mask": batch["coord_mask"],
            "stress": torch.randn(batch_size, 6),
            "descriptor": torch.randn(batch_size, 208),
            "aux_records": [{"name": "toy", "value": torch.randn(batch_size),
                             "mask": torch.ones(batch_size, dtype=torch.bool)}],
        })
        model = MuseEncoder(cfg)
        losses = model.pretraining_losses(
            batch, targets,
            {"mlm": 1.0, "denoise": 1.0, "coordinate": 1.0, "distance": 1.0,
             "energy": 1.0, "force": 30.0, "stress": 1.0,
             "descriptor": 1.0, "aux": 1.0},
        )
        for key in ("mlm", "denoise", "coordinate", "distance", "energy",
                    "force", "stress", "descriptor", "aux", "total"):
            self.assertIn(key, losses)
            self.assertTrue(torch.isfinite(losses[key]), key)
        losses["total"].backward()
        gradient = sum(parameter.grad.abs().sum().item()
                       for parameter in model.parameters() if parameter.grad is not None)
        self.assertGreater(gradient, 0.0)
        self.assertTrue(np.isfinite(gradient))
        model.eval()
        with torch.no_grad():
            encoded = model.encode(batch)
        self.assertEqual(encoded["embedding"].shape, (batch_size, cfg.embed_dim))
        self.assertEqual(encoded["tokens"].shape, (batch_size, length, cfg.embed_dim))
        self.assertTrue(torch.isfinite(encoded["embedding"]).all())


if __name__ == "__main__":
    unittest.main()
