from __future__ import annotations

import unittest

import numpy as np
import torch
from rdkit import Chem

from muse_encoder.config import MuseConfig
from muse_encoder.featurize import ATOM_ELEMENT_INDEX, RecordFeaturizer, collate
from muse_encoder.masking import IGNORE
from muse_encoder.pretrain.batching import BatchBuilder
from muse_encoder.tokenizers import ResidueTokenizer, SPECIALS


def config():
    return MuseConfig(d_model=16, n_layers=1, n_heads=4, d_ffn=16,
                      embed_dim=12, num_experts=1, descriptor_dim=208,
                      max_seq_len=2048, dropout=0.0)


class BatchBuilderTests(unittest.TestCase):
    def test_mixed_mlm_masks_atom_feature_and_uses_token_heads(self):
        cfg = config()
        fz = RecordFeaturizer(cfg)
        record = fz.featurize_protein_ligand("ACDEFGHIKLMNPQRSTVWY", "C" * 50)
        original = collate([record], cfg)
        original_tokens = torch.from_numpy(original["token_ids"])
        original_features = torch.from_numpy(original["atom_features"])
        batch, targets = BatchBuilder(cfg).build([record], seed=17, denoise=False)
        labels = targets["mlm_labels"]
        selected = labels != IGNORE
        self.assertTrue(selected.any())
        self.assertTrue(torch.equal(labels[selected], original_tokens[selected]))
        self.assertTrue(torch.equal(batch["token_ids"][~selected],
                                    original_tokens[~selected]))
        self.assertEqual(int(labels[0, 0]), IGNORE)
        kinds = batch["token_kind"]
        atom_selected = selected & (kinds == 0)
        atom_unselected = ~selected & (kinds == 0)
        self.assertTrue(atom_selected.any())
        self.assertTrue((batch["token_ids"][atom_selected] == cfg.atom_mask_id).all())
        self.assertTrue((batch["atom_features"][..., ATOM_ELEMENT_INDEX][atom_selected]
                         == cfg.atom_mask_id).all())
        self.assertTrue(torch.equal(
            batch["atom_features"][..., ATOM_ELEMENT_INDEX][atom_unselected],
            original_features[..., ATOM_ELEMENT_INDEX][atom_unselected]))
        self.assertTrue(torch.equal(batch["atom_features"][..., 1:],
                                    original_features[..., 1:]))
        self.assertTrue((targets["mlm_head"][kinds == 0] == 3).all())
        self.assertTrue((targets["mlm_head"][kinds == 3] == 2).all())
        self.assertFalse(batch["drop_feature_mask"].any())
        self.assertNotIn("denoise_sigma", batch)

    def test_mlm_uses_injected_tokenizer_special_ids(self):
        cfg = config()
        cfg.residue_vocab_size = 12
        cfg.mlm_mask_prob = 1.0
        ordinary = ("A", "C", "D", "E", "F", "G")
        vocab = {token: index for index, token in enumerate(ordinary)}
        vocab.update({token: index + len(ordinary)
                      for index, token in enumerate(SPECIALS)})
        tokenizers = {"residue": ResidueTokenizer(vocab)}
        record = RecordFeaturizer(cfg, tokenizers=tokenizers).featurize_sequence("A?C")
        original = torch.from_numpy(record.token_ids).unsqueeze(0)
        batch, targets = BatchBuilder(cfg, tokenizers=tokenizers).build([record], seed=5)
        labels = targets["mlm_labels"]
        special = original >= len(ordinary)
        self.assertTrue((labels[special] == IGNORE).all())
        self.assertTrue(torch.equal(batch["token_ids"][special], original[special]))
        self.assertTrue((labels[~special] != IGNORE).all())
        random_replaced = ((labels != IGNORE)
                           & (batch["token_ids"] != tokenizers["residue"].mask_id)
                           & (batch["token_ids"] != original))
        self.assertTrue((batch["token_ids"][random_replaced] < len(ordinary)).all())

    def test_denoise_scope_and_determinism(self):
        cfg = config()
        fz = RecordFeaturizer(cfg)
        coords = np.asarray([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [2.0, 1.0, 0.2]])
        records = [
            fz.featurize_molecule("CCO", coords=coords),
            fz.featurize_sequence("ACD"),
            fz.featurize_crystal(np.eye(3) * 5.0, np.array([[0.0, 0.0, 0.0]]), ["Na"]),
        ]
        original = collate(records, cfg)
        clean = torch.from_numpy(original["coords"])
        first_batch, first_targets = BatchBuilder(cfg).build(records, seed=23, denoise=True)
        second_batch, second_targets = BatchBuilder(cfg).build(records, seed=23, denoise=True)
        for key in ("coords", "denoise_sigma", "drop_feature_mask"):
            np.testing.assert_array_equal(first_batch[key].numpy(), second_batch[key].numpy())
        for key in ("noise", "coords"):
            np.testing.assert_array_equal(first_targets[key].numpy(),
                                          second_targets[key].numpy())
        active = first_batch["denoise_active"]
        np.testing.assert_array_equal(active.numpy(), [True, False, True])
        sigma = first_batch["denoise_sigma"].view(-1, 1, 1)
        noise = first_targets["noise"]
        noise_mask = first_targets["noise_mask"]
        torch.testing.assert_close(first_batch["coords"], clean + sigma * noise)
        recovered_noise = (first_batch["coords"] - clean) / sigma
        torch.testing.assert_close(recovered_noise[noise_mask], noise[noise_mask])
        torch.testing.assert_close(first_batch["coords"][~noise_mask], clean[~noise_mask])
        torch.testing.assert_close(first_targets["coords"], clean)
        expected_coord_mask = torch.from_numpy(original["coord_mask"]) & torch.tensor(
            [True, False, False])[:, None]
        self.assertTrue(torch.equal(first_targets["coord_mask"], expected_coord_mask))
        self.assertTrue(first_targets["coord_mask"][0].any())
        self.assertFalse(first_targets["coord_mask"][1].any())
        self.assertFalse(first_targets["coord_mask"][2].any())
        self.assertTrue(noise_mask[2].any())
        self.assertFalse((first_batch["drop_feature_mask"] & ~active).any())
        self.assertTrue(torch.equal(first_batch["drop_feature_mask"],
                                    first_targets["drop_feature_mask"]))
        noisy = first_batch["coords"]
        expected_pair = noisy[:, None, :, :] - noisy[:, :, None, :]
        shift_cart = torch.einsum(
            "bxyi,bjk->bxyk",
            torch.from_numpy(original["pbc_shift"]),
            torch.from_numpy(original["lattice"]),
        )
        expected_pair = expected_pair + shift_cart
        torch.testing.assert_close(first_batch["pair_disp"], expected_pair)
        self.assertFalse(first_batch["pair_basis_mask"][active].any())
        self.assertGreater(
            (first_batch["pair_disp"] - torch.from_numpy(original["pair_disp"])).abs().max(),
            0.0,
        )

    def test_denoise_uses_configured_sigma_range(self):
        cfg = config()
        cfg.denoise_sigma_min = 0.25
        cfg.denoise_sigma_max = 0.25
        coords = np.asarray([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [2.0, 1.0, 0.2]])
        record = RecordFeaturizer(cfg).featurize_molecule("CCO", coords=coords)
        batch, _targets = BatchBuilder(cfg).build([record], seed=29, mlm=False,
                                                  denoise=True)
        torch.testing.assert_close(batch["denoise_sigma"], torch.full((1,), 0.25))

    def test_drop_feature_probability_and_target_sync(self):
        cfg = config()
        coords = np.asarray([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [2.0, 1.0, 0.2]])
        record = RecordFeaturizer(cfg).featurize_molecule("CCO", coords=coords)
        batch, targets = BatchBuilder(cfg).build([record] * 2048, seed=37,
                                                 mlm=False, denoise=True)
        drop = batch["drop_feature_mask"]
        self.assertTrue(batch["denoise_active"].all())
        self.assertTrue(torch.equal(drop, targets["drop_feature_mask"]))
        self.assertAlmostEqual(float(drop.float().mean()), 0.5, delta=0.05)

    def test_crystal_denoise_has_no_molecular_coordinate_target(self):
        cfg = config()
        record = RecordFeaturizer(cfg).featurize_crystal(
            np.eye(3) * 5.0, np.array([[0.0, 0.0, 0.0]]), ["Na"])
        batch, targets = BatchBuilder(cfg).build([record], seed=41, mlm=False,
                                                 denoise=True)
        self.assertTrue(batch["denoise_active"].all())
        self.assertTrue(targets["noise_mask"].any())
        self.assertNotIn("coords", targets)
        self.assertNotIn("coord_mask", targets)

    def test_peptide_strict_mask_runs_with_real_annotations(self):
        cfg = config()
        fz = RecordFeaturizer(cfg)
        mol = Chem.MolFromSequence("ACDE")
        record = fz.featurize_peptide_atomic(smiles=Chem.MolToSmiles(mol))
        batch, targets = BatchBuilder(cfg).build([record], seed=31)
        selected = targets["mlm_labels"] != IGNORE
        self.assertTrue(selected.any())
        self.assertTrue((batch["token_ids"][selected] == cfg.atom_mask_id).all())
        self.assertTrue((batch["atom_features"][..., ATOM_ELEMENT_INDEX][selected]
                         == cfg.atom_mask_id).all())

    def test_peptide_strict_mask_rejects_fallback(self):
        cfg = config()
        mol = Chem.MolFromSequence("ACDE")
        record = RecordFeaturizer(cfg).featurize_peptide_atomic(
            smiles=Chem.MolToSmiles(mol))
        for key in ("group_ids", "fragment_ids"):
            record.meta[key] = np.full_like(record.meta[key], -1)
        record.meta["sidechain_flags"] = np.zeros_like(
            record.meta["sidechain_flags"])
        with self.assertRaisesRegex(ValueError, "strict"):
            BatchBuilder(cfg).build([record], seed=31)

    def test_metadata_collate_graph_atom_bond_and_efs(self):
        cfg = config()
        cfg.aux_specs = {
            "g": {"kind": "graph", "dim": 1, "task_type": "regression"},
            "a": {"kind": "atom", "dim": 1, "task_type": "regression"},
            "b": {"kind": "bond", "dim": 1, "task_type": "regression"},
        }
        record = RecordFeaturizer(cfg).featurize_molecule("CCO")
        record.meta["sdf_mapping"] = np.asarray([2, 0, 4])
        record.meta["sdf_bond_pairs"] = np.asarray([(0, 1), (0, 2), (1, 2), (0, 4)])
        item = {
            "descriptor": (np.ones(208, dtype=np.float32), np.ones(208, dtype=bool)),
            "energy": (np.ones(1, dtype=np.float32), np.ones(1, dtype=bool)),
            "stress": (np.ones(6, dtype=np.float32), np.ones(6, dtype=bool)),
            "force": (np.ones((3, 3), dtype=np.float32), np.ones((3, 3), dtype=bool)),
            "aux_records": [
                {"name": "g", "value": np.ones(1, dtype=np.float32), "mask": np.ones(1, dtype=bool)},
                {"name": "a", "value": np.arange(5, dtype=np.float32).reshape(5, 1),
                 "mask": np.ones((5, 1), dtype=bool)},
                {"name": "b", "value": np.asarray([10, 20, 30, 40], dtype=np.float32).reshape(4, 1),
                 "mask": np.ones((4, 1), dtype=bool)},
            ],
        }
        _batch, targets = BatchBuilder(cfg).build([record], seed=1, items=[item])
        self.assertEqual(targets["descriptor"].shape, (1, 208))
        self.assertEqual(targets["force"].shape, (1, 4, 3))
        self.assertEqual(targets["force_mask"].sum().item(), 3)
        aux = {record["name"]: record for record in targets["aux_records"]}
        self.assertEqual(aux["g"]["value"].shape, (1, 1))
        self.assertEqual(aux["a"]["value"].shape, (1, 4, 1))
        self.assertEqual(aux["b"]["value"].shape, (1, 4, 4, 1))
        self.assertEqual(aux["a"]["mask"].sum().item(), 3)
        self.assertEqual(aux["b"]["mask"].sum().item(), 4)
        np.testing.assert_array_equal(aux["a"]["value"][0, 1:, 0], [2, 0, 4])
        self.assertEqual(aux["b"]["value"][0, 1, 2, 0], 20)
        self.assertEqual(aux["b"]["value"][0, 2, 1, 0], 20)
        self.assertEqual(aux["b"]["value"][0, 2, 3, 0], 40)
        self.assertEqual(aux["b"]["value"][0, 3, 2, 0], 40)


if __name__ == "__main__":
    unittest.main()
