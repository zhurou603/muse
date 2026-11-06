from __future__ import annotations

import itertools
import unittest
from unittest.mock import patch

import numpy as np
import torch
from rdkit import Chem

from muse_encoder.config import MuseConfig
from muse_encoder.featurize import (
    ATOM_CHIRALITY_INDEX,
    ATOM_ELEMENT_INDEX,
    ATOM_FORMAL_CHARGE_INDEX,
    ATOM_HYBRIDIZATION_INDEX,
    ATOM_RADICAL_INDEX,
    ATOM_RING_INDEX,
    ATOM_AROMATIC_INDEX,
    BOND_PBC,
    D_RESIDUE_CHIRAL_CODE,
    RecordFeaturizer,
    collate,
)
from muse_encoder.inference import _featurize_record, encode_subset
from muse_encoder.structures import (
    PBC_BASIS_CENTERS,
    PBC_BASIS_GAMMA,
    align_coordinates,
    build_pbc_neighbors,
    graph_from_smiles,
)


def small_config(**kwargs):
    values = dict(
        d_model=8,
        embed_dim=8,
        n_layers=1,
        n_heads=2,
        d_ffn=16,
        max_seq_len=8,
        crystal_cutoff=6.0,
        crystal_max_neighbors=32,
    )
    values.update(kwargs)
    return MuseConfig(**values)


def sdf_with_coords(smiles, xyz):
    mol = Chem.MolFromSmiles(smiles)
    conformer = Chem.Conformer(len(xyz))
    for i, point in enumerate(xyz):
        conformer.SetAtomPosition(i, tuple(float(x) for x in point))
    mol.AddConformer(conformer)
    return mol


def brute_neighbors(lattice, frac, cutoff, max_neighbors):
    lattice = np.asarray(lattice, dtype=np.float64)
    cart = np.asarray(frac, dtype=np.float64) @ lattice
    n = len(cart)
    result = []
    for i in range(n):
        candidates = []
        for j, shift in itertools.product(range(n), itertools.product(range(-4, 5), repeat=3)):
            shift = np.asarray(shift, dtype=np.int64)
            if i == j and not shift.any():
                continue
            vector = cart[j] + shift @ lattice - cart[i]
            distance = float(np.linalg.norm(vector))
            if 1e-8 < distance <= cutoff + 1e-8:
                candidates.append((distance, j, shift, vector))
        candidates.sort(key=lambda item: (item[0], item[1], *item[2].tolist()))
        result.append(candidates[:max_neighbors])
    return result


class StructureAlignmentTests(unittest.TestCase):
    def test_sdf_reverse_order_coordinate_recovery(self):
        xyz = np.array([[0.0, 0.0, 0.0], [1.4, 0.2, 0.0], [2.2, 1.1, 0.3]])
        source = sdf_with_coords("CCO", xyz)
        shuffled = Chem.RenumberAtoms(source, [2, 1, 0])
        _, recovered, mapping = align_coordinates("CCO", shuffled)
        self.assertEqual(mapping, (2, 1, 0))
        np.testing.assert_allclose(recovered, xyz, rtol=0, atol=1e-12)
        self.assertFalse(np.array_equal(shuffled.GetConformer().GetPositions(), xyz))

    def test_coordinate_count_mismatch_and_nonfinite_are_rejected(self):
        featurizer = RecordFeaturizer(small_config())
        for count in (2, 4):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "完全一致"):
                featurizer.featurize_molecule("CCO", np.zeros((count, 3)))
        bad = np.zeros((3, 3))
        bad[1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "有限"):
            featurizer.featurize_molecule("CCO", bad)

    def test_coordinates_are_centered_without_truncation(self):
        xyz = np.arange(9, dtype=np.float64).reshape(3, 3)
        record = RecordFeaturizer(small_config()).featurize_molecule("CCO", xyz)
        np.testing.assert_allclose(record.coords[1:].mean(axis=0), 0.0, atol=1e-7)
        np.testing.assert_allclose(record.coords[1:], xyz - xyz.mean(axis=0), atol=1e-7)
        np.testing.assert_allclose(
            record.pair_disp[1, 2], record.coords[2] - record.coords[1], atol=1e-7
        )

    def test_strict_never_uses_fallback(self):
        featurizer = RecordFeaturizer(small_config())
        with self.assertRaises(ValueError):
            featurizer.featurize_molecule("C(", strict=True)
        degraded = featurizer.featurize_molecule("C(", strict=False)
        self.assertTrue(degraded.meta["degraded"])
        with patch("muse_encoder.featurize._HAS_RDKIT", False):
            with self.assertRaises(ImportError):
                featurizer.featurize_molecule("CCO", strict=True)
            degraded = featurizer.featurize_molecule("CCO", strict=False)
            self.assertTrue(degraded.meta["degraded"])

    def test_wrong_sdf_connectivity_is_rejected(self):
        source = sdf_with_coords("CCO", np.eye(3))
        with self.assertRaises(ValueError):
            align_coordinates("COC", source)


class ChemicalRecoveryTests(unittest.TestCase):
    def test_unsanitized_topology_has_real_graph_and_audit_metadata(self):
        fz = RecordFeaturizer(small_config())
        smiles = "O=C1C(=CC2=CC=C(c3ccc(Cl)cc3)O2)SC(=S)N1CCc(o)o"
        with self.assertRaises(ValueError):
            fz.featurize_molecule(smiles)
        record = fz.featurize_molecule(smiles, strict=False)
        self.assertEqual(record.meta["parse_mode"], "unsanitized_topology")
        self.assertTrue(record.meta["degraded"])
        self.assertIsNone(record.coords)
        self.assertGreater(record.n_real_tokens, 10)
        self.assertGreater(int(record.bond_mat.sum()), 0)

    def test_missing_input_is_one_unknown_element_without_fake_bonds(self):
        fz = RecordFeaturizer(small_config())
        record = fz.featurize_molecule("nan", strict=False)
        self.assertEqual(record.n_real_tokens, 1)
        self.assertEqual(record.token_ids[1], 0)
        self.assertTrue(record.meta["missing_input"])
        self.assertTrue(record.meta["degraded"])
        self.assertFalse(record.bond_mat.any())
        self.assertIsNone(record.coords)
        sequence = fz.featurize_polymer(psmiles="nan", sequence=True, strict=False)
        self.assertEqual(sequence.n_real_tokens, 1)
        self.assertTrue(sequence.meta["missing_input"])
        self.assertEqual(sequence.token_ids[1], fz.tk["psmiles"].unk_id)


class AtomFeatureTests(unittest.TestCase):
    def setUp(self):
        self.featurizer = RecordFeaturizer(small_config())

    def test_official_atom_order_and_real_discrete_features(self):
        record = self.featurizer.featurize_molecule("C[C@H](O)c1ccccc1")
        graph = graph_from_smiles("C[C@H](O)c1ccccc1")
        np.testing.assert_array_equal(
            record.token_ids[1:], [atom.GetAtomicNum() for atom in graph.GetAtoms()]
        )
        features = record.atom_features[1:]
        self.assertEqual(features[0, ATOM_ELEMENT_INDEX], 5)
        self.assertGreater(features[1, ATOM_CHIRALITY_INDEX], 0)
        self.assertTrue(features[3:, ATOM_AROMATIC_INDEX].all())
        self.assertTrue(features[3:, ATOM_RING_INDEX].all())
        self.assertEqual(features[0, ATOM_HYBRIDIZATION_INDEX], 2)
        self.assertTrue(np.all(record.token_kind == 0))

    def test_formal_charge_and_radical_are_not_omitted(self):
        charged = self.featurizer.featurize_molecule("[NH4+]")
        self.assertEqual(charged.atom_features[1, ATOM_FORMAL_CHARGE_INDEX], 6)
        radical = self.featurizer.featurize_molecule("[CH3]")
        self.assertEqual(radical.atom_features[1, ATOM_RADICAL_INDEX], 1)


class SequenceAndPolymerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = small_config()
        self.featurizer = RecordFeaturizer(self.cfg)

    def test_smiles_and_psmiles_sequence_token_kind(self):
        molecule = self.featurizer.featurize_molecule("CCO", sequence=True)
        np.testing.assert_array_equal(
            molecule.token_ids, self.featurizer.tk["smiles"].encode("CCO", add_cls=True)
        )
        self.assertTrue(np.all(molecule.token_kind == 1))
        self.assertFalse(molecule.use_atom_features)
        self.assertIsNone(molecule.bond_mat)

        polymer = self.featurizer.featurize_polymer(psmiles="[*]CC[*]", sequence=True)
        np.testing.assert_array_equal(
            polymer.token_ids,
            self.featurizer.tk["psmiles"].encode("[*]CC[*]", add_cls=True),
        )
        self.assertTrue(np.all(polymer.token_kind == 2))
        atomic = self.featurizer.featurize_polymer(psmiles="[*]CC[*]")
        self.assertTrue(np.all(atomic.token_kind == 0))

    def test_copolymer_composition_segments_and_pool_weights(self):
        record = self.featurizer.featurize_polymer(
            monomers=["CC", "COO"], composition=[0.2, 0.8]
        )
        np.testing.assert_array_equal(record.segment_ids, [0, 0, 0, 1, 1, 1])
        np.testing.assert_allclose(
            record.pool_weights,
            [0.0, 0.1, 0.1, 0.8 / 3, 0.8 / 3, 0.8 / 3],
            atol=1e-7,
        )
        np.testing.assert_allclose(record.scalar_cond[:2], [0.2, 0.8], atol=1e-7)
        np.testing.assert_allclose(record.meta["composition"], [0.2, 0.8])
        self.assertEqual(record.spd_mat[1, 3], 15)
        hidden = np.arange(1, 6, dtype=np.float32)
        pooled = np.sum(record.pool_weights[1:] * hidden)
        self.assertAlmostEqual(
            pooled, 0.2 * hidden[:2].mean() + 0.8 * hidden[2:].mean(), delta=1e-6
        )

    def test_sequence_copolymer_keeps_all_token_blocks(self):
        record = self.featurizer.featurize_polymer(
            monomers=["CC", "CO"], composition=[1, 3], sequence=True
        )
        self.assertEqual(record.n_real_tokens, 4)
        self.assertTrue(np.all(record.token_kind == 1))
        self.assertAlmostEqual(record.pool_weights.sum(), 1.0, places=6)
        self.assertIsNone(record.bond_mat)

    def test_actual_copolymer_fields_are_routed(self):
        record = _featurize_record(
            self.featurizer,
            {
                "sample_type": "copolymer",
                "inputs": {
                    "monomer_a_smiles": "CC",
                    "monomer_b_smiles": "CO",
                    "fraction_a": 0.4,
                    "fraction_b": 0.6,
                    "composition": "4A_6B",
                },
            },
        )
        np.testing.assert_allclose(record.scalar_cond[:2], [0.4, 0.6])
        self.assertEqual(record.meta["composition_text"], "4A_6B")
        self.assertEqual(record.n_real_tokens, 4)

    def test_overlong_smiles_sequence_is_preserved_for_windowing(self):
        cfg = small_config(max_seq_len=32)
        fz = RecordFeaturizer(cfg)
        record = fz.featurize_molecule("C" * 600, sequence=True)
        self.assertEqual(len(record.token_ids), 601)
        self.assertFalse(record.meta["truncated"])
        self.assertEqual(record.n_real_tokens, 600)
        short = fz.featurize_molecule("CCO", sequence=True)
        self.assertFalse(short.meta["truncated"])


class ProteinTests(unittest.TestCase):
    def setUp(self):
        self.cfg = small_config()
        self.featurizer = RecordFeaturizer(self.cfg)

    def test_pair_segments_and_lowercase_d_flags(self):
        record = self.featurizer.featurize_sequence("AaC", "dE")
        self.assertEqual(record.n_real_tokens, 5)
        np.testing.assert_array_equal(record.segment_ids, [0, 0, 0, 0, 0, 1, 1])
        lowercase_positions = np.flatnonzero(
            record.atom_features[:, ATOM_CHIRALITY_INDEX] == D_RESIDUE_CHIRAL_CODE
        )
        np.testing.assert_array_equal(lowercase_positions, [2, 5])
        self.assertEqual(record.pool_weights[4], 0.0)
        np.testing.assert_array_equal(record.meta["output_token_indices"], [1, 2, 3, 5, 6])
        batch = collate([record], self.cfg)
        np.testing.assert_array_equal(
            np.flatnonzero(batch["atom_feature_mask"][0]), lowercase_positions
        )

    def test_protein_ligand_is_mixed_residue_and_atom_graph(self):
        record = self.featurizer.featurize_protein_ligand("AaC", "CO")
        np.testing.assert_array_equal(record.token_kind, [3, 3, 3, 3, 0, 0])
        np.testing.assert_array_equal(record.segment_ids, [0, 0, 0, 0, 1, 1])
        self.assertFalse(record.use_atom_features)
        ligand = self.featurizer.featurize_molecule("CO")
        np.testing.assert_array_equal(record.bond_mat[4:, 4:], ligand.bond_mat[1:, 1:])
        self.assertFalse(record.bond_mat[:4, 4:].any())

        routed = _featurize_record(
            self.featurizer,
            {
                "sample_type": "protein_ligand",
                "inputs": {"protein_sequence": "AC", "ligand_smiles": "CO"},
            },
        )
        np.testing.assert_array_equal(routed.token_kind, [3, 3, 3, 0, 0])

    def test_protein_ligand_can_keep_an_unparsed_ligand_as_tokens(self):
        record = self.featurizer.featurize_protein_ligand(
            "ACd", "not-a-molecule", ligand_sequence=True, strict=False)
        self.assertEqual(record.meta["protein_length"], 3)
        self.assertTrue(record.meta["ligand_sequence_frontend"])
        self.assertTrue(record.meta["degraded"])
        self.assertEqual(record.meta["ligand_smiles"], "not-a-molecule")
        self.assertIsNone(record.coords)
        self.assertIsNone(record.bond_mat)
        self.assertEqual(record.n_real_tokens, record.meta["protein_length"]
                         + record.meta["ligand_length"])
        self.assertTrue((record.token_kind[1:4] == 3).all())
        self.assertTrue((record.token_kind[4:] == 1).all())

    def test_peptide_atomic_requires_and_collates_annotations(self):
        with self.assertRaisesRegex(ValueError, "group_ids"):
            self.featurizer.featurize_peptide_atomic(smiles="CCO", annotation={})
        annotation = {
            "group_ids": [0, 0, 1],
            "sidechain_flags": [False, True, True],
            "fragment_ids": [0, 0, 1],
        }
        record = self.featurizer.featurize_peptide_atomic(
            smiles="CCO", annotation=annotation
        )
        batch = collate([record], self.cfg)
        np.testing.assert_array_equal(batch["group_ids"][0], [-1, 0, 0, 1])
        np.testing.assert_array_equal(batch["sidechain_flags"][0], [False, False, True, True])
        np.testing.assert_array_equal(batch["fragment_ids"][0], [-1, 0, 0, 1])

    def test_peptide_atomic_calls_real_mol_annotator(self):
        smiles = Chem.MolToSmiles(Chem.MolFromSequence("AC"))
        record = self.featurizer.featurize_peptide_atomic(smiles=smiles)
        for key in ("group_ids", "sidechain_flags", "fragment_ids"):
            self.assertEqual(record.meta[key].shape, (len(record.token_ids),))
        self.assertTrue(np.all(record.meta["group_ids"][1:] >= 0))
        self.assertTrue(np.all(record.meta["fragment_ids"][1:] >= 0))


class PBCTests(unittest.TestCase):
    def test_cross_cell_images_translation_multiplicity_and_basis(self):
        neighbors = build_pbc_neighbors(np.eye(3) * 4.0, [[0, 0, 0]], cutoff=6.0)
        self.assertEqual(neighbors.neighbor_counts[0], 18)
        self.assertEqual(neighbors.multiplicity[0, 0], 18)
        shifts = neighbors.neighbor_shifts[0]
        self.assertTrue(any(np.array_equal(shift, [1, 0, 0]) for shift in shifts))
        self.assertTrue(any(np.array_equal(shift, [-1, 0, 0]) for shift in shifts))
        self.assertAlmostEqual(np.linalg.norm(neighbors.pair_disp[0, 0]), 4.0, places=6)
        np.testing.assert_allclose(
            neighbors.pair_disp[0, 0], neighbors.pbc_shift[0, 0] @ (np.eye(3) * 4.0)
        )
        expected_basis = np.zeros(32)
        for shift in neighbors.neighbor_shifts[0, neighbors.neighbor_mask[0]]:
            distance = np.linalg.norm(shift @ (np.eye(3) * 4.0))
            expected_basis += np.exp(-PBC_BASIS_GAMMA * (distance - PBC_BASIS_CENTERS) ** 2)
        np.testing.assert_allclose(neighbors.pair_basis[0, 0], expected_basis, atol=1e-6)

    def test_per_center_truncation_statistics(self):
        record = RecordFeaturizer(small_config()).featurize_crystal(
            np.eye(3) * 4.0,
            [[0, 0, 0]],
            ["He"],
            spacegroup_id=221,
            max_neighbors=4,
        )
        stats = record.meta["pbc_stats"]
        np.testing.assert_array_equal(stats["neighbor_counts"], [18])
        np.testing.assert_array_equal(stats["retained_counts"], [4])
        np.testing.assert_array_equal(stats["truncated_counts"], [14])
        self.assertEqual(record.meta["pbc_multiplicity"][1, 1], 4)
        self.assertEqual(record.pair_basis.shape, (2, 2, 32))
        self.assertEqual(record.spacegroup_id, 221)
        self.assertEqual(record.meta["lattice_parameters"].shape, (6,))
        self.assertEqual(record.bond_mat[1, 1], BOND_PBC)
        batch = collate([record], small_config())
        np.testing.assert_array_equal(batch["pbc_truncated_counts"][0], [0, 14])
        np.testing.assert_array_equal(batch["pair_multiplicity"], batch["pbc_multiplicity"])
        self.assertEqual(batch["pbc_shift"].dtype, np.float32)
        self.assertEqual(batch["pbc_neighbor_mask"][0, 1].sum(), 4)

    def test_composition_features_are_extensive_and_zero_for_other_domains(self):
        cfg = small_config(crystal_composition_features=True, crystal_composition_scale=2.0)
        featurizer = RecordFeaturizer(cfg)
        crystal = featurizer.featurize_crystal(
            np.eye(3) * 8.0, [[0.1, 0.1, 0.1], [0.4, 0.4, 0.4], [0.7, 0.7, 0.7]],
            ["H", "C", "H"],
        )
        sequence = featurizer.featurize_sequence("AC")
        batch = collate([crystal, sequence], cfg)
        self.assertEqual(batch["global_features"].shape, (2, cfg.num_elements - 1))
        self.assertEqual(batch["global_features"][0, 0], 1.0)
        self.assertEqual(batch["global_features"][0, 5], 0.5)
        self.assertEqual(batch["global_features"][0, -1], 1.5)
        np.testing.assert_array_equal(batch["global_features"][1], 0.0)
        legacy = RecordFeaturizer(small_config()).featurize_crystal(
            np.eye(3) * 8.0, [[0.1, 0.1, 0.1]], ["H"]
        )
        self.assertIsNone(legacy.global_features)

    def test_independent_brute_force_reference(self):
        lattice = np.array([[3.0, 0.0, 0.0], [0.4, 3.2, 0.0], [0.0, 0.0, 5.0]])
        frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
        actual = build_pbc_neighbors(lattice, frac, cutoff=4.0, max_neighbors=32)
        expected = brute_neighbors(lattice, frac, 4.0, 32)
        for i in range(2):
            self.assertEqual(actual.neighbor_counts[i], len(expected[i]))
            self.assertEqual(actual.retained_counts[i], len(expected[i]))
            for k, (distance, j, shift, vector) in enumerate(expected[i]):
                self.assertEqual(actual.neighbor_indices[i, k], j)
                np.testing.assert_array_equal(actual.neighbor_shifts[i, k], shift)
                self.assertAlmostEqual(actual.neighbor_distances[i, k], distance, places=6)
                if k == 0 or not np.array_equal(shift, actual.neighbor_shifts[i, k - 1]):
                    self.assertTrue(np.isfinite(vector).all())
            multiplicity = np.zeros(2, dtype=np.int64)
            for _, j, _, _ in expected[i]:
                multiplicity[j] += 1
            np.testing.assert_array_equal(actual.multiplicity[i], multiplicity)


class CollateTests(unittest.TestCase):
    def test_pure_sequence_has_no_pair_matrix_and_no_data_loss(self):
        cfg = small_config(max_seq_len=5)
        featurizer = RecordFeaturizer(cfg)
        long_record = featurizer.featurize_sequence("ACDEFGHIKLMNPQRSTVWY")
        short_record = featurizer.featurize_sequence("AC")
        batch = collate([long_record, short_record], cfg)
        self.assertEqual(batch["token_ids"].shape, (2, 21))
        np.testing.assert_array_equal(batch["token_ids"][0], long_record.token_ids)
        np.testing.assert_array_equal(batch["pool_weights"][0], long_record.pool_weights)
        self.assertEqual(batch["n_real_tokens"].tolist(), [20, 2])
        for key in ("bond_mat", "spd_mat", "pair_disp", "pbc_shift", "pair_basis"):
            self.assertNotIn(key, batch)

    def test_normal_pool_weights_and_copolymer_weights_survive_collate(self):
        cfg = small_config()
        featurizer = RecordFeaturizer(cfg)
        normal = featurizer.featurize_molecule("CCO")
        copolymer = featurizer.featurize_polymer(
            monomers=["CC", "CO"], composition=[0.25, 0.75]
        )
        batch = collate([normal, copolymer], cfg)
        np.testing.assert_array_equal(batch["pool_weights"][0], [0, 1, 1, 1, 0])
        np.testing.assert_allclose(batch["pool_weights"][1], copolymer.pool_weights)
        self.assertEqual(batch["pool_weights"][0, 0], 0)


class DummyEncoder(torch.nn.Module):
    def encode(self, batch):
        self.last_batch = batch
        token_ids = batch["token_ids"]
        x = torch.zeros((*token_ids.shape, 8), dtype=torch.float32)
        x[..., 0] = token_ids.float()
        x[..., 1] = batch["token_kind"].float()
        x[..., 2] = batch["segment_ids"].float()
        weights = batch["pool_weights"].unsqueeze(-1)
        embedding = (x * weights).sum(1) / weights.sum(1).clamp(min=1.0)
        return {"embedding": embedding, "tokens": x}


class LegacyDummyEncoder(DummyEncoder):
    def encode(self, batch):
        _ = batch["bond_mat"], batch["spd_mat"]
        return super().encode(batch)


class InferenceTests(unittest.TestCase):
    def test_long_sequence_windows_preserve_offsets_and_token_ids(self):
        cfg = small_config(max_seq_len=5)
        featurizer = RecordFeaturizer(cfg)
        sequence = "ACDEFGHIKLMNP"
        records = [
            {"id": "long", "sample_type": "protein", "inputs": {"sequence": sequence}},
            {"id": "short", "sample_type": "protein", "inputs": {"sequence": "WY"}},
        ]
        _, embeddings, tokens, offsets = encode_subset(
            DummyEncoder(), featurizer, cfg, records, batch_size=2, token_level=True
        )
        self.assertEqual(tokens.shape, (15, 8))
        np.testing.assert_array_equal(offsets, [0, 13, 15])
        expected = featurizer.tk["residue"].encode(sequence + "WY", add_cls=False)
        np.testing.assert_array_equal(tokens[:, 0], expected)
        self.assertTrue(np.isfinite(embeddings).all())

    def test_protein_pair_routes_actual_fields_and_excludes_sep_from_tokens(self):
        cfg = small_config()
        featurizer = RecordFeaturizer(cfg)
        records = [{
            "id": "pair",
            "sample_type": "protein_pair",
            "inputs": {"sequence_1": "AC", "sequence_2": "DE"},
        }]
        _, _, tokens, offsets = encode_subset(
            DummyEncoder(), featurizer, cfg, records, token_level=True
        )
        np.testing.assert_array_equal(offsets, [0, 4])
        expected = featurizer.tk["residue"].encode("ACDE", add_cls=False)
        np.testing.assert_array_equal(tokens[:, 0], expected)

    def test_legacy_model_gets_only_required_zero_pairs(self):
        cfg = small_config()
        records = [{"id": "p", "sample_type": "protein", "inputs": {"sequence": "AC"}}]
        encoder = LegacyDummyEncoder()
        _, embeddings, _, _ = encode_subset(encoder, RecordFeaturizer(cfg), cfg, records)
        self.assertTrue(np.isfinite(embeddings).all())
        self.assertIn("bond_mat", encoder.last_batch)

    def test_crystal_receives_spacegroup(self):
        featurizer = RecordFeaturizer(small_config())
        record = _featurize_record(
            featurizer,
            {
                "sample_type": "crystal",
                "inputs": {
                    "lattice": (np.eye(3) * 4).tolist(),
                    "frac_coords": [[0, 0, 0]],
                    "elements": ["Si"],
                    "spacegroup": 227,
                },
            },
        )
        self.assertEqual(record.spacegroup_id, 227)
        self.assertEqual(record.meta["spacegroup_id"], 227)


if __name__ == "__main__":
    unittest.main()
