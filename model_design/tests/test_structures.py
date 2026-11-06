from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muse_encoder.structures import align_coordinates, _topology_mapping

from rdkit import Chem


REAL_OPENBABEL_TRIAZOLE = """muse__qm9__molecule__c62502601daab4a23b0dab243c7a5762
 OpenBabel06302521273D

  8  8  0  0  0  0  0  0  0  0999 V2000
    0.0069    1.3544    0.0098 N   0  0  0  0  0  0  0  0  0  0  0  0
    1.2877    1.7751   -0.0006 C   0  0  0  0  0  3  0  0  0  0  0  0
    2.1022    0.7379   -0.0144 N   0  0  0  0  0  0  0  0  0  0  0  0
    1.2387   -0.3172   -0.0116 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0453    0.0015    0.0029 N   0  0  0  0  0  2  0  0  0  0  0  0
   -0.8460    1.8891    0.0212 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.5694    2.8173    0.0023 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.5661   -1.3458   -0.0204 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  6  1  0  0  0  0
  2  7  1  0  0  0  0
  2  1  1  0  0  0  0
  3  4  2  0  0  0  0
  3  2  1  0  0  0  0
  4  5  1  0  0  0  0
  5  1  1  0  0  0  0
  8  4  1  0  0  0  0
M  END
"""


def _with_coordinates(mol):
    mol = Chem.Mol(mol)
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(i, (0.31 * i, -0.17 * i, 0.07 * i))
    mol.AddConformer(conformer)
    return mol


class AromaticRepresentationAlignmentTests(unittest.TestCase):
    def test_real_openbabel_sample_recovers_original_coordinates(self):
        source = Chem.MolFromMolBlock(
            REAL_OPENBABEL_TRIAZOLE, removeHs=False, sanitize=True
        )
        self.assertIsNotNone(source)
        original_positions = source.GetConformer().GetPositions().copy()
        original_smiles = Chem.MolToSmiles(source)
        original_radicals = [
            atom.GetNumRadicalElectrons() for atom in source.GetAtoms()
        ]

        graph, coords, mapping = align_coordinates("c1nc[nH]n1", source)

        self.assertEqual(Chem.MolToSmiles(graph), "c1nc[nH]n1")
        self.assertEqual(mapping, (3, 2, 1, 0, 4))
        expected = np.array([
            [1.2387, -0.3172, -0.0116],
            [2.1022, 0.7379, -0.0144],
            [1.2877, 1.7751, -0.0006],
            [0.0069, 1.3544, 0.0098],
            [-0.0453, 0.0015, 0.0029],
        ])
        np.testing.assert_array_equal(coords, expected)
        np.testing.assert_array_equal(
            source.GetConformer().GetPositions(), original_positions
        )
        self.assertEqual(Chem.MolToSmiles(source), original_smiles)
        self.assertEqual(
            [atom.GetNumRadicalElectrons() for atom in source.GetAtoms()],
            original_radicals,
        )

    def test_radicals_without_valence_marker_are_not_cleared(self):
        source = Chem.RemoveAllHs(Chem.MolFromMolBlock(
            REAL_OPENBABEL_TRIAZOLE, removeHs=False, sanitize=True
        ))
        changed = []
        for atom in source.GetAtoms():
            if atom.GetNumRadicalElectrons():
                atom.SetNoImplicit(False)
                changed.append(atom.GetIdx())
        self.assertEqual(changed, [1, 4])

        with self.assertRaisesRegex(ValueError, "非立体化学结构不一致"):
            align_coordinates("c1nc[nH]n1", source)

    def test_valid_kekule_graph_preserves_double_bond_stereo(self):
        source = Chem.MolFromSmiles(r"F/C=C/C1=CC=CC=C1")
        self.assertIsNotNone(source)
        Chem.Kekulize(source, clearAromaticFlags=True)
        source = _with_coordinates(source)
        original = source.GetConformer().GetPositions().copy()

        _, coords, mapping = align_coordinates(r"F/C=C/c1ccccc1", source)

        np.testing.assert_array_equal(
            coords, original[np.asarray(mapping, dtype=np.int64)]
        )

    def test_opposite_double_bond_stereo_is_rejected(self):
        source = Chem.MolFromSmiles(r"F/C=C\C1=CC=CC=C1")
        self.assertIsNotNone(source)
        Chem.Kekulize(source, clearAromaticFlags=True)
        source = _with_coordinates(source)

        with self.assertRaises(ValueError):
            align_coordinates(r"F/C=C/c1ccccc1", source)

    def test_exocyclic_bond_and_radical_difference_is_rejected(self):
        source = _with_coordinates(Chem.MolFromSmiles("[NH][C]1N=COC=N1"))

        with self.assertRaisesRegex(ValueError, "非立体化学结构不一致"):
            align_coordinates("N=c1ncocn1", source)

    def test_neutral_sdf_does_not_match_zwitterionic_smiles(self):
        source = _with_coordinates(Chem.MolFromSmiles("CNCC(=O)O"))

        with self.assertRaisesRegex(ValueError, "非立体化学结构不一致"):
            align_coordinates("C[NH2+]CC(=O)[O-]", source)

    def test_disconnected_and_missing_bonds_are_not_repaired(self):
        source = _with_coordinates(Chem.MolFromSmiles("[CH].[CH]1C=CC=N1"))

        with self.assertRaisesRegex(ValueError, "原子数或键数不一致"):
            align_coordinates("Cn1cccc1", source)

    def test_unsanitized_topology_is_explicit_and_annotations_are_preserved(self):
        from muse_encoder.structures import (
            MissingChemicalInput, graph_from_smiles, mol_degraded, mol_mode,
        )

        smiles = "O=C1C(=CC2=CC=C(c3ccc(Cl)cc3)O2)SC(=S)N1CCc(o)o"
        with self.assertRaises(ValueError):
            graph_from_smiles(smiles)
        mol = graph_from_smiles(smiles, allow_unsanitized=True)
        self.assertEqual(mol_mode(mol), "unsanitized_topology")
        self.assertTrue(mol_degraded(mol))
        self.assertGreater(mol.GetNumAtoms(), 10)
        annotated = graph_from_smiles(
            "[NH3+]Cc1ccc(cc1)C(=O)N[C@H]1Cc2cccc(c2O[B]1(O)O)C(=O)O\t5FQB_LIGAND",
            allow_unsanitized=True,
        )
        self.assertEqual(annotated.GetProp("_muse_annotation_0"), "5FQB_LIGAND")
        with self.assertRaises(MissingChemicalInput):
            graph_from_smiles("nan", allow_unsanitized=True)

    def test_coordinate_mismatch_requires_an_explicit_1d_decision(self):
        from muse_encoder.structures import alignment_mode

        source = _with_coordinates(Chem.MolFromSmiles("CCC"))
        with self.assertRaises(ValueError):
            align_coordinates("CCO", source)
        target, coords, mapping = align_coordinates("CCO", source, allow_1d_fallback=True)
        self.assertIsNone(coords)
        self.assertIsNone(mapping)
        self.assertEqual(alignment_mode(target), "smiles_1d_unmatched")


class TopologySearchBudgetTests(unittest.TestCase):
    def test_partial_search_states_have_a_hard_budget(self):
        source = Chem.MolFromSmiles("CC")
        target = Chem.MolFromSmiles("CC")
        with self.assertRaisesRegex(RuntimeError, "搜索节点超过2"):
            _topology_mapping(Chem, source, target, max_candidates=2)

    def test_valid_mapping_is_unchanged_within_budget(self):
        source = Chem.MolFromSmiles("CC")
        target = Chem.MolFromSmiles("CC")
        mapping = _topology_mapping(Chem, source, target)
        self.assertIsNotNone(mapping)
        self.assertEqual(len(mapping), 2)


if __name__ == "__main__":
    unittest.main()
