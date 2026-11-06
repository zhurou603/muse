from __future__ import annotations

import unittest

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from muse_encoder.pretrain.molecule3d import _official_graph_variant


def with_coordinates(mol):
    mol = Chem.AddHs(mol)
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(index, Point3D(index * 0.25, index * 0.5, index * 0.75))
    mol.RemoveAllConformers()
    mol.AddConformer(conformer)
    return mol


class OfficialGraphVariantTest(unittest.TestCase):
    def test_charge_separated_sulfoxide_uses_official_graph(self):
        mol = with_coordinates(Chem.MolFromSmiles("C[S@@+](C)[O-]"))
        result = _official_graph_variant("CS(C)=O", mol)
        self.assertIsNotNone(result)
        graph, coords, mapping = result
        self.assertEqual(graph.GetNumAtoms(), 4)
        self.assertEqual(mapping, tuple(
            atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1))
        self.assertEqual(coords.shape, (4, 3))
        heavy_positions = np.asarray(mol.GetConformer().GetPositions())[list(mapping)]
        np.testing.assert_array_equal(coords, heavy_positions)

    def test_different_formula_is_rejected(self):
        mol = with_coordinates(Chem.MolFromSmiles("CCC"))
        self.assertIsNone(_official_graph_variant("CCO", mol))


if __name__ == "__main__":
    unittest.main()
