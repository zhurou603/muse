from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "launches/stage2-20261031/dbaasp.py"
SPEC = importlib.util.spec_from_file_location("stage2_dbaasp", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DBAASPAuxiliaryTests(unittest.TestCase):
    def test_descriptors_are_finite_and_case_insensitive(self):
        upper = MODULE.peptide_descriptors("KVFKWP")
        lower = MODULE.peptide_descriptors("kvfkwp")
        self.assertEqual(upper.shape, (len(MODULE.DESCRIPTOR_NAMES),))
        self.assertTrue(np.isfinite(upper).all())
        np.testing.assert_array_equal(upper, lower)
        self.assertAlmostEqual(float(upper[0]), 6.0)
        self.assertGreater(float(upper[2]), 0.0)

    def test_robust_stats_use_mad_with_safe_fallback(self):
        median, scale = MODULE.robust_stats(np.asarray([1, 2, 3, 100], dtype=np.float32))
        self.assertEqual(float(median), 2.5)
        self.assertEqual(float(scale), 1.0)
        median, scale = MODULE.robust_stats(np.asarray([2, 2, 2, 2], dtype=np.float32))
        self.assertEqual((float(median), float(scale)), (2.0, 1.0))


if __name__ == "__main__":
    unittest.main()
