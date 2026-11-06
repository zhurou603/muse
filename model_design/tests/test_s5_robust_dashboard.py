from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

# The reviewed sample omits the official scorer package; regression compliance
# does not call it.  The real repository supplies the official implementation.
try:
    from muse import tasks as _tasks
except ImportError:
    _tasks = types.ModuleType("muse.tasks")
    _tasks.ROOT = Path("/nonexistent")
    _tasks.DATA_DIR = Path("/nonexistent")
    _tasks.EXTERNAL_DIR = Path("/nonexistent")
    sys.modules["muse.tasks"] = _tasks
from muse_encoder.pretrain import dashboard as D
if D._OFFICIAL_SCORER is None and not D.OFFICIAL_SCORER_PATH.exists():
    D._OFFICIAL_SCORER = object()

from muse_encoder.s5.robust_dashboard import fit_probe_robust


class RobustDashboardComplianceTest(unittest.TestCase):
    def test_single_seed_and_parameter_budget(self):
        x_train = np.arange(80, dtype=np.float32).reshape(20, 4) / 80
        y_train = np.linspace(-2, 3, 20, dtype=np.float32)
        x_validation = x_train[:3]
        with patch.object(D, "_train_probe_seed", wraps=D._train_probe_seed) as train_seed:
            prediction, meta = fit_probe_robust(
                x_train, y_train, x_validation, "regression", device="cpu")
        self.assertEqual(train_seed.call_count, 1)
        self.assertEqual(meta["seeds"], [0])
        self.assertFalse(meta["seed_ensemble"])
        self.assertLessEqual(meta["params_per_head"], 100_000)
        self.assertEqual(meta["protocol"], "muse_encoder.s5_fixed_robust_probe.v1")
        self.assertEqual(prediction.shape, (3,))
        self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
