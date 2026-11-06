from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from muse_encoder.pretrain.aux import NamespacedRecord
from muse_encoder.pretrain.targets import TargetNormalizer, compute_target_stats


def record(record_id, labels):
    return NamespacedRecord("toy", record_id, {"id": record_id, "labels": labels})


class TargetTests(unittest.TestCase):
    def test_stats_normalization_units_shadow_and_missing(self):
        records = [
            record("a", {
                "energy": [{"value": 1.0, "unit": "eV", "task_type": "regression"}],
                "force": [{"value": [[1.0, 2.0], [3.0, 4.0]], "task_type": "atom_regression"}],
                "stable": [{"value": True, "task_type": "binary_classification"}],
                "conc": [{"value": 2.0, "unit": "µM", "task_type": "regression"},
                         {"value": 5.0, "unit": "µg/ml", "task_type": "regression"}],
            }),
            record("b", {
                "energy": [{"value": 3.0, "unit": "eV", "task_type": "regression"}],
                "force": [{"value": [[5.0, 6.0]], "task_type": "atom_regression"}],
                "conc": [{"value": 4.0, "unit": "µM", "task_type": "regression"}],
            }),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "targets.json"
            with patch("muse_encoder.pretrain.targets.iter_source", return_value=iter(records)):
                payload = compute_target_stats(["toy"], output, {"toy::energy"})
            normalizer = TargetNormalizer.from_path(output)
            self.assertEqual(payload["stats"]["toy::force::"]["mean"], [3.0, 4.0])
            energy_spec = normalizer.specs["toy::energy::eV"]
            self.assertTrue(energy_spec.shadow)
            first = normalizer.aux_records(records[0])
            self.assertEqual({r["name"] for r in first},
                             {"toy::force::", "toy::stable::", "toy::conc::µM", "toy::conc::µg/ml"})
            second = normalizer.aux_records(records[1])
            by_name = {r["name"]: r for r in second}
            self.assertIsNone(by_name.get("toy::energy::eV"))
            self.assertFalse(by_name["toy::stable::"]["mask"].any())
            stable = {r["name"]: r for r in first}["toy::stable::"]
            self.assertEqual(stable["value"].item(), 1.0)
            self.assertAlmostEqual(normalizer.normalize("toy::conc::µM", 3.0)[0].item(), 0.0)
            self.assertEqual(len(normalizer.specs), 5)

    def test_nonfinite_values_are_masked(self):
        records = [record("a", {"x": [{"value": [1.0, 2.0], "task_type": "vector_regression"}]}),
                   record("b", {"x": [{"value": [1.0, 4.0], "task_type": "vector_regression"}]})]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "targets.json"
            with patch("muse_encoder.pretrain.targets.iter_source", return_value=iter(records)):
                compute_target_stats(["toy"], output)
            normalizer = TargetNormalizer.from_path(output)
            value, mask = normalizer.normalize("toy::x::", [float("nan"), 3.0])
            self.assertFalse(mask[0, 0])
            self.assertTrue(mask[0, 1])
            self.assertEqual(value[0, 0], 0.0)
            self.assertTrue(np.isfinite(value).all())


if __name__ == "__main__":
    unittest.main()
