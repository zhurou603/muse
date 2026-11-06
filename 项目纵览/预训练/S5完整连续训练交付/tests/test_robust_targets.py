from __future__ import annotations

import unittest
import sys
import types
from pathlib import Path

import numpy as np

# The reviewed sample snapshot intentionally omits the official muse/tasks.py;
# the real repository provides it.  Regression-only unit tests need no scorer.
try:
    from muse import tasks as _tasks
except ImportError:
    _tasks = types.ModuleType("muse.tasks")
    _tasks.ROOT = Path("/nonexistent")
    _tasks.DATA_DIR = Path("/nonexistent")
    _tasks.EXTERNAL_DIR = Path("/nonexistent")
    sys.modules["muse.tasks"] = _tasks
from muse_encoder.pretrain import dashboard as _dashboard
if _dashboard._OFFICIAL_SCORER is None and not _dashboard.OFFICIAL_SCORER_PATH.exists():
    _dashboard._OFFICIAL_SCORER = object()

from muse_encoder.pretrain.a_train import ATrainTaskSpec
from s5.robust_targets import S5TargetNormalizer, regression_value, robust_location_scale


def spec():
    return ATrainTaskSpec(
        source="ds/ss", name="value", dataset="ds", subset="ss", domain="small_molecule",
        kind="graph", task_type="regression", dim=1, train_observations=3,
    )


class RobustTargetTest(unittest.TestCase):
    def test_median_record_and_mad(self):
        task = spec()
        record = {"id": "x", "labels": {"value": [
            {"task_type": "regression", "value": 1, "split": "train"},
            {"task_type": "regression", "value": 2, "split": "train"},
            {"task_type": "regression", "value": 100, "split": "train"},
        ]}}
        self.assertEqual(regression_value(record, task), 2.0)
        median, scale = robust_location_scale([1, 2, 100])
        self.assertEqual((median, scale), (2.0, 1.0))

    def test_normalizer_replaces_legacy_mean(self):
        task = spec()
        payload = {
            "schema_version": "muse_encoder.a_train_targets.v1", "split": "train",
            "specs": {task.key: {**task.__dict__, "class_labels": (), "positive_label": None}},
            "stats": {task.key: {"mean": [2.0], "std": [1.0], "mask": [True]}},
            "sources": {},
        }
        normalizer = S5TargetNormalizer(payload)
        record = {"id": "x", "labels": {"value": [
            {"task_type": "regression", "value": 1},
            {"task_type": "regression", "value": 2},
            {"task_type": "regression", "value": 100},
        ]}}
        item = normalizer.aux_records("ds/ss", record, None)[0]
        self.assertEqual(float(item["value"][0]), 0.0)
        self.assertTrue(bool(item["mask"][0]))

    def test_non_train_rejected(self):
        task = spec()
        record = {"id": "x", "labels": {"value": [
            {"task_type": "regression", "value": 1, "split": "validation"},
        ]}}
        with self.assertRaises(ValueError):
            regression_value(record, task)


if __name__ == "__main__":
    unittest.main()
