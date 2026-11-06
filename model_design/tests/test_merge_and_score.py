from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[1] / "launches/eval16-20261027/merge_and_score.py"
SPEC = importlib.util.spec_from_file_location("merge_and_score", MODULE_PATH)
MERGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MERGE)


class MergeValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.tasks = [
            SimpleNamespace(task_id="Toy/a/x", prediction_file="Toy/a/x.csv.gz"),
            SimpleNamespace(task_id="Toy/b/y", prediction_file="Toy/b/y.csv.gz"),
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def shard(self, name, task, sha="checkpoint-a", seeds=(0,)):
        root = self.root / name
        (root / "predictions").mkdir(parents=True)
        (root / "task_cache").mkdir()
        (root / "run_state.json").write_text(json.dumps({
            "checkpoint": {"sha256": sha},
            "model_config": {"d_model": 32},
            "probe_config": {"seeds": list(seeds)},
        }), encoding="utf-8")
        (root / "predictions/manifest.json").write_text(json.dumps({
            "board": "A", "split": "test", "heads": {},
        }), encoding="utf-8")
        path = root / "predictions" / task.prediction_file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"prediction")
        return root

    def test_collects_each_official_task_exactly_once(self):
        first = self.shard("01", self.tasks[0])
        second = self.shard("02", self.tasks[1])
        predictions, _, provenance = MERGE.collect_validated_predictions(
            [first, second], self.tasks, "test")
        self.assertEqual(set(predictions), {task.prediction_file for task in self.tasks})
        self.assertEqual(len(provenance), 2)

    def test_rejects_duplicate_missing_checkpoint_and_probe_mismatch(self):
        first = self.shard("01", self.tasks[0])
        duplicate = self.shard("02", self.tasks[0])
        with self.assertRaisesRegex(ValueError, "重复任务预测"):
            MERGE.collect_validated_predictions([first, duplicate], self.tasks, "test")
        with self.assertRaisesRegex(ValueError, "缺少 1 个任务"):
            MERGE.collect_validated_predictions([first], self.tasks, "test")
        second = self.shard("03", self.tasks[1], sha="checkpoint-b")
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            MERGE.collect_validated_predictions([first, second], self.tasks, "test")
        second = self.shard("04", self.tasks[1], seeds=(0, 1))
        with self.assertRaisesRegex(ValueError, "单种子"):
            MERGE.collect_validated_predictions([first, second], self.tasks, "test")


if __name__ == "__main__":
    unittest.main()
