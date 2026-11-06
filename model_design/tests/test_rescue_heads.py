from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "launches/stage2-20261031/rescue_heads.py"
SPEC = importlib.util.spec_from_file_location("stage2_rescue_heads", MODULE_PATH)
RESCUE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RESCUE
SPEC.loader.exec_module(RESCUE)


class RescueHeadTests(unittest.TestCase):
    def test_mad_scale_and_constant_fallback(self):
        median, mad, scale, kind = RESCUE.robust_location_scale(np.asarray([1, 2, 3, 100]))
        self.assertEqual((median, mad, scale, kind), (2.5, 1.0, 1.0, "mad"))
        median, mad, scale, kind = RESCUE.robust_location_scale(np.asarray([2, 2, 2, 8]))
        self.assertEqual((median, mad, kind), (2.0, 0.0, "std_fallback"))
        self.assertGreater(scale, 0)

    def test_search_heads_fit_parameter_budget(self):
        for task_type in ("regression", "binary_classification"):
            for spec in RESCUE.screen_specs(task_type):
                self.assertLessEqual(RESCUE.parameter_count(1024, spec), 100_000)
        self.assertEqual(len(RESCUE.screen_specs("regression")), 18)
        self.assertEqual(len(RESCUE.screen_specs("binary_classification")), 9)
        linear = RESCUE.HeadSpec(0, "linear", "mae", 3e-4, 1e-4)
        self.assertEqual(RESCUE.parameter_count(1024, linear), 1025)

    def test_record_split_keeps_repeated_observations_together(self):
        ids = ["a", "a", "b", "c", "c", "d", "e", "f", "g", "h"]
        fit, stop = RESCUE._record_split(ids, fraction=0.2)
        for record_id in set(ids):
            selected = [index for index, value in enumerate(ids) if value == record_id]
            self.assertEqual(len(set(fit[selected])), 1)
            self.assertEqual(len(set(stop[selected])), 1)

    def test_loader_uses_every_observation_without_averaging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            cache = run / "embeddings/D/s"
            cache.mkdir(parents=True)
            for split in ("train", "validation"):
                ids = [f"{split}-1"]
                (cache / f"{split}.ids.json").write_text(json.dumps(ids), encoding="utf-8")
                np.save(cache / f"{split}.embeddings.npy", np.asarray([[1, 2]], dtype=np.float32))
                if split == "train":
                    (cache / "train.encoding.json").write_text(
                        json.dumps({"limit_per_split": 1}), encoding="utf-8")
                data = root / "data/D/s"
                data.mkdir(parents=True, exist_ok=True)
                record = {
                    "id": ids[0],
                    "labels": {"y": [
                        {"value": 1.0, "task_type": "regression"},
                        {"value": 3.0, "task_type": "regression"},
                    ]},
                }
                (data / f"{split}.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            task = SimpleNamespace(
                task_id="D/s/y", dataset="D", subset="s", task="y",
                task_type="regression", positive_label=None, train_median=2.0,
            )
            board = SimpleNamespace(
                data_root=root / "data",
                subsets={("D", "s"): SimpleNamespace(directory=Path("D/s"))},
            )
            loaded = RESCUE.load_task_data(board, run, task)
            self.assertEqual(loaded.train_ids, ["train-1", "train-1"])
            self.assertEqual(loaded.y_train.tolist(), [1.0, 3.0])
            self.assertEqual(loaded.x_train.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
