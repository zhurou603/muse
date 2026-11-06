from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DESIGN = PROJECT_ROOT / "model_design"
LAUNCH = MODEL_DESIGN / "launches/eval16-20261027"
for path in (str(PROJECT_ROOT), str(MODEL_DESIGN), str(LAUNCH)):
    if path not in sys.path:
        sys.path.insert(0, path)

from muse_encoder.config import MuseConfig
from muse_encoder.pretrain import dashboard as D


class FastEvalTest(unittest.TestCase):
    def test_repairs_and_per_record_failures_reach_disk(self):
        spec = importlib.util.spec_from_file_location(
            "fast_eval_under_test", LAUNCH / "fast_eval.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            (data_root / "manifest.json").write_text("test-manifest", encoding="utf-8")
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"test-checkpoint")
            repairs = root / "repairs.json"
            repairs.write_text("{}", encoding="utf-8")
            output = root / "output"
            raw = [{"id": "bad-record", "sample_type": "molecule",
                    "inputs": {"smiles": "not-a-molecule"}}]
            failed = [{"id": "bad-record", "error": "synthetic encoding error"}]
            encoded = D.EncodedSplit([], np.empty((0, 8), dtype=np.float32),
                                     None, None, failed)
            cfg = MuseConfig(embed_dim=8)
            subset = SimpleNamespace(dataset="dataset", subset="subset", directory="dataset/subset",
                                     domain="small_molecule")

            def fake_run(_checkpoint, output_dir, **kwargs):
                D.encode_split_cached(
                    None, None, cfg, subset, data_root, "train", Path(output_dir),
                    "checkpoint-sha", None, False, 8, "cuda:0",
                    probe_config=kwargs["probe_config"])
                return {"cached_splits": ["dataset/subset/train"],
                        "summary": {"final_score": None, "scored_tasks": 0, "num_tasks": 90}}

            argv = [
                "fast_eval.py", "--checkpoint", str(checkpoint), "--output-dir", str(output),
                "--devices", "0", "--cpu-workers", "2", "--batch-size", "8",
                "--train-cap", "17", "--single-seed", "--official-graph-repairs", str(repairs),
            ]
            with patch.object(D, "DATA_ROOT", data_root), \
                    patch.object(D, "_read_jsonl", return_value=raw), \
                    patch.object(D, "run_dashboard", side_effect=fake_run), \
                    patch.object(module, "encode_records_parallel",
                                 return_value=(encoded, {"central_hit": False})) as encode, \
                    patch.object(sys, "argv", argv):
                module.main()
            self.assertEqual(encode.call_args.kwargs["graph_repairs"], repairs)
            cache = output / "embeddings/dataset/subset"
            self.assertEqual(json.loads((cache / "train.failed.json").read_text()), failed)
            metadata = json.loads((cache / "train.encoding.json").read_text())
            self.assertEqual(metadata["failed"], failed)
            self.assertEqual(metadata["limit_per_split"], 17)
            self.assertEqual(metadata["parallel"]["devices"], ["cuda:0"])


if __name__ == "__main__":
    unittest.main()
