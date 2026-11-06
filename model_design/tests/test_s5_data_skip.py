from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from muse_encoder.pretrain.sampling import SamplePointer, SourceSampler, SourceSpec
from muse_encoder.s5.data_skip import DataSkipLogger, fetch_valid_batch, load_skipped_rows


class _Dataset:
    name = "toy"
    kind = "aux_molecule"

    def __init__(self, error=ValueError("bad record")):
        self.error = error
        self.cfg = SimpleNamespace(aux_specs={})

    def get(self, pointer):
        if pointer.row == 1:
            raise self.error
        return SimpleNamespace(meta={}, coord_mask=np.asarray([True])), {"row": pointer.row}

    def get_2d_view(self, pointer):
        return SimpleNamespace(meta={}, coord_mask=None), {}


class DataSkipTest(unittest.TestCase):
    def test_replaces_bad_row_persists_audit_and_restores_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _Dataset()
            sampler = SourceSampler(SourceSpec("toy", "small_molecule", 8))
            logger = DataSkipLogger(tmp, 3)
            pointers = [SamplePointer("toy", "small_molecule", 1),
                        SamplePointer("toy", "small_molecule", 2)]
            records, metadata, accepted, align_pairs, skipped = fetch_valid_batch(
                dataset, sampler, np.random.default_rng(7), pointers, True, logger, 11)
            self.assertEqual(len(records), 4)
            self.assertEqual(len(metadata), 4)
            self.assertNotIn(1, [pointer.row for pointer in accepted])
            self.assertEqual(align_pairs, [[0, 2], [1, 3]])
            self.assertEqual(len(skipped), 1)
            self.assertIn(1, sampler.exclude)
            lines = (Path(tmp) / "data_skips.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 1)
            event = json.loads(lines[0])
            self.assertEqual(event["micro_step"], 11)
            self.assertEqual(event["rank"], 3)
            self.assertEqual(event["stage"], "dataset.get")
            self.assertEqual(load_skipped_rows(tmp), {"toy": {1}})

    def test_does_not_disguise_io_error_as_bad_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _Dataset(OSError("temporary read failure"))
            sampler = SourceSampler(SourceSpec("toy", "small_molecule", 8))
            pointer = SamplePointer("toy", "small_molecule", 1)
            with self.assertRaisesRegex(OSError, "temporary read failure"):
                fetch_valid_batch(dataset, sampler, np.random.default_rng(7), [pointer],
                                  False, DataSkipLogger(tmp, 0), 0)
            self.assertFalse((Path(tmp) / "data_skips.jsonl").exists())

    def test_runtime_data_error_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _Dataset(RuntimeError("原子映射候选超过10000，停止，不能确认完整搜索"))
            sampler = SourceSampler(SourceSpec("toy", "small_molecule", 8))
            pointer = SamplePointer("toy", "small_molecule", 1)
            records, _metadata, accepted, _align_pairs, skipped = fetch_valid_batch(
                dataset, sampler, np.random.default_rng(7), [pointer],
                False, DataSkipLogger(tmp, 0), 0)
            self.assertEqual(len(records), 1)
            self.assertNotEqual(accepted[0].row, 1)
            self.assertEqual(len(skipped), 1)
            self.assertEqual(load_skipped_rows(tmp), {"toy": {1}})

    def test_does_not_disguise_nccl_error_as_bad_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _Dataset(RuntimeError("NCCL collective timeout"))
            sampler = SourceSampler(SourceSpec("toy", "small_molecule", 8))
            pointer = SamplePointer("toy", "small_molecule", 1)
            with self.assertRaisesRegex(RuntimeError, "NCCL collective timeout"):
                fetch_valid_batch(dataset, sampler, np.random.default_rng(7), [pointer],
                                  False, DataSkipLogger(tmp, 0), 0)
            self.assertFalse((Path(tmp) / "data_skips.jsonl").exists())

    def test_unmappable_bond_is_rejected_during_metadata_validation(self):
        class MappingDataset(_Dataset):
            def __init__(self):
                super().__init__()
                self.cfg = SimpleNamespace(aux_specs={"bond": {"kind": "bond", "dim": 1}})

            def get(self, pointer):
                if pointer.row != 1:
                    return super().get(pointer)
                record = SimpleNamespace(
                    token_ids=np.zeros(3), coord_mask=None,
                    meta={"bond_pairs": np.asarray([[0, 1]]),
                          "sdf_mapping": np.asarray([0, 1]),
                          "sdf_bond_pairs": np.asarray([[0, 2], [1, 2]])})
                item = {"aux_records": [{"name": "bond", "value": np.zeros((2, 1)),
                                        "mask": np.ones((2, 1), dtype=bool)}]}
                return record, item

        with tempfile.TemporaryDirectory() as tmp:
            dataset = MappingDataset()
            sampler = SourceSampler(SourceSpec("toy", "small_molecule", 8))
            pointer = SamplePointer("toy", "small_molecule", 1)
            _records, _metadata, accepted, _align_pairs, skipped = fetch_valid_batch(
                dataset, sampler, np.random.default_rng(7), [pointer],
                False, DataSkipLogger(tmp, 0), 0)
            self.assertNotEqual(accepted[0].row, 1)
            self.assertEqual(skipped[0]["stage"], "dataset.metadata")
            self.assertIn("找不到目标键对应的源结构键", skipped[0]["error"])
            self.assertEqual(load_skipped_rows(tmp), {"toy": {1}})

    def test_explicit_cfg_supports_dataset_without_cfg_attribute(self):
        dataset = _Dataset()
        dataset.cfg = None
        sampler = SourceSampler(SourceSpec("toy", "small_molecule", 8))
        pointer = SamplePointer("toy", "small_molecule", 2)
        records, _metadata, accepted, _align_pairs, skipped = fetch_valid_batch(
            dataset, sampler, np.random.default_rng(7), [pointer], False,
            DataSkipLogger(None, 0), 0, cfg=SimpleNamespace(aux_specs={}))
        self.assertEqual(len(records), 1)
        self.assertEqual(accepted[0].row, 2)
        self.assertEqual(skipped, [])


if __name__ == "__main__":
    unittest.main()
