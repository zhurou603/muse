import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from muse_encoder.pretrain.omc25_source import OMC25Source
from muse_encoder.pretrain.sampling import SamplePointer


class OMC25SourceTests(unittest.TestCase):
    def _paths(self, root, missing=False):
        root = Path(root)
        sample = root / "sample.jsonl"
        sample.write_text(
            '{"id":"a","starting_heap":"H1"}\n'
            '{"id":"b","starting_heap":"H1"}\n'
            '{"id":"c","starting_heap":"H2"}\n', encoding="utf-8")
        recovery = root / "recovery.jsonl"
        rows = [
            {"starting_heap": "H1", "spacegroup": 4, "missing": []},
            {"starting_heap": "H2", "spacegroup": 0 if missing else 225,
             "missing": ["parent"] if missing else []},
        ]
        recovery.write_text("".join(json.dumps(row) + "\n" for row in rows),
                            encoding="utf-8")
        exclusions = root / "exclude.jsonl"
        exclusions.write_text("", encoding="utf-8")
        return {"sample_jsonl": sample, "parent_recovery": recovery,
                "suggested_exclude_heaps": exclusions,
                "cache_dir": root / "cache", "stats": root / "stats.json"}

    @patch("muse_encoder.pretrain.omc25_source.FrameTargetStats.from_json")
    @patch("muse_encoder.pretrain.omc25_source.OMC25CrystalDataset")
    def test_sample_spacegroups_and_dataset_contract(self, dataset_type, stats):
        dataset_type.return_value.__len__.return_value = 3
        dataset_type.return_value.get.return_value = ("record", {"energy": 1})
        with tempfile.TemporaryDirectory() as root:
            source = OMC25Source(None, self._paths(root))
            self.assertEqual(len(source), 3)
            self.assertEqual(source.spec.source.domain, "crystal")
            pointer = SamplePointer("omc25", "crystal", 1)
            self.assertEqual(source.get(pointer), ("record", {"energy": 1}))
            kwargs = dataset_type.call_args.kwargs
            self.assertEqual(kwargs["spacegroups"], {"H1": 4, "H2": 225})
            self.assertTrue(kwargs["strict_targets"])
            self.assertTrue(kwargs["verify_digests"])
            source.close()

    def test_missing_spacegroup_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "空间群"):
                OMC25Source(None, self._paths(root, missing=True))


if __name__ == "__main__":
    unittest.main()
