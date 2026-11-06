from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from muse_encoder.s5.prepare_data import write_a_train_failure_rows


class ATrainFailureRowsTest(unittest.TestCase):
    def test_maps_failure_ids_to_official_train_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.jsonl"
            train.write_text("".join(
                json.dumps({"id": value}) + "\n" for value in ("a", "b", "c", "d")), encoding="utf-8")
            failures = root / "failures.jsonl"
            failures.write_text('{"id":"d"}\n{"id":"a"}\n', encoding="utf-8")
            output = root / "rows.txt"
            rows = write_a_train_failure_rows(SimpleNamespace(train_path=train), failures, output)
            self.assertEqual(rows, [0, 3])
            self.assertEqual(output.read_text(), "0\n3\n")


if __name__ == "__main__":
    unittest.main()
