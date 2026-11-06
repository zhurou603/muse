import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import wait_for_smoke


class PipelineGateTests(unittest.TestCase):
    def test_only_matching_completed_smoke_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "control").mkdir()
            state = {"state": "completed", "mode": "smoke-start", "session": "frozen.json"}
            (root / "control/cluster_state.json").write_text(json.dumps(state))
            self.assertEqual(wait_for_smoke(root, 12345), Path("frozen.json"))

    def test_failed_smoke_does_not_pass_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "control").mkdir()
            (root / "control/cluster_state.json").write_text(json.dumps({"state": "failed"}))
            with self.assertRaisesRegex(RuntimeError, "did not complete"):
                wait_for_smoke(root, 12345)


if __name__ == "__main__":
    unittest.main()
