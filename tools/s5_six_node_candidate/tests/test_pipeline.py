import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import wait_for_smoke


class PipelineGateTests(unittest.TestCase):
    def write_session(self, root, mode="smoke-start"):
        path = root / "frozen.json"
        path.write_text(json.dumps({"mode": mode, "candidate": str(root),
                                   "run_dir": str(root / "runs/smoke")}))
        return path

    def test_completed_state_without_mode_uses_immutable_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "control").mkdir()
            session = self.write_session(root)
            state = {"state": "completed", "session": str(session)}
            (root / "control/cluster_state.json").write_text(json.dumps(state))
            self.assertEqual(wait_for_smoke(root, 12345), session)

    def test_session_mode_overrides_claimed_state_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "control").mkdir()
            session = self.write_session(root, mode="train-start")
            (root / "control/cluster_state.json").write_text(json.dumps({
                "state": "completed", "mode": "smoke-start", "session": str(session)}))
            with self.assertRaisesRegex(RuntimeError, "did not complete"):
                wait_for_smoke(root, 12345)

    def test_failed_smoke_does_not_pass_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "control").mkdir()
            (root / "control/cluster_state.json").write_text(json.dumps({"state": "failed"}))
            with self.assertRaisesRegex(RuntimeError, "did not complete"):
                wait_for_smoke(root, 12345)


if __name__ == "__main__":
    unittest.main()
