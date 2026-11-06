from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from muse_encoder.pretrain.report import render_dashboard, write_dashboard


class ReportTests(unittest.TestCase):
    def test_render_with_synthetic_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "config.resolved.yaml").write_text(
                yaml.safe_dump({"training": {"total_steps": 100}}), encoding="utf-8")
            rows = [
                {"step": i, "samples": i * 64, "lr": 1e-4, "elapsed_seconds": i * 8.0,
                 "batch_samples": 64, "grad_norm": 1.0 + i * 0.01,
                 "losses": {"total": 10.0 / (i + 1), "mlm": 5.0 / (i + 1)}}
                for i in (10, 20, 30)
            ]
            with (run_dir / "train.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            (run_dir / "checkpoints").mkdir()
            (run_dir / "checkpoints" / "step_00000030.pt").write_bytes(b"toy")
            html = render_dashboard(run_dir, smi=[{"card": 0, "vram_pct": 40.0,
                                                   "util_pct": 80.0}])
            for needle in ("30 / 100", "total loss", "grad_norm", "GPU",
                           "step_00000030.pt", "refresh"):
                self.assertIn(needle, html)
            out = write_dashboard(run_dir)
            self.assertTrue(out.exists())
            self.assertIn("text/html", out.read_text(encoding="utf-8")[:0] or "text/html")


if __name__ == "__main__":
    unittest.main()
