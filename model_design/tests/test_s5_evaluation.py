from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from muse_encoder.s5.evaluate_candidate import (
    gate, resume_frozen_checkpoint, summarize_result, validate_bootstrap, validate_full,
)
from muse_encoder.s5.provenance import sha256_file


CFG = {
    "shadow_properties": 2, "min_j_delta": 0.02,
    "max_encoder_missing_fraction": 0.001,
    "domain_forget_thresholds": {"small_molecule": 0.2, "organic_crystal": 0.1},
}


def result(score=52.2, small=30.0, complete=True):
    return {
        "smoke": False, "fixed_probe_protocol": "muse_encoder.s5_fixed_robust_probe.v1",
        "summary": {
            "final_score": score, "complete": complete, "is_full_a_board": complete,
            "scored_tasks": 90 if complete else 5, "num_tasks": 90,
            "domains": {"small_molecule": {"score": small}, "organic_crystal": {"score": 98.0}},
        },
        "tasks": {
            "a": {"missing_predictions": 0, "n_validation_total": 1000},
            "b": {"missing_predictions": 0, "n_validation_total": 1000},
        },
        "shadow": {"status": "ok", "properties": {
            "x": {"status": "ok", "normalized_score": 0.4},
            "y": {"status": "ok", "metrics": {"score": 0.3}},
        }},
    }


class EvaluationGateTest(unittest.TestCase):
    def test_summary_and_accept(self):
        baseline = summarize_result(result(score=52.0))
        candidate = summarize_result(result(score=52.2))
        self.assertEqual(validate_full(candidate, CFG), [])
        self.assertEqual(gate(candidate, baseline, CFG), [])

    def test_reject_incomplete_and_regression(self):
        baseline = summarize_result(result(score=52.0, small=30.0))
        incomplete = summarize_result(result(score=99.0, complete=False))
        self.assertTrue(validate_full(incomplete, CFG))
        forgetting = summarize_result(result(score=52.2, small=29.0))
        self.assertTrue(any("遗忘" in reason for reason in gate(forgetting, baseline, CFG)))

    def test_reject_low_delta(self):
        baseline = summarize_result(result(score=52.0))
        candidate = summarize_result(result(score=52.01))
        self.assertTrue(any("J_encoder_fixed" in reason for reason in gate(candidate, baseline, CFG)))

    def test_bootstrap_cannot_overwrite_best(self):
        candidate = summarize_result(result(score=99.0))
        self.assertEqual(validate_bootstrap(candidate, {}, CFG), [])
        reasons = validate_bootstrap(candidate, {"best_fixed": candidate}, CFG)
        self.assertTrue(any("禁止" in reason for reason in reasons))

    def test_resume_requires_identical_checkpoint_and_preserves_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "parent.pt"
            parent.write_bytes(b"self-trained-parent")
            candidate = root / "evaluation" / "20261104T000000_full"
            (candidate / "dashboard" / "predictions").mkdir(parents=True)
            (candidate / "encoder.pt").write_bytes(parent.read_bytes())
            sha256 = sha256_file(parent)
            (candidate / "dashboard" / "run_state.json").write_text(
                json.dumps({"checkpoint": {"sha256": sha256}}), encoding="utf-8")
            (candidate / "dashboard" / "predictions" / "manifest.json").write_text(
                json.dumps({"name": "s5-full-original"}), encoding="utf-8")
            frozen, name = resume_frozen_checkpoint(parent, candidate, root / "evaluation")
            self.assertEqual(frozen["sha256"], sha256)
            self.assertEqual(name, "s5-full-original")
            parent.write_bytes(b"different-parent")
            with self.assertRaisesRegex(RuntimeError, "SHA256"):
                resume_frozen_checkpoint(parent, candidate, root / "evaluation")
            with self.assertRaisesRegex(ValueError, "evaluation-root"):
                resume_frozen_checkpoint(parent, candidate, root / "other-evaluation")


if __name__ == "__main__":
    unittest.main()
