from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "launches/stage3-polymer-20261101/run_polymer.py"
SPEC = importlib.util.spec_from_file_location("stage3_run_polymer", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Stage3PolymerTests(unittest.TestCase):
    def test_task_weights_and_independence(self):
        self.assertAlmostEqual(sum(MODULE.TASKS.values()) + 0.10, 1.0)
        self.assertEqual(len(MODULE.TASKS), 13)
        self.assertNotIn("TransPolymer/EPS/eps", MODULE.TASKS)
        self.assertNotIn("TransPolymer/Nc/nc", MODULE.TASKS)
        self.assertNotIn("TransPolymer/OPV/pce_ave", MODULE.TASKS)
        self.assertNotEqual(
            "a_train::Polymer-Genome-2016/DFT-HSE06/electronic_band_gap",
            "a_train::Polymer-Genome-2016/DFT-rPW86/electronic_band_gap",
        )

    def test_sampler_shares_task_but_not_rows_across_ranks(self):
        sources = [
            MODULE.SourceSampler(MODULE.SourceSpec("a", "polymer", 1000)),
            MODULE.SourceSampler(MODULE.SourceSpec("b", "polymer", 1000)),
        ]
        sampler = MODULE.RankSynchronizedWeightedSampler(
            sources, {"a": 0.5, "b": 0.5}, seed=43, stage=3)
        first = sampler.batch(7, 0, 4, 16)
        second = sampler.batch(7, 1, 4, 16)
        self.assertEqual({pointer.source for pointer in first},
                         {pointer.source for pointer in second})
        self.assertNotEqual([pointer.row for pointer in first],
                            [pointer.row for pointer in second])

    def test_linear_warmup_and_decay(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=1e-5)
        scheduler = MODULE.linear_scheduler(optimizer, 300, 15)
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 1e-5 / 15)
        for _ in range(15):
            scheduler.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 1e-5)
        for _ in range(285):
            scheduler.step()
        self.assertEqual(scheduler.get_last_lr()[0], 0.0)


if __name__ == "__main__":
    unittest.main()
