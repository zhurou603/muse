from __future__ import annotations

import collections
import unittest
from pathlib import Path

import yaml

from muse_encoder.s5.curriculum import Curriculum


class CurriculumTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parents[1] / "configs" / "s5_continuous.yaml"
        cls.config = yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_domain_slots_and_accumulation(self):
        course = Curriculum.from_config(self.config)
        self.assertEqual(len(course.slots), 20)
        for step in range(100):
            choice = course.resolve(step)
            for micro in range(4):
                self.assertEqual(course.resolve((step * 4 + micro) // 4), choice)

    def test_dbaasp_exact_micro_ratios(self):
        course = Curriculum.from_config(self.config)
        counts = collections.Counter()
        for step in range(200):
            choice = course.resolve(step)
            if choice.group == "dbaasp":
                counts[choice.source] += 1
        sources = self.config["data"]["dbaasp_sources"]
        self.assertEqual(dict(counts), {
            sources["main"]: 9, sources["hemo"]: 2, sources["nonmic"]: 2,
            sources["short_uniref"]: 3, sources["short_pdb"]: 1, sources["long_pdb"]: 3,
        })

    def test_resume_is_deterministic(self):
        course = Curriculum.from_config(self.config)
        expected = [course.resolve(step) for step in range(1000)]
        resumed = [course.resolve(step) for step in range(417, 1000)]
        self.assertEqual(expected[417:], resumed)


if __name__ == "__main__":
    unittest.main()
