from __future__ import annotations

import unittest

import numpy as np

from muse_encoder.pretrain.sampling import (
    HierarchicalSampler,
    ScaffoldIndex,
    SourceSampler,
    SourceSpec,
    temperature_weights,
)


class SamplingTests(unittest.TestCase):
    def test_temperature_balances_long_tail(self):
        weights = temperature_weights([1, 100], 0.005)
        self.assertLess(weights[1] / weights[0], 1.03)
        self.assertAlmostEqual(weights.sum(), 1.0)

    def test_scaffold_sampling_is_deterministic_and_balanced(self):
        groups = np.array([0] * 90 + [1] * 10)
        source = SourceSampler(SourceSpec("x", "small_molecule", 100), ScaffoldIndex(groups))
        rng = np.random.default_rng(42)
        rows = [source.sample(rng).row for _ in range(4000)]
        self.assertGreater(np.mean(np.array(rows) >= 90), 0.4)
        other = SourceSampler(SourceSpec("x", "small_molecule", 100), ScaffoldIndex(groups))
        self.assertEqual(source.sample(np.random.default_rng(7)).row,
                         other.sample(np.random.default_rng(7)).row)

    def test_hierarchical_batch_resume_and_views(self):
        sources = [
            SourceSampler(SourceSpec("pi1m", "polymer", 100, views=5)),
            SourceSampler(SourceSpec("uniref50", "bio", 200)),
        ]
        sampler = HierarchicalSampler(sources, seed=42, stage=3,
                                      domain_weights={"polymer": 0.4, "bio": 0.6})
        first = sampler.batch(17, 1, 5, 32)
        second = sampler.batch(17, 1, 5, 32)
        self.assertEqual(first, second)
        self.assertNotEqual(first, sampler.batch(17, 2, 5, 32))
        for pointer in first:
            if pointer.source == "pi1m":
                self.assertLess(pointer.row, 100)
                self.assertLess(pointer.view, 5)
            else:
                self.assertLess(pointer.row, 200)
                self.assertEqual(pointer.view, 0)

    def test_domain_weights_must_match_stage(self):
        sources = [SourceSampler(SourceSpec("x", "crystal", 10))]
        with self.assertRaises(ValueError):
            HierarchicalSampler(sources, domain_weights={"bio": 1.0})

    def test_scaffold_dedup_is_deterministic_across_sources(self):
        hashes = np.asarray([[11, 101], [22, 202]], dtype=np.uint64)
        sources = [
            SourceSampler(SourceSpec(name, "small_molecule", 4),
                          ScaffoldIndex(np.asarray([0, 0, 1, 1]), hashes))
            for name in ("x", "y")
        ]
        sampler = HierarchicalSampler(sources, seed=42, unique_scaffolds=True)
        first = sampler.batch(7, 0, 1, 2)
        second = sampler.batch(7, 0, 1, 2)
        self.assertEqual(first, second)
        fingerprints = {sampler.by_name[p.source].fingerprint(p) for p in first}
        self.assertEqual(fingerprints, {(11, 101), (22, 202)})

    def test_scaffold_dedup_has_bounded_rejection(self):
        source = SourceSampler(
            SourceSpec("x", "small_molecule", 2),
            ScaffoldIndex(np.asarray([0, 0]), np.asarray([[1, 2]], dtype=np.uint64)),
        )
        sampler = HierarchicalSampler([source], unique_scaffolds=True, max_attempts=4)
        with self.assertRaisesRegex(RuntimeError, "骨架去重"):
            sampler.batch(0, 0, 1, 2)

    def test_source_grouped_batches_single_source_and_deterministic(self):
        sources = [
            SourceSampler(SourceSpec("pubchem", "small_molecule", 100)),
            SourceSampler(SourceSpec("uniref50", "bio", 100)),
            SourceSampler(SourceSpec("pi1m", "polymer", 100, views=2)),
        ]
        sampler = HierarchicalSampler(
            sources, seed=42, source_grouped=True,
            domain_weights={"small_molecule": 0.35, "bio": 0.45, "polymer": 0.20},
        )
        batch = sampler.batch(5, 1, 4, 8)
        self.assertEqual(batch, sampler.batch(5, 1, 4, 8))
        self.assertEqual({pointer.source for pointer in batch}, {batch[0].source})
        # 多个 micro-batch 的源分布应覆盖全部三个域
        seen = {sampler.batch(i, 0, 1, 4)[0].source for i in range(200)}
        self.assertEqual(seen, {"pubchem", "uniref50", "pi1m"})
        with self.assertRaisesRegex(ValueError, "同时开启"):
            HierarchicalSampler(sources, unique_scaffolds=True, source_grouped=True)

    def test_exclude_rows_redraws_and_stays_deterministic(self):
        excluded = frozenset({0, 1, 2, 3, 4})
        source = SourceSampler(SourceSpec("x", "small_molecule", 10), exclude=excluded)
        import numpy as np
        rng = np.random.default_rng(0)
        rows = [source.sample(rng).row for _ in range(200)]
        self.assertTrue(all(row not in excluded for row in rows))
        rng2 = np.random.default_rng(0)
        rows2 = [source.sample(rng2).row for _ in range(200)]
        self.assertEqual(rows, rows2)
        # 排除集合覆盖全部记录时构造即拒绝
        with self.assertRaisesRegex(ValueError, "排除行数"):
            SourceSampler(SourceSpec("y", "bio", 3), exclude=frozenset({0, 1, 2}))


if __name__ == "__main__":
    unittest.main()
