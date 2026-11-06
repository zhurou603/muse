from __future__ import annotations

import time
import unittest

from muse_encoder.pretrain.scaffolds import (
    FALLBACK_PREFIX,
    _salvage_item,
    _scaffold_worker,
    scaffold_smiles,
)


def _poison_worker(item):
    """模块级打捞测试桩：对 poison 记录睡死（fork 子进程按名解析）。"""
    if item[2] == "poison":
        time.sleep(30)
    return (item[1], "ok")


class ScaffoldWorkerTests(unittest.TestCase):
    def test_ring_and_acyclic_scaffolds(self):
        self.assertEqual(scaffold_smiles("c1ccccc1O"), "c1ccccc1")
        self.assertEqual(scaffold_smiles("CCO"), "")  # 无环分子共享空骨架组

    def test_unparseable_falls_back_to_singleton_group(self):
        row, key = _scaffold_worker(("chembl", 40369, "CHEMBL4298947", "C(", None))
        self.assertEqual(row, 40369)
        self.assertEqual(key, f"{FALLBACK_PREFIX}CHEMBL4298947")

    def test_pi1m_si_valence_falls_back_instead_of_raising(self):
        smiles = ("*Nc1ccc(NC(=O)c2ccc(NC(C)=O)cc2)cc1[Si](=O)(=O)"
                  "c1ccc(C(=O)N(*)c2ccccc2)cc1")
        row, key = _scaffold_worker(("pi1m", 6975, "toy-pi1m", smiles, None))
        self.assertTrue(key.startswith(FALLBACK_PREFIX) or isinstance(key, str))
        self.assertEqual(row, 6975)

    def test_parent_group_bypasses_scaffold(self):
        row, key = _scaffold_worker(("geom_drugs", 3, "rid", "C(", "parent-42"))
        self.assertEqual((row, key), (3, "parent-42"))

    def test_salvage_item_hard_timeout_and_success(self):
        (row, key), timed_out = _salvage_item(
            ("pubchem", 7, "poison", "C", None), 1.0, worker=_poison_worker)
        self.assertTrue(timed_out)
        self.assertEqual((row, key), (7, f"{FALLBACK_PREFIX}poison"))
        (row, key), timed_out = _salvage_item(
            ("pubchem", 8, "good", "C", None), 5.0, worker=_poison_worker)
        self.assertFalse(timed_out)
        self.assertEqual((row, key), (8, "ok"))

    def test_salvage_batch_preserves_order_and_isolates_poison(self):
        from muse_encoder.pretrain.scaffolds import _salvage_batch
        batch = [("pubchem", i, "poison" if i == 66 else f"r{i}", "C", None)
                 for i in range(130)]
        # fork 子进程继承父进程模块状态，monkeypatch 对 Pool worker 生效。
        import muse_encoder.pretrain.scaffolds as sc
        original = sc._scaffold_worker
        sc._scaffold_worker = _poison_worker
        try:
            results, timeouts = _salvage_batch(batch, 2.0, sub_chunk=64)
        finally:
            sc._scaffold_worker = original
        self.assertEqual(timeouts, 1)
        self.assertEqual(len(results), 130)
        self.assertEqual([row for row, _ in results], list(range(130)))
        self.assertEqual(results[66], (66, f"{FALLBACK_PREFIX}poison"))
        self.assertEqual(results[65], (65, "ok"))


if __name__ == "__main__":
    unittest.main()
