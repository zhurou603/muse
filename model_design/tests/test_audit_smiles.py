from __future__ import annotations

import time
import unittest

import muse_encoder.pretrain.audit_smiles as audit


def _fake_scan(args):
    """模拟：凡包含 row 5 的范围都"卡死"（睡 30s 超过超时），其余秒回。"""
    _name, start, end = args
    if start <= 5 < end:
        time.sleep(30)
    return (start, [])


class AuditSmilesRescueTests(unittest.TestCase):
    def test_rescue_bisects_to_poison_row(self):
        original = audit._scan_chunk
        audit._scan_chunk = _fake_scan
        try:
            bad: list[int] = []
            audit._rescue_chunk("pubchem", 0, 8, bad)
        finally:
            audit._scan_chunk = original
        self.assertEqual(sorted(bad), [5])

    def test_rescue_clean_range_returns_fast(self):
        original = audit._scan_chunk
        audit._scan_chunk = _fake_scan
        try:
            bad: list[int] = []
            audit._rescue_chunk("pubchem", 8, 16, bad)
        finally:
            audit._scan_chunk = original
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
