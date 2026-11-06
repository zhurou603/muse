from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from muse_encoder.pretrain.aux import AuxSource
from muse_encoder.pretrain.aux_index import AuxIndex, build_aux_index


class AuxIndexTests(unittest.TestCase):
    def test_filtered_byte_offsets_and_random_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a.jsonl"
            second = root / "b.jsonl"
            hits_path = root / "hits.jsonl"
            first.write_text('{"id":"keep-0","v":0}\n{"id":"drop-1","v":1}\n', encoding="utf-8")
            second.write_text('{"id":"drop-2","v":2}\n{"id":"keep-3","v":3}\n', encoding="utf-8")
            hits_path.write_text('{"id":"drop-1","reason":"unparseable"}\n'
                                 '{"id":"drop-2","reason":"future"}\n', encoding="utf-8")
            spec = AuxSource("toy", (first, second), hits_path, 4, 2, 2)
            with patch.dict("muse_encoder.pretrain.aux_index.AUX_SOURCES", {"toy": spec}, clear=True):
                path = build_aux_index("toy", root / "toy.npy")
                index = AuxIndex("toy", path)
                self.assertEqual(len(index), 2)
                self.assertEqual(index.get(1).id, "keep-3")
                self.assertEqual(index.get(0).payload["v"], 0)
                self.assertEqual(index.get(1).payload["v"], 3)
                index.hits.add("keep-3")
                with self.assertRaisesRegex(RuntimeError, "排除 ID"):
                    index.get(1)
                index.close()

    def test_count_mismatch_does_not_publish_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "a.jsonl"
            hits = root / "hits.jsonl"
            source.write_text('{"id":"a"}\n', encoding="utf-8")
            hits.write_text("", encoding="utf-8")
            spec = AuxSource("bad", (source,), hits, 2, 0, 2)
            with patch.dict("muse_encoder.pretrain.aux_index.AUX_SOURCES", {"bad": spec}, clear=True):
                with self.assertRaisesRegex(ValueError, "实际计数"):
                    build_aux_index("bad", root / "bad.npy")
                self.assertFalse((root / "bad.npy").exists())
                self.assertFalse((root / "bad.json").exists())
    def test_concurrent_get_is_thread_safe(self):
        import random
        from concurrent.futures import ThreadPoolExecutor

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "a.jsonl"
            hits = root / "hits.jsonl"
            count = 2000
            source.write_text(
                "".join(f'{{"id":"r-{i}","v":{i},"pad":"{"x" * 64}"}}\n'
                        for i in range(count)),
                encoding="utf-8",
            )
            hits.write_text("", encoding="utf-8")
            spec = AuxSource("conc", (source,), hits, count, 0, count)
            with patch.dict("muse_encoder.pretrain.aux_index.AUX_SOURCES",
                            {"conc": spec}, clear=True):
                path = build_aux_index("conc", root / "conc.npy")
                index = AuxIndex("conc", path)
                rng = random.Random(20261024)
                rows = [rng.randrange(count) for _ in range(4000)]

                def check(row):
                    record = index.get(row)
                    return record.id == f"r-{row}" and record.payload["v"] == row

                with ThreadPoolExecutor(max_workers=32) as pool:
                    results = list(pool.map(check, rows))
                self.assertTrue(all(results))
                index.close()
                self.assertEqual(index._all_handles, [])


if __name__ == "__main__":
    unittest.main()
