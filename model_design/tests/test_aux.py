import copy
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muse_encoder.pretrain.aux import (
    AUX_SOURCES,
    AuxSource,
    NamespacedRecord,
    SourceCounts,
    extract_labels,
    iter_source,
    iter_sources,
    label_key,
    load_hit_ids,
    validate_source,
)


class TemporaryAuxTest(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)

    def tearDown(self):
        self._temporary_directory.cleanup()

    def write_jsonl(self, name, records):
        path = self.root / name
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path

    def make_source(self, name, records, hits):
        hit_ids = {hit["id"] for hit in hits}
        excluded = sum(record["id"] in hit_ids for record in records)
        return AuxSource(
            name=name,
            paths=(self.write_jsonl(f"{name}.jsonl", records),),
            hits_path=self.write_jsonl(f"{name}.hits.jsonl", hits),
            expected_raw=len(records),
            expected_excluded=excluded,
            expected_kept=len(records) - excluded,
        )

    def test_every_hit_reason_is_excluded_without_payload_mutation(self):
        reasons = [
            "valtest_fingerprint_hit", "kmer_cover_0.75", "exact_match",
            "unparseable", "future_reason",
        ]
        records = (
            [{"id": "keep-a", "labels": {"x": [{"value": 1}]}}]
            + [{"id": f"drop-{i}", "nested": {"id": "keep-a"}}
               for i in range(len(reasons))]
            + [{"id": "keep-b", "value": False}]
        )
        hits = [
            {"id": f"drop-{i}", "reason": reason}
            for i, reason in enumerate(reasons)
        ]
        spec = self.make_source("temporary", records, hits)

        kept = list(iter_source(spec))

        self.assertEqual(load_hit_ids(spec.hits_path), {f"drop-{i}" for i in range(5)})
        self.assertEqual([record.id for record in kept], ["keep-a", "keep-b"])
        self.assertEqual([record.payload for record in kept],
                         [records[0], records[-1]])
        self.assertTrue(all(isinstance(record, NamespacedRecord) for record in kept))
        self.assertTrue(all(record.source == "temporary" for record in kept))
        self.assertNotIn("source", kept[0].payload)

    def test_same_id_from_different_sources_is_not_deduplicated(self):
        first = self.make_source("source_a", [{"id": "shared", "value": 1}], [])
        second = self.make_source("source_b", [{"id": "shared", "value": 2}], [])

        records = list(iter_sources([first, second]))

        self.assertEqual([(record.source, record.id) for record in records],
                         [("source_a", "shared"), ("source_b", "shared")])
        self.assertEqual([record.payload["value"] for record in records], [1, 2])

    def test_raw_jsonl_is_consumed_lazily(self):
        raw_path = self.root / "stream.jsonl"
        raw_path.write_text('{"id": "first"}\nnot-json\n', encoding="utf-8")
        spec = AuxSource("stream", (raw_path,), self.write_jsonl("empty.jsonl", []),
                         2, 0, 2)

        records = iter_source(spec)
        self.assertEqual(next(records).id, "first")
        with self.assertRaises(json.JSONDecodeError):
            next(records)
        records.close()

    def test_validate_source_asserts_all_expected_counts(self):
        spec = self.make_source(
            "counts",
            [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            [{"id": "b", "reason": "any_reason"}],
        )
        hits = load_hit_ids(spec.hits_path)
        self.assertEqual(validate_source(spec, hit_ids=hits), SourceCounts(3, 1, 2))

        for field in ("expected_raw", "expected_excluded", "expected_kept"):
            with self.subTest(field=field):
                invalid = replace(spec, **{field: getattr(spec, field) + 1})
                with self.assertRaisesRegex(ValueError, "count mismatch"):
                    validate_source(invalid, hit_ids=hits)

    def test_unit_grouping_and_missing_mask(self):
        delta_g = label_key("delta_g", "kcal/mol")
        stabilizing = label_key("stabilizing_mutation")
        micromolar = label_key("concentration", "µM")
        micrograms = label_key("concentration", "µg/ml")
        payload = {
            "labels": {
                "delta_g": [{"value": 1.25, "unit": "kcal/mol"}],
                "stabilizing_mutation": [{"value": False}],
                "concentration": [
                    {"value": 6.0, "unit": "µM"},
                    {"value": 7.0, "unit": "µg/ml"},
                ],
            }
        }
        original = copy.deepcopy(payload)
        keys = [delta_g, stabilizing, micromolar, micrograms]

        values, missing = extract_labels(payload, keys)

        self.assertEqual(payload, original)
        self.assertEqual(values[delta_g], 1.25)
        self.assertIs(values[stabilizing], False)
        self.assertEqual(values[micromolar], 6.0)
        self.assertEqual(values[micrograms], 7.0)
        self.assertEqual(set(values), set(keys))
        self.assertFalse(any(missing.values()))

        d23_missing = {"labels": {"delta_g": payload["labels"]["delta_g"]}}
        values, missing = extract_labels(d23_missing, [delta_g, stabilizing])
        self.assertEqual(values[delta_g], 1.25)
        self.assertIsNone(values[stabilizing])
        self.assertFalse(missing[delta_g])
        self.assertTrue(missing[stabilizing])


class RegistryAndRealDataTest(unittest.TestCase):
    def test_registry_paths_and_counts(self):
        expected = {
            "geom_drugs": (32305220, 592688, 31712532),
            "qmugs": (1912850, 14785, 1898065),
            "qm9": (90965, 1431, 89534),
            "pubchemqc": (3970758, 192995, 3777763),
            "pcqm4v2": (3445326, 1148, 3444178),
            "megascale_d1": (1838174, 346534, 1491640),
            "megascale_d23": (775516, 300751, 474765),
            "dbaasp_hemo": (13848, 2441, 11407),
            "dbaasp_nonmic": (24568, 4569, 19999),
        }
        self.assertEqual(set(AUX_SOURCES), set(expected))
        for name, counts in expected.items():
            with self.subTest(source=name):
                spec = AUX_SOURCES[name]
                self.assertEqual(
                    (spec.expected_raw, spec.expected_excluded, spec.expected_kept),
                    counts,
                )
                self.assertTrue(spec.paths)
                self.assertTrue(all(path.is_file() for path in spec.paths))
                self.assertTrue(spec.hits_path.is_file())

    def check_real_prefix(self, name):
        spec = AUX_SOURCES[name]
        hits = load_hit_ids(spec.hits_path)
        with spec.paths[0].open(encoding="utf-8") as handle:
            raw = [json.loads(line) for line in islice(handle, 6)]
        expected = [record for record in raw if record["id"] not in hits]

        records = iter_source(spec, hit_ids=hits)
        try:
            actual = list(islice(records, len(expected)))
            self.assertEqual([record.payload for record in actual], expected)
            self.assertTrue(all(record.source == name for record in actual))
            self.assertTrue(all(record.id == record.payload["id"] for record in actual))
            if not expected:
                first_kept = next(records)
                self.assertNotIn(first_kept.id, hits)
                self.assertEqual(first_kept.source, name)
        finally:
            records.close()

    def test_real_qm9_and_megascale_prefix_smoke(self):
        for name in ("qm9", "megascale_d1", "megascale_d23"):
            with self.subTest(source=name):
                self.check_real_prefix(name)


if __name__ == "__main__":
    unittest.main()
