import io
import json
import sqlite3
import sys
import tarfile
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muse_encoder.pretrain.omc25 import (
    EXPECTED_AUX_FRAMES,
    EXPECTED_ENERGY_RECORDS,
    SamplingConfig,
    audit_triples,
    audit_valtest_conservative,
    build_index,
    compute_statistics,
    extract_lmdb_member,
    index_aux_tar,
    index_energy_jsonl,
    initialize_index,
    read_ase_lmdb_row,
    reduced_formula,
    row_field_report,
    sample_frames,
    starting_heap_id,
    stratified_quotas,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MUSE_ROOT = PROJECT_ROOT / "data/muse-a/MUSE-A"
REAL_AUX = MUSE_ROOT / "aux_labels/A_datasets/organic_crystal/OMC25/omc25/part_00.tar"
REAL_TRAIN = MUSE_ROOT / "data/A_datasets/organic_crystal/OMC25/omc25/train.jsonl"
REAL_LMDB = PROJECT_ROOT / "data/pretrain/cache/omc25/part_00/train/data0000.aselmdb"


def add_bytes(tf, name, payload, kind=tarfile.REGTYPE, linkname=""):
    member = tarfile.TarInfo(name)
    member.type = kind
    member.linkname = linkname
    member.mode = 0o644
    if kind == tarfile.REGTYPE:
        member.size = len(payload)
        tf.addfile(member, io.BytesIO(payload))
    else:
        member.size = 0
        tf.addfile(member)


def npy_bytes(array):
    handle = io.BytesIO()
    np.save(handle, array, allow_pickle=False)
    return handle.getvalue()


def make_aux_tar(path, ids, offsets, forces, stress, prefix="train/data0000"):
    with tarfile.open(path, "w") as tf:
        add_bytes(tf, f"{prefix}/forces.npy", npy_bytes(forces))
        add_bytes(tf, f"{prefix}/ids.txt", "".join(f"{value}\n" for value in ids).encode())
        add_bytes(tf, f"{prefix}/offsets.npy", npy_bytes(offsets))
        add_bytes(tf, f"{prefix}/stress.npy", npy_bytes(stress))
    return path


def make_record(record_id, sid, energy, formula="C2H4", atom_count=2):
    return {
        "id": record_id,
        "inputs": {"chemical_formula": formula, "atom_count": atom_count},
        "external_inputs": [{
            "id": record_id,
            "role": "periodic_structure",
            "format": "ase_lmdb",
            "selector": {
                "archive_member": "train/data0000.aselmdb",
                "lmdb_row_id": 1,
                "sid": sid,
            },
            "source_path": "assets/omc_train_250802/part_00.tar",
        }],
        "labels": {"energy": [{"value": energy, "unit": "eV"}]},
    }


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    return path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ase_payload(sid="AAA-1-gener-hash-0"):
    return {
        "numbers": {"__ndarray__": [[2], "int64", [1, 6]]},
        "positions": {"__ndarray__": [[2, 3], "float64", [0, 0, 0, 1, 2, 3]]},
        "cell": {"__ndarray__": [[3, 3], "float64", [5, 0, 0, 0, 6, 0, 0, 0, 7]]},
        "pbc": {"__ndarray__": [[3], "bool", [True, True, True]]},
        "unique_id": "temporary",
        "data": {"sid": sid},
    }


def make_lmdb(path, rows=1):
    import lmdb

    environment = lmdb.open(str(path), subdir=False, map_size=4 * 1024 * 1024)
    with environment.begin(write=True) as transaction:
        for row_id in range(1, rows + 1):
            payload = ase_payload(f"AAA-1-gener-hash-{row_id - 1}")
            transaction.put(str(row_id).encode(), zlib.compress(
                json.dumps(payload, separators=(",", ":")).encode()
            ))
    environment.close()
    return path


class TemporaryOMC25Test(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)

    def tearDown(self):
        self._temporary_directory.cleanup()

    def memory_index(self):
        connection = sqlite3.connect(":memory:")
        initialize_index(connection)
        return connection

    def test_id_join_offsets_and_complete_bidirectional_missing(self):
        aux = make_aux_tar(
            self.root / "aux.tar", ["a", "b", "c"],
            np.asarray([0, 2, 5, 6], dtype=np.int64),
            np.arange(18, dtype=np.float32).reshape(6, 3),
            np.zeros((3, 6), dtype=np.float32),
        )
        records = [
            make_record("c", "AAA-1-gener-hash-2", 30, atom_count=1),
            make_record("a", "AAA-1-gener-hash-0", 10, atom_count=2),
            make_record("d", "AAA-1-gener-hash-3", 40, atom_count=4),
        ]
        train = write_jsonl(self.root / "train.jsonl", records)
        output = self.root / "index.sqlite3"

        report = build_index(train, [aux], output, expected_energy=3, expected_aux=3)

        self.assertEqual(report["matched_by_id"], 2)
        self.assertEqual(report["energy_without_aux"], 1)
        self.assertEqual(report["aux_without_energy"], 1)
        self.assertEqual(report["count_difference_energy_minus_aux"], 0)
        self.assertEqual([row["id"] for row in read_jsonl(str(output) + ".missing_aux.jsonl")], ["d"])
        self.assertEqual([row["id"] for row in read_jsonl(str(output) + ".missing_energy.jsonl")], ["b"])
        connection = sqlite3.connect(output)
        rows = connection.execute(
            """SELECT e.id,a.frame_row,a.atom_start,a.atom_end,e.energy
               FROM energy_records e JOIN aux_frames a ON a.id=e.id ORDER BY e.jsonl_row"""
        ).fetchall()
        shard = connection.execute(
            "SELECT tar_path,member_prefix,frame_count,atom_count,stress_width FROM aux_shards"
        ).fetchone()
        connection.close()
        self.assertEqual(rows, [("c", 2, 5, 6, 30.0), ("a", 0, 0, 2, 10.0)])
        self.assertEqual(shard[1:], ("train/data0000", 3, 6, 6))
        self.assertTrue(Path(shard[0]).is_absolute())
        self.assertEqual(EXPECTED_ENERGY_RECORDS - EXPECTED_AUX_FRAMES, 9544)

    def test_strict_aux_shape_offset_and_id_validation(self):
        cases = {
            "forces": {
                "forces": np.zeros((2, 2), dtype=np.float32), "error": "forces",
            },
            "offsets": {
                "offsets": np.asarray([0, 2, 1], dtype=np.int64), "error": "offsets",
            },
            "stress": {
                "stress": np.zeros((1, 6), dtype=np.float32), "error": "stress",
            },
            "extra_ids": {"ids": ["a", "b", "c"], "error": "IDs"},
            "missing_ids": {"ids": ["a"], "error": "IDs"},
        }
        for name, case in cases.items():
            with self.subTest(case=name):
                values = {
                    "ids": ["a", "b"],
                    "offsets": np.asarray([0, 1, 2], dtype=np.int64),
                    "forces": np.zeros((2, 3), dtype=np.float32),
                    "stress": np.zeros((2, 6), dtype=np.float32),
                }
                values.update({key: value for key, value in case.items() if key != "error"})
                path = make_aux_tar(self.root / f"{name}.tar", **values)
                connection = self.memory_index()
                with self.assertRaisesRegex(ValueError, case["error"]):
                    index_aux_tar(connection, path)
                connection.close()

    def test_safe_atomic_lmdb_extraction_ase_row_and_unsafe_paths(self):
        source_lmdb = make_lmdb(self.root / "source.aselmdb", rows=2)
        archive = self.root / "structures.tar"
        with tarfile.open(archive, "w") as tf:
            tf.add(source_lmdb, arcname="train/data0000.aselmdb", recursive=False)

        extracted = extract_lmdb_member(archive, "train/data0000.aselmdb", self.root / "cache")
        row = read_ase_lmdb_row(extracted, 2)

        self.assertEqual(extracted, self.root / "cache/structures/train/data0000.aselmdb")
        self.assertEqual(row.atoms.positions.tolist(), [[0, 0, 0], [1, 2, 3]])
        self.assertEqual(row.atoms.get_chemical_formula(), "CH")
        self.assertEqual(row.raw["data"]["sid"], "AAA-1-gener-hash-1")
        self.assertEqual(row_field_report(row)["data_fields"], ["sid"])
        with self.assertRaises(KeyError):
            read_ase_lmdb_row(extracted, 3)

        for index, name in enumerate(("../escape.aselmdb", "/absolute.aselmdb")):
            with self.subTest(path=name):
                unsafe = self.root / f"unsafe-{index}.tar"
                with tarfile.open(unsafe, "w") as tf:
                    add_bytes(tf, name, b"bad")
                with self.assertRaises(ValueError):
                    extract_lmdb_member(unsafe, name, self.root / "cache")
                self.assertFalse((self.root / "escape.aselmdb").exists())
        link_archive = self.root / "link.tar"
        with tarfile.open(link_archive, "w") as tf:
            add_bytes(tf, "train/link.aselmdb", b"", tarfile.SYMTYPE, "target")
        with self.assertRaisesRegex(ValueError, "链接"):
            extract_lmdb_member(link_archive, "train/link.aselmdb", self.root / "cache")

    def test_triple_audit_and_explicit_conservative_fallback(self):
        references = [{"id": "ref", "chemical_formula": "C4H8", "spacegroup": 2,
                       "parent_inchikey": "PARENT-KEY"}]
        records = [
            {"id": "hit", "chemical_formula": "C2H4", "spacegroup": "2",
             "parent_inchikey": "PARENT-KEY"},
            {"id": "sg-miss", "chemical_formula": "C2H4", "spacegroup": "1",
             "parent_inchikey": "PARENT-KEY"},
        ]
        self.assertEqual([row["id"] for row in audit_triples(records, references)], ["hit"])
        self.assertEqual(reduced_formula("Br2C16H12N2"), "C8H6BrN")
        self.assertEqual(starting_heap_id("AAA-1-gener-hash-17"), "AAA-1-gener-hash")

        aux = make_aux_tar(
            self.root / "audit-aux.tar", ["a", "b", "c"],
            np.asarray([0, 1, 2, 3], dtype=np.int64),
            np.zeros((3, 3), dtype=np.float32),
            np.zeros((3, 6), dtype=np.float32),
        )
        train = write_jsonl(self.root / "audit-train.jsonl", [
            make_record("a", "ZZZ-1-gener-hash-0", 1, "C2H4", 1),
            make_record("b", "AAA-2-gener-hash-0", 2, "C6H6", 1),
            make_record("c", "BBB-1-gener-hash-0", 3, "C6H6", 1),
        ])
        database = self.root / "audit.sqlite3"
        build_index(train, [aux], database)
        crystals = self.root / "crystals.tsv"
        crystals.write_text(
            "C4H8\t8\tZZZ-1-gener-hash-0\tOMC25/omc25/validation/z\n"
            "H2\t2\tAAA-9-gener-hash-3\tOMC25/omc25/test/y\n",
            encoding="utf-8",
        )
        hits_path = self.root / "hits.jsonl"
        report = audit_valtest_conservative(database, crystals, hits_path)

        hits = {row["id"]: row["reasons"] for row in read_jsonl(hits_path)}
        self.assertIn("reduced_formula", hits["a"])
        self.assertIn("sid", hits["a"])
        self.assertIn("csd_refcode", hits["b"])
        self.assertNotIn("c", hits)
        self.assertFalse(report["triple_audit"]["completed"])
        self.assertEqual(report["triple_audit"]["status"], "blocked_missing_source_fields")

    def test_streaming_statistics_and_reproducible_stratified_sampling(self):
        frame_count = 120
        counts = np.asarray([index % 3 + 1 for index in range(frame_count)], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        forces = np.zeros((int(offsets[-1]), 3), dtype=np.float32)
        forces[offsets[1], 0] = 100.0
        ids = [f"frame-{index:03d}" for index in range(frame_count)]
        aux = make_aux_tar(
            self.root / "sample-aux.tar", ids, offsets, forces,
            np.zeros((frame_count, 6), dtype=np.float32),
        )
        records = []
        for index in reversed(range(frame_count)):
            heap = index % 4
            records.append(make_record(
                ids[index], f"REF{heap}-1-gener-hash-{index}",
                100.0 if index == 0 else 0.0, atom_count=int(counts[index]),
            ))
        train = write_jsonl(self.root / "sample-train.jsonl", records)
        database = self.root / "sample.sqlite3"
        build_index(train, [aux], database)

        statistics = compute_statistics(database, self.root / "stats.json", chunk_atoms=7)
        self.assertEqual(statistics["energy"]["count"], frame_count)
        self.assertEqual(statistics["forces"]["count"], int(offsets[-1]) * 3)
        self.assertGreater(statistics["energy"]["std"], 0)
        quotas = stratified_quotas({"a": 10, "b": 20, "c": 30}, 17)
        self.assertEqual(sum(quotas.values()), 17)
        self.assertTrue(all(value >= 1 for value in quotas.values()))

        config = SamplingConfig(8, 6, 8, seed=1729, z_threshold=10, chunk_atoms=7)
        first = sample_frames(database, statistics, self.root / "first.jsonl", config)
        second = sample_frames(database, statistics, self.root / "second.jsonl", config)
        self.assertEqual((self.root / "first.jsonl").read_bytes(),
                         (self.root / "second.jsonl").read_bytes())
        self.assertEqual((self.root / "first.jsonl.exclusions.jsonl").read_bytes(),
                         (self.root / "second.jsonl.exclusions.jsonl").read_bytes())
        self.assertEqual((self.root / "first.jsonl.strata.jsonl").read_bytes(),
                         (self.root / "second.jsonl.strata.jsonl").read_bytes())
        self.assertEqual(first["selected"], 8)
        self.assertEqual(first["reason_counts"]["energy_z"], 1)
        self.assertEqual(first["reason_counts"]["force_z"], 1)
        output_keys = {"sample_jsonl", "exclusions_jsonl", "strata_jsonl"}
        self.assertEqual({key: value for key, value in first.items() if key not in output_keys},
                         {key: value for key, value in second.items() if key not in output_keys})
        selected = read_jsonl(self.root / "first.jsonl")
        self.assertEqual(len({row["starting_heap"] for row in selected}), 4)
        self.assertNotIn("frame-000", {row["id"] for row in selected})
        self.assertNotIn("frame-001", {row["id"] for row in selected})
        excluded = {row["id"]: row["reasons"] for row in read_jsonl(
            self.root / "first.jsonl.exclusions.jsonl"
        )}
        self.assertIn("energy_z", excluded["frame-000"])
        self.assertIn("force_z", excluded["frame-001"])

    def test_prior_exclusions_apply_before_quota_generation(self):
        frame_count = 40
        counts = np.full(frame_count, 2, dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        ids = [f"f-{index:03d}" for index in range(frame_count)]
        aux = make_aux_tar(
            self.root / "prior-aux.tar", ids, offsets,
            np.zeros((int(offsets[-1]), 3), dtype=np.float32),
            np.zeros((frame_count, 6), dtype=np.float32),
        )
        records = [
            make_record(ids[index], f"REF{index % 4}-1-gener-hash-{index}", 0.0,
                        atom_count=2)
            for index in range(frame_count)
        ]
        train = write_jsonl(self.root / "prior-train.jsonl", records)
        database = self.root / "prior.sqlite3"
        build_index(train, [aux], database)
        statistics = compute_statistics(database, self.root / "prior-stats.json")

        prior = write_jsonl(self.root / "prior-exclusions.jsonl", [
            {"id": "f-005"},
            {"starting_heap": "REF2-1-gener-hash", "reasons": ["triple_hit"]},
        ])
        config = SamplingConfig(8, 6, 8, seed=7, z_threshold=10)
        report = sample_frames(
            database, statistics, self.root / "prior-sample.jsonl", config,
            prior_exclusions_jsonl=prior,
        )

        # 1 个 id + 整个 REF2 heap（10 帧）在配额生成之前被排除。
        self.assertEqual(report["prior_exclusion_ids"], 1)
        self.assertEqual(report["prior_exclusion_heaps"], 1)
        self.assertEqual(report["prior_deleak_excluded_frames"], 11)
        self.assertEqual(report["reason_counts"]["prior_deleak"], 11)
        self.assertEqual(report["eligible_frames"], frame_count - 11)
        self.assertEqual(report["selected"], 8)
        self.assertEqual(report["prior_exclusions_jsonl"], str(prior.resolve()))

        strata = read_jsonl(self.root / "prior-sample.jsonl.strata.jsonl")
        eligible = {row["starting_heap"]: row["eligible"] for row in strata}
        self.assertEqual(eligible, {
            "REF0-1-gener-hash": 10,
            "REF1-1-gener-hash": 9,
            "REF3-1-gener-hash": 10,
        })
        selected = read_jsonl(self.root / "prior-sample.jsonl")
        selected_ids = {row["id"] for row in selected}
        self.assertEqual(len(selected), 8)
        self.assertNotIn("f-005", selected_ids)
        self.assertTrue(all(row["starting_heap"] != "REF2-1-gener-hash" for row in selected))
        excluded = {row["id"]: row["reasons"] for row in read_jsonl(
            self.root / "prior-sample.jsonl.exclusions.jsonl"
        )}
        self.assertIn("prior_deleak", excluded["f-005"])
        for index in range(frame_count):
            if index % 4 == 2:
                self.assertIn("prior_deleak", excluded[f"f-{index:03d}"])

        # 对照：无先验排除时 4 个 heap 都参与配额，证明排除先于配额生成。
        baseline = sample_frames(
            database, statistics, self.root / "baseline-sample.jsonl", config,
        )
        self.assertEqual(baseline["eligible_frames"], frame_count)
        self.assertNotIn("prior_deleak", baseline["reason_counts"])
        baseline_strata = read_jsonl(self.root / "baseline-sample.jsonl.strata.jsonl")
        self.assertEqual(len(baseline_strata), 4)


@unittest.skipUnless(REAL_AUX.is_file() and REAL_TRAIN.is_file() and REAL_LMDB.is_file(),
                     "真实 OMC25 shard 或已解包 LMDB smoke member 不可用")
class RealOMC25SmokeTest(unittest.TestCase):
    def test_one_real_aux_shard_train_prefix_and_lmdb_row(self):
        connection = sqlite3.connect(":memory:")
        initialize_index(connection)
        summary = index_aux_tar(connection, REAL_AUX, only_prefixes={"train/data0000"})
        first_aux = connection.execute(
            """SELECT id,frame_row,atom_start,atom_end FROM aux_frames
               WHERE shard_id=1 AND frame_row=0"""
        ).fetchone()
        connection.close()

        self.assertEqual(summary, {"shards": 1, "frames": 596_690, "atoms": 55_116_446})
        self.assertEqual(first_aux, (
            "muse__omc25__periodic_frame__OQECAP_1_press_97b75093ceb3ffd_0", 0, 0, 24,
        ))
        row = read_ase_lmdb_row(REAL_LMDB, 1)
        report = row_field_report(row)
        self.assertEqual(len(row.atoms), 24)
        self.assertTrue(row.atoms.pbc.all())
        self.assertEqual(report["geometry_fields"], ["numbers", "positions", "cell", "pbc"])
        self.assertEqual(report["data_fields"], ["sid"])
        self.assertEqual(report["spacegroup_fields"], [])
        self.assertEqual(report["potential_parent_molecule_fields"], [])
        self.assertFalse(report["triple_audit_ready"])
        expected_id = "muse__omc25__periodic_frame__" + report["sid"].replace("-", "_")
        self.assertEqual(expected_id, first_aux[0])

        energy_connection = sqlite3.connect(":memory:")
        initialize_index(energy_connection)
        self.assertEqual(index_energy_jsonl(energy_connection, REAL_TRAIN, limit=2), 2)
        selector = energy_connection.execute(
            """SELECT id,chemical_formula,source_path,archive_member,lmdb_row_id,sid
               FROM energy_records ORDER BY jsonl_row LIMIT 1"""
        ).fetchone()
        energy_connection.close()
        self.assertEqual(selector[1:], (
            "Br2C16H12N2", "assets/omc_train_250802/part_01.tar",
            "train/data0003.aselmdb", 1, "AJEXIX-2-gener-d95bafcf2a4d27b-17",
        ))


class LMDBConcurrencyTest(unittest.TestCase):
    @unittest.skipUnless(REAL_LMDB.is_file(), "真实 ASE-LMDB 不可用")
    def test_concurrent_reads_reuse_one_environment_per_file(self):
        from concurrent.futures import ThreadPoolExecutor
        from muse_encoder.pretrain.omc25 import close_lmdb_environments

        close_lmdb_environments()
        with ThreadPoolExecutor(max_workers=4) as executor:
            rows = list(executor.map(lambda _: read_ase_lmdb_row(REAL_LMDB, 1), range(32)))
        self.assertEqual([len(row.atoms) for row in rows], [24] * 32)
        close_lmdb_environments()


if __name__ == "__main__":
    unittest.main()
