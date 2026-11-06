from __future__ import annotations

import io
import json
import sqlite3
import sys
import tarfile
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muse_encoder.config import DOMAIN_CRYSTAL, MuseConfig
from muse_encoder.pretrain import crystal_dataset
from muse_encoder.pretrain.batching import BatchBuilder
from muse_encoder.pretrain.crystal_dataset import (
    FrameTargetStats,
    OMC25CrystalDataset,
    build_frame_cache,
    compute_frame_stats,
    load_exclusion_spec,
)
from muse_encoder.pretrain.crystal_references import RESIDUAL_MODE, ElementReferences
from muse_encoder.pretrain.omc25 import build_index


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_INDEX = PROJECT_ROOT / "data/pretrain/index/omc25.sqlite3"
REAL_EXTERNAL = PROJECT_ROOT / "data/muse-a/MUSE-A/external_inputs/A_datasets/organic_crystal/OMC25"
REAL_ARCHIVE = REAL_EXTERNAL / "assets/omc_train_250802/part_00.tar"
REAL_LMDB_CACHE = PROJECT_ROOT / "data/pretrain/cache/omc25"
REAL_LMDB = REAL_LMDB_CACHE / "part_00/train/data0000.aselmdb"


def config():
    return MuseConfig(d_model=16, n_layers=1, n_heads=4, d_ffn=16,
                      embed_dim=12, num_experts=1, max_seq_len=2048, dropout=0.0)


def stats():
    return FrameTargetStats(
        energy_mean=5.0, energy_std=2.0, force_mean=1.0, force_std=4.0,
        stress_mean=(0.0,), stress_std=(2.0,),
    )


def add_bytes(tf, name, payload):
    member = tarfile.TarInfo(name)
    member.type = tarfile.REGTYPE
    member.mode = 0o644
    member.size = len(payload)
    tf.addfile(member, io.BytesIO(payload))


def npy_bytes(array):
    handle = io.BytesIO()
    np.save(handle, array, allow_pickle=False)
    return handle.getvalue()


def make_lmdb(path, rows):
    """rows: [(sid, numbers)]，cell 固定 20Å 立方，坐标落在晶胞内部。"""
    import lmdb

    environment = lmdb.open(str(path), subdir=False, map_size=16 * 1024 * 1024)
    with environment.begin(write=True) as transaction:
        for row_id, (sid, numbers) in enumerate(rows, start=1):
            count = len(numbers)
            positions = []
            for index in range(count):
                positions += [
                    1.0 + (index % 3) * 1.1,
                    1.0 + ((index // 3) % 3) * 1.1,
                    1.0 + (index // 9) * 1.1,
                ]
            payload = {
                "numbers": {"__ndarray__": [[count], "int64", list(numbers)]},
                "positions": {"__ndarray__": [[count, 3], "float64", positions]},
                "cell": {"__ndarray__": [[3, 3], "float64",
                                          [20.0, 0, 0, 0, 20.0, 0, 0, 0, 20.0]]},
                "pbc": {"__ndarray__": [[3], "bool", [True, True, True]]},
                "unique_id": "temporary",
                "data": {"sid": sid},
            }
            transaction.put(str(row_id).encode(), zlib.compress(
                json.dumps(payload, separators=(",", ":")).encode()
            ))
    environment.close()
    return path


def make_record(record_id, sid, energy, member, lmdb_row_id, formula, atom_count):
    return {
        "id": record_id,
        "inputs": {"chemical_formula": formula, "atom_count": atom_count},
        "external_inputs": [{
            "id": record_id,
            "role": "periodic_structure",
            "format": "ase_lmdb",
            "selector": {
                "archive_member": member,
                "lmdb_row_id": lmdb_row_id,
                "sid": sid,
            },
            "source_path": "assets/omc_train_250802/part_00.tar",
        }],
        "labels": {"energy": [{"value": energy, "unit": "eV"}]},
    }


def write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return path


# (id, heap, frame_in_shard, atom_numbers, energy)
SHARD_A_FRAMES = [
    ("fa0", "HEAPA-1-gener-aa", 0, [1, 6], 10.0),
    ("fa1", "HEAPA-1-gener-aa", 1, [8], 20.0),
    ("fa2", "HEAPA-9-gener-cc", 2, [1, 6, 7], 30.0),
    ("fa3", "HEAPA-9-gener-cc", 3, [6, 6], 40.0),
]
SHARD_B_FRAMES = [
    ("fb0", "HEAPB-2-gener-bb", 0, [1, 1], 50.0),
    ("fb1", "HEAPB-2-gener-bb", 1, [6, 8], 60.0),
]
OFFSETS_A = np.asarray([0, 2, 3, 6, 8], dtype=np.int64)
FORCES_A = np.arange(24, dtype=np.float32).reshape(8, 3)
STRESS_A = np.arange(24, dtype=np.float32).reshape(4, 6) * np.float32(0.5)
OFFSETS_B = np.asarray([0, 2, 4], dtype=np.int64)
FORCES_B = 1000 + np.arange(12, dtype=np.float32).reshape(4, 3)
STRESS_B = 2000 + np.arange(12, dtype=np.float32).reshape(2, 6)


def build_world(root: Path):
    """构造临时 SQLite 索引 + aux tar + 结构 tar，返回关键路径与每帧元数据。"""
    aux = root / "aux.tar"
    with tarfile.open(aux, "w") as tf:
        add_bytes(tf, "train/data0000/forces.npy", npy_bytes(FORCES_A))
        add_bytes(tf, "train/data0000/ids.txt",
                  "".join(f"{row[0]}\n" for row in SHARD_A_FRAMES).encode())
        add_bytes(tf, "train/data0000/offsets.npy", npy_bytes(OFFSETS_A))
        add_bytes(tf, "train/data0000/stress.npy", npy_bytes(STRESS_A))
        add_bytes(tf, "train/data0001/forces.npy", npy_bytes(FORCES_B))
        add_bytes(tf, "train/data0001/ids.txt",
                  "".join(f"{row[0]}\n" for row in SHARD_B_FRAMES).encode())
        add_bytes(tf, "train/data0001/offsets.npy", npy_bytes(OFFSETS_B))
        add_bytes(tf, "train/data0001/stress.npy", npy_bytes(STRESS_B))

    lmdb_a = make_lmdb(root / "a.aselmdb", [
        (f"{heap}-{frame}", numbers) for _id, heap, frame, numbers, _e in SHARD_A_FRAMES
    ])
    lmdb_b = make_lmdb(root / "b.aselmdb", [
        (f"{heap}-{frame}", numbers) for _id, heap, frame, numbers, _e in SHARD_B_FRAMES
    ])
    archive = root / "assets/omc_train_250802/part_00.tar"
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w") as tf:
        tf.add(lmdb_a, arcname="train/data0000.aselmdb", recursive=False)
        tf.add(lmdb_b, arcname="train/data0001.aselmdb", recursive=False)

    records = []
    for member, frames in (("train/data0000.aselmdb", SHARD_A_FRAMES),
                           ("train/data0001.aselmdb", SHARD_B_FRAMES)):
        for record_id, heap, frame, numbers, energy in frames:
            records.append(make_record(
                record_id, f"{heap}-{frame}", energy, member, frame + 1,
                "C", len(numbers),
            ))
    train = write_jsonl(root / "train.jsonl", records)
    index = root / "index.sqlite3"
    build_index(train, [aux], index, expected_energy=6, expected_aux=6)

    connection = sqlite3.connect(f"file:{index}?mode=ro", uri=True)
    rows = connection.execute(
        """SELECT s.shard_id, s.member_prefix, a.frame_row, a.id, e.starting_heap, e.energy
           FROM aux_shards s JOIN aux_frames a ON a.shard_id=s.shard_id
           JOIN energy_records e ON e.id=a.id ORDER BY s.shard_id, a.frame_row"""
    ).fetchall()
    connection.close()
    frames = {
        row[3]: {"shard_id": row[0], "member_prefix": row[1], "frame_row": row[2],
                 "starting_heap": row[4], "energy": row[5]}
        for row in rows
    }
    return {"aux": aux, "archive": archive, "index": index, "frames": frames}


def sample_rows(frames, ids):
    return [{
        "id": record_id,
        "starting_heap": frames[record_id]["starting_heap"],
        "seed": 42,
        "priority": 0,
        "shard_id": frames[record_id]["shard_id"],
        "frame_row": frames[record_id]["frame_row"],
        "stratum_rank": 1,
        "selection_reason": "unit_test",
    } for record_id in ids]


class CrystalDatasetTest(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)
        self.world = build_world(self.root)
        self.frames = self.world["frames"]
        self.lmdb_cache = self.root / "lmdb-cache"

    def tearDown(self):
        self._temporary_directory.cleanup()

    def write_sample(self, ids, name="sample.jsonl"):
        return write_jsonl(self.root / name, sample_rows(self.frames, ids))

    def build(self, ids, cache_name="cache", **kwargs):
        sample = self.write_sample(ids)
        report = build_frame_cache(sample, self.world["index"], self.root / cache_name, **kwargs)
        return sample, self.root / cache_name, report

    def dataset(self, cache_dir, sample=None, **kwargs):
        kwargs.setdefault("stats", stats())
        return OMC25CrystalDataset(
            cache_dir, config(), external_root=self.root,
            lmdb_cache_root=self.lmdb_cache, sample_jsonl=sample, **kwargs
        )

    def expected_force(self, record_id):
        frame = self.frames[record_id]
        if frame["member_prefix"] == "train/data0000":
            row = frame["frame_row"]
            return FORCES_A[int(OFFSETS_A[row]):int(OFFSETS_A[row + 1])]
        row = frame["frame_row"]
        return FORCES_B[int(OFFSETS_B[row]):int(OFFSETS_B[row + 1])]

    def expected_stress(self, record_id):
        frame = self.frames[record_id]
        if frame["member_prefix"] == "train/data0000":
            return STRESS_A[frame["frame_row"]]
        return STRESS_B[frame["frame_row"]]

    def test_build_and_getitem_roundtrip_id_join_offsets_normalization(self):
        ids = ["fb1", "fa0", "fa3", "fa2", "fb0", "fa1"]  # 故意打乱抽样顺序
        sample, cache_dir, report = self.build(ids)
        self.assertEqual(report["frames"], 6)
        self.assertEqual(report["atoms"], 12)
        self.assertEqual(report["shards_built"], 2)
        self.assertEqual(report["shards_resumed"], 0)

        manifest = json.loads((cache_dir / "manifest.json").read_text())
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["selected_frames"], 6)
        shard_a = manifest["shards"][str(self.frames["fa0"]["shard_id"])]
        self.assertEqual(shard_a["frame_count"], 4)
        self.assertEqual(shard_a["atom_count"], 8)
        frames_path = cache_dir / shard_a["frames_file"]
        frame_records = [json.loads(line) for line in frames_path.read_text().splitlines()]
        self.assertEqual([row["id"] for row in frame_records], ["fa0", "fa1", "fa2", "fa3"])
        self.assertEqual([row["atom_start"] for row in frame_records], [0, 2, 3, 6])
        self.assertEqual([row["force_shape"] for row in frame_records],
                         [[2, 3], [1, 3], [3, 3], [2, 3]])
        self.assertTrue(all(len(row["target_digest"]) == 32 for row in frame_records))
        self.assertEqual([row["energy"] for row in frame_records], [10.0, 20.0, 30.0, 40.0])

        dataset = self.dataset(cache_dir, sample)
        self.assertEqual(len(dataset), 6)
        for row, record_id in enumerate(ids):  # master 顺序必须与抽样 JSONL 一致
            record, item = dataset.get(row)
            self.assertEqual(item["id"], record_id)
            self.assertEqual(item["starting_heap"], self.frames[record_id]["starting_heap"])
            self.assertEqual(record.domain, DOMAIN_CRYSTAL)
            self.assertEqual(record.n_real_tokens, len(self.expected_force(record_id)))
            energy_value, energy_mask = item["energy"]
            self.assertAlmostEqual(
                float(energy_value[0]), (self.frames[record_id]["energy"] - 5.0) / 2.0, places=6
            )
            self.assertTrue(bool(energy_mask[0]))
            force_value, force_mask = item["force"]
            np.testing.assert_allclose(
                force_value, (self.expected_force(record_id) - 1.0) / 4.0, atol=1e-6
            )
            self.assertTrue(force_mask.all())
            stress_value, stress_mask = item["stress"]
            np.testing.assert_allclose(
                stress_value, self.expected_stress(record_id) / 2.0, atol=1e-6
            )
            self.assertTrue(stress_mask.all())

    def test_element_reference_residual_target_and_hash_handshake(self):
        sample, cache_dir, _ = self.build(["fa0"])
        references = ElementReferences(
            tuple([None, 1.0, None, None, None, None, 6.0] + [None] * 112),
            intercept=0.5,
            sha256="a" * 64,
        )
        residual_stats = FrameTargetStats(
            energy_mean=0.5, energy_std=2.0, force_mean=1.0, force_std=4.0,
            stress_mean=(0.0,), stress_std=(2.0,),
            energy_mode=RESIDUAL_MODE, reference_sha256="a" * 64,
        )
        dataset = self.dataset(cache_dir, sample, stats=residual_stats, references=references)
        _record, item = dataset.get(0)
        self.assertEqual(item["energy_reference_ev"], 7.5)
        self.assertEqual(item["energy_residual_ev"], 2.5)
        self.assertEqual(float(item["energy"][0][0]), 1.0)
        with self.assertRaisesRegex(ValueError, "reference_sha256 不符"):
            self.dataset(
                cache_dir, sample,
                stats=FrameTargetStats(
                    energy_mean=0.5, energy_std=2.0, force_mean=1.0, force_std=4.0,
                    stress_mean=(0.0,), stress_std=(2.0,),
                    energy_mode=RESIDUAL_MODE, reference_sha256="b" * 64,
                ),
                references=references,
            )

    def test_exclusions_strict_matching_and_digest_handshake(self):
        suggested = self.root / "suggested_exclude_heaps.jsonl"
        write_jsonl(suggested, [{"starting_heap": "HEAPA-9-gener-cc",
                                 "reasons": ["triple_hit"]}])
        exclusions = load_exclusion_spec(suggested)
        sample, cache_dir, report = self.build(
            ["fb1", "fa0", "fa3", "fa2", "fb0", "fa1"], exclusions=exclusions
        )
        self.assertEqual(report["frames"], 4)
        self.assertEqual(report["excluded_rows"], 2)
        dataset = self.dataset(cache_dir, sample, exclusions=exclusions)
        present = {dataset.get(row)[1]["id"] for row in range(len(dataset))}
        self.assertEqual(present, {"fa0", "fa1", "fb0", "fb1"})

        with self.assertRaisesRegex(ValueError, "排除清单与缓存构建时不符"):
            self.dataset(cache_dir, sample)  # 静默省略排除清单必须报错
        with self.assertRaisesRegex(ValueError, "排除清单与缓存构建时不符"):
            self.dataset(cache_dir, sample,
                         exclusions=load_exclusion_spec(exclude_heaps=["HEAPB-2-gener-bb"]))

        with self.assertRaisesRegex(ValueError, "未命中任何抽样帧"):
            self.build(["fa0"], cache_name="unmatched",
                       exclusions=load_exclusion_spec(exclude_heaps=["NOPE-1-gener-zz"]))
        _, _, report = self.build(
            ["fa0"], cache_name="unmatched-ok",
            exclusions=load_exclusion_spec(exclude_heaps=["NOPE-1-gener-zz"]),
            allow_unmatched_exclusions=True,
        )
        self.assertEqual(report["frames"], 1)
        self.assertEqual(report["unmatched_exclusions"], ["NOPE-1-gener-zz"])

        bad = self.root / "bad_exclusions.jsonl"
        write_jsonl(bad, [{"reasons": ["no_key"]}])
        with self.assertRaisesRegex(ValueError, "starting_heap/id"):
            load_exclusion_spec(bad)

        exclusions_both = load_exclusion_spec(suggested, exclude_ids=["fb0"])
        _, cache_dir2, report2 = self.build(
            ["fb1", "fa0", "fa3", "fa2", "fb0", "fa1"], cache_name="cache-ids",
            exclusions=exclusions_both,
        )
        self.assertEqual(report2["frames"], 3)
        dataset2 = self.dataset(cache_dir2, exclusions=exclusions_both)
        self.assertEqual({dataset2.get(row)[1]["id"] for row in range(len(dataset2))},
                         {"fa0", "fa1", "fb1"})

    def test_cache_resume_reuse_and_sample_binding(self):
        ids = ["fa0", "fa1", "fb0"]
        sample, cache_dir, report = self.build(ids)
        forces_path = cache_dir / "shard_000001/forces.npy"
        before = forces_path.read_bytes()
        second = build_frame_cache(sample, self.world["index"], cache_dir)
        self.assertTrue(second["resumed_only"])
        self.assertEqual(second["shards_built"], 0)
        self.assertEqual(second["shards_resumed"], 2)
        self.assertEqual(forces_path.read_bytes(), before)

        import shutil
        shutil.rmtree(cache_dir / "shard_000002")
        third = build_frame_cache(sample, self.world["index"], cache_dir)
        self.assertEqual(third["shards_built"], 1)
        self.assertEqual(third["shards_resumed"], 1)
        dataset = self.dataset(cache_dir, sample)
        self.assertEqual(len(dataset), 3)

        self.write_sample(ids + ["fa2"])  # 覆盖同一文件 → 内容变化
        with self.assertRaisesRegex(ValueError, "抽样 JSONL 内容与已有缓存不符"):
            build_frame_cache(sample, self.world["index"], cache_dir)
        with self.assertRaisesRegex(ValueError, "抽样 JSONL 内容与已有缓存不符"):
            build_frame_cache(sample, self.world["index"], cache_dir,
                              exclusions=load_exclusion_spec(exclude_heaps=["HEAPA-1-gener-aa"]))

    def test_lmdb_extract_once_and_sid_guard(self):
        ids = ["fa0", "fa1", "fb0"]
        sample, cache_dir, _ = self.build(ids)
        real_extract = crystal_dataset.extract_lmdb_member
        with mock.patch.object(crystal_dataset, "extract_lmdb_member",
                               wraps=real_extract) as spy:
            dataset = self.dataset(cache_dir, sample)
            dataset.get(0)
            dataset.get(1)  # 同一 member 第二次取帧不再解包
            self.assertEqual(spy.call_count, 1)
            dataset.get(2)  # 另一 member 才再次解包
            self.assertEqual(spy.call_count, 2)

        existing = self.lmdb_cache / "part_00/train/data0000.aselmdb"
        self.assertTrue(existing.is_file())
        dataset2 = self.dataset(cache_dir, sample)
        with mock.patch.object(crystal_dataset, "extract_lmdb_member",
                               side_effect=AssertionError("不应重复解包")):
            record, item = dataset2.get(0)
        self.assertEqual(item["id"], "fa0")

        frames_path = cache_dir / "shard_000001/frames.jsonl"
        tampered = frames_path.read_text(encoding="utf-8").replace(
            '"sid":"HEAPA-1-gener-aa-0"', '"sid":"WRONG-1-gener-aa-0"'
        )
        frames_path.write_text(tampered, encoding="utf-8")
        dataset3 = self.dataset(cache_dir, sample)
        with self.assertRaisesRegex(ValueError, "sid"):
            dataset3.get(0)

    def test_digest_tamper_and_strict_targets(self):
        ids = ["fa0", "fa1"]
        sample, cache_dir, _ = self.build(ids)
        forces_path = cache_dir / "shard_000001/forces.npy"
        payload = bytearray(forces_path.read_bytes())
        payload[-1] ^= 0xFF
        forces_path.write_bytes(bytes(payload))
        dataset = self.dataset(cache_dir, sample)
        with self.assertRaisesRegex(ValueError, "digest"):
            dataset.get(1)
        relaxed = self.dataset(cache_dir, sample, verify_digests=False)
        relaxed.get(1)  # 显式关闭校验后才可使用

        forces = np.load(forces_path, allow_pickle=False)
        forces[0, 0] = np.nan
        np.save(forces_path, forces, allow_pickle=False)
        strict = self.dataset(cache_dir, sample, verify_digests=False, strict_targets=True)
        with self.assertRaisesRegex(ValueError, "非有限"):
            strict.get(0)
        lenient = self.dataset(cache_dir, sample, verify_digests=False, strict_targets=False)
        _, item = lenient.get(0)
        self.assertEqual(item["force"][1].tolist(), [False, True])
        self.assertEqual(float(item["force"][0][0, 0]), 0.0)
        self.assertTrue(bool(item["energy"][1][0]))

    def test_compute_frame_stats_only_allowed_frames(self):
        exclusions = load_exclusion_spec(exclude_heaps=["HEAPA-9-gener-cc"],
                                         exclude_ids=["fb0"])
        _, cache_dir, _ = self.build(["fa0", "fa1", "fa2", "fa3", "fb0", "fb1"],
                                     exclusions=exclusions)
        payload = compute_frame_stats(cache_dir, self.root / "stats.json")
        self.assertEqual(payload["frames"], 3)  # fa0, fa1, fb1
        self.assertEqual(payload["atoms"], 5)
        self.assertEqual(payload["energy"]["count"], 3)
        self.assertAlmostEqual(payload["energy"]["mean"], 30.0, places=9)
        self.assertNotAlmostEqual(payload["energy"]["mean"], 35.0, places=9)  # 非全量 6 帧
        self.assertAlmostEqual(payload["energy"]["std"],
                               float(np.std([10.0, 20.0, 60.0])), places=9)
        self.assertEqual(payload["forces"]["count"], 15)
        expected_forces = np.concatenate([
            FORCES_A[0:3].reshape(-1), FORCES_B[2:4].reshape(-1)
        ]).astype(np.float64)
        self.assertAlmostEqual(payload["forces"]["mean"], float(expected_forces.mean()), places=9)
        self.assertAlmostEqual(payload["forces"]["std"], float(expected_forces.std()), places=9)
        expected_stress = np.stack([STRESS_A[0], STRESS_A[1], STRESS_B[1]])
        np.testing.assert_allclose(payload["stress"]["mean"],
                                   expected_stress.mean(axis=0), atol=1e-6)
        np.testing.assert_allclose(payload["stress"]["std"],
                                   expected_stress.std(axis=0), atol=1e-6)
        target_stats = FrameTargetStats.from_mapping(payload)
        self.assertEqual(target_stats.energy_mean, payload["energy"]["mean"])
        self.assertEqual(len(target_stats.stress_std), 6)
        self.assertTrue((np.asarray(target_stats.stress_std) > 0).all())
        self.assertTrue((self.root / "stats.json").is_file())

    def test_collate_metadata_compatible_batch(self):
        ids = ["fa0", "fa2", "fb1"]
        sample, cache_dir, _ = self.build(ids)
        dataset = self.dataset(cache_dir, sample)
        records, items = [], []
        for row in range(len(dataset)):
            record, item = dataset.get(row)
            records.append(record)
            items.append(item)
        cfg = config()
        batch, targets = BatchBuilder(cfg).build(records, seed=7, items=items)
        self.assertEqual(tuple(targets["energy"].shape), (3,))
        self.assertEqual(tuple(targets["stress"].shape), (3, 6))
        length = batch["token_ids"].shape[1]
        self.assertEqual(tuple(targets["force"].shape), (3, length, 3))
        for index, record_id in enumerate(ids):
            count = len(self.expected_force(record_id))
            np.testing.assert_allclose(
                targets["force"][index, 1:count + 1].numpy(),
                (self.expected_force(record_id) - 1.0) / 4.0, atol=1e-5,
            )
            self.assertEqual(int(targets["force_mask"][index].sum().item()), count)
            np.testing.assert_allclose(
                targets["stress"][index].numpy(),
                self.expected_stress(record_id) / 2.0, atol=1e-5,
            )
            self.assertAlmostEqual(
                float(targets["energy"][index]),
                (self.frames[record_id]["energy"] - 5.0) / 2.0, places=5,
            )
        self.assertIn("mlm_labels", targets)
        self.assertTrue((batch["domain"] == DOMAIN_CRYSTAL).all())


@unittest.skipUnless(
    REAL_INDEX.is_file() and REAL_LMDB.is_file() and REAL_ARCHIVE.is_file(),
    "真实 OMC25 index / data0000 LMDB / 结构 tar 不可用",
)
class RealOMC25FrameSmokeTest(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)

    def tearDown(self):
        self._temporary_directory.cleanup()

    def _npy_member_view(self, tf, name):
        import ast as ast_module

        member = tf.getmember(name)
        stream = tf.extractfile(member)
        assert stream is not None
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError("不是 NPY")
        version = stream.read(2)
        length = int.from_bytes(stream.read(2 if version[0] == 1 else 4), "little")
        header = ast_module.literal_eval(stream.read(length).decode("latin1").strip())
        return stream, tuple(header["shape"]), np.dtype(header["descr"])

    def test_real_frames_end_to_end(self):
        connection = sqlite3.connect(f"file:{REAL_INDEX}?mode=ro", uri=True)
        connection.execute("PRAGMA query_only = ON")
        rows = connection.execute(
            """SELECT a.shard_id, a.frame_row, a.id, e.starting_heap, e.energy,
                      e.atom_count, e.sid, e.source_path, e.archive_member, e.lmdb_row_id,
                      a.atom_start, a.atom_end, s.tar_path
               FROM aux_shards s JOIN aux_frames a ON a.shard_id=s.shard_id
               JOIN energy_records e ON e.id=a.id
               WHERE s.member_prefix='train/data0000'
                 AND e.archive_member='train/data0000.aselmdb'
                 AND a.frame_row % 500 = 0
               LIMIT 64"""
        ).fetchall()
        connection.close()
        self.assertTrue(rows)
        heaps = {}
        for row in rows:  # 取 8 个不同起始堆积的帧，保证统计方差非零
            heaps.setdefault(row[3], row)
        picked = list(heaps.values())[:8]
        self.assertGreaterEqual(len(picked), 4)
        order = [3, 0, 5, 1, 2, 4, 6, 7][:len(picked)]
        picked = [picked[index] for index in order]

        sample = self.root / "sample.jsonl"
        write_jsonl(sample, [{
            "id": row[2], "starting_heap": row[3], "seed": 42, "priority": 0,
            "shard_id": row[0], "frame_row": row[1], "stratum_rank": 1,
            "selection_reason": "smoke",
        } for row in picked])
        cache_dir = self.root / "cache"
        report = build_frame_cache(sample, REAL_INDEX, cache_dir)
        self.assertEqual(report["frames"], len(picked))

        # 独立交叉校验：不经缓存直接读 tar，逐字节比较 forces/stress。
        probe = picked[0]
        with tarfile.open(probe[12], "r:") as tf:
            stream, shape, dtype = self._npy_member_view(tf, "train/data0000/offsets.npy")
            offsets = np.frombuffer(stream.read(shape[0] * dtype.itemsize), dtype=dtype)
            fstream, fshape, fdtype = self._npy_member_view(tf, "train/data0000/forces.npy")
            sstream, sshape, sdtype = self._npy_member_view(tf, "train/data0000/stress.npy")
            atom_start, atom_end = int(offsets[probe[1]]), int(offsets[probe[1] + 1])
            self.assertEqual((atom_start, atom_end), (probe[10], probe[11]))
            fstream.seek(atom_start * 3 * fdtype.itemsize, io.SEEK_CUR)
            expected_forces = np.frombuffer(
                fstream.read((atom_end - atom_start) * 3 * fdtype.itemsize), fdtype
            ).reshape(-1, 3)
            sstream.seek(probe[1] * sshape[1] * sdtype.itemsize, io.SEEK_CUR)
            expected_stress = np.frombuffer(
                sstream.read(sshape[1] * sdtype.itemsize), sdtype
            )

        frame_stats = compute_frame_stats(cache_dir)
        target_stats = FrameTargetStats.from_mapping(frame_stats)
        dataset = OMC25CrystalDataset(
            cache_dir, config(), target_stats,
            external_root=REAL_EXTERNAL, lmdb_cache_root=REAL_LMDB_CACHE,
            sample_jsonl=sample,
        )
        self.assertEqual(len(dataset), len(picked))
        with mock.patch.object(crystal_dataset, "extract_lmdb_member",
                               side_effect=AssertionError("不应重复解包已存在的 LMDB")):
            first_record, first_item = dataset.get(0)
        self.assertEqual(dataset._lmdb_path(probe[7], probe[8]), REAL_LMDB)
        self.assertEqual(first_item["id"], probe[2])
        self.assertEqual(first_record.domain, DOMAIN_CRYSTAL)
        self.assertEqual(first_record.n_real_tokens, probe[5])
        # forces/stress 反归一化后必须与独立 tar 读取一致
        raw_forces = first_item["force"][0] * np.float32(target_stats.force_std) + np.float32(target_stats.force_mean)
        np.testing.assert_allclose(raw_forces, expected_forces, atol=1e-4)
        raw_stress = first_item["stress"][0] * target_stats.stress_arrays(6)[1] + target_stats.stress_arrays(6)[0]
        np.testing.assert_allclose(raw_stress, expected_stress, atol=1e-3)
        self.assertTrue(np.isfinite(first_item["force"][0]).all())
        energy_z = (probe[4] - target_stats.energy_mean) / target_stats.energy_std
        self.assertAlmostEqual(float(first_item["energy"][0][0]), energy_z, places=5)

        records, items = [], []
        for row in range(len(dataset)):
            record, item = dataset.get(row)
            self.assertEqual(record.n_real_tokens, item["force"][0].shape[0])
            self.assertTrue(item["energy"][1][0])
            records.append(record)
            items.append(item)
        batch, targets = BatchBuilder(config()).build(records, seed=11, items=items)
        self.assertEqual(tuple(targets["energy"].shape), (len(picked),))
        self.assertEqual(tuple(targets["stress"].shape), (len(picked), 6))
        self.assertEqual(int(targets["force_mask"].sum().item()),
                         sum(row[5] for row in picked))

        with self.assertRaisesRegex(ValueError, "排除清单与缓存构建时不符"):
            OMC25CrystalDataset(
                cache_dir, config(), target_stats,
                external_root=REAL_EXTERNAL, lmdb_cache_root=REAL_LMDB_CACHE,
                exclusions=load_exclusion_spec(exclude_heaps=["X-1-gener-y"]),
            )


if __name__ == "__main__":
    unittest.main()
