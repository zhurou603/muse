from __future__ import annotations

import io
import json
import pickle
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muse_encoder.config import MuseConfig
from muse_encoder.featurize import RecordFeaturizer
from muse_encoder.structures import align_coordinates
from muse_encoder.pretrain.aux import AUX_SOURCES, AuxSource
from muse_encoder.pretrain.aux_index import AuxIndex
from muse_encoder.pretrain import molecule3d
from muse_encoder.pretrain.molecule3d import (
    Molecule3DStore,
    build_geom_index,
    build_sdf_index,
    read_geom_conformer_at,
    read_sdf_mol_at,
)

from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MUSE_ROOT = PROJECT_ROOT / "data/muse-a/MUSE-A"
REAL_QM9_SDF = MUSE_ROOT / "external_inputs/A_datasets/small_molecule/QM9/assets/qm9.sdf"
REAL_QMUGS_SDF = MUSE_ROOT / "external_inputs/A_datasets/small_molecule/QMugs/assets/qmugs.sdf"
REAL_GEOM_DIR = MUSE_ROOT / "external_inputs/A_datasets/small_molecule/GEOM-Drugs/assets/rdkit_folder"


def _mol_with_coords(smiles, xyz):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None and mol.GetNumAtoms() == len(xyz)
    conf = Chem.Conformer(len(xyz))
    for i, point in enumerate(xyz):
        conf.SetAtomPosition(i, tuple(float(v) for v in point))
    mol.AddConformer(conf)
    return mol


ETHANOL_XYZ = np.array([[0.0, 0.0, 0.0], [1.4, 0.2, 0.0], [2.2, 1.1, 0.3]])
METHANE_XYZ = np.array([[0.1, -0.2, 0.3]])
PROPANE_XYZ = np.array([[0.0, 0.0, 0.0], [1.5, 0.1, 0.0], [2.6, -0.4, 0.2]])


def _write_sdf(path, entries):
    """entries: [(record_id, mol)]；按官方格式写出（首行名称 + mol block + $$$$）。"""
    writer = Chem.SDWriter(str(path))
    for record_id, mol in entries:
        mol.SetProp("_Name", record_id)
        writer.write(mol)
    writer.close()


def _aux_record(record_id, smiles, ref):
    return {
        "schema_version": "muse.final.v1",
        "id": record_id,
        "sample_type": "molecule",
        "inputs": {"smiles": smiles},
        "external_inputs": [ref],
        "labels": {},
    }


def _sdf_ref(record_id, source_path="assets/test.sdf"):
    return {
        "id": record_id,
        "role": "molecular_structure",
        "format": "sdf",
        "selector": {"record_id": record_id},
        "source_path": source_path,
    }


def _geom_ref(record_id, member, conformer_index, geom_id, part="part_00.tar"):
    return {
        "id": record_id,
        "role": "molecular_conformer",
        "format": "rdkit_pickle",
        "selector": {
            "archive_member": member,
            "conformer_index": conformer_index,
            "geom_id": geom_id,
        },
        "source_path": f"assets/rdkit_folder/{part}",
    }


def _sha256(path):
    import hashlib

    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


class _TempAux:
    """构造临时 aux JSONL + hits + AuxIndex（patch AUX_SOURCES 注册表）。"""

    def __init__(self, case, tmp, name, records, hits=()):
        root = Path(tmp)
        aux_path = root / f"{name}.aux.jsonl"
        with aux_path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")
        hits_path = root / f"{name}.hits.jsonl"
        with hits_path.open("w", encoding="utf-8") as stream:
            for hit in hits:
                stream.write(json.dumps({"id": hit}) + "\n")
        self.spec = AuxSource(name, (aux_path,), hits_path,
                              len(records), len(hits), len(records) - len(hits))
        self._patcher = patch.dict(
            "muse_encoder.pretrain.aux_index.AUX_SOURCES", {name: self.spec}, clear=False
        )
        self._patcher.start()
        case.addCleanup(self._patcher.stop)
        self.index = AuxIndex(name, root / f"{name}.locations.npy")
        case.addCleanup(self.index.close)


def _build_sdf_fixture(case, tmp, sdf_entries, aux_records, name="toy_sdf", hits=()):
    root = Path(tmp)
    ext = root / "ext"
    (ext / "assets").mkdir(parents=True)
    _write_sdf(ext / "assets/test.sdf", sdf_entries)
    aux = _TempAux(case, tmp, name, aux_records, hits)
    index_dir = root / "idx"
    return aux, ext, index_dir


class SdfIndexTests(unittest.TestCase):
    def _records(self):
        return [
            _aux_record("mol-0", "C", _sdf_ref("mol-0")),
            _aux_record("mol-1", "CCO", _sdf_ref("mol-1")),
            _aux_record("mol-2", "CCC", _sdf_ref("mol-2")),
        ]

    def _sdf_entries(self, ethanol_mol=None, propane_mol=None):
        return [
            ("mol-0", _mol_with_coords("C", METHANE_XYZ)),
            ("mol-1", ethanol_mol or _mol_with_coords("CCO", ETHANOL_XYZ)),
            ("mol-2", propane_mol or _mol_with_coords("CCC", PROPANE_XYZ)),
        ]

    def test_build_and_random_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            aux, ext, index_dir = _build_sdf_fixture(
                self, tmp, self._sdf_entries(), self._records()
            )
            path = build_sdf_index(aux.spec, index_dir, aux_index=aux.index, base=ext)
            self.assertEqual(path, index_dir / "toy_sdf.records.npy")
            with Molecule3DStore(aux.spec, index_dir, aux_index=aux.index, base=ext) as store:
                self.assertEqual(len(store), 3)
                one = store.get(1)
                self.assertEqual(one.id, "mol-1")
                self.assertEqual(one.smiles, "CCO")
                np.testing.assert_allclose(one.coords, ETHANOL_XYZ, rtol=0, atol=1e-12)
                self.assertEqual(one.mapping, (0, 1, 2))
                self.assertEqual(one.provenance["record_id"], "mol-1")
                self.assertEqual(one.provenance["n_atoms"], 3)
                two = store.get_by_id("mol-2")
                np.testing.assert_allclose(two.coords, PROPANE_XYZ, rtol=0, atol=1e-12)
                self.assertEqual(store.get(2).id, two.id)
                # 抽样确定性
                a = store.sample(2, seed=7)
                b = store.sample(2, seed=7)
                self.assertEqual([s.id for s in a], [s.id for s in b])
                self.assertEqual(len({s.id for s in a}), 2)

    def test_reversed_atom_order_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            shuffled = Chem.RenumberAtoms(_mol_with_coords("CCO", ETHANOL_XYZ), [2, 1, 0])
            self.assertFalse(
                np.allclose(shuffled.GetConformer().GetPositions(), ETHANOL_XYZ)
            )
            aux, ext, index_dir = _build_sdf_fixture(
                self, tmp, self._sdf_entries(ethanol_mol=shuffled), self._records()
            )
            build_sdf_index(aux.spec, index_dir, aux_index=aux.index, base=ext)
            with Molecule3DStore(aux.spec, index_dir, aux_index=aux.index, base=ext) as store:
                got = store.get_by_id("mol-1")
                self.assertEqual(got.mapping, (2, 1, 0))
                np.testing.assert_allclose(got.coords, ETHANOL_XYZ, rtol=0, atol=1e-12)
                self.assertEqual(got.provenance["heavy_atom_mapping"], [2, 1, 0])

    def test_featurizer_accepts_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            aux, ext, index_dir = _build_sdf_fixture(
                self, tmp, self._sdf_entries(), self._records()
            )
            build_sdf_index(aux.spec, index_dir, aux_index=aux.index, base=ext)
            with Molecule3DStore(aux.spec, index_dir, aux_index=aux.index, base=ext) as store:
                structure = store.get(1)
            featurizer = RecordFeaturizer(MuseConfig())
            record = featurizer.featurize_molecule(**structure.featurizer_kwargs())
            self.assertEqual(record.n_real_tokens, 3)
            centered = ETHANOL_XYZ - ETHANOL_XYZ.mean(axis=0, keepdims=True)
            np.testing.assert_allclose(
                record.coords[1:], centered.astype(np.float32), rtol=1e-6, atol=1e-6
            )
            self.assertTrue(record.coord_mask[1:].all())
            self.assertEqual(record.meta["sdf_mapping"], None)

    def test_unparseable_sdf_record_raises_at_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ext = root / "ext"
            (ext / "assets").mkdir(parents=True)
            mol = _mol_with_coords("C", METHANE_XYZ)
            mol.SetProp("_Name", "mol-ok")
            good_block = Chem.MolToMolBlock(mol)
            bad_lines = good_block.split("\n")
            bad_lines[0] = "mol-bad"
            bad_lines[3] = "  9  9  0  0  0  0            999 V2000"  # 声称9原子9键
            (ext / "assets/test.sdf").write_text(
                good_block + "$$$$\n" + "\n".join(bad_lines) + "$$$$\n",
                encoding="utf-8",
            )
            records = [
                _aux_record("mol-ok", "C", _sdf_ref("mol-ok")),
                _aux_record("mol-bad", "C", _sdf_ref("mol-bad")),
            ]
            aux = _TempAux(self, tmp, "toy_bad_sdf", records)
            # 名称级索引照常建立；严格解析失败在读取时抛出
            build_sdf_index(aux.spec, root / "idx", aux_index=aux.index, base=ext)
            with Molecule3DStore(aux.spec, root / "idx",
                                 aux_index=aux.index, base=ext) as store:
                np.testing.assert_allclose(
                    store.get_by_id("mol-ok").coords, METHANE_XYZ, rtol=0, atol=1e-12
                )
                with self.assertRaisesRegex(ValueError, "严格解析失败"):
                    store.get_by_id("mol-bad")

    def test_missing_structure_fails_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._records() + [_aux_record("mol-x", "CO", _sdf_ref("mol-x"))]
            aux, ext, index_dir = _build_sdf_fixture(
                self, tmp, self._sdf_entries(), records
            )
            with self.assertRaisesRegex(ValueError, "SDF 中不存在 record_id mol-x"):
                build_sdf_index(aux.spec, index_dir, aux_index=aux.index, base=ext)
            self.assertFalse((index_dir / "toy_sdf.records.npy").exists())

    def test_duplicate_sdf_record_id_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = self._sdf_entries() + [
                ("mol-1", _mol_with_coords("CCO", ETHANOL_XYZ))
            ]
            aux, ext, index_dir = _build_sdf_fixture(self, tmp, entries, self._records())
            with self.assertRaisesRegex(ValueError, "重复 SDF record_id mol-1"):
                build_sdf_index(aux.spec, index_dir, aux_index=aux.index, base=ext)

    def test_duplicate_aux_id_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._records() + [_aux_record("mol-0", "C", _sdf_ref("mol-0"))]
            aux, ext, index_dir = _build_sdf_fixture(
                self, tmp, self._sdf_entries(), records
            )
            with self.assertRaisesRegex(ValueError, "重复 aux id mol-0"):
                build_sdf_index(aux.spec, index_dir, aux_index=aux.index, base=ext)

    def test_resume_and_atomic_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            aux, ext, index_dir = _build_sdf_fixture(
                self, tmp, self._sdf_entries(), self._records()
            )
            # 阶段 A 断点：只扫 2 条 SDF 记录
            self.assertIsNone(build_sdf_index(
                aux.spec, index_dir, aux_index=aux.index, base=ext, sdf_scan_limit=2
            ))
            self.assertFalse((index_dir / "toy_sdf.json").exists())
            state = json.loads((index_dir / "toy_sdf.structures.state.json").read_text())
            self.assertEqual(state["records"], 2)
            self.assertFalse(state["done"])
            # 续跑完成阶段 A，阶段 B 限制 1 条
            self.assertIsNone(build_sdf_index(
                aux.spec, index_dir, aux_index=aux.index, base=ext, aux_limit=1
            ))
            rstate = json.loads((index_dir / "toy_sdf.records.state.json").read_text())
            self.assertEqual(rstate["next_row"], 1)
            # 完整续跑后原子发布，断点文件被清理
            path = build_sdf_index(aux.spec, index_dir, aux_index=aux.index, base=ext)
            self.assertTrue((index_dir / "toy_sdf.json").exists())
            for leftover in index_dir.glob("*.tmp.*"):
                self.fail(f"发布后仍残留断点文件 {leftover}")
            meta = json.loads((index_dir / "toy_sdf.json").read_text())
            self.assertEqual(meta["records"], 3)
            self.assertEqual(meta["kind"], "sdf")
            # 已发布索引重复构建是幂等 no-op
            self.assertEqual(build_sdf_index(
                aux.spec, index_dir, aux_index=aux.index, base=ext
            ), path)

    def test_unpublished_store_and_bad_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            aux, ext, index_dir = _build_sdf_fixture(
                self, tmp, self._sdf_entries(), self._records()
            )
            with self.assertRaisesRegex(FileNotFoundError, "未发布"):
                Molecule3DStore(aux.spec, index_dir, aux_index=aux.index, base=ext)
            build_sdf_index(aux.spec, index_dir, aux_index=aux.index, base=ext)
            with Molecule3DStore(aux.spec, index_dir, aux_index=aux.index, base=ext) as store:
                with self.assertRaises(IndexError):
                    store.get(3)
                with self.assertRaises(IndexError):
                    store.get(-1)
                with self.assertRaises(KeyError):
                    store.get_by_id("mol-404")
                with self.assertRaises(ValueError):
                    store.sample(4)


def _geom_pickle(smiles, conformers, charge=0):
    return pickle.dumps({"smiles": smiles, "charge": charge, "conformers": conformers})


def _geom_conformer(geom_id, mol):
    return {"geom_id": geom_id, "rd_mol": mol}


class _GeomFixture:
    """两个 parent、三个构象的临时 GEOM 结构（part_00 + part_01 两个 tar）。"""

    MEMBER_A = "rdkit_folder/drugs/CCO.pickle"
    MEMBER_B = "rdkit_folder/drugs/C.pickle"

    def __init__(self, case, tmp, name="toy_geom"):
        root = Path(tmp)
        self.base = root / "ext"
        folder = self.base / "assets/rdkit_folder"
        folder.mkdir(parents=True)
        ethanol_a = _mol_with_coords("CCO", ETHANOL_XYZ)
        ethanol_b = _mol_with_coords("CCO", ETHANOL_XYZ + np.array([10.0, 0.0, 0.0]))
        methane = _mol_with_coords("C", METHANE_XYZ)
        parts = [
            ("part_00.tar", [(self.MEMBER_A, _geom_pickle("CCO", [
                _geom_conformer(101, ethanol_a), _geom_conformer(102, ethanol_b),
            ]))]),
            ("part_01.tar", [(self.MEMBER_B, _geom_pickle("C", [
                _geom_conformer(201, methane),
            ]))]),
        ]
        manifest_parts = []
        for file_name, members in parts:
            tar_path = folder / file_name
            with tarfile.open(tar_path, "w") as tf:
                for member_name, payload in members:
                    info = tarfile.TarInfo(member_name)
                    info.size = len(payload)
                    tf.addfile(info, io.BytesIO(payload))
            manifest_parts.append({
                "file": file_name,
                "bytes": tar_path.stat().st_size,
                "sha256": _sha256(tar_path),
                "members": ["rdkit_folder", "rdkit_folder/drugs"]
                           + [m for m, _ in members],
            })
        (folder / "parts.json").write_text(json.dumps({
            "schema_version": "muse.external_parts.v1",
            "source": "toy", "limit_bytes": 0, "parts": manifest_parts,
        }), encoding="utf-8")
        self.records = [
            _aux_record("conf-a0", "CCO", _geom_ref("conf-a0", self.MEMBER_A, 0, 101)),
            _aux_record("conf-a1", "CCO", _geom_ref("conf-a1", self.MEMBER_A, 1, 102)),
            _aux_record("conf-b0", "C", _geom_ref("conf-b0", self.MEMBER_B, 0, 201,
                                                  part="part_01.tar")),
        ]
        self.aux = _TempAux(case, tmp, name, self.records)
        self.index_dir = root / "idx"

    def build(self, **kwargs):
        return build_geom_index(self.aux.spec, self.index_dir,
                                aux_index=self.aux.index, base=self.base, **kwargs)

    def store(self):
        return Molecule3DStore(self.aux.spec, self.index_dir,
                               aux_index=self.aux.index, base=self.base)


class GeomIndexTests(unittest.TestCase):
    def test_build_and_conformer_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _GeomFixture(self, tmp)
            path = fixture.build()
            self.assertEqual(path, fixture.index_dir / "toy_geom.records.npy")
            with fixture.store() as store:
                self.assertEqual(len(store), 3)
                conf0 = store.get(0)
                conf1 = store.get(1)
                # 同 parent 的不同构象必须各取各的坐标
                np.testing.assert_allclose(conf0.coords, ETHANOL_XYZ, rtol=0, atol=1e-6)
                np.testing.assert_allclose(
                    conf1.coords, ETHANOL_XYZ + np.array([10.0, 0.0, 0.0]),
                    rtol=0, atol=1e-6,
                )
                self.assertEqual(conf0.provenance["geom_id"], 101)
                self.assertEqual(conf1.provenance["geom_id"], 102)
                self.assertEqual(conf1.provenance["conformer_index"], 1)
                self.assertEqual(conf0.provenance["archive_member"], _GeomFixture.MEMBER_A)
                other = store.get_by_id("conf-b0")
                self.assertEqual(other.provenance["source_path"],
                                 "assets/rdkit_folder/part_01.tar")
                np.testing.assert_allclose(other.coords, METHANE_XYZ, rtol=0, atol=1e-6)
                meta = store.meta
                self.assertTrue(meta["parts_sha256_verified"])
                self.assertEqual(meta["members"], 2)

    def test_parent_pickle_lru_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _GeomFixture(self, tmp)
            fixture.build()
            with Molecule3DStore(fixture.aux.spec, fixture.index_dir,
                                 aux_index=fixture.aux.index, base=fixture.base,
                                 parent_cache_size=4) as store:
                store.get_by_id("conf-a0")
                store.get_by_id("conf-a1")  # 同 parent，第二次应命中缓存
                store.get_by_id("conf-b0")
                self.assertEqual(store.cache_stats, {"hits": 1, "misses": 2})
            with Molecule3DStore(fixture.aux.spec, fixture.index_dir,
                                 aux_index=fixture.aux.index, base=fixture.base,
                                 parent_cache_size=0) as store:
                np.testing.assert_allclose(
                    store.get_by_id("conf-a1").coords,
                    ETHANOL_XYZ + np.array([10.0, 0.0, 0.0]), rtol=0, atol=1e-6,
                )
                store.get_by_id("conf-a0")
                self.assertEqual(store.cache_stats, {"hits": 0, "misses": 2})

    def test_conformer_index_out_of_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _GeomFixture(self, tmp)
            fixture.records.append(_aux_record(
                "conf-oob", "CCO", _geom_ref("conf-oob", _GeomFixture.MEMBER_A, 5, 105)
            ))
            fixture.aux = _TempAux(self, tmp, "toy_oob", fixture.records)
            # 越界只有在真实读取构象时才能发现：构建通过、读取拒绝
            build_geom_index(fixture.aux.spec, fixture.index_dir,
                             aux_index=fixture.aux.index, base=fixture.base)
            with Molecule3DStore(fixture.aux.spec, fixture.index_dir,
                                 aux_index=fixture.aux.index, base=fixture.base) as store:
                with self.assertRaisesRegex(ValueError, "越界（共 2 个）"):
                    store.get_by_id("conf-oob")
                np.testing.assert_allclose(
                    store.get_by_id("conf-a0").coords, ETHANOL_XYZ, rtol=0, atol=1e-6
                )

    def test_geom_id_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _GeomFixture(self, tmp)
            records = [
                _aux_record("conf-bad", "CCO", _geom_ref("conf-bad", _GeomFixture.MEMBER_A, 0, 999))
            ]
            aux = _TempAux(self, tmp, "toy_bad", records)
            build_geom_index(aux.spec, fixture.index_dir, aux_index=aux.index,
                             base=fixture.base)
            with Molecule3DStore(aux.spec, fixture.index_dir,
                                 aux_index=aux.index, base=fixture.base) as store:
                with self.assertRaisesRegex(ValueError, "geom_id 不符"):
                    store.get(0)

    def test_missing_member_fails_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _GeomFixture(self, tmp)
            records = [_aux_record(
                "conf-missing", "CCO",
                _geom_ref("conf-missing", "rdkit_folder/drugs/Nope.pickle", 0, 1),
            )]
            aux = _TempAux(self, tmp, "toy_missing", records)
            with self.assertRaisesRegex(ValueError, "tar 中不存在成员"):
                build_geom_index(aux.spec, fixture.index_dir,
                                 aux_index=aux.index, base=fixture.base)

    def test_part_sha256_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _GeomFixture(self, tmp)
            parts_path = fixture.base / "assets/rdkit_folder/parts.json"
            manifest = json.loads(parts_path.read_text())
            manifest["parts"][0]["sha256"] = "0" * 64
            parts_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sha256 与 parts.json 不符"):
                fixture.build()
            # 显式跳过校验后可构建（仅用于已核验过的镜像）
            fixture.build(verify_part_sha256=False)
            with fixture.store() as store:
                self.assertFalse(store.meta["parts_sha256_verified"])
                self.assertEqual(store.get(0).provenance["geom_id"], 101)

    def test_resume_by_part(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _GeomFixture(self, tmp)
            self.assertIsNone(fixture.build(max_parts=1))
            state = json.loads(
                (fixture.index_dir / "toy_geom.members.state.json").read_text()
            )
            self.assertEqual(state["next_part"], 1)
            self.assertEqual(state["members"], 1)
            self.assertIsNotNone(fixture.build())
            with fixture.store() as store:
                self.assertEqual(len(store), 3)
                np.testing.assert_allclose(
                    store.get_by_id("conf-b0").coords, METHANE_XYZ, rtol=0, atol=1e-6
                )


@unittest.skipUnless(REAL_QM9_SDF.is_file(), "真实 QM9 SDF 不可用")
class RealQm9SmokeTests(unittest.TestCase):
    def test_full_build_and_cross_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            aux = AuxIndex("qm9")
            self.addCleanup(aux.close)
            index_dir = Path(tmp)
            path = build_sdf_index("qm9", index_dir, aux_index=aux)
            self.assertEqual(path, index_dir / "qm9.records.npy")
            with Molecule3DStore("qm9", index_dir, aux_index=aux) as store:
                self.assertEqual(len(store), AUX_SOURCES["qm9"].expected_kept)
                # 随机抽样 + 与底层独立读取交叉验证
                for item in store.sample(8, seed=0):
                    self.assertTrue(np.isfinite(item.coords).all())
                    self.assertTrue(item.id.startswith("muse__qm9__"))
                    self.assertEqual(item.provenance["record_id"],
                                     item.id)
                    offset = store._sdf_offset(item.provenance["record_id"])
                    with REAL_QM9_SDF.open("rb") as handle:
                        mol = read_sdf_mol_at(handle, offset, item.provenance["record_id"])
                    _, coords, mapping = align_coordinates(item.smiles, mol)
                    np.testing.assert_allclose(item.coords, coords, rtol=0, atol=0)
                    self.assertEqual(tuple(mapping), item.mapping)
                # get_by_id 与 get 行路径一致
                row = int(store.records["aux_row"][0])
                by_row = store.get(row)
                by_id = store.get_by_id(str(store.records["id"][0].decode("ascii")))
                np.testing.assert_allclose(by_row.coords, by_id.coords, rtol=0, atol=0)
            meta = json.loads((index_dir / "qm9.json").read_text())
            self.assertEqual(meta["records"], AUX_SOURCES["qm9"].expected_kept)
            self.assertGreaterEqual(meta["structures"], meta["records"])
            self.assertTrue(meta["external_files"][0]["sha256"])


@unittest.skipUnless(REAL_QMUGS_SDF.is_file(), "真实 QMugs SDF 不可用")
class RealQmugsSmokeTests(unittest.TestCase):
    def test_partial_scan_resume_and_align(self):
        with tempfile.TemporaryDirectory() as tmp:
            aux = AuxIndex("qmugs")
            self.addCleanup(aux.close)
            index_dir = Path(tmp)
            # 部分扫描 → 断点；扩大 limit 续跑
            self.assertIsNone(build_sdf_index(
                "qmugs", index_dir, aux_index=aux, sdf_scan_limit=256
            ))
            state = json.loads((index_dir / "qmugs.structures.state.json").read_text())
            self.assertEqual(state["records"], 256)
            first_offset = state["next_offset"]
            self.assertIsNone(build_sdf_index(
                "qmugs", index_dir, aux_index=aux, sdf_scan_limit=256
            ))
            state = json.loads((index_dir / "qmugs.structures.state.json").read_text())
            self.assertEqual(state["records"], 512)
            self.assertGreater(state["next_offset"], first_offset)
            entries = {
                e["record_id"]: e["offset"]
                for e in molecule3d._read_jsonl(index_dir / "qmugs.structures.tmp.jsonl")
            }
            # 前若干条 aux 记录应对齐成功且 provenance 正确
            aligned = 0
            with REAL_QMUGS_SDF.open("rb") as handle:
                for row in range(512):
                    record = aux.get(row)
                    record_id = record.payload["external_inputs"][0]["selector"]["record_id"]
                    if record_id not in entries:
                        continue
                    mol = read_sdf_mol_at(handle, entries[record_id], record_id)
                    graph, coords, mapping = align_coordinates(
                        record.payload["inputs"]["smiles"], mol
                    )
                    self.assertTrue(np.isfinite(coords).all())
                    self.assertEqual(graph.GetNumAtoms(), len(mapping))
                    aligned += 1
                    if aligned >= 4:
                        break
            self.assertEqual(aligned, 4)


@unittest.skipUnless(REAL_GEOM_DIR.is_dir(), "真实 GEOM-Drugs tar 不可用")
class RealGeomSmokeTests(unittest.TestCase):
    def test_real_member_read_and_conformer_isolation(self):
        import rdkit

        self.assertTrue(rdkit.__version__.startswith("2026"))
        part = REAL_GEOM_DIR / "part_00.tar"
        manifest = json.loads((REAL_GEOM_DIR / "parts.json").read_text())
        declared = [m for m in manifest["parts"][0]["members"] if m.endswith(".pickle")]
        scanned = []
        with tarfile.open(part, "r:") as tf:
            for member in tf:
                if member.isfile() and member.name.endswith(".pickle"):
                    scanned.append((member.name, member.offset_data, member.size))
                if len(scanned) >= 1024:
                    break
        # tar 头扫描顺序与 parts.json 清单一致
        self.assertEqual([s[0] for s in scanned], declared[:1024])

        aux = AuxIndex("geom_drugs")
        self.addCleanup(aux.close)
        by_member = {}
        for row in range(1024):
            record = aux.get(row)
            selector = record.payload["external_inputs"][0]["selector"]
            by_member.setdefault(selector["archive_member"], []).append((row, record))
        read = 0
        with part.open("rb") as handle:
            for name, offset, size in scanned:
                if read >= 16:
                    break
                for row, record in by_member.get(name, []):
                    if read >= 16:
                        break
                    selector = record.payload["external_inputs"][0]["selector"]
                    mol, obj = read_geom_conformer_at(
                        handle, offset, size,
                        selector["conformer_index"], selector["geom_id"],
                    )
                    graph, coords, mapping = align_coordinates(
                        record.payload["inputs"]["smiles"], mol
                    )
                    self.assertTrue(np.isfinite(coords).all())
                    self.assertEqual(
                        obj["conformers"][selector["conformer_index"]]["geom_id"],
                        selector["geom_id"],
                    )
                    read += 1
        self.assertGreaterEqual(read, 8)


if __name__ == "__main__":
    unittest.main()
