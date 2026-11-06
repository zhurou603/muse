"""官方分子 3D 结构的随机访问读取，供正式预训练 S2 使用。

覆盖三类小分子 aux 源：
- QM9 / QMugs：官方 SDF（record 首行名称 = selector.record_id）；
- GEOM-Drugs：rdkit_pickle（tar 分包，selector 含
  source_path/archive_member/conformer_index/geom_id）。

设计约束：
- 坐标一律来自官方文件，经 structures.align_coordinates 完成 SMILES 图与
  官方结构的重原子/键/立体/原子属性对齐；绝不生成构象替代官方坐标。
- 索引按 aux 顶层 id 与 SDF record_id / GEOM selector 建立，不做行号猜测。
- 索引构建可断点续跑（checkpoint 文件），完成并校验后以 os.replace 原子
  发布；发布物为零拷贝 mmap 的排序数组（id→aux_row、record_id→offset），
  支持 O(log n) 随机读取与抽样验证。

失败语义：构建阶段只核验 record_id/成员存在性与 selector 形态；严格解析与
对齐在读取时进行（官方坐标逐条还原），任何失败都抛异常，绝不静默跳过或
降级。个别保留记录可能因官方结构无法通过严格 sanitize（如价态超限）而在
读取时被拒绝——调用方应把每条记录的异常视为该记录不可用，而不是索引损坏。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import random
import tarfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..structures import align_coordinates, mol_from_sdf_bytes, require_rdkit
from .aux import AUX_ROOT, AUX_SOURCES, AuxSource, NamespacedRecord
from .aux_index import AuxIndex
from .corpora import PRETRAIN_ROOT, PROJECT_ROOT
from .trainer import sha256_file

EXTERNAL_ROOT = PROJECT_ROOT / "data/muse-a/MUSE-A/external_inputs/A_datasets"
INDEX_ROOT = PRETRAIN_ROOT / "index" / "molecule3d"

SDF_SOURCES = ("qm9", "qmugs")
PICKLE_SOURCES = ("geom_drugs",)
SDF_ROLE = "molecular_structure"
PICKLE_ROLE = "molecular_conformer"

ID_BYTES = 64  # muse 顶层 id / record_id 均为 ASCII，当前最长 62 字符
RECORDS_DTYPE = np.dtype([("id", f"S{ID_BYTES}"), ("aux_row", "<i8")])
SDF_STRUCT_DTYPE = np.dtype([("record_id", f"S{ID_BYTES}"), ("offset", "<i8")])
INDEX_SCHEMA = "muse.molecule3d.v1"

SourceRef = Union[str, AuxSource]

_CHECKPOINT_ROWS = 100_000


def _resolve_source(source: SourceRef) -> AuxSource:
    return AUX_SOURCES[source] if isinstance(source, str) else source


def external_base(source: SourceRef) -> Path:
    """由 aux 路径推导 external_inputs 根目录；自定义 AuxSource 需显式传入。"""
    spec = _resolve_source(source)
    bases = set()
    for path in spec.paths:
        try:
            rel = Path(path).resolve().relative_to(AUX_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(
                f"{spec.name}: aux 路径 {path} 不在 AUX_ROOT 下，"
                "请显式提供 external_base"
            ) from exc
        bases.add(rel.parent.parent)
    if len(bases) != 1:
        raise ValueError(f"{spec.name}: aux 文件不在同一数据集目录，无法确定 external base")
    base = EXTERNAL_ROOT / bases.pop()
    if not base.is_dir():
        raise FileNotFoundError(f"{spec.name}: external base 不存在: {base}")
    return base


def _resolve_external_file(base: Path, source_path: str) -> Path:
    base = Path(base).resolve()
    candidate = (base / source_path).resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError(f"source_path 越出 external base: {source_path}")
    return candidate


def _encode_id(value: str, what: str = "id") -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} 必须是非空字符串: {value!r}")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{what} 必须是 ASCII: {value!r}") from exc
    if len(encoded) > ID_BYTES:
        raise ValueError(f"{what} 超过 {ID_BYTES} 字节: {value!r}")
    return encoded


@dataclass(frozen=True)
class StructureRef:
    role: str
    format: str
    source_path: str
    selector: Mapping[str, Any]


def structure_ref(payload: Mapping[str, Any], role: str) -> StructureRef:
    """提取指定 role 的唯一 external_inputs 引用；数量不符即拒绝。"""
    refs = [r for r in payload.get("external_inputs") or () if r.get("role") == role]
    record_id = payload.get("id", "<unknown>")
    if len(refs) != 1:
        raise ValueError(f"{record_id}: role={role} 的结构引用数={len(refs)}")
    ref = refs[0]
    selector = ref.get("selector")
    if not isinstance(selector, Mapping):
        raise ValueError(f"{record_id}: 结构引用缺少 selector")
    return StructureRef(role, ref.get("format"), ref.get("source_path"), selector)


@dataclass(frozen=True)
class AlignedStructure:
    """一条对齐后的官方 3D 结构；coords 为原始参考系（中心化由 Featurizer 完成）。"""

    source: str
    id: str
    smiles: str
    coords: np.ndarray  # [n, 3] float64，按 SMILES 图序，未中心化
    mapping: Tuple[int, ...]  # target(SMILES 图序) -> source(官方结构) 原子下标
    provenance: Dict[str, Any] = field(default_factory=dict)

    def featurizer_kwargs(self) -> Dict[str, Any]:
        """直接展开给 RecordFeaturizer.featurize_molecule 的关键字参数。"""
        return {"smiles": self.smiles, "coords": self.coords}


def _official_graph_variant(smiles: str, mol):
    chem = require_rdkit()
    from rdkit.Chem import rdMolDescriptors
    from ..structures import graph_from_smiles
    target = graph_from_smiles(smiles)
    source = chem.RemoveAllHs(mol)
    if source.GetNumConformers() != 1:
        return None
    if (source.GetNumAtoms(), source.GetNumBonds()) != (
            target.GetNumAtoms(), target.GetNumBonds()):
        return None
    if rdMolDescriptors.CalcMolFormula(source) != rdMolDescriptors.CalcMolFormula(target):
        return None

    def connectivity(value):
        normalized = chem.RWMol(value)
        for atom in normalized.GetAtoms():
            atom.SetFormalCharge(0)
            atom.SetIsAromatic(False)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(True)
            atom.SetNumRadicalElectrons(0)
        for bond in normalized.GetBonds():
            bond.SetBondType(chem.BondType.SINGLE)
            bond.SetIsAromatic(False)
        return chem.MolToSmiles(normalized, isomericSmiles=False)

    if connectivity(source) != connectivity(target):
        return None
    mapping = tuple(
        int(atom.GetIdx()) for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    coords = np.asarray(source.GetConformer().GetPositions(), dtype=np.float64)
    if len(mapping) != source.GetNumAtoms() or coords.shape != (len(mapping), 3) \
            or not np.isfinite(coords).all():
        raise ValueError("官方构象图的原子映射或坐标非法")
    return source, coords, mapping


# ---------------------------------------------------------------------------
# 底层单点读取（构建器与 Store 共用）
# ---------------------------------------------------------------------------


def read_sdf_mol_at(handle, offset: int, record_id: str):
    """在 SDF 文件 offset 处读取单条记录，按首行名称严格核验 record_id。"""
    handle.seek(int(offset))
    block = []
    while True:
        line = handle.readline()
        if not line:
            raise ValueError(f"SDF offset={offset}: 记录缺少 $$$$ 分隔符")
        block.append(line)
        if line.strip() == b"$$$$":
            break
    name = block[0].decode("utf-8").strip()
    if name != record_id:
        raise ValueError(f"SDF offset={offset}: 记录名 {name!r} != 期望 {record_id!r}")
    return mol_from_sdf_bytes(b"".join(block), record_id, allow_unsanitized=True)


def load_geom_pickle(handle, offset: int, size: int):
    """读取 tar 成员字节并反序列化；兼容 RDKit 2026 直接还原官方 pickle。"""
    handle.seek(int(offset))
    data = handle.read(int(size))
    if len(data) != int(size):
        raise ValueError(f"tar 成员读取不完整: {len(data)} != {size}")
    try:
        obj = pickle.loads(data)
    except Exception as exc:
        raise ValueError(f"rdkit_pickle 反序列化失败: {exc}") from exc
    conformers = obj.get("conformers")
    if not isinstance(conformers, list) or not conformers:
        raise ValueError("rdkit_pickle 缺少非空 conformers 列表")
    return obj


def pick_geom_conformer(obj, conformer_index: int, geom_id: int):
    """从 parent pickle 对象中取指定构象；geom_id 双重核验，不混用其他构象。"""
    conformers = obj["conformers"]
    if isinstance(conformer_index, bool) or not 0 <= int(conformer_index) < len(conformers):
        raise ValueError(
            f"conformer_index {conformer_index} 越界（共 {len(conformers)} 个）"
        )
    entry = conformers[int(conformer_index)]
    if int(entry.get("geom_id", -1)) != int(geom_id):
        raise ValueError(
            f"geom_id 不符: 期望 {geom_id}，实际 {entry.get('geom_id')}"
        )
    mol = entry.get("rd_mol")
    if mol is None:
        raise ValueError("conformer 缺少 rd_mol")
    if mol.GetNumConformers() != 1:
        raise ValueError(f"构象数 {mol.GetNumConformers()} != 1")
    return mol


def read_geom_conformer_at(handle, offset: int, size: int,
                           conformer_index: int, geom_id: int):
    """从 tar 成员字节还原指定构象，返回 (rd_mol, pickle 对象)。"""
    obj = load_geom_pickle(handle, offset, size)
    return pick_geom_conformer(obj, conformer_index, geom_id), obj


# ---------------------------------------------------------------------------
# checkpoint 工具（写 entries → flush → 写 state；恢复时按 state 截断溢出）
# ---------------------------------------------------------------------------


def _state_load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _state_save(path: Path, state: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(dict(state), indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _load_checkpoint_entries(entries_path: Path, state: Mapping[str, Any],
                             count_key: str) -> List[Dict[str, Any]]:
    """读取断点 entries；若条目数多于 state（崩溃窗口），截断到 state 计数。"""
    entries = list(_read_jsonl(entries_path)) if entries_path.exists() else []
    expected = int(state[count_key])
    if len(entries) < expected:
        raise ValueError(
            f"{entries_path}: 断点条目 {len(entries)} 少于 state {expected}，文件损坏"
        )
    if len(entries) > expected:
        kept = entries[:expected]
        with entries_path.open("w", encoding="utf-8") as stream:
            for entry in kept:
                stream.write(json.dumps(entry, separators=(",", ":")) + "\n")
        entries = kept
    return entries


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _sorted_structured(keys: List[str], values: Sequence[int], dtype) -> np.ndarray:
    """按键排序成结构化数组；重复键（含排序后相邻重复）一律拒绝。"""
    if len(keys) != len(values):
        raise ValueError("keys/values 长度不一致")
    field0, field1 = dtype.names
    key_array = np.empty(len(keys), dtype=dtype[field0])
    for i, key in enumerate(keys):
        key_array[i] = _encode_id(key, field0)
    order = np.argsort(key_array, kind="stable")
    sorted_keys = key_array[order]
    if len(sorted_keys) > 1 and (sorted_keys[1:] == sorted_keys[:-1]).any():
        raise ValueError("存在重复键")
    output = np.empty(len(keys), dtype=dtype)
    output[field0] = sorted_keys
    output[field1] = np.asarray(values, dtype=np.int64)[order]
    return output


def _publish_array(array: np.ndarray, final: Path) -> None:
    tmp = final.with_suffix(".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, final)


def _publish_json(payload: Mapping[str, Any], final: Path) -> None:
    tmp = final.with_suffix(final.suffix + ".part")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, final)


# ---------------------------------------------------------------------------
# SDF 索引（QM9 / QMugs）
# ---------------------------------------------------------------------------


def _scan_sdf(sdf_path: Path, start_offset: int, seen: set,
              stream, scan_limit: Optional[int]) -> Tuple[int, int, bool]:
    """从 start_offset 继续扫描 SDF；返回 (next_offset, 新写入条数, 是否扫描完)。"""
    written = 0
    with sdf_path.open("rb") as handle:
        handle.seek(start_offset)
        while True:
            offset = handle.tell()
            first = handle.readline()
            if not first:
                return offset, written, True
            record_id = first.decode("utf-8").strip()
            _encode_id(record_id, "SDF record_id")
            while True:
                line = handle.readline()
                if not line:
                    raise ValueError(f"{sdf_path}:{offset}: 文件尾部记录不完整，缺少 $$$$")
                if line.strip() == b"$$$$":
                    break
            if record_id in seen:
                raise ValueError(f"{sdf_path}:{offset}: 重复 SDF record_id {record_id}")
            seen.add(record_id)
            stream.write(json.dumps(
                {"record_id": record_id, "offset": offset}, separators=(",", ":")
            ) + "\n")
            written += 1
            if scan_limit is not None and written >= scan_limit:
                stream.flush()
                return handle.tell(), written, False
            if written % 200_000 == 0:
                stream.flush()
                print(f"  {sdf_path.name}: 已扫描 {len(seen)} 条", flush=True)


def build_sdf_index(source: SourceRef, index_dir: Optional[Path] = None,
                    aux_index: Optional[AuxIndex] = None,
                    base: Optional[Path] = None,
                    sdf_scan_limit: Optional[int] = None,
                    aux_limit: Optional[int] = None) -> Optional[Path]:
    """构建 SDF 源的 molecule3d 索引；返回发布的 records.npy 路径。

    两阶段：A 扫描 SDF 得 record_id→offset；B 遍历 aux（经 AuxIndex）把
    顶层 id 与 selector.record_id 关联。两阶段均可断点续跑；全部完成并
    校验通过后原子发布。到达 *_limit 时保存断点并返回 None（供分批/冒烟）。
    """
    spec = _resolve_source(source)
    index_dir = Path(index_dir) if index_dir is not None else INDEX_ROOT / spec.name
    index_dir.mkdir(parents=True, exist_ok=True)
    records_npy = index_dir / f"{spec.name}.records.npy"
    meta_path = index_dir / f"{spec.name}.json"
    if meta_path.exists() and records_npy.exists():
        return records_npy
    aux_index = aux_index or AuxIndex(spec.name)
    base = Path(base) if base is not None else external_base(spec)

    # --- 阶段 A：SDF record_id -> offset ---
    struct_tmp = index_dir / f"{spec.name}.structures.tmp.jsonl"
    struct_state_path = index_dir / f"{spec.name}.structures.state.json"
    state = _state_load(struct_state_path) or {}
    seen = set()
    if state:
        for entry in _load_checkpoint_entries(struct_tmp, state, "records"):
            seen.add(entry["record_id"])
    sdf_done = bool(state.get("done"))
    if not sdf_done:
        ref0 = structure_ref(aux_index.get(0).payload, SDF_ROLE)
        if ref0.format != "sdf":
            raise ValueError(f"{spec.name}: 非 SDF 结构 {ref0.format}")
        sdf_path = _resolve_external_file(base, ref0.source_path)
        with struct_tmp.open("a", encoding="utf-8") as stream:
            next_offset, _, sdf_done = _scan_sdf(
                sdf_path, int(state.get("next_offset", 0)), seen, stream, sdf_scan_limit
            )
        _state_save(struct_state_path, {
            "next_offset": next_offset, "records": len(seen), "done": sdf_done,
            "sdf_path": str(sdf_path),
        })
        if not sdf_done:
            print(f"{spec.name}: SDF 扫描到达 limit，断点已保存", flush=True)
            return None
    else:
        sdf_path = Path(state["sdf_path"])

    # --- 阶段 B：aux 顶层 id 与 record_id 关联 ---
    records_tmp = index_dir / f"{spec.name}.records.tmp.jsonl"
    records_state_path = index_dir / f"{spec.name}.records.state.json"
    rstate = _state_load(records_state_path) or {}
    start_row = int(rstate.get("next_row", 0))
    ids: List[str] = []
    rows: List[int] = []
    if rstate:
        for entry in _load_checkpoint_entries(records_tmp, rstate, "records"):
            ids.append(entry["id"])
            rows.append(int(entry["aux_row"]))
    id_set = set(ids)
    total = len(aux_index)
    end_row = total if aux_limit is None else min(total, start_row + aux_limit)
    buffer: List[Dict[str, Any]] = []
    with records_tmp.open("a", encoding="utf-8") as stream:
        for row in range(start_row, end_row):
            aux = aux_index.get(row)
            payload = aux.payload
            ref = structure_ref(payload, SDF_ROLE)
            if ref.format != "sdf":
                raise ValueError(f"{aux.id}: 非 SDF 结构 {ref.format}")
            path = _resolve_external_file(base, ref.source_path)
            if path != sdf_path:
                raise ValueError(f"{aux.id}: source_path 不一致 {ref.source_path}")
            record_id = _encode_id(ref.selector.get("record_id"), "record_id").decode("ascii")
            if record_id not in seen:
                raise ValueError(f"{aux.id}: SDF 中不存在 record_id {record_id}")
            if aux.id in id_set:
                raise ValueError(f"{spec.name}: 重复 aux id {aux.id}")
            id_set.add(aux.id)
            ids.append(aux.id)
            rows.append(row)
            buffer.append({"id": aux.id, "aux_row": row})
            if len(buffer) >= _CHECKPOINT_ROWS:
                for entry in buffer:
                    stream.write(json.dumps(entry, separators=(",", ":")) + "\n")
                buffer.clear()
                stream.flush()
                _state_save(records_state_path, {"next_row": row + 1, "records": len(ids)})
        for entry in buffer:
            stream.write(json.dumps(entry, separators=(",", ":")) + "\n")
        stream.flush()
    _state_save(records_state_path, {"next_row": end_row, "records": len(ids)})
    if end_row < total:
        print(f"{spec.name}: aux 关联到达 limit ({end_row}/{total})，断点已保存", flush=True)
        return None

    # --- 校验与原子发布 ---
    if len(ids) != total or len(ids) != len(id_set):
        raise ValueError(f"{spec.name}: 记录数 {len(ids)} != aux 保留数 {total}")
    struct_entries = list(_read_jsonl(struct_tmp))
    struct_array = _sorted_structured(
        [e["record_id"] for e in struct_entries],
        [int(e["offset"]) for e in struct_entries],
        SDF_STRUCT_DTYPE,
    )
    record_array = _sorted_structured(ids, rows, RECORDS_DTYPE)
    _publish_array(struct_array, index_dir / f"{spec.name}.structures.npy")
    _publish_array(record_array, records_npy)
    metadata = {
        "schema": INDEX_SCHEMA,
        "source": spec.name,
        "kind": "sdf",
        "records": len(ids),
        "structures": len(struct_entries),
        "external_files": [{"source_path": str(sdf_path), "sha256": sha256_file(sdf_path)}],
        "aux_index": str(aux_index.index_path),
        "aux_index_sha256": sha256_file(aux_index.index_path),
        "aux_kept": total,
        "rdkit_version": require_rdkit().rdBase.rdkitVersion,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _publish_json(metadata, meta_path)
    for tmp in (struct_tmp, struct_state_path, records_tmp, records_state_path):
        if tmp.exists():
            tmp.unlink()
    print(f"{spec.name}: 索引已发布 records={len(ids)} structures={len(struct_entries)}",
          flush=True)
    return records_npy


# ---------------------------------------------------------------------------
# GEOM tar pickle 索引
# ---------------------------------------------------------------------------


def _geom_parts(base: Path) -> List[Dict[str, Any]]:
    parts_path = Path(base) / "assets/rdkit_folder/parts.json"
    if not parts_path.is_file():
        raise FileNotFoundError(f"GEOM 索引需要 {parts_path}")
    manifest = json.loads(parts_path.read_text(encoding="utf-8"))
    parts = manifest.get("parts")
    if not parts:
        raise ValueError(f"{parts_path}: parts 为空")
    return parts


def build_geom_index(source: SourceRef, index_dir: Optional[Path] = None,
                     aux_index: Optional[AuxIndex] = None,
                     base: Optional[Path] = None,
                     max_parts: Optional[int] = None,
                     aux_limit: Optional[int] = None,
                     verify_part_sha256: bool = True) -> Optional[Path]:
    """构建 GEOM-Drugs 的 molecule3d 索引；返回发布的 records.npy 路径。

    阶段 A 按 parts.json 顺序逐 part 扫描 tar 头，记录
    archive_member→(part, offset, size)（只读头并校验 sha256/大小/成员清单，
    不展开 pickle）；阶段 B 遍历 aux，校验每条 selector 的 source_path/
    archive_member/conformer_index/geom_id 形态并关联顶层 id。
    均可断点续跑，完成后原子发布。
    """
    spec = _resolve_source(source)
    index_dir = Path(index_dir) if index_dir is not None else INDEX_ROOT / spec.name
    index_dir.mkdir(parents=True, exist_ok=True)
    records_npy = index_dir / f"{spec.name}.records.npy"
    meta_path = index_dir / f"{spec.name}.json"
    if meta_path.exists() and records_npy.exists():
        return records_npy
    aux_index = aux_index or AuxIndex(spec.name)
    base = Path(base) if base is not None else external_base(spec)
    parts = _geom_parts(base)

    # --- 阶段 A：tar 成员偏移（以整个 part 为断点粒度） ---
    members_tmp = index_dir / f"{spec.name}.members.tmp.jsonl"
    members_state_path = index_dir / f"{spec.name}.members.state.json"
    state = _state_load(members_state_path) or {}
    start_part = int(state.get("next_part", 0))
    members: Dict[str, Tuple[int, int, int]] = {}
    if state:
        for entry in _load_checkpoint_entries(members_tmp, state, "members"):
            members[entry["archive_member"]] = (
                int(entry["part"]), int(entry["offset"]), int(entry["size"]),
            )
    part_names = [p["file"] for p in parts]
    end_part = len(parts) if max_parts is None else min(len(parts), max_parts)
    for part_index in range(start_part, end_part):
        part = parts[part_index]
        tar_path = _resolve_external_file(base, f"assets/rdkit_folder/{part['file']}")
        if os.path.getsize(tar_path) != int(part["bytes"]):
            raise ValueError(f"{tar_path}: 大小与 parts.json 不符")
        if verify_part_sha256 and sha256_file(tar_path) != part["sha256"]:
            raise ValueError(f"{tar_path}: sha256 与 parts.json 不符")
        declared = {m for m in part["members"] if m.endswith(".pickle")}
        found: List[Dict[str, Any]] = []
        with tarfile.open(tar_path, "r:") as tf:
            for member in tf:
                if not member.isfile() or not member.name.endswith(".pickle"):
                    continue
                if member.name in members:
                    raise ValueError(f"重复 archive_member: {member.name}")
                found.append({
                    "archive_member": member.name, "part": part_index,
                    "offset": int(member.offset_data), "size": int(member.size),
                })
        if {f["archive_member"] for f in found} != declared:
            missing = sorted(declared - {f["archive_member"] for f in found})[:5]
            extra = sorted({f["archive_member"] for f in found} - declared)[:5]
            raise ValueError(
                f"{tar_path}: 成员与 parts.json 不一致 missing={missing} extra={extra}"
            )
        _append = members_tmp.open("a", encoding="utf-8")
        with _append as stream:
            for entry in found:
                stream.write(json.dumps(entry, separators=(",", ":")) + "\n")
        for entry in found:
            members[entry["archive_member"]] = (entry["part"], entry["offset"], entry["size"])
        _state_save(members_state_path, {
            "next_part": part_index + 1, "members": len(members), "parts": part_names,
        })
        print(f"  {part['file']}: 成员 {len(found)}，累计 {len(members)}", flush=True)
    if end_part < len(parts):
        print(f"{spec.name}: tar 扫描到达 max_parts={max_parts}，断点已保存", flush=True)
        return None

    # --- 阶段 B：aux 顶层 id 关联与 selector 校验 ---
    records_tmp = index_dir / f"{spec.name}.records.tmp.jsonl"
    records_state_path = index_dir / f"{spec.name}.records.state.json"
    rstate = _state_load(records_state_path) or {}
    start_row = int(rstate.get("next_row", 0))
    ids: List[str] = []
    rows: List[int] = []
    if rstate:
        for entry in _load_checkpoint_entries(records_tmp, rstate, "records"):
            ids.append(entry["id"])
            rows.append(int(entry["aux_row"]))
    id_set = set(ids)
    total = len(aux_index)
    end_row = total if aux_limit is None else min(total, start_row + aux_limit)
    buffer = []
    prefix: Optional[str] = None
    with records_tmp.open("a", encoding="utf-8") as stream:
        for row in range(start_row, end_row):
            aux = aux_index.get(row)
            payload = aux.payload
            ref = structure_ref(payload, PICKLE_ROLE)
            if ref.format != "rdkit_pickle":
                raise ValueError(f"{aux.id}: 非 rdkit_pickle 结构 {ref.format}")
            selector = ref.selector
            member = selector.get("archive_member")
            if not isinstance(member, str) or not member:
                raise ValueError(f"{aux.id}: selector 缺少 archive_member")
            if member not in members:
                raise ValueError(f"{aux.id}: tar 中不存在成员 {member}")
            part_index = members[member][0]
            if prefix is None:
                prefix = str(Path(ref.source_path).parent)
            if str(Path(ref.source_path).parent) != prefix:
                raise ValueError(f"{aux.id}: source_path 前缀不一致 {ref.source_path}")
            if Path(ref.source_path).name != part_names[part_index]:
                raise ValueError(
                    f"{aux.id}: source_path {ref.source_path} 与成员所在 part "
                    f"{part_names[part_index]} 不符"
                )
            conformer_index = selector.get("conformer_index")
            geom_id = selector.get("geom_id")
            if isinstance(conformer_index, bool) or not isinstance(conformer_index, int) \
                    or conformer_index < 0:
                raise ValueError(f"{aux.id}: conformer_index 非法 {conformer_index!r}")
            if isinstance(geom_id, bool) or not isinstance(geom_id, int):
                raise ValueError(f"{aux.id}: geom_id 非法 {geom_id!r}")
            if aux.id in id_set:
                raise ValueError(f"{spec.name}: 重复 aux id {aux.id}")
            id_set.add(aux.id)
            ids.append(aux.id)
            rows.append(row)
            buffer.append({"id": aux.id, "aux_row": row})
            if len(buffer) >= _CHECKPOINT_ROWS:
                for entry in buffer:
                    stream.write(json.dumps(entry, separators=(",", ":")) + "\n")
                buffer.clear()
                stream.flush()
                _state_save(records_state_path, {"next_row": row + 1, "records": len(ids)})
                print(f"  {spec.name}: 已关联 {len(ids)}/{total}", flush=True)
        for entry in buffer:
            stream.write(json.dumps(entry, separators=(",", ":")) + "\n")
        stream.flush()
    _state_save(records_state_path, {"next_row": end_row, "records": len(ids)})
    if end_row < total:
        print(f"{spec.name}: aux 关联到达 limit ({end_row}/{total})，断点已保存", flush=True)
        return None

    if len(ids) != total or len(ids) != len(id_set):
        raise ValueError(f"{spec.name}: 记录数 {len(ids)} != aux 保留数 {total}")
    record_array = _sorted_structured(ids, rows, RECORDS_DTYPE)
    _publish_array(record_array, records_npy)
    _publish_json(
        {key: list(value) for key, value in members.items()},
        index_dir / f"{spec.name}.members.json",
    )
    metadata = {
        "schema": INDEX_SCHEMA,
        "source": spec.name,
        "kind": "rdkit_pickle",
        "records": len(ids),
        "members": len(members),
        "parts": part_names,
        "source_path_prefix": prefix,
        "parts_sha256": {p["file"]: p["sha256"] for p in parts},
        "parts_sha256_verified": bool(verify_part_sha256),
        "aux_index": str(aux_index.index_path),
        "aux_index_sha256": sha256_file(aux_index.index_path),
        "aux_kept": total,
        "rdkit_version": require_rdkit().rdBase.rdkitVersion,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _publish_json(metadata, meta_path)
    for tmp in (members_tmp, members_state_path, records_tmp, records_state_path):
        if tmp.exists():
            tmp.unlink()
    print(f"{spec.name}: 索引已发布 records={len(ids)} members={len(members)}",
          flush=True)
    return records_npy


def build_index(source: SourceRef, **kwargs) -> Optional[Path]:
    """按源类型分发到对应构建器。"""
    spec = _resolve_source(source)
    if spec.name in SDF_SOURCES:
        return build_sdf_index(spec, **kwargs)
    if spec.name in PICKLE_SOURCES:
        return build_geom_index(spec, **kwargs)
    raise ValueError(f"{spec.name}: 不是已知的 3D 分子源（{SDF_SOURCES + PICKLE_SOURCES}）")


# ---------------------------------------------------------------------------
# 随机访问 Store
# ---------------------------------------------------------------------------


class Molecule3DStore:
    """官方 3D 结构的只读随机访问。

    get(row) 按 aux 保留行号读取；get_by_id(id) 按 aux 顶层 id 读取；
    sample(count, seed) 抽样验证。所有坐标经 align_coordinates 对齐后输出。
    GEOM 源对 parent pickle 反序列化结果做有界 LRU 缓存（parent_cache_size，
    0 关闭）；缓存对象只读共享，命中统计见 cache_stats。
    """

    def __init__(self, source: SourceRef, index_dir: Optional[Path] = None,
                 aux_index: Optional[AuxIndex] = None, base: Optional[Path] = None,
                 parent_cache_size: int = 32):
        if isinstance(parent_cache_size, bool) or int(parent_cache_size) < 0:
            raise ValueError("parent_cache_size 必须是非负整数")
        self.spec = _resolve_source(source)
        self.index_dir = Path(index_dir) if index_dir is not None else INDEX_ROOT / self.spec.name
        meta_path = self.index_dir / f"{self.spec.name}.json"
        records_path = self.index_dir / f"{self.spec.name}.records.npy"
        if not meta_path.exists() or not records_path.exists():
            raise FileNotFoundError(
                f"{self.spec.name}: molecule3d 索引未发布于 {self.index_dir}，"
                "请先运行 build_index / build_sdf_index / build_geom_index"
            )
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if self.meta.get("schema") != INDEX_SCHEMA or self.meta.get("source") != self.spec.name:
            raise ValueError(f"{meta_path}: schema/source 不符")
        self.kind = self.meta["kind"]
        self.records = np.load(records_path, mmap_mode="r")
        if len(self.records) != int(self.meta["records"]):
            raise ValueError(f"{self.spec.name}: records.npy 条数与元数据不符")
        self._aux = aux_index or AuxIndex(self.spec.name)
        if len(self._aux) != len(self.records):
            raise ValueError(
                f"{self.spec.name}: molecule3d 索引 {len(self.records)} 条 "
                f"!= aux 索引 {len(self._aux)} 条"
            )
        self.base = Path(base) if base is not None else external_base(self.spec)
        # 句柄按线程隔离：并发 get 共享句柄会让 seek/read 交错（与 AuxIndex 一致）。
        self._local = threading.local()
        self._handles_lock = threading.Lock()
        self._all_handles: List[Any] = []
        self._parent_cache: "OrderedDict[Tuple[str, str], Any]" = OrderedDict()
        self.parent_cache_size = int(parent_cache_size)
        self.cache_stats = {"hits": 0, "misses": 0}
        if self.kind == "sdf":
            structures_path = self.index_dir / f"{self.spec.name}.structures.npy"
            self.structures = np.load(structures_path, mmap_mode="r")
            if len(self.structures) != int(self.meta["structures"]):
                raise ValueError(f"{self.spec.name}: structures.npy 条数与元数据不符")
            self._members = None
        elif self.kind == "rdkit_pickle":
            members_path = self.index_dir / f"{self.spec.name}.members.json"
            self._members = {
                key: tuple(value)
                for key, value in json.loads(members_path.read_text(encoding="utf-8")).items()
            }
            if len(self._members) != int(self.meta["members"]):
                raise ValueError(f"{self.spec.name}: members.json 条数与元数据不符")
            self.structures = None
        else:
            raise ValueError(f"{self.spec.name}: 未知索引类型 {self.kind}")

    def __len__(self) -> int:
        return len(self.records)

    def close(self) -> None:
        with self._handles_lock:
            for handle in self._all_handles:
                try:
                    handle.close()
                except OSError:
                    pass
            self._all_handles.clear()
        if hasattr(self._local, "handles"):
            self._local.handles.clear()
        if self._aux is not None:
            self._aux.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- 内部 --

    def _handle(self, path: Path):
        handles = getattr(self._local, "handles", None)
        if handles is None:
            handles = self._local.handles = {}
        handle = handles.get(path)
        if handle is None:
            handle = path.open("rb")
            handles[path] = handle
            with self._handles_lock:
                self._all_handles.append(handle)
        return handle

    def _geom_obj(self, path: Path, member: str, offset: int, size: int):
        """带 LRU 缓存的 parent pickle 反序列化；缓存对象只读共享。"""
        key = (str(path), member)
        with self._handles_lock:
            cached = self._parent_cache.get(key)
            if cached is not None:
                self._parent_cache.move_to_end(key)
                self.cache_stats["hits"] += 1
                return cached
            self.cache_stats["misses"] += 1
        obj = load_geom_pickle(self._handle(path), offset, size)
        with self._handles_lock:
            self._parent_cache[key] = obj
            while len(self._parent_cache) > self.parent_cache_size:
                self._parent_cache.popitem(last=False)
        return obj

    def _aux_row_for_id(self, record_id: str) -> int:
        key = _encode_id(record_id)
        pos = int(np.searchsorted(self.records["id"], key))
        if pos >= len(self.records) or self.records["id"][pos] != key:
            raise KeyError(f"{self.spec.name}: 索引中不存在 id {record_id}")
        return int(self.records["aux_row"][pos])

    def _sdf_offset(self, record_id: str) -> int:
        key = _encode_id(record_id)
        pos = int(np.searchsorted(self.structures["record_id"], key))
        if pos >= len(self.structures) or self.structures["record_id"][pos] != key:
            raise KeyError(f"{self.spec.name}: SDF 索引中不存在 record_id {record_id}")
        return int(self.structures["offset"][pos])

    @staticmethod
    def _smiles(payload: Mapping[str, Any]) -> str:
        smiles = (payload.get("inputs") or {}).get("smiles")
        if not isinstance(smiles, str) or not smiles:
            raise ValueError(f"{payload.get('id')}: inputs 缺少 smiles")
        return smiles

    def _align(self, aux: NamespacedRecord, expect_row: int) -> AlignedStructure:
        aux_row = self._aux_row_for_id(aux.id)
        if aux_row != expect_row:
            raise RuntimeError(
                f"{self.spec.name}: molecule3d 索引(aux_row={aux_row}) 与 "
                f"aux 索引(row={expect_row}) 不一致，需重建"
            )
        payload = aux.payload
        smiles = self._smiles(payload)

        def align_source():
            try:
                graph, coords, mapping = align_coordinates(smiles, mol)
                return graph, coords, mapping, smiles, "official_smiles", None
            except ValueError as exc:
                variant = _official_graph_variant(smiles, mol)
                if variant is None:
                    raise
                graph, coords, mapping = variant
                return (graph, coords, mapping, chem.MolToSmiles(graph, isomericSmiles=True),
                        "official_conformer", str(exc))

        chem = require_rdkit()
        if self.kind == "sdf":
            ref = structure_ref(payload, SDF_ROLE)
            if ref.format != "sdf":
                raise ValueError(f"{aux.id}: 非 SDF 结构 {ref.format}")
            record_id = _encode_id(ref.selector.get("record_id"), "record_id").decode("ascii")
            path = _resolve_external_file(self.base, ref.source_path)
            mol = read_sdf_mol_at(self._handle(path), self._sdf_offset(record_id), record_id)
            graph, coords, mapping, smiles, graph_source, alignment_note = align_source()
            extra = {"record_id": record_id}
        else:
            ref = structure_ref(payload, PICKLE_ROLE)
            if ref.format != "rdkit_pickle":
                raise ValueError(f"{aux.id}: 非 rdkit_pickle 结构 {ref.format}")
            selector = ref.selector
            member = selector.get("archive_member")
            if member not in self._members:
                raise KeyError(f"{aux.id}: tar 成员索引中不存在 {member}")
            part_index, offset, size = self._members[member]
            part_file = self.meta["parts"][part_index]
            if Path(ref.source_path).name != part_file:
                raise ValueError(f"{aux.id}: source_path 与成员 part 不符")
            path = _resolve_external_file(
                self.base, f"{self.meta['source_path_prefix']}/{part_file}"
            )
            conformer_index = selector.get("conformer_index")
            geom_id = selector.get("geom_id")
            if isinstance(conformer_index, bool) or not isinstance(conformer_index, int):
                raise ValueError(f"{aux.id}: conformer_index 非法 {conformer_index!r}")
            if isinstance(geom_id, bool) or not isinstance(geom_id, int):
                raise ValueError(f"{aux.id}: geom_id 非法 {geom_id!r}")
            obj = self._geom_obj(path, member, offset, size)
            mol = pick_geom_conformer(obj, conformer_index, geom_id)
            graph, coords, mapping, smiles, graph_source, alignment_note = align_source()
            inputs = payload.get("inputs") or {}
            extra = {
                "archive_member": member,
                "conformer_index": int(conformer_index),
                "geom_id": int(geom_id),
                "parent_molecule_id": inputs.get("parent_molecule_id"),
                "charge": inputs.get("charge"),
                "pickle_smiles": obj.get("smiles"),
            }
        extra["graph_source"] = graph_source
        if alignment_note:
            extra["alignment_note"] = alignment_note
        n = graph.GetNumAtoms()
        center = coords.mean(axis=0, keepdims=True)
        provenance = {
            "source": self.spec.name,
            "id": aux.id,
            "role": ref.role,
            "format": ref.format,
            "source_path": ref.source_path,
            "rdkit_version": self.meta.get("rdkit_version"),
            **extra,
            "n_atoms": int(n),
            "heavy_atom_mapping": [int(m) for m in mapping],
            "source_bond_pairs": [
                (int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx()))
                for bond in mol.GetBonds()
            ],
            "coordinate_center": center[0].tolist(),
        }
        return AlignedStructure(
            source=self.spec.name,
            id=aux.id,
            smiles=smiles,
            coords=coords,
            mapping=tuple(int(m) for m in mapping),
            provenance=provenance,
        )

    # -- 公开接口 --

    def get(self, row: int) -> AlignedStructure:
        """按 aux 保留行号随机读取。"""
        if isinstance(row, bool) or not 0 <= int(row) < len(self):
            raise IndexError(row)
        return self._align(self._aux.get(int(row)), int(row))

    def get_by_id(self, record_id: str) -> AlignedStructure:
        """按 aux 顶层 id 随机读取。"""
        aux_row = self._aux_row_for_id(record_id)
        aux = self._aux.get(aux_row)
        if aux.id != record_id:
            raise RuntimeError(
                f"{self.spec.name}: aux 索引第 {aux_row} 行 id={aux.id} != {record_id}"
            )
        return self._align(aux, aux_row)

    def sample(self, count: int, seed: int = 0) -> List[AlignedStructure]:
        """确定性抽样读取，用于抽样验证。"""
        if isinstance(count, bool) or not 1 <= int(count) <= len(self):
            raise ValueError(f"count 必须在 [1, {len(self)}] 内")
        rows = random.Random(seed).sample(range(len(self)), int(count))
        return [self.get(row) for row in rows]


# ---------------------------------------------------------------------------
# CLI：全量构建与抽样验证
# ---------------------------------------------------------------------------


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="官方分子 3D 结构索引构建与抽样验证")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="可断点续跑的索引构建（完成后原子发布）")
    build.add_argument("source", choices=sorted(SDF_SOURCES + PICKLE_SOURCES))
    build.add_argument("--no-verify-sha256", action="store_true",
                       help="GEOM：跳过 part tar 的 sha256 校验（默认校验）")
    sample = sub.add_parser("sample", help="随机抽样读取并验证")
    sample.add_argument("source", choices=sorted(SDF_SOURCES + PICKLE_SOURCES))
    sample.add_argument("--count", type=int, default=8)
    sample.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.command == "build":
        path = (build_geom_index(args.source, verify_part_sha256=not args.no_verify_sha256)
                if args.source in PICKLE_SOURCES else build_sdf_index(args.source))
        if path is None:
            print(f"{args.source}: 构建未完成，断点已保存，重跑同一命令即可续跑")
            return 1
        print(f"{args.source}: 索引已发布: {path}")
        return 0

    with Molecule3DStore(args.source) as store:
        structures = store.sample(args.count, args.seed)
        for item in structures:
            if not np.isfinite(item.coords).all():
                raise RuntimeError(f"{item.id}: 坐标含非有限值")
            brief = {k: v for k, v in item.provenance.items()
                     if k not in ("heavy_atom_mapping", "coordinate_center")}
            print(f"PASS {item.id} atoms={item.provenance['n_atoms']} "
                  f"mapping_changed={tuple(item.mapping) != tuple(range(len(item.mapping)))} "
                  f"provenance={json.dumps(brief, ensure_ascii=False)}")
        print(f"{args.source}: 抽样 {len(structures)} 条全部通过对齐与有限性检查")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "AlignedStructure",
    "EXTERNAL_ROOT",
    "INDEX_ROOT",
    "Molecule3DStore",
    "PICKLE_SOURCES",
    "RECORDS_DTYPE",
    "SDF_SOURCES",
    "SDF_STRUCT_DTYPE",
    "StructureRef",
    "build_geom_index",
    "build_index",
    "build_sdf_index",
    "external_base",
    "load_geom_pickle",
    "pick_geom_conformer",
    "read_geom_conformer_at",
    "read_sdf_mol_at",
    "structure_ref",
]
