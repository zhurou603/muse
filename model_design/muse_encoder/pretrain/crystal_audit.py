"""OMC25 起始晶体元数据、母分子几何恢复与三元组审计。

母分子 InChIKey 只由原子几何经共价连通分片和 RDKit 键级推断产生；
化学式仅用于选择片段，绝不用于合成或猜测 InChIKey。
"""
from __future__ import annotations

import csv
import math
import re
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from .omc25 import audit_triples, reduced_formula, starting_heap_id


@dataclass(frozen=True)
class StartingCrystal:
    starting_heap: str
    csd_refcode: str
    z_value: int
    genarris_step: str
    xtal_id: str
    split: str
    nframes: int | None
    mol_composition: str | None
    xtal_composition: str | None
    mol_natoms: int | None
    xtal_natoms: int | None
    spacegroup: int | None
    csv_row: int

    @property
    def metadata(self) -> tuple[str, int | None, str | None]:
        return self.csd_refcode, self.spacegroup, self.mol_composition

    @property
    def mol_formula(self) -> str | None:
        return formula_from_composition(self.mol_composition)

    @property
    def xtal_formula(self) -> str | None:
        return formula_from_composition(self.xtal_composition or self.mol_composition)


@dataclass(frozen=True)
class StructureSample:
    record_id: str
    starting_heap: str
    atoms: Any
    sid: str | None = None

    @classmethod
    def from_ase_row(cls, row: Any) -> "StructureSample":
        sid = (row.raw.get("data") or {}).get("sid")
        if not sid:
            raise ValueError("ASE row 缺少 data.sid")
        record_id = "muse__omc25__periodic_frame__" + sid.replace("-", "_")
        return cls(record_id, starting_heap_id(sid), row.atoms, sid)


@dataclass(frozen=True)
class MolecularFragment:
    indices: tuple[int, ...]
    positions: np.ndarray
    pbc_conflicts: int


@dataclass(frozen=True)
class ParentRecoveryResult:
    status: str
    target_composition: str
    inchikey: str | None = None
    smiles: str | None = None
    failure_reason: str | None = None
    matching_fragments: int = 0
    successful_fragments: int = 0
    fragment_inchikeys: tuple[str, ...] = ()
    attempts: tuple[dict[str, Any], ...] = ()
    fragments: tuple[dict[str, Any], ...] = ()
    method: str = "ASE PBC covalent components + RDKit rdDetermineBonds"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_REQUIRED_COLUMNS = {
    "csd_refcode", "z_value", "genarris_step", "xtal.id", "split",
    "nframes", "mol.composition", "xtal.composition", "mol.natoms",
    "xtal.natoms", "xtal.spacegroup",
}
_COMPOSITION_TOKEN = re.compile(r"\s*([A-Z][a-z]?)\s*([0-9]+)?")


def composition_counts(composition: str) -> dict[str, int]:
    """解析 ``C12 H8 F4`` 或 ``C12H8F4``，不接受括号或推断氢数。"""
    if not isinstance(composition, str) or not composition.strip():
        raise ValueError("空 mol.composition")
    counts: dict[str, int] = {}
    position = 0
    for match in _COMPOSITION_TOKEN.finditer(composition):
        if match.start() != position:
            raise ValueError(f"非法 composition: {composition!r}")
        element, raw_count = match.groups()
        count = int(raw_count or 1)
        if count < 1:
            raise ValueError(f"非法 composition 计数: {composition!r}")
        counts[element] = counts.get(element, 0) + count
        position = match.end()
    if not counts or composition[position:].strip():
        raise ValueError(f"非法 composition: {composition!r}")
    return counts


def formula_from_counts(counts: Mapping[str, int]) -> str:
    if not counts or any(count < 1 for count in counts.values()):
        raise ValueError("空 composition counts")
    if "C" in counts:
        order = ["C"] + (["H"] if "H" in counts else []) + sorted(
            element for element in counts if element not in ("C", "H")
        )
    else:
        order = sorted(counts)
    return "".join(
        element + (str(counts[element]) if counts[element] != 1 else "")
        for element in order
    )


def formula_from_composition(composition: str | None) -> str | None:
    return formula_from_counts(composition_counts(composition)) if composition else None


def _optional_int(value: Any, field: str, csv_row: int) -> int | None:
    if value is None or not str(value).strip():
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"CSV 第 {csv_row} 行 {field} 不是整数: {value!r}") from exc


def iter_starting_crystals(
    path: str | Path, limit: int | None = None
) -> Iterator[StartingCrystal]:
    """流式读取官方 starting-crystals CSV，并构造精确 starting_heap。"""
    count = 0
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = _REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: 缺少 CSV 字段 {sorted(missing)}")
        for row in reader:
            if limit is not None and count >= limit:
                break
            csv_row = reader.line_num
            required = ("csd_refcode", "z_value", "genarris_step", "xtal.id")
            if any(not (row.get(field) or "").strip() for field in required):
                raise ValueError(f"CSV 第 {csv_row} 行缺少 heap 连接字段")
            try:
                z_value = int(row["z_value"])
            except ValueError as exc:
                raise ValueError(f"CSV 第 {csv_row} 行 z_value 非整数") from exc
            csd_refcode = row["csd_refcode"].strip()
            genarris_step = row["genarris_step"].strip()
            xtal_id = row["xtal.id"].strip()
            heap = f"{csd_refcode}-{z_value}-{genarris_step}-{xtal_id}"
            spacegroup = _optional_int(row.get("xtal.spacegroup"), "xtal.spacegroup", csv_row)
            if spacegroup is not None and not 1 <= spacegroup <= 230:
                raise ValueError(f"CSV 第 {csv_row} 行空间群越界: {spacegroup}")
            yield StartingCrystal(
                starting_heap=heap,
                csd_refcode=csd_refcode,
                z_value=z_value,
                genarris_step=genarris_step,
                xtal_id=xtal_id,
                split=(row.get("split") or "").strip(),
                nframes=_optional_int(row.get("nframes"), "nframes", csv_row),
                mol_composition=(row.get("mol.composition") or "").strip() or None,
                xtal_composition=(row.get("xtal.composition") or "").strip() or None,
                mol_natoms=_optional_int(row.get("mol.natoms"), "mol.natoms", csv_row),
                xtal_natoms=_optional_int(row.get("xtal.natoms"), "xtal.natoms", csv_row),
                spacegroup=spacegroup,
                csv_row=csv_row,
            )
            count += 1


class StartingCrystalIndex:
    """218k 行的轻量内存索引；构建过程仍为逐行流式读取。"""

    def __init__(self, records: Iterable[StartingCrystal]):
        self._records: dict[str, StartingCrystal] = {}
        for record in records:
            if record.starting_heap in self._records:
                raise ValueError(f"重复 starting_heap: {record.starting_heap}")
            self._records[record.starting_heap] = record

    @classmethod
    def from_csv(
        cls, path: str | Path, expected_count: int | None = None,
        limit: int | None = None,
    ) -> "StartingCrystalIndex":
        index = cls(iter_starting_crystals(path, limit=limit))
        if expected_count is not None and len(index) != expected_count:
            raise ValueError(f"starting-crystals 行数 {len(index)} != {expected_count}")
        return index

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, starting_heap: str) -> StartingCrystal:
        return self._records[starting_heap]

    def get(self, starting_heap: str) -> StartingCrystal | None:
        return self._records.get(starting_heap)

    def values(self) -> Iterator[StartingCrystal]:
        return iter(self._records.values())


def find_molecular_fragments(
    atoms: Any, connectivity_factor: float = 1.2, minimum_distance: float = 0.4
) -> list[MolecularFragment]:
    """以 PBC 共价半径图分分子，并沿生成树消除周期边界位移。"""
    from ase.neighborlist import natural_cutoffs, neighbor_list

    if not 0 < connectivity_factor or not 0 <= minimum_distance:
        raise ValueError("非法 connectivity_factor/minimum_distance")
    if len(atoms) == 0:
        return []
    cutoffs = natural_cutoffs(atoms, mult=connectivity_factor)
    first, second, distances, shifts = neighbor_list(
        "ijdS", atoms, cutoffs, self_interaction=False
    )
    adjacency: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(len(atoms))]
    for i, j, distance, shift in zip(first, second, distances, shifts):
        if distance < minimum_distance:
            continue
        i, j = int(i), int(j)
        adjacency[i].append((j, np.asarray(shift, dtype=np.int64)))
        adjacency[j].append((i, -np.asarray(shift, dtype=np.int64)))
    assigned: dict[int, np.ndarray] = {}
    fragments = []
    for root in range(len(atoms)):
        if root in assigned:
            continue
        offsets = {root: np.zeros(3, dtype=np.int64)}
        assigned[root] = offsets[root]
        queue = deque([root])
        conflicts = 0
        while queue:
            i = queue.popleft()
            for j, shift in adjacency[i]:
                expected = offsets[i] + shift
                if j not in offsets:
                    offsets[j] = expected
                    assigned[j] = expected
                    queue.append(j)
                elif not np.array_equal(offsets[j], expected):
                    conflicts += 1
        indices = tuple(sorted(offsets))
        unwrapped = atoms.positions[list(indices)] + np.asarray(
            [offsets[index] for index in indices]
        ) @ atoms.cell.array
        unwrapped = np.asarray(unwrapped, dtype=np.float64)
        unwrapped -= unwrapped.mean(axis=0)
        fragments.append(MolecularFragment(indices, unwrapped, conflicts))
    return fragments


def _fragment_counts(atoms: Any, fragment: MolecularFragment) -> dict[int, int]:
    counts: dict[int, int] = {}
    for number in atoms.numbers[list(fragment.indices)]:
        counts[int(number)] = counts.get(int(number), 0) + 1
    return counts


def _target_atomic_numbers(composition: Mapping[str, int]) -> dict[int, int]:
    from ase.data import atomic_numbers

    return {atomic_numbers[element]: count for element, count in composition.items()}


def _rdkit_molecule(atoms: Any, fragment: MolecularFragment) -> Any:
    from rdkit import Chem

    molecule = Chem.RWMol()
    for index in fragment.indices:
        molecule.AddAtom(Chem.Atom(int(atoms.numbers[index])))
    conformer = Chem.Conformer(len(fragment.indices))
    for index, position in enumerate(fragment.positions):
        conformer.SetAtomPosition(index, tuple(float(value) for value in position))
    molecule.AddConformer(conformer)
    return molecule


def recover_parent_inchikey(
    atoms: Any,
    mol_composition: str,
    connectivity_factor: float = 1.2,
    bond_factors: Iterable[float] = (1.2, 1.3),
    use_vdw: Iterable[bool] = (False, True),
    charge: int = 0,
    max_iterations: int = 1000,
    exhaustive: bool = False,
) -> ParentRecoveryResult:
    """从晶体内几何恢复母分子；任何失败均返回无 InChIKey 的结果。

    多个符合组成的片段必须给出一致 InChIKey。默认每片段首个成功 factor
    后停止后续 factor，``exhaustive=True`` 检查全部参数组合的一致性。
    """
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    bond_factors = tuple(bond_factors)
    use_vdw = tuple(use_vdw)
    if (not bond_factors or any(factor <= 0 for factor in bond_factors)
            or not use_vdw or max_iterations < 1):
        raise ValueError("bond_factors/use_vdw 不能为空，factor/max_iterations 必须为正")
    target = composition_counts(mol_composition)
    target_numbers = _target_atomic_numbers(target)
    fragments = find_molecular_fragments(atoms, connectivity_factor)
    fragment_summaries = []
    matching = []
    for index, fragment in enumerate(fragments):
        counts = _fragment_counts(atoms, fragment)
        fragment_summaries.append({
            "fragment": index,
            "atom_count": len(fragment.indices),
            "atomic_numbers": {str(key): value for key, value in sorted(counts.items())},
            "pbc_conflicts": fragment.pbc_conflicts,
            "composition_match": counts == target_numbers,
        })
        if counts == target_numbers and not fragment.pbc_conflicts:
            matching.append((index, fragment))
    if not matching:
        return ParentRecoveryResult(
            "failed", mol_composition, failure_reason="no_composition_matching_fragment",
            fragments=tuple(fragment_summaries),
        )
    attempts = []
    fragment_keys: dict[int, str] = {}
    fragment_smiles: dict[int, str] = {}
    disagreement = False
    for fragment_index, fragment in matching:
        successful_key = None
        for factor in bond_factors:
            for vdw in use_vdw:
                attempt = {"fragment": fragment_index, "cov_factor": factor, "use_vdw": vdw}
                try:
                    molecule = _rdkit_molecule(atoms, fragment)
                    rdDetermineBonds.DetermineBonds(
                        molecule, charge=charge, covFactor=factor, useVdw=vdw,
                        allowChargedFragments=True, embedChiral=True,
                        maxIterations=max_iterations,
                    )
                    if len(Chem.GetMolFrags(molecule)) != 1:
                        raise ValueError("RDKit connectivity 产生多个片段")
                    inchikey = Chem.MolToInchiKey(molecule)
                    if not inchikey:
                        raise ValueError("RDKit 返回空 InChIKey")
                    smiles = Chem.MolToSmiles(molecule)
                    attempt.update(status="success", inchikey=inchikey, smiles=smiles)
                    if successful_key is not None and successful_key != inchikey:
                        disagreement = True
                    if successful_key is None:
                        successful_key = inchikey
                        fragment_smiles[fragment_index] = smiles
                except Exception as exc:
                    attempt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                attempts.append(attempt)
            if successful_key is not None and not exhaustive:
                break
        if successful_key is not None:
            fragment_keys[fragment_index] = successful_key
    unique_keys = sorted(set(fragment_keys.values()))
    if len(unique_keys) > 1:
        disagreement = True
    if disagreement:
        return ParentRecoveryResult(
            "failed", mol_composition, failure_reason="inconsistent_inferred_inchikey",
            matching_fragments=len(matching), successful_fragments=len(fragment_keys),
            fragment_inchikeys=tuple(unique_keys), attempts=tuple(attempts),
            fragments=tuple(fragment_summaries),
        )
    if not unique_keys:
        return ParentRecoveryResult(
            "failed", mol_composition, failure_reason="rdkit_bond_inference_failed",
            matching_fragments=len(matching), attempts=tuple(attempts),
            fragments=tuple(fragment_summaries),
        )
    first_fragment = sorted(fragment_keys)[0]
    return ParentRecoveryResult(
        "success", mol_composition, inchikey=unique_keys[0],
        smiles=fragment_smiles[first_fragment], matching_fragments=len(matching),
        successful_fragments=len(fragment_keys), fragment_inchikeys=tuple(unique_keys),
        attempts=tuple(attempts), fragments=tuple(fragment_summaries),
    )


def build_triple_record(
    sample: StructureSample, metadata: StartingCrystal | None,
    recovery_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构建一条 triple 记录；缺字段与恢复失败均显式保留。"""
    options = dict(recovery_options or {})
    if metadata is None:
        return {
            "id": sample.record_id,
            "sid": sample.sid,
            "starting_heap": sample.starting_heap,
            "chemical_formula": sample.atoms.get_chemical_formula(),
            "spacegroup": None,
            "parent_inchikey": None,
            "missing": ["starting_heap_metadata", "spacegroup", "parent_inchikey"],
        }
    recovery = None
    parent_inchikey = None
    parent_smiles = None
    missing = []
    if metadata.spacegroup is None:
        missing.append("spacegroup")
    if not metadata.mol_composition:
        missing.extend(["mol_composition", "parent_inchikey"])
    else:
        recovery = recover_parent_inchikey(
            sample.atoms, metadata.mol_composition, **options
        )
        if recovery.inchikey is None:
            missing.append("parent_inchikey")
        else:
            parent_inchikey = recovery.inchikey
            parent_smiles = recovery.smiles
    return {
        "id": sample.record_id,
        "sid": sample.sid,
        "starting_heap": sample.starting_heap,
        "csd_refcode": metadata.csd_refcode,
        "split": metadata.split,
        "chemical_formula": metadata.xtal_formula,
        "parent_composition": metadata.mol_formula,
        "spacegroup": metadata.spacegroup,
        "parent_inchikey": parent_inchikey,
        "parent_smiles": parent_smiles,
        "missing": missing,
        "recovery": recovery.to_dict() if recovery is not None else None,
    }


def build_crystal_triples(
    samples: Iterable[StructureSample],
    index: StartingCrystalIndex,
    splits: Iterable[str] | None = None,
    heaps: Iterable[str] | None = None,
    recovery_options: Mapping[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """对给定结构流式构建 triples；不会隐式扫描 25M JSONL/LMDB。"""
    split_set = set(splits) if splits is not None else None
    heap_set = set(heaps) if heaps is not None else None
    for sample in samples:
        if heap_set is not None and sample.starting_heap not in heap_set:
            continue
        metadata = index.get(sample.starting_heap)
        if metadata is None:
            yield build_triple_record(sample, None, recovery_options)
        elif split_set is None or metadata.split in split_set:
            yield build_triple_record(sample, metadata, recovery_options)


def build_training_triples(
    samples: Iterable[StructureSample], index: StartingCrystalIndex,
    heaps: Iterable[str] | None = None,
    recovery_options: Mapping[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    return build_crystal_triples(samples, index, ("train",), heaps, recovery_options)


def build_reference_triples(
    samples: Iterable[StructureSample], index: StartingCrystalIndex,
    splits: Iterable[str] | None = ("val", "test"),
    heaps: Iterable[str] | None = None,
    recovery_options: Mapping[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """默认按官方 CSV val/test；MUSE 重切分用 splits=None 并显式给 heaps。"""
    return build_crystal_triples(samples, index, splits, heaps, recovery_options)


def _complete(record: Mapping[str, Any]) -> bool:
    return all(record.get(field) for field in (
        "chemical_formula", "spacegroup", "parent_inchikey"
    ))


def audit_crystal_triples(
    training_records: Iterable[Mapping[str, Any]],
    reference_records: Iterable[Mapping[str, Any]],
    conservative_missing: bool = False,
) -> dict[str, Any]:
    """复用 audit_triples；缺失记录单列，可选择明确的保守 fallback。"""
    training = list(training_records)
    references = list(reference_records)
    complete_training = [record for record in training if _complete(record)]
    complete_references = [record for record in references if _complete(record)]
    missing_training = [record for record in training if not _complete(record)]
    missing_references = [record for record in references if not _complete(record)]
    triple_hits = list(audit_triples(complete_training, complete_references))
    conservative_hits = []
    if conservative_missing:
        reference_formula_sg = set()
        reference_formulas = set()
        reference_refcodes = set()
        for record in references:
            formula = record.get("chemical_formula")
            if formula:
                reduced = reduced_formula(formula)
                reference_formulas.add(reduced)
                if record.get("spacegroup") is not None:
                    reference_formula_sg.add((reduced, str(record["spacegroup"])))
            if record.get("csd_refcode"):
                reference_refcodes.add(record["csd_refcode"])
        for record in missing_training:
            reasons = []
            formula = record.get("chemical_formula")
            if formula:
                reduced = reduced_formula(formula)
                if record.get("spacegroup") is not None and (
                    reduced, str(record["spacegroup"])
                ) in reference_formula_sg:
                    reasons.append("conservative_reduced_formula_spacegroup")
                if reduced in reference_formulas:
                    reasons.append("conservative_reduced_formula")
            if record.get("csd_refcode") in reference_refcodes:
                reasons.append("conservative_csd_refcode")
            if reasons:
                conservative_hits.append({
                    "id": record["id"], "starting_heap": record.get("starting_heap"),
                    "reasons": reasons, "missing": record.get("missing", []),
                })
    return {
        "triple_hits": triple_hits,
        "conservative_hits": conservative_hits,
        "missing_training": missing_training,
        "missing_reference": missing_references,
        "summary": {
            "training_records": len(training),
            "reference_records": len(references),
            "complete_training": len(complete_training),
            "complete_reference": len(complete_references),
            "missing_training": len(missing_training),
            "missing_reference": len(missing_references),
            "triple_hits": len(triple_hits),
            "conservative_hits": len(conservative_hits),
            "conservative_missing_enabled": conservative_missing,
        },
    }


__all__ = [
    "MolecularFragment", "ParentRecoveryResult", "StartingCrystal",
    "StartingCrystalIndex", "StructureSample", "audit_crystal_triples",
    "build_crystal_triples", "build_reference_triples", "build_training_triples",
    "build_triple_record", "composition_counts", "find_molecular_fragments",
    "formula_from_composition", "formula_from_counts", "iter_starting_crystals",
    "recover_parent_inchikey",
]
