"""基于 RDKit 化学图的肽残基、骨架与 AdaFrag 标注。

annotate_peptide 直接接收 RDKit Mol，不接收或规范化序列，因此不会改变
D/L 手性、同位素、显式氢或自定义原子属性。当前自动识别 alpha 氨基酸的
N-CA-C(=O) 主链；非 alpha/歧义修饰单元保留为非标准结构组，完全无肽主链才报错。
"""
from __future__ import annotations

from typing import Iterable, Sequence, Set, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS

_CORE = Chem.MolFromSmarts('[#7;X3,X4:1]-[#6;X4:2]-[#6;X3:3](=[#8;X1:4])')
_PEPTIDE_BOND = Chem.MolFromSmarts(
    '[#7;X3,X4:1]-[#6;X4:2]-[#6;X3:3](=[#8;X1:4])'
    '-[#7;X3,X4:5]-[#6;X4:6]')
_AMIDE = Chem.MolFromSmarts('[CX3:1](=[OX1:2])-[NX3,NX4:3]')
_ORIGIN_PROP = '_muse_adafrag_origin'


def _components(mol: Chem.Mol, cut_bonds: Iterable[int]) -> Tuple[Tuple[int, ...], ...]:
    cut_bonds = sorted(set(cut_bonds))
    fragmented = Chem.FragmentOnBonds(mol, cut_bonds, addDummies=False) if cut_bonds else mol
    return Chem.GetMolFrags(fragmented)


def _amide_pairs(mol: Chem.Mol) -> Set[Tuple[int, int]]:
    return {(match[0], match[2]) for match in mol.GetSubstructMatches(_AMIDE)}


def _backbone_bonds(mol: Chem.Mol) -> Set[Tuple[int, int]]:
    pairs = set()
    for match in mol.GetSubstructMatches(_PEPTIDE_BOND):
        carbonyl_c, amide_n = match[2], match[4]
        bond = mol.GetBondBetweenAtoms(carbonyl_c, amide_n)
        if bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
            pairs.add((carbonyl_c, amide_n))
    return pairs


def _residue_annotation(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray, Set[Tuple[int, int]]]:
    core_matches = mol.GetSubstructMatches(_CORE)
    if not core_matches:
        raise ValueError('未识别到 alpha 氨基酸 N-CA-C(=O) 主链')
    backbone_bonds = _backbone_bonds(mol)
    bond_ids = []
    for carbonyl_c, amide_n in backbone_bonds:
        bond = mol.GetBondBetweenAtoms(carbonyl_c, amide_n)
        bond_ids.append(bond.GetIdx())
    residues = _components(mol, bond_ids)

    left_carbonyls = {pair[0] for pair in backbone_bonds}
    right_nitrogens = {pair[1] for pair in backbone_bonds}
    protected = set()
    for carbonyl_c, amide_n in _amide_pairs(mol):
        protected.update((carbonyl_c, amide_n))
        for neighbor in mol.GetAtomWithIdx(carbonyl_c).GetNeighbors():
            if neighbor.GetAtomicNum() == 8:
                protected.add(neighbor.GetIdx())
    group_ids = np.full(mol.GetNumAtoms(), -1, dtype=np.int64)
    sidechain_flags = np.zeros(mol.GetNumAtoms(), dtype=bool)
    for group_id, atoms in enumerate(residues):
        atom_set = set(atoms)
        candidates = [match for match in core_matches
                      if all(index in atom_set for index in match)]
        if not candidates:
            for index in atoms:
                group_ids[index] = group_id
                sidechain_flags[index] = index not in protected
            continue
        scored = []
        for match in candidates:
            n, _, carbonyl_c, _ = match
            terminal_oxygen = any(
                neighbor.GetAtomicNum() == 8
                and mol.GetBondBetweenAtoms(carbonyl_c, neighbor.GetIdx()).GetBondType()
                == Chem.BondType.SINGLE
                for neighbor in mol.GetAtomWithIdx(carbonyl_c).GetNeighbors())
            score = (2 * int(n in right_nitrogens)
                     + 2 * int(carbonyl_c in left_carbonyls) + int(terminal_oxygen))
            scored.append((score, match))
        best_score = max(score for score, _ in scored)
        best = [match for score, match in scored if score == best_score]
        if len(best) > 1:
            for index in atoms:
                group_ids[index] = group_id
                sidechain_flags[index] = index not in protected
            continue
        n, alpha_c, carbonyl_c, carbonyl_o = best[0]
        backbone = {n, alpha_c, carbonyl_c, carbonyl_o}
        for neighbor in mol.GetAtomWithIdx(carbonyl_c).GetNeighbors():
            if neighbor.GetAtomicNum() == 8:
                backbone.add(neighbor.GetIdx())
        for index in atoms:
            group_ids[index] = group_id
            atom = mol.GetAtomWithIdx(index)
            if index in backbone:
                continue
            if atom.GetAtomicNum() == 1 and any(
                    neighbor.GetIdx() in backbone for neighbor in atom.GetNeighbors()):
                continue
            sidechain_flags[index] = True
    for index in protected:
        sidechain_flags[index] = False
    return group_ids, sidechain_flags, backbone_bonds


def _base_adafrag_cuts(mol: Chem.Mol,
                       amide_pairs: Set[Tuple[int, int]]) -> Set[int]:
    protected = {frozenset(pair) for pair in amide_pairs}
    cuts = set()
    for carbonyl_c, amide_n in amide_pairs:
        for bond in mol.GetAtomWithIdx(carbonyl_c).GetBonds():
            other = bond.GetOtherAtomIdx(carbonyl_c)
            if (bond.GetBondType() == Chem.BondType.SINGLE
                    and mol.GetAtomWithIdx(other).GetAtomicNum() == 6
                    and frozenset((carbonyl_c, other)) not in protected):
                cuts.add(bond.GetIdx())
        for bond in mol.GetAtomWithIdx(amide_n).GetBonds():
            other = bond.GetOtherAtomIdx(amide_n)
            if (bond.GetBondType() == Chem.BondType.SINGLE
                    and mol.GetAtomWithIdx(other).GetAtomicNum() == 6
                    and frozenset((amide_n, other)) not in protected):
                cuts.add(bond.GetIdx())
    return cuts


def _adafrag_fragment_ids(mol: Chem.Mol, amide_pairs: Set[Tuple[int, int]],
                          brics_heavy_atom_threshold: int = 8) -> np.ndarray:
    if brics_heavy_atom_threshold < 1:
        raise ValueError('brics_heavy_atom_threshold 必须为正整数')
    cuts = _base_adafrag_cuts(mol, amide_pairs)
    protected = {frozenset(pair) for pair in amide_pairs}

    work = Chem.Mol(mol)
    for atom in work.GetAtoms():
        atom.SetProp(_ORIGIN_PROP, str(atom.GetIdx()))
    fragmented = (Chem.FragmentOnBonds(work, sorted(cuts), addDummies=False)
                  if cuts else work)
    fragments = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=False)
    for fragment in fragments:
        if fragment.GetNumHeavyAtoms() < brics_heavy_atom_threshold:
            continue
        try:
            Chem.SanitizeMol(fragment)
        except (ValueError, RuntimeError) as exc:
            raise ValueError('大侧链片段无法 sanitize，不能可靠执行 BRICS') from exc
        origin = [int(atom.GetProp(_ORIGIN_PROP)) for atom in fragment.GetAtoms()]
        for (begin, end), _ in BRICS.FindBRICSBonds(fragment):
            original_pair = (origin[begin], origin[end])
            if frozenset(original_pair) in protected:
                continue
            bond = mol.GetBondBetweenAtoms(*original_pair)
            if bond is not None:
                cuts.add(bond.GetIdx())

    fragment_ids = np.full(mol.GetNumAtoms(), -1, dtype=np.int64)
    for fragment_id, atoms in enumerate(_components(mol, cuts)):
        fragment_ids[list(atoms)] = fragment_id
    return fragment_ids


def annotate_peptide(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 group_ids、sidechain_flags、fragment_ids 三个等长 numpy 数组。

    group_ids/fragment_ids 为从 0 开始的 int64，sidechain_flags 为 bool；
    数组仅对应 Mol 原子，不含 CLS。所有酰胺 C(=O)-N 键在 AdaFrag 中保留，
    大片段再用 BRICS 切分。输入 Mol 不会被修改。
    """
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError('annotate_peptide 需要非空 RDKit Mol')
    group_ids, sidechain_flags, _ = _residue_annotation(mol)
    fragment_ids = _adafrag_fragment_ids(mol, _amide_pairs(mol))
    if not (np.all(group_ids >= 0) and np.all(fragment_ids >= 0)):
        raise ValueError('肽标注未覆盖全部原子')
    return group_ids, sidechain_flags, fragment_ids


def _connected_subset(mol: Chem.Mol, atoms: Sequence[int]) -> bool:
    remaining = set(atoms)
    if not remaining:
        return False
    stack = [next(iter(remaining))]
    seen = {stack[0]}
    while stack:
        index = stack.pop()
        for neighbor in mol.GetAtomWithIdx(index).GetNeighbors():
            other = neighbor.GetIdx()
            if other in remaining and other not in seen:
                seen.add(other)
                stack.append(other)
    return seen == remaining


def fragment_recovery_rate(mol: Chem.Mol,
                           fragment_ids: np.ndarray = None) -> float:
    """返回可由片段原子划分无损恢复的原子比例。

    可恢复片段必须覆盖原原子、在原图上连通并能导出片段 SMILES；任何酰胺
    键被切开都视为不可恢复。所有跨片段原键仍由原分子图提供，重连不会改写
    原子属性。该指标用于本模块的完整性验收，不等同于论文片段词表覆盖率。
    """
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError('需要非空 RDKit Mol')
    if fragment_ids is None:
        fragment_ids = annotate_peptide(mol)[2]
    fragment_ids = np.asarray(fragment_ids)
    if fragment_ids.shape != (mol.GetNumAtoms(),):
        raise ValueError('fragment_ids 必须与原子数等长')
    for carbonyl_c, amide_n in _amide_pairs(mol):
        if fragment_ids[carbonyl_c] != fragment_ids[amide_n]:
            return 0.0

    recovered = 0
    for fragment_id in np.unique(fragment_ids[fragment_ids >= 0]):
        atoms = np.flatnonzero(fragment_ids == fragment_id).tolist()
        if not _connected_subset(mol, atoms):
            continue
        try:
            smiles = Chem.MolFragmentToSmiles(mol, atoms)
        except (ValueError, RuntimeError):
            continue
        if smiles:
            recovered += len(atoms)
    return recovered / mol.GetNumAtoms()


def check_fragment_recovery(mol: Chem.Mol, fragment_ids: np.ndarray = None,
                            min_rate: float = 0.97) -> float:
    """恢复率低于 min_rate 时抛出 ValueError，否则返回恢复率。"""
    if not 0.0 <= min_rate <= 1.0:
        raise ValueError('min_rate 必须在 [0,1]')
    rate = fragment_recovery_rate(mol, fragment_ids)
    if rate < min_rate:
        raise ValueError(f'片段恢复率 {rate:.4f} 低于要求 {min_rate:.4f}')
    return rate
