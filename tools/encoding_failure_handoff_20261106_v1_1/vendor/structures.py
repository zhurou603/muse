"""严格的分子结构对齐与周期性邻居构建。"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    from rdkit import Chem
    _HAS_RDKIT = True
except ImportError:
    Chem = None
    _HAS_RDKIT = False

PBC_BASIS_CENTERS = np.linspace(0.0, 10.0, 32, dtype=np.float64)
PBC_BASIS_GAMMA = 10.0


def require_rdkit():
    if not _HAS_RDKIT:
        raise ImportError("RDKit 是正式特征化的必需依赖")
    return Chem


def graph_from_smiles(smiles: str):
    """Return the heavy-atom graph in the official Uni-Mol atom order."""
    chem = require_rdkit()
    if not isinstance(smiles, str) or not smiles:
        raise ValueError("SMILES 必须是非空字符串")
    mol = chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit 无法解析: {smiles[:80]}")
    return chem.RemoveAllHs(chem.AddHs(mol, addCoords=True))


def atom_signature(atom):
    return (
        atom.GetAtomicNum(),
        atom.GetIsotope(),
        atom.GetFormalCharge(),
        atom.GetIsAromatic(),
        atom.GetTotalNumHs(),
    )


def _aromatic_representation_mapping(chem, source, target):
    """Match only a fully validated aromatic/Kekulé representation difference."""
    if not any(bond.GetIsAromatic() for bond in target.GetBonds()):
        return None

    query = chem.RWMol(target)
    generic_bond = chem.MolFromSmarts("[*]~[*]").GetBondWithIdx(0)
    for bond in list(query.GetBonds()):
        if bond.GetIsAromatic():
            query.ReplaceBond(bond.GetIdx(), generic_bond, preserveProps=False)
    candidates = source.GetSubstructMatches(
        query.GetMol(), useChirality=True, uniquify=False, maxMatches=10001
    )
    if len(candidates) >= 10001:
        raise RuntimeError("原子映射候选超过10000，停止，不能确认完整搜索")

    allowed_source_bonds = (
        chem.BondType.SINGLE,
        chem.BondType.DOUBLE,
        chem.BondType.AROMATIC,
    )
    n = target.GetNumAtoms()
    for candidate in candidates:
        if len(candidate) != n or len(set(candidate)) != n:
            continue
        valid = True
        for i, j in enumerate(candidate):
            target_atom = target.GetAtomWithIdx(i)
            source_atom = source.GetAtomWithIdx(j)
            stable_signature = (
                target_atom.GetAtomicNum(),
                target_atom.GetIsotope(),
                target_atom.GetFormalCharge(),
                target_atom.GetTotalNumHs(),
            )
            source_signature = (
                source_atom.GetAtomicNum(),
                source_atom.GetIsotope(),
                source_atom.GetFormalCharge(),
                source_atom.GetTotalNumHs(),
            )
            if stable_signature != source_signature:
                valid = False
                break
            target_radicals = target_atom.GetNumRadicalElectrons()
            source_radicals = source_atom.GetNumRadicalElectrons()
            if not target_atom.GetIsAromatic():
                if (
                    target_atom.GetIsAromatic() != source_atom.GetIsAromatic()
                    or target_radicals != source_radicals
                ):
                    valid = False
                    break
            elif target_radicals != source_radicals:
                if (
                    target_radicals != 0
                    or source_radicals != 1
                    or not source_atom.GetNoImplicit()
                ):
                    valid = False
                    break
                tied_to_changed_bond = False
                for bond in target_atom.GetBonds():
                    if not bond.GetIsAromatic():
                        continue
                    other = source.GetBondBetweenAtoms(
                        j, candidate[bond.GetOtherAtomIdx(i)]
                    )
                    if other is not None and other.GetBondType() in allowed_source_bonds \
                            and (bond.GetBondType(), bond.GetIsAromatic()) != (
                                other.GetBondType(), other.GetIsAromatic()
                            ):
                        tied_to_changed_bond = True
                        break
                if not tied_to_changed_bond:
                    valid = False
                    break
        if not valid:
            continue

        for bond in target.GetBonds():
            other = source.GetBondBetweenAtoms(
                candidate[bond.GetBeginAtomIdx()],
                candidate[bond.GetEndAtomIdx()],
            )
            if other is None:
                valid = False
                break
            if bond.GetIsAromatic():
                if other.GetBondType() not in allowed_source_bonds or not other.IsInRing():
                    valid = False
                    break
            elif (bond.GetBondType(), bond.GetIsAromatic()) != (
                other.GetBondType(), other.GetIsAromatic()
            ):
                valid = False
                break
        if not valid:
            continue

        original_h_counts = [atom.GetTotalNumHs() for atom in source.GetAtoms()]
        normalized = chem.RWMol(source)
        for atom, h_count in zip(normalized.GetAtoms(), original_h_counts):
            atom.SetNumExplicitHs(h_count)
            atom.SetNoImplicit(True)
        for i, j in enumerate(candidate):
            target_atom = target.GetAtomWithIdx(i)
            if target_atom.GetIsAromatic():
                source_atom = normalized.GetAtomWithIdx(j)
                source_atom.SetIsAromatic(True)
                source_atom.SetNumRadicalElectrons(
                    target_atom.GetNumRadicalElectrons()
                )
        for bond in target.GetBonds():
            if bond.GetIsAromatic():
                other = normalized.GetBondBetweenAtoms(
                    candidate[bond.GetBeginAtomIdx()],
                    candidate[bond.GetEndAtomIdx()],
                )
                other.SetBondType(chem.BondType.AROMATIC)
                other.SetIsAromatic(True)
        normalized = normalized.GetMol()
        try:
            normalized.UpdatePropertyCache(strict=True)
            chem.SanitizeMol(normalized)
        except (ValueError, RuntimeError):
            continue
        if any(
            atom_signature(target.GetAtomWithIdx(i))
            != atom_signature(normalized.GetAtomWithIdx(j))
            or target.GetAtomWithIdx(i).GetNumRadicalElectrons()
            != normalized.GetAtomWithIdx(j).GetNumRadicalElectrons()
            for i, j in enumerate(candidate)
        ):
            continue
        if any(
            (bond.GetBondType(), bond.GetIsAromatic())
            != (
                normalized.GetBondBetweenAtoms(
                    candidate[bond.GetBeginAtomIdx()],
                    candidate[bond.GetEndAtomIdx()],
                ).GetBondType(),
                normalized.GetBondBetweenAtoms(
                    candidate[bond.GetBeginAtomIdx()],
                    candidate[bond.GetEndAtomIdx()],
                ).GetIsAromatic(),
            )
            for bond in target.GetBonds()
        ):
            continue
        exact = normalized.GetSubstructMatches(
            target, useChirality=True, uniquify=False, maxMatches=10001
        )
        if len(exact) >= 10001:
            raise RuntimeError("原子映射候选超过10000，停止，不能确认完整搜索")
        if candidate in exact:
            return tuple(candidate)
    return None


def align_coordinates(smiles: str, sdf_mol):
    """Align original SDF coordinates to the official SMILES atom order.

    The returned mapping maps every target/SMILES atom index to its source/SDF
    atom index. No coordinates are generated, reordered in place, or optimized.
    Aromatic/Kekulé differences require full sanitization and exact atom, bond,
    radical, hydrogen and stereo checks on the same mapping.
    """
    chem = require_rdkit()
    if sdf_mol is None:
        raise ValueError("SDF 解析失败")
    target = graph_from_smiles(smiles)
    source = chem.RemoveAllHs(sdf_mol)
    if source.GetNumConformers() != 1:
        raise ValueError("需要恰好一个SDF构象")
    n = target.GetNumAtoms()
    if (source.GetNumAtoms(), source.GetNumBonds()) != (
        n, target.GetNumBonds()
    ):
        raise ValueError("SMILES/SDF原子数或键数不一致")

    source_graph, target_graph = chem.Mol(source), chem.Mol(target)
    chem.RemoveStereochemistry(source_graph)
    chem.RemoveStereochemistry(target_graph)
    same_graph = chem.MolToSmiles(
        source_graph, isomericSmiles=True
    ) == chem.MolToSmiles(target_graph, isomericSmiles=True)
    if not same_graph:
        mapping = _aromatic_representation_mapping(chem, source, target)
        if mapping is None:
            raise ValueError("SMILES/SDF非立体化学结构不一致（连接、键、电荷或同位素等）")
    else:
        candidates = source.GetSubstructMatches(
            target, useChirality=True, uniquify=False, maxMatches=10001
        )
        if len(candidates) >= 10001:
            raise RuntimeError("原子映射候选超过10000，停止，不能确认完整搜索")
        mapping = None
        for candidate in candidates:
            if len(candidate) != n or len(set(candidate)) != n:
                continue
            if any(
                atom_signature(target.GetAtomWithIdx(i))
                != atom_signature(source.GetAtomWithIdx(j))
                or target.GetAtomWithIdx(i).GetNumRadicalElectrons()
                != source.GetAtomWithIdx(j).GetNumRadicalElectrons()
                for i, j in enumerate(candidate)
            ):
                continue
            valid = True
            for bond in target.GetBonds():
                other = source.GetBondBetweenAtoms(
                    candidate[bond.GetBeginAtomIdx()],
                    candidate[bond.GetEndAtomIdx()],
                )
                if other is None or (
                    bond.GetBondType(), bond.GetIsAromatic()
                ) != (other.GetBondType(), other.GetIsAromatic()):
                    valid = False
                    break
            if valid:
                mapping = tuple(candidate)
                break
        if mapping is None:
            raise ValueError("没有满足原子属性、化学键、自由基及已指定立体信息的完整映射")

    original = np.asarray(source.GetConformer().GetPositions(), dtype=np.float64)
    if original.shape != (n, 3) or not np.isfinite(original).all():
        raise ValueError("SDF坐标非法")
    aligned = original[np.asarray(mapping, dtype=np.int64)]
    return target, aligned, mapping


align_sdf_coordinates = align_coordinates


@dataclass
class PBCNeighbors:
    """Directed periodic neighbors; row i contains images around center atom i."""

    lattice: np.ndarray
    frac_coords: np.ndarray
    coords: np.ndarray
    neighbor_indices: np.ndarray
    neighbor_shifts: np.ndarray
    neighbor_distances: np.ndarray
    neighbor_mask: np.ndarray
    neighbor_counts: np.ndarray
    retained_counts: np.ndarray
    truncated_counts: np.ndarray
    multiplicity: np.ndarray
    candidate_multiplicity: np.ndarray
    pair_disp: np.ndarray
    pbc_shift: np.ndarray
    pair_disp_mean: np.ndarray
    pbc_shift_mean: np.ndarray
    pair_basis: np.ndarray

    @property
    def truncation_stats(self) -> Dict[str, np.ndarray]:
        return {
            "neighbor_counts": self.neighbor_counts,
            "retained_counts": self.retained_counts,
            "truncated_counts": self.truncated_counts,
            "truncated": self.truncated_counts > 0,
        }


def _validated_cell(lattice, frac_coords):
    lattice = np.asarray(lattice, dtype=np.float64)
    frac = np.asarray(frac_coords, dtype=np.float64)
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise ValueError("lattice 必须是有限的 [3,3] 矩阵")
    if frac.ndim != 2 or frac.shape[1] != 3 or len(frac) == 0:
        raise ValueError("frac_coords 必须是非空的 [N,3] 数组")
    if not np.isfinite(frac).all():
        raise ValueError("分数坐标必须全部有限")
    det = float(np.linalg.det(lattice))
    if not math.isfinite(det) or abs(det) < 1e-10:
        raise ValueError("晶胞必须非退化")
    return lattice, np.mod(frac, 1.0)


def build_pbc_neighbors(
    lattice,
    frac_coords,
    cutoff: float = 6.0,
    max_neighbors: int = 32,
) -> PBCNeighbors:
    """Build a complete cutoff candidate list, then retain nearest images per center.

    Integer shifts use row-vector lattice convention: image displacement is
    ``cart[j] + shift @ lattice - cart[i]``. ``multiplicity[i,j]`` counts all
    retained images of atom j around center i, not merely the nearest image.
    """
    lattice, frac = _validated_cell(lattice, frac_coords)
    cutoff = float(cutoff)
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("PBC cutoff 必须是有限正数")
    if isinstance(max_neighbors, bool) or int(max_neighbors) != max_neighbors:
        raise ValueError("max_neighbors 必须是正整数")
    max_neighbors = int(max_neighbors)
    if max_neighbors <= 0:
        raise ValueError("max_neighbors 必须是正整数")

    cart = frac @ lattice
    n = len(cart)
    inverse = np.linalg.inv(lattice)
    reps = [
        int(math.ceil(cutoff * np.linalg.norm(inverse[:, axis]))) + 1
        for axis in range(3)
    ]
    translations = np.asarray(
        list(
            itertools.product(
                range(-reps[0], reps[0] + 1),
                range(-reps[1], reps[1] + 1),
                range(-reps[2], reps[2] + 1),
            )
        ),
        dtype=np.int64,
    )
    cart_shifts = translations @ lattice
    all_t = np.repeat(translations, n, axis=0)
    all_id = np.tile(np.arange(n, dtype=np.int64), len(translations))
    all_c = (cart[None, :, :] + cart_shifts[:, None, :]).reshape(-1, 3)

    neighbor_indices = np.full((n, max_neighbors), -1, dtype=np.int64)
    neighbor_shifts = np.zeros((n, max_neighbors, 3), dtype=np.int64)
    neighbor_distances = np.zeros((n, max_neighbors), dtype=np.float32)
    neighbor_mask = np.zeros((n, max_neighbors), dtype=bool)
    neighbor_counts = np.zeros(n, dtype=np.int64)
    retained_counts = np.zeros(n, dtype=np.int64)
    multiplicity = np.zeros((n, n), dtype=np.int64)
    candidate_multiplicity = np.zeros((n, n), dtype=np.int64)
    pair_disp = np.zeros((n, n, 3), dtype=np.float32)
    pbc_shift = np.zeros((n, n, 3), dtype=np.int64)
    pair_disp_mean = np.zeros((n, n, 3), dtype=np.float32)
    pbc_shift_mean = np.zeros((n, n, 3), dtype=np.float32)
    pair_basis = np.zeros((n, n, 32), dtype=np.float32)

    cutoff2 = cutoff * cutoff
    for i in range(n):
        delta = all_c - cart[i]
        distance2 = np.einsum("ij,ij->i", delta, delta)
        candidate = np.flatnonzero(
            (distance2 > 1e-16) & (distance2 <= cutoff2 + 1e-10)
        )
        distances = np.sqrt(distance2[candidate])
        js = all_id[candidate]
        shifts = all_t[candidate]
        vectors = delta[candidate]
        order = np.lexsort(
            (shifts[:, 2], shifts[:, 1], shifts[:, 0], js, distances)
        )
        candidate = candidate[order]
        distances = distances[order]
        js = js[order]
        shifts = shifts[order]
        vectors = vectors[order]

        total = len(candidate)
        retained = min(total, max_neighbors)
        neighbor_counts[i] = total
        retained_counts[i] = retained
        np.add.at(candidate_multiplicity[i], js, 1)
        if retained == 0:
            continue

        js = js[:retained]
        shifts = shifts[:retained]
        vectors = vectors[:retained]
        distances = distances[:retained]
        neighbor_indices[i, :retained] = js
        neighbor_shifts[i, :retained] = shifts
        neighbor_distances[i, :retained] = distances
        neighbor_mask[i, :retained] = True
        np.add.at(multiplicity[i], js, 1)

        disp_sum = np.zeros((n, 3), dtype=np.float64)
        shift_sum = np.zeros((n, 3), dtype=np.float64)
        np.add.at(disp_sum, js, vectors)
        np.add.at(shift_sum, js, shifts)
        nearest_seen = np.zeros(n, dtype=bool)
        for j, shift, vector, distance in zip(js, shifts, vectors, distances):
            if not nearest_seen[j]:
                pair_disp[i, j] = vector
                pbc_shift[i, j] = shift
                nearest_seen[j] = True
            pair_basis[i, j] += np.exp(
                -PBC_BASIS_GAMMA * (distance - PBC_BASIS_CENTERS) ** 2
            )
        occupied = multiplicity[i] > 0
        pair_disp_mean[i, occupied] = (
            disp_sum[occupied] / multiplicity[i, occupied, None]
        )
        pbc_shift_mean[i, occupied] = (
            shift_sum[occupied] / multiplicity[i, occupied, None]
        )

    return PBCNeighbors(
        lattice=lattice,
        frac_coords=frac,
        coords=(cart - cart.mean(axis=0, keepdims=True)).astype(np.float32),
        neighbor_indices=neighbor_indices,
        neighbor_shifts=neighbor_shifts,
        neighbor_distances=neighbor_distances,
        neighbor_mask=neighbor_mask,
        neighbor_counts=neighbor_counts,
        retained_counts=retained_counts,
        truncated_counts=neighbor_counts - retained_counts,
        multiplicity=multiplicity,
        candidate_multiplicity=candidate_multiplicity,
        pair_disp=pair_disp,
        pbc_shift=pbc_shift,
        pair_disp_mean=pair_disp_mean,
        pbc_shift_mean=pbc_shift_mean,
        pair_basis=pair_basis,
    )


def build_pbc_graph(
    lattice,
    frac_coords,
    elements: List,
    cutoff: float = 6.0,
    max_neighbors: int = 32,
    max_atoms=None,
) -> PBCNeighbors:
    """Compatibility entry point; oversized cells are rejected, never subsampled."""
    n = len(frac_coords)
    if len(elements) != n:
        raise ValueError("elements 数量必须与 frac_coords 完全一致")
    if max_atoms is not None and n > max_atoms:
        raise ValueError(
            f"晶体原子数 {n} 超过 max_atoms={max_atoms}；禁止静默子采样"
        )
    return build_pbc_neighbors(lattice, frac_coords, cutoff, max_neighbors)
