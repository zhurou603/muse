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


class MissingChemicalInput(ValueError):
    """The input is an explicit missing sentinel, not a molecule."""


_PARSE_MODE = "_muse_parse_mode"
_DEGRADED = "_muse_degraded"
_ALIGNMENT_MODE = "_muse_alignment_mode"
_TOTAL_HS = "_muse_total_hs"
_MISSING_TOKENS = {"", "nan", "na", "n/a", "none", "null", "."}


def is_missing_chemical_input(value) -> bool:
    return not isinstance(value, str) or value.strip().casefold() in _MISSING_TOKENS


def _set_mol_mode(mol, mode: str, degraded: bool = False):
    mol.SetProp(_PARSE_MODE, mode)
    mol.SetBoolProp(_DEGRADED, bool(degraded))
    return mol


def mol_degraded(mol) -> bool:
    return bool(mol is not None and mol.HasProp(_DEGRADED) and mol.GetBoolProp(_DEGRADED))


def mol_mode(mol, default="strict") -> str:
    return mol.GetProp(_PARSE_MODE) if mol is not None and mol.HasProp(_PARSE_MODE) else default


def alignment_mode(mol) -> str | None:
    return mol.GetProp(_ALIGNMENT_MODE) if mol is not None and mol.HasProp(_ALIGNMENT_MODE) else None


def alignment_error(mol) -> str | None:
    return mol.GetProp("_muse_alignment_error") \
        if mol is not None and mol.HasProp("_muse_alignment_error") else None


def _strip_ligand_annotation(smiles: str):
    """Separate only explicit PDB ligand annotations, never arbitrary SMILES text."""
    annotations = []
    chemical = smiles.strip()
    if "\t" in chemical:
        chemical, suffix = chemical.split("\t", 1)
        if suffix.isascii() and suffix.replace("_", "").isalnum() and suffix.endswith("_LIGAND"):
            annotations.append(suffix)
        else:
            chemical = smiles.strip()
    experimental = "_experimental"
    if chemical.endswith(experimental):
        candidate = chemical[:-len(experimental)]
        if candidate:
            chemical = candidate
            annotations.append(experimental.lstrip("_"))
    return chemical, annotations


def _heavy_with_hydrogen_metadata(chem, mol, sanitize: bool):
    """Remove H atoms while retaining their original per-heavy-atom counts."""
    mol.UpdatePropertyCache(strict=bool(sanitize))
    counts = [atom.GetTotalNumHs(includeNeighbors=True)
              for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1]
    heavy = chem.RemoveAllHs(mol, sanitize=bool(sanitize))
    if heavy.GetNumAtoms() != len(counts):
        raise ValueError("重原子去氢后数量发生变化")
    for atom, count in zip(heavy.GetAtoms(), counts):
        atom.SetIntProp(_TOTAL_HS, int(count))
    return heavy


def _hydrogen_count(atom) -> int:
    return int(atom.GetIntProp(_TOTAL_HS)) if atom.HasProp(_TOTAL_HS) else int(atom.GetTotalNumHs())


def graph_from_smiles(smiles: str, allow_unsanitized: bool = False):
    """Return a heavy graph without inventing atoms, bonds or connectivity.

    Strict sanitization remains the first path. Only an explicit caller may allow
    ``sanitize=False`` parsing, which is marked ``unsanitized_topology``. Missing
    sentinels and syntax that RDKit cannot tokenize still fail closed.
    """
    chem = require_rdkit()
    if is_missing_chemical_input(smiles):
        raise MissingChemicalInput("化学输入缺失或显式为空")
    chemical, annotations = _strip_ligand_annotation(smiles)
    if is_missing_chemical_input(chemical):
        raise MissingChemicalInput("化学输入缺失或显式为空")
    mol = chem.MolFromSmiles(chemical)
    if mol is not None:
        heavy = _heavy_with_hydrogen_metadata(chem, chem.AddHs(mol, addCoords=True), True)
        _set_mol_mode(heavy, "strict", False)
    else:
        if not allow_unsanitized:
            raise ValueError(f"RDKit 无法解析: {chemical[:80]}")
        mol = chem.MolFromSmiles(chemical, sanitize=False)
        if mol is None:
            raise ValueError(f"RDKit 无法解析: {chemical[:80]}")
        heavy = _heavy_with_hydrogen_metadata(chem, mol, False)
        _set_mol_mode(heavy, "unsanitized_topology", True)
    if heavy.GetNumAtoms() < 1:
        raise ValueError("化学图没有重原子")
    for index, annotation in enumerate(annotations):
        heavy.SetProp(f"_muse_annotation_{index}", annotation)
    return heavy


def mol_from_sdf_bytes(block: bytes, record_id: str, allow_unsanitized: bool = True):
    """Parse one official SDF block, retaining coordinates and parse provenance."""
    import io

    chem = require_rdkit()
    supplier = chem.ForwardSDMolSupplier(io.BytesIO(block), removeHs=False, sanitize=True)
    mol = next(supplier, None)
    if mol is not None:
        return _set_mol_mode(mol, "strict", False)
    if not allow_unsanitized:
        raise ValueError(f"SDF 严格解析失败: {record_id}")
    supplier = chem.ForwardSDMolSupplier(io.BytesIO(block), removeHs=False, sanitize=False)
    mol = next(supplier, None)
    if mol is None:
        raise ValueError(f"SDF 严格解析失败，非严格解析也失败: {record_id}")
    mol.UpdatePropertyCache(strict=False)
    return _set_mol_mode(mol, "unsanitized_topology", True)


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
                _hydrogen_count(target_atom),
            )
            source_signature = (
                source_atom.GetAtomicNum(),
                source_atom.GetIsotope(),
                source_atom.GetFormalCharge(),
                _hydrogen_count(source_atom),
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

        original_h_counts = [_hydrogen_count(atom) for atom in source.GetAtoms()]
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


def _topology_mapping(chem, source, target, max_candidates=10000):
    """Map target->source using connectivity and conserved atom attributes.

    Bond order is intentionally compared only by connectivity here: this path is
    for aromatic/Kekulé and SDF bond-perception differences.  Atomic number,
    isotope, formal charge, original hydrogen count and radical count must match
    at every mapped atom.  Specified stereo conflicts are rejected below.
    """
    n = target.GetNumAtoms()
    if source.GetNumAtoms() != n or source.GetNumBonds() != target.GetNumBonds():
        return None

    def signature(atom):
        return (
            atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge(),
            _hydrogen_count(atom), atom.GetNumRadicalElectrons(),
        )

    candidates = {
        index: [j for j in range(n)
                if signature(target.GetAtomWithIdx(index)) == signature(source.GetAtomWithIdx(j))]
        for index in range(n)
    }
    if any(not values for values in candidates.values()):
        return None
    order = sorted(range(n), key=lambda index: (
        len(candidates[index]), -target.GetAtomWithIdx(index).GetDegree(), index))
    mapping = [-1] * n
    used = set()
    complete = 0

    def stereo_compatible(candidate):
        try:
            if source.GetNumConformers() == 1:
                chem.AssignStereochemistryFrom3D(source)
            chem.AssignStereochemistry(target, cleanIt=True, force=True)
        except (ValueError, RuntimeError):
            pass
        for i, j in enumerate(candidate):
            target_atom = target.GetAtomWithIdx(i)
            if not target_atom.HasProp("_CIPCode"):
                continue
            source_atom = source.GetAtomWithIdx(j)
            if not source_atom.HasProp("_CIPCode") \
                    or source_atom.GetProp("_CIPCode") != target_atom.GetProp("_CIPCode"):
                return False
        for bond in target.GetBonds():
            if bond.GetStereo() == chem.BondStereo.STEREONONE:
                continue
            other = source.GetBondBetweenAtoms(
                candidate[bond.GetBeginAtomIdx()], candidate[bond.GetEndAtomIdx()])
            if other is None or other.GetStereo() != bond.GetStereo():
                return False
        return True

    def search(depth):
        nonlocal complete
        if depth == n:
            complete += 1
            return tuple(mapping) if stereo_compatible(mapping) else None
        if complete >= max_candidates:
            return None
        index = order[depth]
        for source_index in candidates[index]:
            if source_index in used:
                continue
            valid = True
            for bond in target.GetAtomWithIdx(index).GetBonds():
                other_index = bond.GetOtherAtomIdx(index)
                if mapping[other_index] >= 0 and source.GetBondBetweenAtoms(
                        source_index, mapping[other_index]) is None:
                    valid = False
                    break
            if not valid:
                continue
            mapping[index] = source_index
            used.add(source_index)
            result = search(depth + 1)
            if result is not None:
                return result
            used.remove(source_index)
            mapping[index] = -1
        return None

    return search(0)


def _alignment_fallback(target, message, allow_1d_fallback):
    if not allow_1d_fallback:
        raise ValueError(message)
    target.SetProp(_ALIGNMENT_MODE, "smiles_1d_unmatched")
    target.SetProp("_muse_alignment_error", str(message)[:500])
    _set_mol_mode(target, mol_mode(target), True)
    return target, None, None


def align_coordinates(smiles: str, sdf_mol, allow_1d_fallback: bool = False):
    """Align official coordinates only after proving atom-level graph identity.

    Exact RDKit matching is tried first.  A bounded topology matcher additionally
    handles unsanitized SDF and aromatic/Kekulé differences while conserving the
    original per-atom hydrogen counts.  If identity cannot be proved, the default
    strict path raises.  A-test inference may explicitly request a marked 1D
    fallback; it never receives reordered, generated or mismatched coordinates.
    """
    chem = require_rdkit()
    if sdf_mol is None:
        raise ValueError("SDF 解析失败")
    target = graph_from_smiles(smiles)
    try:
        source = _heavy_with_hydrogen_metadata(chem, sdf_mol, False)
        if source.GetNumConformers() != 1:
            raise ValueError("需要恰好一个SDF构象")
        n = target.GetNumAtoms()
        mapping = None
        exact_error = None
        if (source.GetNumAtoms(), source.GetNumBonds()) == (n, target.GetNumBonds()):
            try:
                source_graph, target_graph = chem.Mol(source), chem.Mol(target)
                chem.RemoveStereochemistry(source_graph)
                chem.RemoveStereochemistry(target_graph)
                same_graph = chem.MolToSmiles(source_graph, isomericSmiles=True) \
                    == chem.MolToSmiles(target_graph, isomericSmiles=True)
                if same_graph:
                    candidates = source.GetSubstructMatches(
                        target, useChirality=True, uniquify=False, maxMatches=10001)
                    if len(candidates) >= 10001:
                        raise RuntimeError("原子映射候选超过10000，停止，不能确认完整搜索")
                    for candidate in candidates:
                        if len(candidate) != n or len(set(candidate)) != n:
                            continue
                        if any(
                                (target.GetAtomWithIdx(i).GetAtomicNum(),
                                 target.GetAtomWithIdx(i).GetIsotope(),
                                 target.GetAtomWithIdx(i).GetFormalCharge(),
                                 target.GetAtomWithIdx(i).GetIsAromatic(),
                                 _hydrogen_count(target.GetAtomWithIdx(i)))
                                != (source.GetAtomWithIdx(j).GetAtomicNum(),
                                    source.GetAtomWithIdx(j).GetIsotope(),
                                    source.GetAtomWithIdx(j).GetFormalCharge(),
                                    source.GetAtomWithIdx(j).GetIsAromatic(),
                                    _hydrogen_count(source.GetAtomWithIdx(j)))
                                or target.GetAtomWithIdx(i).GetNumRadicalElectrons()
                                != source.GetAtomWithIdx(j).GetNumRadicalElectrons()
                                for i, j in enumerate(candidate)):
                            continue
                        valid = True
                        for bond in target.GetBonds():
                            other = source.GetBondBetweenAtoms(
                                candidate[bond.GetBeginAtomIdx()], candidate[bond.GetEndAtomIdx()])
                            if other is None or (bond.GetBondType(), bond.GetIsAromatic()) != (
                                    other.GetBondType(), other.GetIsAromatic()):
                                valid = False
                                break
                        if valid:
                            mapping = tuple(candidate)
                            break
                else:
                    mapping = _aromatic_representation_mapping(chem, source, target)
            except (ValueError, RuntimeError) as exc:
                exact_error = exc
            if mapping is None:
                mapping = _topology_mapping(chem, source, target)
                mode = "topology_h_conserving" if mapping is not None else None
            else:
                mode = "exact"
        else:
            exact_error = ValueError("SMILES/SDF原子数或键数不一致")
            mode = None
        if mapping is None:
            message = exact_error or "SMILES/SDF非立体化学结构不一致（连接、键、电荷或同位素等）"
            return _alignment_fallback(target, message, allow_1d_fallback)
        original = np.asarray(source.GetConformer().GetPositions(), dtype=np.float64)
        if original.shape != (n, 3) or not np.isfinite(original).all():
            return _alignment_fallback(target, "SDF坐标非法", allow_1d_fallback)
        aligned = original[np.asarray(mapping, dtype=np.int64)]
        target.SetProp(_ALIGNMENT_MODE, mode)
        _set_mol_mode(target, mol_mode(target),
                      mol_degraded(target) or mol_degraded(source) or mode != "exact")
        return target, aligned, tuple(mapping)
    except MissingChemicalInput:
        raise
    except Exception as exc:
        return _alignment_fallback(target, str(exc), allow_1d_fallback)


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
