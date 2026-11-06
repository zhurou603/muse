from __future__ import annotations

import numpy as np
import torch

from ..config import DOMAIN_CRYSTAL
from ..featurize import ATOM_ELEMENT_INDEX, collate
from ..masking import (
    IGNORE,
    peptide_aware_mask,
    random_atom_mask,
    random_mlm_mask,
    sample_coordinate_noise,
)
from ..tokenizers import default_tokenizers


def _align_aux_to_record(record, value, mask, kind, dim, name):
    if kind == "graph":
        return value, mask
    count = len(record.token_ids) - 1
    expected_count = count if kind == "atom" else len(record.meta["bond_pairs"])
    expected = (expected_count, dim)
    if value.shape == expected and mask.shape == expected:
        return value, mask
    mapping = record.meta.get("sdf_mapping")
    if mapping is None:
        raise ValueError(f"aux[{name}] 与{'原子' if kind == 'atom' else '键'}数不一致，且缺少结构映射")
    mapping = np.asarray(mapping, dtype=np.int64)
    if kind == "atom":
        if value.ndim != 2 or value.shape != mask.shape or value.shape[1] != dim \
                or len(mapping) != count or (mapping.size and
                (mapping.min() < 0 or mapping.max() >= value.shape[0])):
            raise ValueError(f"aux[{name}] 原子标签或结构映射形状非法")
        value, mask = value[mapping], mask[mapping]
    else:
        source_pairs = record.meta.get("sdf_bond_pairs")
        if source_pairs is None or value.ndim != 2 or value.shape != mask.shape \
                or value.shape[1] != dim or len(source_pairs) != value.shape[0] \
                or len(mapping) != count:
            raise ValueError(f"aux[{name}] 键标签或结构映射形状非法")
        lookup = {
            tuple(sorted((int(left), int(right)))): index
            for index, (left, right) in enumerate(source_pairs)
        }
        if len(lookup) != len(source_pairs):
            raise ValueError(f"aux[{name}] 源结构包含重复键")
        indices = []
        for left, right in record.meta["bond_pairs"]:
            pair = tuple(sorted((int(mapping[left]), int(mapping[right]))))
            if pair not in lookup:
                raise ValueError(f"aux[{name}] 找不到目标键对应的源结构键: {pair}")
            indices.append(lookup[pair])
        value, mask = value[indices], mask[indices]
    if value.shape != expected or mask.shape != expected:
        raise ValueError(f"aux[{name}] 映射后形状 {value.shape}/{mask.shape} != {expected}")
    return value, mask


class BatchBuilder:
    def __init__(self, cfg, tokenizers=None):
        self.cfg = cfg
        self.tokenizers = tokenizers or default_tokenizers()

    def _mask(self, batch, seed):
        token_ids = torch.from_numpy(batch["token_ids"])
        pad_mask = torch.from_numpy(batch["pad_mask"])
        token_kind = torch.from_numpy(batch["token_kind"])
        generator = torch.Generator().manual_seed(seed)
        masked = token_ids.clone()
        labels = torch.full_like(token_ids, IGNORE)
        peptide_records = np.asarray([
            record.meta.get("sample_type") == "peptide_atomic"
            for record in batch["_records"]
        ])
        peptide = torch.from_numpy(peptide_records)
        head_map = torch.tensor([3, 0, 1, 2], dtype=torch.long)
        mlm_head = head_map[token_kind]

        for kind, name, head in ((1, "smiles", 0), (2, "psmiles", 1), (3, "residue", 2)):
            eligible = pad_mask & (token_kind == kind)
            if not eligible.any():
                continue
            tokenizer = self.tokenizers[name]
            candidate, candidate_labels = random_mlm_mask(
                token_ids, eligible, tokenizer.mask_id,
                getattr(self.cfg, f"{name}_vocab_size"),
                self.cfg.mlm_mask_prob, generator,
                special_token_ids=(
                    tokenizer.pad_id, tokenizer.unk_id, tokenizer.mask_id,
                    tokenizer.cls_id, tokenizer.sep_id, tokenizer.eos_id,
                ),
            )
            selected = candidate_labels != IGNORE
            masked[selected] = candidate[selected]
            labels[selected] = candidate_labels[selected]

        atom_eligible = pad_mask & (token_kind == 0) & (~peptide)[:, None]
        if atom_eligible.any():
            candidate, candidate_labels = random_atom_mask(
                token_ids, atom_eligible, self.cfg.mlm_mask_prob, generator,
                self.cfg.atom_mask_id,
            )
            selected = candidate_labels != IGNORE
            masked[selected] = candidate[selected]
            labels[selected] = candidate_labels[selected]

        for index in np.flatnonzero(peptide_records):
            eligible = pad_mask[index:index + 1] & (token_kind[index:index + 1] == 0)
            kwargs = {
                key: torch.from_numpy(batch[key][index:index + 1])
                for key in ("group_ids", "sidechain_flags", "fragment_ids")
            }
            candidate, candidate_labels = peptide_aware_mask(
                token_ids[index:index + 1], eligible,
                mask_prob=self.cfg.mlm_mask_prob, generator=generator,
                atom_mask_id=self.cfg.atom_mask_id, strict=True, **kwargs,
            )
            selected = candidate_labels != IGNORE
            masked[index:index + 1][selected] = candidate[selected]
            labels[index:index + 1][selected] = candidate_labels[selected]

        batch["token_ids"] = masked.numpy()
        atom_selected = (labels != IGNORE) & (token_kind == 0)
        features = torch.from_numpy(batch["atom_features"])
        features[..., ATOM_ELEMENT_INDEX][atom_selected] = self.cfg.atom_mask_id
        batch["atom_features"] = features.numpy()
        return {
            "mlm_labels": labels,
            "mlm_head": mlm_head,
        }

    def _denoise(self, batch, targets, seed):
        coords = torch.from_numpy(batch["coords"])
        coord_mask = torch.from_numpy(batch["coord_mask"])
        active = coord_mask.any(dim=1)
        generator = torch.Generator().manual_seed(seed)
        noisy, noise, sigma, noise_mask = sample_coordinate_noise(
            coords, coord_mask,
            sigma_min=self.cfg.denoise_sigma_min,
            sigma_max=self.cfg.denoise_sigma_max,
            generator=generator, return_noise_mask=True,
        )
        batch["coords"] = noisy.numpy()
        batch["denoise_sigma"] = sigma.numpy()
        batch["denoise_active"] = active.numpy()
        if "pair_disp" in batch:
            pair_disp = noisy[:, None, :, :] - noisy[:, :, None, :]
            if "pbc_shift" in batch:
                shift_cart = torch.einsum(
                    "bxyi,bjk->bxyk",
                    torch.from_numpy(batch["pbc_shift"]).to(dtype=noisy.dtype),
                    torch.from_numpy(batch["lattice"]).to(dtype=noisy.dtype),
                )
                pair_disp = pair_disp + shift_cart
            batch["pair_disp"] = pair_disp.numpy()
        if "pair_basis_mask" in batch:
            batch["pair_basis_mask"] = (
                torch.from_numpy(batch["pair_basis_mask"]) & ~active).numpy()
        drop = (torch.rand(len(active), generator=generator) < self.cfg.pair_mask_prob) & active
        batch["drop_feature_mask"] = drop.numpy()
        targets["drop_feature_mask"] = drop
        if active.any():
            targets["noise"] = noise
            targets["noise_mask"] = noise_mask
        molecular = active & torch.from_numpy(batch["domain"] != DOMAIN_CRYSTAL)
        if molecular.any():
            targets["coords"] = coords
            targets["coord_mask"] = coord_mask & molecular[:, None]

    def build(self, records, seed: int, mlm: bool = True, denoise: bool = False,
              items=None):
        arrays = collate(records, self.cfg)
        arrays["_records"] = records
        targets = {}
        if mlm:
            targets.update(self._mask(arrays, seed))
        if denoise:
            self._denoise(arrays, targets, seed + 1)
        else:
            arrays["drop_feature_mask"] = np.zeros(len(records), dtype=bool)
        arrays.pop("_records")
        batch = {
            key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            for key, value in arrays.items()
        }
        targets = {
            key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            for key, value in targets.items()
        }
        if items is not None:
            targets.update(collate_metadata(records, items, self.cfg))
        return batch, targets


def collate_metadata(records, items, cfg):
    if len(records) != len(items):
        raise ValueError("records/items 数量不一致")
    output = {}
    if any("descriptor" in item for item in items):
        values = np.zeros((len(items), cfg.descriptor_dim), dtype=np.float32)
        masks = np.zeros_like(values, dtype=bool)
        for index, item in enumerate(items):
            if "descriptor" in item:
                value, mask = item["descriptor"]
                values[index], masks[index] = value, mask
        output["descriptor"] = torch.from_numpy(values)
        output["descriptor_mask"] = torch.from_numpy(masks)

    for key in ("energy", "stress"):
        if any(key in item for item in items):
            width = 1 if key == "energy" else 6
            values = np.zeros((len(items), width), dtype=np.float32)
            masks = np.zeros_like(values, dtype=bool)
            for index, item in enumerate(items):
                if key in item:
                    value, mask = item[key]
                    values[index], masks[index] = value, mask
            if key == "energy":
                values, masks = values[:, 0], masks[:, 0]
            output[key] = torch.from_numpy(values)
            output[f"{key}_mask"] = torch.from_numpy(masks)
    if any("force" in item for item in items):
        length = max(len(record.token_ids) for record in records)
        values = np.zeros((len(items), length, 3), dtype=np.float32)
        masks = np.zeros((len(items), length), dtype=bool)
        for index, (record, item) in enumerate(zip(records, items)):
            if "force" not in item:
                continue
            value, mask = item["force"]
            if mask.ndim == 2:
                mask = mask.any(axis=1)
            count = len(record.token_ids) - 1
            if value.shape != (count, 3) or mask.shape != (count,):
                raise ValueError("force 与原子数不一致")
            values[index, 1:count + 1] = value
            masks[index, 1:count + 1] = mask
        output["force"] = torch.from_numpy(values)
        output["force_mask"] = torch.from_numpy(masks)

    grouped = {}
    for index, (record, item) in enumerate(zip(records, items)):
        for aux in item.get("aux_records", []):
            grouped.setdefault(aux["name"], []).append((index, record, aux))
    for name, entries in grouped.items():
        spec = cfg.aux_specs[name]
        dim = spec["dim"]
        kind = spec["kind"]
        if kind == "graph":
            values = np.zeros((len(items), dim), dtype=np.float32)
            masks = np.zeros_like(values, dtype=bool)
        elif kind == "atom":
            length = max(len(record.token_ids) for record in records)
            values = np.zeros((len(items), length, dim), dtype=np.float32)
            masks = np.zeros_like(values, dtype=bool)
        else:
            length = max(len(record.token_ids) for record in records)
            values = np.zeros((len(items), length, length, dim), dtype=np.float32)
            masks = np.zeros_like(values, dtype=bool)
        for index, record, aux in entries:
            value, mask = _align_aux_to_record(
                record, aux["value"], aux["mask"], kind, dim, name)
            if not mask.any():
                continue
            if kind == "graph":
                values[index], masks[index] = value.reshape(dim), mask.reshape(dim)
            elif kind == "atom":
                count = len(record.token_ids) - 1
                if value.shape != (count, dim) or mask.shape != (count, dim):
                    raise ValueError(f"aux[{name}] 与原子数不一致")
                values[index, 1:count + 1] = value
                masks[index, 1:count + 1] = mask
            else:
                pairs = record.meta["bond_pairs"]
                if value.shape != (len(pairs), dim) or mask.shape != (len(pairs), dim):
                    raise ValueError(f"aux[{name}] 与键数不一致")
                for pair_index, (left, right) in enumerate(pairs):
                    for a, b in ((left + 1, right + 1), (right + 1, left + 1)):
                        values[index, a, b] = value[pair_index]
                        masks[index, a, b] = mask[pair_index]
        output.setdefault("aux_records", []).append({
            "name": name, "value": torch.from_numpy(values),
            "mask": torch.from_numpy(masks),
        })
    return output
