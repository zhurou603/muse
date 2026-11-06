"""冻结 Encoder 推理；长序列不丢 token，表征只依赖单条记录。"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import MuseConfig
from .featurize import FeaturizedRecord, RecordFeaturizer, collate
from .model import MuseEncoder


def _value(inputs: Dict, names, required=True):
    for name in names:
        if name in inputs and inputs[name] is not None:
            return inputs[name]
    if required:
        raise ValueError(f"输入缺少字段: {'/'.join(names)}")
    return None


def _sequence_frontend(inputs: Dict) -> bool:
    if "sequence_frontend" in inputs:
        return bool(inputs["sequence_frontend"])
    if "use_sequence" in inputs:
        return bool(inputs["use_sequence"])
    return isinstance(inputs.get("sequence"), bool) and inputs["sequence"]


def _protein_sequence(inputs: Dict, second: bool = False):
    if second:
        return _value(inputs, ("sequence2", "sequence_2", "protein_sequence_2", "seq2"))
    return _value(
        inputs,
        ("sequence", "protein_sequence", "sequence_1", "protein_sequence_1", "seq1"),
    )


def _copolymer_record(fz, inputs, sequence):
    if "monomers" in inputs:
        record = fz.featurize_polymer(
            monomers=inputs["monomers"],
            composition=inputs["composition"],
            sequence=sequence,
        )
    else:
        monomers = [inputs["monomer_a_smiles"], inputs["monomer_b_smiles"]]
        composition = [inputs["fraction_a"], inputs["fraction_b"]]
        record = fz.featurize_polymer(
            monomers=monomers, composition=composition, sequence=sequence
        )
    for key in ("polymer_type",):
        if key in inputs:
            record.meta[key] = inputs[key]
    if isinstance(inputs.get("composition"), str):
        record.meta["composition_text"] = inputs["composition"]
    return record


def _featurize_record(fz: RecordFeaturizer, rec: Dict) -> FeaturizedRecord:
    """Route by sample_type only; no subset/id-dependent representation logic."""
    sample_type = rec["sample_type"]
    inputs = rec["inputs"]
    if sample_type in ("molecule", "conformer"):
        return fz.featurize_molecule(
            inputs["smiles"],
            coords=inputs.get("coords"),
            sequence=_sequence_frontend(inputs),
            sdf_mol=inputs.get("sdf_mol"),
        )
    if sample_type == "copolymer":
        return _copolymer_record(fz, inputs, _sequence_frontend(inputs))
    if sample_type == "polymer":
        sequence = _sequence_frontend(inputs)
        if "psmiles" in inputs:
            return fz.featurize_polymer(psmiles=inputs["psmiles"], sequence=sequence)
        return _copolymer_record(fz, inputs, sequence)
    if sample_type in ("crystal", "periodic_structure", "periodic_frame"):
        return fz.featurize_crystal(
            inputs["lattice"],
            inputs["frac_coords"],
            inputs["elements"],
            spacegroup_id=inputs.get("spacegroup_id", 0),
            spacegroup=inputs.get("spacegroup"),
            cutoff=inputs.get("cutoff"),
            max_neighbors=inputs.get("max_neighbors"),
        )
    if sample_type in ("protein", "protein_domain", "peptide", "peptide_measurement"):
        return fz.featurize_sequence(_protein_sequence(inputs))
    if sample_type == "protein_pair":
        return fz.featurize_sequence(
            _protein_sequence(inputs), _protein_sequence(inputs, second=True)
        )
    if sample_type == "protein_ligand":
        return fz.featurize_protein_ligand(
            _protein_sequence(inputs),
            _value(inputs, ("ligand_smiles", "smiles", "ligand")),
            ligand_coords=inputs.get("ligand_coords", inputs.get("coords")),
            sdf_mol=inputs.get("sdf_mol"),
        )
    if sample_type == "peptide_atomic":
        annotation = inputs.get("annotation", inputs.get("peptide_annotation"))
        if annotation is None and any(
            key in inputs for key in ("group_ids", "sidechain_flags", "fragment_ids")
        ):
            annotation = {
                key: inputs[key] for key in ("group_ids", "sidechain_flags", "fragment_ids")
                if key in inputs
            }
        return fz.featurize_peptide_atomic(
            smiles=_value(inputs, ("smiles", "peptide_smiles"), required=False),
            sequence=_value(inputs, ("sequence", "protein_sequence"), required=False),
            coords=inputs.get("coords"),
            annotation=annotation,
        )
    raise ValueError(f"未知 sample_type: {sample_type}")


def _torch_batch(records, cfg, device):
    return {
        key: torch.from_numpy(value).to(device)
        for key, value in collate(records, cfg).items()
    }


def _token_bounded_ranges(lengths, cfg, max_batch_size=16):
    if max_batch_size < 1 or cfg.max_seq_len < 1:
        raise ValueError("推理 token 预算必须为正数")
    budget = max_batch_size * cfg.max_seq_len
    start = 0
    maximum = 0
    for index, length in enumerate(lengths):
        if index > start and max(maximum, int(length)) * (index - start + 1) > budget:
            yield start, index
            start = index
            maximum = 0
        maximum = max(maximum, int(length))
    if start < len(lengths):
        yield start, len(lengths)


def _encode_compatible(encoder, batch):
    """New batches stay pair-free; add legacy zero pairs only if an old model asks."""
    try:
        return encoder.encode(batch)
    except KeyError as exc:
        if not exc.args or exc.args[0] not in ("bond_mat", "spd_mat"):
            raise
    legacy = dict(batch)
    token_ids = legacy["token_ids"]
    shape = token_ids.shape
    for key in ("bond_mat", "spd_mat"):
        if key not in legacy:
            legacy[key] = torch.zeros(
                (shape[0], shape[1], shape[1]),
                dtype=torch.int64,
                device=token_ids.device,
            )
    return encoder.encode(legacy)


def _can_window(record: FeaturizedRecord, cfg: MuseConfig) -> bool:
    if cfg.max_seq_len < 2 or len(record.token_ids) <= cfg.max_seq_len:
        return False
    if record.use_atom_features or any(
        value is not None for value in (
            record.coords, record.bond_mat, record.spd_mat,
            record.pair_disp, record.pbc_shift, record.pair_basis,
        )
    ):
        return False
    indices = record.meta.get("output_token_indices")
    if indices is None or record.pool_weights is None:
        return False
    weights = record.pool_weights[1:]
    if len(weights) == 0 or not np.all(weights == weights[0]):
        return False
    return np.array_equal(indices, np.arange(1, len(record.token_ids)))


def _window(record: FeaturizedRecord, start: int, stop: int) -> FeaturizedRecord:
    real = slice(start + 1, stop + 1)
    token_ids = np.r_[record.token_ids[0], record.token_ids[real]].astype(np.int64)
    length = len(token_ids)
    meta = dict(record.meta)
    meta.update({
        "window_start": start,
        "window_stop": stop,
        "output_token_indices": np.arange(1, length, dtype=np.int64),
        "expected_token_count": length - 1,
    })
    return FeaturizedRecord(
        domain=record.domain,
        modality=record.modality,
        token_ids=token_ids,
        use_atom_features=False,
        n_real_tokens=length - 1,
        meta=meta,
        token_kind=np.r_[record.token_kind[0], record.token_kind[real]].astype(np.int64),
        atom_features=np.concatenate([
            record.atom_features[:1], record.atom_features[real]
        ]).astype(np.int64),
        segment_ids=np.r_[record.segment_ids[0], record.segment_ids[real]].astype(np.int64),
        scalar_cond=np.asarray(record.scalar_cond, dtype=np.float32).copy(),
        spacegroup_id=record.spacegroup_id,
        pool_weights=np.r_[0.0, record.pool_weights[real]].astype(np.float32),
        global_features=record.global_features,
    )


def _token_output(record: FeaturizedRecord, encoded: np.ndarray) -> np.ndarray:
    indices = np.asarray(
        record.meta.get(
            "output_token_indices", np.arange(1, record.n_real_tokens + 1)
        ),
        dtype=np.int64,
    )
    if len(indices) != record.n_real_tokens or (
        len(indices) and (indices.min() < 1 or indices.max() >= len(record.token_ids))
    ):
        raise ValueError("output_token_indices 与真实 token 数不一致")
    return encoded[indices]


def _encode_long_record(encoder, fz, cfg, record, batch_size, token_level, device):
    count = len(record.token_ids) - 1
    width = cfg.max_seq_len - 1
    windows = [
        _window(record, start, min(start + width, count))
        for start in range(0, count, width)
    ]
    embedding = np.zeros(cfg.embed_dim, dtype=np.float64)
    total = 0
    token_chunks = []
    for start in range(0, len(windows), batch_size):
        chunk = windows[start:start + batch_size]
        output = _encode_compatible(encoder, _torch_batch(chunk, cfg, device))
        embeddings = output["embedding"].float().cpu().numpy()
        encoded = output["tokens"].float().cpu().numpy()
        for i, window in enumerate(chunk):
            n = window.n_real_tokens
            embedding += embeddings[i].astype(np.float64) * n
            total += n
            if token_level:
                token_chunks.append(_token_output(window, encoded[i]))
    if total != count:
        raise RuntimeError("分窗编码丢失了序列 token")
    tokens = np.concatenate(token_chunks, axis=0) if token_level else None
    return (embedding / total).astype(np.float32), tokens


def _expected_tokens(raw: Dict, record: FeaturizedRecord) -> int:
    sample_type = raw["sample_type"]
    inputs = raw["inputs"]
    if sample_type in ("protein", "peptide", "peptide_measurement"):
        sequence = _protein_sequence(inputs)
        return sum(not char.isspace() for char in sequence.strip())
    if sample_type == "protein_pair":
        first = _protein_sequence(inputs)
        second = _protein_sequence(inputs, second=True)
        return sum(not char.isspace() for text in (first, second) for char in text.strip())
    return int(record.meta.get("expected_token_count", record.n_real_tokens))


@torch.no_grad()
def encode_subset(
    encoder: MuseEncoder,
    fz: RecordFeaturizer,
    cfg: MuseConfig,
    records: List[Dict],
    batch_size: int = 256,
    token_level: bool = False,
    device: str = "cpu",
):
    """Return ids, graph embeddings, optional real-token embeddings and offsets.

    Splittable sequences longer than ``max_seq_len`` use deterministic adjacent
    windows and length-weighted window embeddings. Other records use an explicit
    full-length path. Neither path drops or truncates tokens.
    """
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正数")
    encoder.eval()
    ids = [record["id"] for record in records]
    embeddings = np.zeros((len(records), cfg.embed_dim), dtype=np.float32)
    token_chunks: List[Optional[np.ndarray]] = [None] * len(records)
    featurized: List[Optional[FeaturizedRecord]] = [None] * len(records)

    for start in range(0, len(records), batch_size):
        raw_chunk = records[start:start + batch_size]
        regular = []
        for local, raw in enumerate(raw_chunk):
            index = start + local
            record = _featurize_record(fz, raw)
            featurized[index] = record
            if _can_window(record, cfg):
                embedding, tokens = _encode_long_record(
                    encoder, fz, cfg, record, batch_size, token_level, device
                )
                embeddings[index] = embedding
                if token_level:
                    token_chunks[index] = tokens
            else:
                regular.append((index, record))
        if regular:
            batch = _torch_batch([record for _, record in regular], cfg, device)
            output = _encode_compatible(encoder, batch)
            batch_embeddings = output["embedding"].float().cpu().numpy()
            encoded = output["tokens"].float().cpu().numpy()
            for i, (index, record) in enumerate(regular):
                embeddings[index] = batch_embeddings[i]
                if token_level:
                    token_chunks[index] = _token_output(record, encoded[i])

    if not np.isfinite(embeddings).all():
        raise ValueError("表征含 NaN/Inf")
    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    tokens = None
    if token_level:
        for i, (raw, chunk) in enumerate(zip(records, token_chunks)):
            if chunk is None:
                raise RuntimeError(f"记录 {raw['id']} 缺少逐 token 输出")
            expected = _expected_tokens(raw, featurized[i])
            if len(chunk) != expected:
                raise RuntimeError(
                    f"记录 {raw['id']} 的 token 数 {len(chunk)} != 原始长度 {expected}"
                )
            offsets[i + 1] = offsets[i] + len(chunk)
        if token_chunks:
            tokens = np.concatenate(token_chunks, axis=0).astype(np.float32)
        else:
            tokens = np.empty((0, cfg.embed_dim), dtype=np.float32)
        if not np.isfinite(tokens).all():
            raise ValueError("逐残基表征含 NaN/Inf")
    return ids, embeddings, tokens, offsets


def write_subset(
    out_dir: str,
    ids: List[str],
    embs: np.ndarray,
    tokens: Optional[np.ndarray] = None,
    offsets: Optional[np.ndarray] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ids.txt"), "w", encoding="utf-8") as stream:
        stream.write("\n".join(str(value) for value in ids) + "\n")
    np.save(os.path.join(out_dir, "embeddings.npy"), embs)
    if tokens is not None:
        if offsets is None:
            raise ValueError("写出 tokens.npy 时必须提供 token_offsets")
        np.save(os.path.join(out_dir, "tokens.npy"), tokens)
        np.save(os.path.join(out_dir, "token_offsets.npy"), offsets)


def encode_jsonl_file(
    encoder,
    fz,
    cfg,
    jsonl_path: str,
    out_dir: str,
    token_level: bool = False,
    batch_size: int = 256,
    device: str = "cpu",
):
    with open(jsonl_path, encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    ids, embeddings, tokens, offsets = encode_subset(
        encoder, fz, cfg, records, batch_size, token_level, device
    )
    write_subset(out_dir, ids, embeddings, tokens, offsets)
    return len(records)
