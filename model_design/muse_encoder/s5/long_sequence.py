"""Training-only deterministic chemical-token and residue-sequence crop.

The base featurizer retains complete sequences for inference. This subclass is
installed only on training datasets, so long inputs cannot OOM the training batch.
"""
from __future__ import annotations

import hashlib

import numpy as np

from muse_encoder.featurize import RecordFeaturizer


class TrainingRecordFeaturizer(RecordFeaturizer):
    def __init__(self, *args, crop_seed: int = 42, **kwargs):
        super().__init__(*args, **kwargs)
        self.crop_seed = int(crop_seed)

    def _crop_residue(self, text: str, capacity: int):
        chars = [char for char in text.strip() if not char.isspace()]
        count = len(chars)
        if count <= capacity:
            return "".join(chars), 0, count, count
        digest = hashlib.sha256(f"{self.crop_seed}\0{text}".encode("utf-8")).digest()
        start = int.from_bytes(digest[:8], "little") % (count - capacity + 1)
        return "".join(chars[start:start + capacity]), start, start + capacity, count

    def featurize_sequence(self, seq: str, seq2=None):
        first_count = len([char for char in seq.strip() if not char.isspace()])
        second_count = 0 if seq2 is None else len([
            char for char in seq2.strip() if not char.isspace()])
        capacity = int(self.cfg.max_seq_len) - (1 if seq2 is None else 2)
        if capacity < 1:
            raise ValueError("max_seq_len 不足以容纳 CLS/SEP 与真实残基")
        if first_count + second_count <= capacity:
            return super().featurize_sequence(seq, seq2)
        if seq2 is None:
            first_capacity, second_capacity = capacity, 0
        else:
            second_capacity = min(second_count, capacity // 2)
            first_capacity = min(first_count, capacity - second_capacity)
            second_capacity = min(second_count, capacity - first_capacity)
        first, first_start, first_stop, first_original = self._crop_residue(
            seq, first_capacity)
        second = None
        second_start = second_stop = second_original = 0
        if seq2 is not None:
            second, second_start, second_stop, second_original = self._crop_residue(
                seq2, second_capacity)
        record = super().featurize_sequence(first, second)
        record.meta.update({
            "truncated": True, "training_crop": True,
            "training_crop_start": first_start, "training_crop_stop": first_stop,
            "training_original_tokens": first_original + second_original,
        })
        if seq2 is not None:
            record.meta.update({
                "training_crop_start_2": second_start, "training_crop_stop_2": second_stop,
                "training_original_tokens_1": first_original,
                "training_original_tokens_2": second_original,
            })
        return record

    def _chemical_sequence(self, text, domain, kind, tokenizer_name, strict):
        full = super()._chemical_sequence(text, domain, kind, tokenizer_name, strict)
        capacity = int(self.cfg.max_seq_len) - 1
        if capacity < 1:
            raise ValueError("max_seq_len 必须至少为 2（CLS + 一个真实 token）")
        count = len(full.token_ids) - 1
        if count <= capacity:
            return full
        digest = hashlib.sha256(f"{self.crop_seed}\0{text}".encode("utf-8")).digest()
        start = int.from_bytes(digest[:8], "little") % (count - capacity + 1)
        stop = start + capacity
        real = slice(start + 1, stop + 1)
        length = capacity + 1
        meta = dict(full.meta)
        meta.update({
            "truncated": True,
            "training_crop": True,
            "training_crop_start": start,
            "training_crop_stop": stop,
            "training_original_tokens": count,
            "output_token_indices": np.arange(1, length, dtype=np.int64),
            "expected_token_count": capacity,
        })
        return type(full)(
            domain=full.domain,
            modality=full.modality,
            token_ids=np.r_[full.token_ids[0], full.token_ids[real]].astype(np.int64),
            use_atom_features=False,
            n_real_tokens=capacity,
            meta=meta,
            token_kind=np.r_[full.token_kind[0], full.token_kind[real]].astype(np.int64),
            atom_features=np.concatenate([full.atom_features[:1], full.atom_features[real]]),
            segment_ids=np.r_[full.segment_ids[0], full.segment_ids[real]].astype(np.int64),
            scalar_cond=np.asarray(full.scalar_cond, dtype=np.float32).copy(),
            spacegroup_id=full.spacegroup_id,
            pool_weights=np.r_[0.0, full.pool_weights[real]].astype(np.float32),
            global_features=full.global_features,
        )
