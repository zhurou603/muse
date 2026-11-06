"""Training-only deterministic chemical token crop.

The base featurizer must retain complete chemical sequences for inference.  This
subclass is installed only on training datasets, after featurization and before
batching, so a long SMILES/PSMILES cannot OOM the training batch.
"""
from __future__ import annotations

import hashlib

import numpy as np

from muse_encoder.featurize import RecordFeaturizer


class TrainingRecordFeaturizer(RecordFeaturizer):
    def __init__(self, *args, crop_seed: int = 42, **kwargs):
        super().__init__(*args, **kwargs)
        self.crop_seed = int(crop_seed)

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
