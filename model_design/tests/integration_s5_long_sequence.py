"""Run after install.sh on the real repo; CPU is sufficient, RDKit not required."""
from __future__ import annotations

import argparse

import numpy as np
import torch

from muse_encoder.config import MuseConfig
from muse_encoder.featurize import RecordFeaturizer
from muse_encoder.inference import _can_window, _window, encode_subset
from muse_encoder.model import MuseEncoder
from muse_encoder.s5.long_sequence import TrainingRecordFeaturizer
from muse_encoder.tokenizers import default_tokenizers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-tokenizers", action="store_true",
                        help="在原机使用生产 load_tokenizers()，替代审查快照的内置词表")
    args = parser.parse_args()
    torch.manual_seed(7)
    cfg = MuseConfig(
        d_model=32, n_layers=2, n_heads=4, d_ffn=64, num_experts=2, mole_every=2,
        embed_dim=32, contrastive_dim=16, max_seq_len=16, gradient_checkpointing=False,
    )
    # Synthetic integration uses built-in vocabularies so the reviewed code
    # snapshot can run without the separately stored production vocabulary.
    if args.production_tokenizers:
        from muse_encoder.tokenizers import load_tokenizers
        tokenizers = load_tokenizers()
    else:
        tokenizers = default_tokenizers()
    fz = RecordFeaturizer(cfg, tokenizers, strict=False)
    text = "C" * 80
    record = fz.featurize_molecule(text, sequence=True, strict=False)
    assert record.n_real_tokens > 15, "化学序列仍在特征化阶段截断"
    assert _can_window(record, cfg)
    count = record.n_real_tokens
    width = cfg.max_seq_len - 1
    windows = [_window(record, start, min(start + width, count)) for start in range(0, count, width)]
    assert sum(item.n_real_tokens for item in windows) == count
    assert windows[-1].token_ids[-1] == record.token_ids[-1]

    train_fz = TrainingRecordFeaturizer(cfg, tokenizers, crop_seed=42)
    cropped = train_fz.featurize_molecule(text, sequence=True, strict=False)
    assert len(cropped.token_ids) == cfg.max_seq_len
    assert cropped.meta["training_original_tokens"] == count

    model = MuseEncoder(cfg).eval()
    long_raw = {"id": "long", "sample_type": "molecule", "inputs": {"smiles": text},
                "external_inputs": {}, "labels": {}}
    short_raw = {"id": "short", "sample_type": "molecule", "inputs": {"smiles": "CCO"},
                 "external_inputs": {}, "labels": {}}
    tail_raw = {"id": "tail", "sample_type": "molecule", "inputs": {"smiles": text + "C"},
                "external_inputs": {}, "labels": {}}
    _, single_embeddings, single_tokens, _ = encode_subset(
        model, fz, cfg, [long_raw], batch_size=4, token_level=True)
    _, batch_embeddings, _, _ = encode_subset(
        model, fz, cfg, [short_raw, long_raw], batch_size=4, token_level=True)
    _, tail_embeddings, tail_tokens, _ = encode_subset(
        model, fz, cfg, [tail_raw], batch_size=4, token_level=True)
    np.testing.assert_allclose(single_embeddings[0], batch_embeddings[1], atol=2e-5, rtol=2e-5)
    assert single_tokens.shape[0] == count
    assert tail_tokens.shape[0] > count
    assert np.max(np.abs(single_embeddings[0] - tail_embeddings[0])) > 1e-7
    print({"status": "ok", "tokens": count, "windows": len(windows),
           "single_batch_max_delta": float(np.max(np.abs(single_embeddings[0] - batch_embeddings[1]))),
           "tail_embedding_delta": float(np.max(np.abs(single_embeddings[0] - tail_embeddings[0])))})


if __name__ == "__main__":
    main()
