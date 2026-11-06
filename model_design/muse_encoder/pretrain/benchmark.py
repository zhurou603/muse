from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from ..config import MuseConfig
from ..featurize import RecordFeaturizer
from ..model import MuseEncoder
from ..tokenizers import load_tokenizers
from .batching import BatchBuilder
from .smoke_dcu import move


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("model_design/runs/benchmark-full.json"))
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    overrides = dict(
        d_model=32, n_layers=2, n_heads=4, d_ffn=48, num_experts=2,
        embed_dim=24, contrastive_dim=8, max_seq_len=2048,
    ) if args.small else {}
    cfg = MuseConfig(
        dropout=0.0,
        aux_specs={"toy": {"kind": "graph", "dim": 1, "task_type": "regression"}},
        **overrides,
    )
    tokenizers = load_tokenizers()
    fz = RecordFeaturizer(cfg, tokenizers)
    coords = np.asarray([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [2.0, 1.0, 0.2]])
    peptide = Chem.MolFromSequence("ACDE")
    records = [
        fz.featurize_molecule("CCO", coords=coords),
        fz.featurize_polymer(psmiles="*CCO*"),
        fz.featurize_crystal(np.eye(3) * 5.0, np.array([[0.0, 0.0, 0.0]]), ["Na"]),
        fz.featurize_sequence("ACDEFGHIKLMNPQRSTVWY"),
        fz.featurize_protein_ligand("ACDE", "CCO"),
        fz.featurize_peptide_atomic(smiles=Chem.MolToSmiles(peptide)),
    ]
    parameters = sum(parameter.numel() for parameter in MuseEncoder(cfg).parameters())
    model = MuseEncoder(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    measurements = []
    for step in range(args.steps):
        batch, targets = BatchBuilder(cfg, tokenizers).build(records, seed=step, denoise=True)
        batch_size, length = batch["token_ids"].shape
        targets.update({
            "energy": torch.randn(batch_size),
            "force": torch.randn(batch_size, length, 3),
            "force_mask": batch["coord_mask"],
            "stress": torch.randn(batch_size, 6),
            "descriptor": torch.randn(batch_size, 208),
            "aux_records": [{"name": "toy", "value": torch.randn(batch_size),
                             "mask": torch.ones(batch_size, dtype=torch.bool)}],
        })
        batch = move(batch, device)
        targets = move(targets, device)
        started = time.monotonic()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            losses = model.pretraining_losses(batch, targets, {})
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        measurement = {"step": step, "seconds": elapsed,
                       "samples_per_second": batch_size / elapsed,
                       "total_loss": float(losses["total"].detach()),
                       "max_memory_mib": torch.cuda.max_memory_allocated() / 2**20}
        measurements.append(measurement)
        print(json.dumps(measurement), flush=True)
    result = {"parameters": parameters, "config": cfg.to_dict(),
              "measurements": measurements}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
