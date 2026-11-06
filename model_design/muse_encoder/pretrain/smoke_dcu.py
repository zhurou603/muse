from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from rdkit import Chem

from ..config import MuseConfig
from ..featurize import RecordFeaturizer
from ..model import MuseEncoder
from .batching import BatchBuilder


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move(item, device) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(42)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    cfg = MuseConfig(
        d_model=32, n_layers=2, n_heads=4, d_ffn=48, num_experts=2,
        mole_every=1, embed_dim=24, contrastive_dim=8, descriptor_dim=208,
        max_seq_len=2048, dropout=0.0,
        aux_specs={"toy": {"kind": "graph", "dim": 1, "task_type": "regression"}},
    )
    fz = RecordFeaturizer(cfg)
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
    batch, targets = BatchBuilder(cfg).build(records, seed=41, denoise=True)
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
    model = MuseEncoder(cfg).to(device)
    batch = move(batch, device)
    targets = move(targets, device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        losses = model.pretraining_losses(batch, targets, {})
    losses["total"].backward()
    torch.cuda.synchronize()
    gradient = sum(parameter.grad.abs().sum().item()
                   for parameter in model.parameters() if parameter.grad is not None)
    result = {
        "device": torch.cuda.get_device_name(device),
        "losses": {key: float(value.detach()) for key, value in losses.items()},
        "gradient_sum": gradient,
        "max_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
    }
    if not np.isfinite(np.asarray(list(result["losses"].values()) + [gradient])).all():
        raise RuntimeError(json.dumps(result))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
