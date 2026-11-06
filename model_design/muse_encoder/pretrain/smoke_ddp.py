from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from rdkit import Chem
from torch.nn.parallel import DistributedDataParallel

from ..config import MuseConfig
from ..featurize import RecordFeaturizer
from ..model import MuseEncoder
from .batching import BatchBuilder
from .smoke_dcu import move
from .trainer import PretrainWrapper, Trainer, warmup_cosine_scheduler


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(__import__("os").environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
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
    encoder = MuseEncoder(cfg).to(device)
    model = DistributedDataParallel(PretrainWrapper(encoder, {}), device_ids=[local_rank],
                                    find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = warmup_cosine_scheduler(optimizer, 2, 1)
    run_dir = Path("model_design/runs/ddp-smoke")
    trainer = Trainer(model, optimizer, scheduler, run_dir, 1, 2, device,
                      rank=rank, log_interval=1, checkpoint_interval=1)

    def stream(start):
        for micro_step in range(start, 2):
            batch, targets = BatchBuilder(cfg).build(
                records, seed=micro_step * world_size + rank, denoise=True)
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
            yield move(batch, device), move(targets, device)

    trainer.fit(stream, {"smoke": "ddp", "world_size": world_size})
    dist.barrier()
    if rank == 0:
        result = {"world_size": world_size, "step": trainer.state.step,
                  "checkpoint": str(run_dir / "checkpoints" / "final.pt")}
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
