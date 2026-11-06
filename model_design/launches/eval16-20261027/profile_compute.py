"""对单 micro=32 fwd+bwd 做算子级 profiling，找真正热点。"""
import time
from pathlib import Path

import torch
import yaml

from muse_encoder.config import MuseConfig
from muse_encoder.model import MuseEncoder
from muse_encoder.pretrain.run_pretrain import build_components, microbatch_seed
from muse_encoder.pretrain.batching import BatchBuilder
from muse_encoder.pretrain.smoke_dcu import move
from muse_encoder.tokenizers import load_tokenizers

ROOT = Path("/yinghepool/tech-dcu/zhaoyao")
CONFIG = ROOT / "model_design/launches/s1a-ddp16-20261026/pretrain_s1a_16gpu.yaml"
DEVICE = "cuda:3"


def main():
    config = yaml.safe_load(CONFIG.read_text())
    config["_model"] = MuseConfig(**config.get("model", {}))
    datasets, sampler = build_components(config, 0, 16)
    builder = BatchBuilder(config["_model"], load_tokenizers())
    by_name = {d.name: d for d in datasets}
    pointers = sampler.batch(20000, 0, 16, 32)
    records, metadata = zip(*[by_name[p.source].get(p) for p in pointers])
    batch, targets = builder.build(list(records), microbatch_seed(config, 20000, 0),
                                   mlm=True, denoise=False, items=list(metadata))
    batch, targets = move(batch, DEVICE), move(targets, DEVICE)
    model = MuseEncoder(config["_model"]).to(DEVICE)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    weights = {"mlm": 1.0, "descriptor": 1.0}
    for _ in range(2):
        losses = model.pretraining_losses(batch, targets, weights)
        losses["total"].backward()
        opt.step()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True) as prof:
        losses = model.pretraining_losses(batch, targets, weights)
        losses["total"].backward()
        opt.step()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18))
    for dataset in datasets:
        dataset.close()


if __name__ == "__main__":
    main()
