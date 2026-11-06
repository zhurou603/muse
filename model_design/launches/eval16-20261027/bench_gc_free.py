"""单卡 fwd+bwd 实测：micro 32/64/128 × gradient_checkpointing 开关。

不碰 DDP/正式配置；用真实批次测每样本 GPU 耗时与峰值显存。
"""
import os
import time
from pathlib import Path

import torch
import yaml

torch.use_deterministic_algorithms(True)

from muse_encoder.config import MuseConfig
from muse_encoder.model import MuseEncoder
from muse_encoder.pretrain.run_pretrain import build_components, microbatch_seed
from muse_encoder.pretrain.batching import BatchBuilder
from muse_encoder.pretrain.smoke_dcu import move
from muse_encoder.tokenizers import load_tokenizers

ROOT = Path("/yinghepool/tech-dcu/zhaoyao")
CONFIG = ROOT / "model_design/launches/s1a-ddp16-20261026/pretrain_s1a_16gpu.yaml"
DEVICE = "cuda:0"


def main():
    config = yaml.safe_load(CONFIG.read_text())
    config["_model"] = MuseConfig(**config.get("model", {}))
    datasets, sampler = build_components(config, 0, 16)
    tokenizers = load_tokenizers()
    builder = BatchBuilder(config["_model"], tokenizers)
    by_name = {d.name: d for d in datasets}

    def make_batch(micro_step, n):
        pointers = sampler.batch(micro_step, 0, 16, 32)
        records, metadata = zip(*[by_name[p.source].get(p) for p in pointers])
        records = (list(records) * (n // 32))[:n]
        metadata = (list(metadata) * (n // 32))[:n]
        batch, targets = builder.build(records, microbatch_seed(config, micro_step, 0),
                                       mlm=True, denoise=False, items=metadata)
        return move(batch, DEVICE), move(targets, DEVICE)

    weights = {"mlm": 1.0, "descriptor": 1.0}
    for gc, n in ((False, 32), (True, 32), (False, 64), (True, 64)):
            cfg = MuseConfig(**{**config.get("model", {}), "gradient_checkpointing": gc})
            model = MuseEncoder(cfg).to(DEVICE)
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            batch, targets = make_batch(20000, n)
            try:
                for _ in range(3):
                    losses = model.pretraining_losses(batch, targets, weights)
                    losses["total"].backward()
                    opt.step()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(DEVICE)
                times = []
                for _ in range(5):
                    t0 = time.monotonic()
                    losses = model.pretraining_losses(batch, targets, weights)
                    losses["total"].backward()
                    opt.step()
                    torch.cuda.synchronize()
                    times.append(time.monotonic() - t0)
                dt = sum(times) / len(times)
                mem = torch.cuda.max_memory_allocated(DEVICE) / 2**30
                print(f"gc={gc} micro={n}: {dt:.2f}s/micro = {dt / n * 1000:.1f}ms/rec "
                      f"peak={mem:.1f}GiB loss={losses['total'].item():.3f} times={['%.2f' % t for t in times]}", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"gc={gc} micro={n}: OOM", flush=True)
            del model, opt, batch, targets
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(DEVICE)
    for dataset in datasets:
        dataset.close()


if __name__ == "__main__":
    main()
