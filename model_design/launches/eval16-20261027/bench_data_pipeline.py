"""独立数据管道测速：不初始化 GPU/DDP，不影响正式训练。

测 prepare() 在 1/4/8/16/32 线程下的 records/s，判断预取扩展性与 GIL 瓶颈。
"""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from muse_encoder.pretrain.run_pretrain import build_components, microbatch_seed
from muse_encoder.pretrain.batching import BatchBuilder
from muse_encoder.tokenizers import load_tokenizers
from muse_encoder.config import MuseConfig

ROOT = Path("/yinghepool/tech-dcu/zhaoyao")
CONFIG = ROOT / "model_design/configs/pretrain_s1b_16.yaml"
N_MICRO = 200
MICRO_BATCH = 32


def main():
    config = yaml.safe_load(CONFIG.read_text())
    config["_model"] = MuseConfig(**config.get("model", {}))
    training = config["training"]
    datasets, sampler = build_components(config, 0, 16)
    builder = BatchBuilder(config["_model"], load_tokenizers())
    by_name = {d.name: d for d in datasets}

    def prepare(micro_step):
        pointers = sampler.batch(micro_step, 0, 16, MICRO_BATCH)
        records, metadata = zip(*[by_name[p.source].get(p) for p in pointers])
        batch, targets = builder.build(list(records), microbatch_seed(config, micro_step, 0),
                                       mlm=True, denoise=False, items=list(metadata))
        return len(records)

    for workers in (4,):
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            counts = list(pool.map(prepare, range(0, N_MICRO)))
        dt = time.monotonic() - start
        n = sum(counts)
        print(f"workers={workers:2d}: {n} records in {dt:.1f}s = {n / dt:.1f} rec/s "
              f"(per micro {dt / N_MICRO:.2f}s)", flush=True)
    for dataset in datasets:
        dataset.close()


if __name__ == "__main__":
    main()
