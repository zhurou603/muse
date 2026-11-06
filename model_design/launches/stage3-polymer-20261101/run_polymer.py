"""Run the gated polymer electronic treatment or replay-only control."""
import argparse
import datetime
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[3]
for path in (str(ROOT), str(ROOT / "model_design"), str(ROOT / "model_design/launches/stage2-20261031")):
    if path not in sys.path:
        sys.path.insert(0, path)

from muse_encoder.config import MuseConfig
from muse_encoder.model import MuseEncoder
from muse_encoder.pretrain.a_train import ATrainDataset, ATrainTargetNormalizer
from muse_encoder.pretrain.batching import BatchBuilder
from muse_encoder.pretrain.corpora import DescriptorNormalizer
from muse_encoder.pretrain.dataset import PretrainDataset, PretrainSourceSpec
from muse_encoder.pretrain.run_pretrain import load_init_weights, microbatch_seed
from muse_encoder.pretrain.sampling import SamplePointer, SourceSampler, SourceSpec
from muse_encoder.pretrain.smoke_dcu import move
from muse_encoder.pretrain.trainer import PretrainWrapper, Trainer, seed_everything
from muse_encoder.tokenizers import load_tokenizers
from prepare_polymer import SOURCES, TASKS
from run_dbaasp_pilot import WeightedSampler

DATA = ROOT / "data/pretrain/stage3-polymer"
PARENT = ROOT / "model_design/runs/s3-residual-pilot-02/checkpoints/final.pt"


class FilteredTaskDataset:
    def __init__(self, task_id, cfg, normalizer, tokenizers):
        self.task_id = task_id
        source_name = task_id.rsplit("/", 1)[0]
        self.base = ATrainDataset(source_name, cfg, normalizer, tokenizers=tokenizers, seed=43)
        self.name = task_id
        self.kind = "a_train"
        self.spec = PretrainSourceSpec(
            SourceSpec(task_id, "polymer", len(self.base), 1), self.kind)
        self.allowed = {
            key for key, spec in normalizer.specs.items() if spec.task_id == task_id
        }
        if len(self.allowed) != 1:
            raise ValueError(f"{task_id}: 任务头映射异常")

    def get(self, pointer):
        translated = SamplePointer(self.base.name, pointer.domain, pointer.row, pointer.view)
        record, item = self.base.get(translated)
        item["aux_records"] = [aux for aux in item["aux_records"] if aux["name"] in self.allowed]
        if not item["aux_records"]:
            raise ValueError(f"{self.task_id}: 过滤后没有监督目标")
        record.meta["stage3_task"] = self.task_id
        return record, item

    def close(self):
        self.base.close()


def linear_scheduler(optimizer, total_steps, warmup_steps):
    def factor(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class RankSynchronizedWeightedSampler(WeightedSampler):
    def batch(self, micro_step, rank, world_size, micro_batch):
        if min(micro_step, rank, micro_batch) < 0 or world_size < 1 or rank >= world_size:
            raise ValueError("非法 DDP batch 参数")
        source = self._sample_source(self._rng(micro_step))
        row_rng = self._rng(micro_step * world_size + rank, attempt=1)
        return [source.sample(row_rng) for _ in range(micro_batch)]


def build_datasets(mode, cfg, tokenizers):
    normalizer = ATrainTargetNormalizer.from_path(DATA / "targets.json")
    datasets = [FilteredTaskDataset(task_id, cfg, normalizer, tokenizers) for task_id in TASKS]
    descriptor = DescriptorNormalizer.from_path(ROOT / "data/pretrain/targets/mtr_norm.json")
    replay = PretrainDataset(
        "pi1m", cfg, descriptor_normalizer=descriptor, tokenizers=tokenizers, seed=43)
    aux_specs = normalizer.model_specs()
    if mode == "control":
        for dataset in datasets:
            dataset.close()
        datasets = []
        weights = {replay.name: 1.0}
    else:
        weights = dict(TASKS)
    datasets.append(replay)
    weights[replay.name] = 0.10 if mode == "treatment" else 1.0
    return datasets, weights, aux_specs


def verify_manifest():
    from muse_encoder.pretrain.trainer import sha256_file
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(DATA / "targets.json") != manifest["targets_sha256"]:
        raise ValueError("polymer targets hash 不匹配")
    if manifest["external_data"] or manifest["validation_test_labels_used"]:
        raise ValueError("polymer 数据合规标记异常")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("treatment", "control"), required=True)
    parser.add_argument("--init-weights", type=Path, default=PARENT)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int, choices=(1, 50, 100, 200, 300), default=300)
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--prefetch", type=int, default=4)
    args = parser.parse_args()
    if args.resume and args.init_weights != PARENT:
        parser.error("--resume 与显式 --init-weights 互斥")
    data_manifest = verify_manifest()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.micro_batch * world_size * args.accumulation_steps != 64:
        raise ValueError("micro_batch × world_size × accumulation_steps 必须等于64")
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=3600))
    rank = dist.get_rank() if distributed else 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    seed_everything(43, True)
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = MuseConfig(
        d_model=1024, n_layers=24, n_heads=16, d_ffn=2048,
        num_experts=8, mole_every=4, embed_dim=1024,
        contrastive_dim=256, descriptor_dim=208, max_seq_len=1024,
        dropout=0.1, crystal_composition_features=True,
        crystal_composition_scale=128.0,
    )
    tokenizers = load_tokenizers()
    datasets, source_weights, aux_specs = build_datasets(args.mode, cfg, tokenizers)
    cfg.aux_specs = aux_specs
    model = MuseEncoder(cfg).to(device)
    if args.resume:
        incompatible = None
    else:
        incompatible = load_init_weights(model, args.init_weights, rank)
        if incompatible.unexpected_keys or any(
                not key.startswith("heads.aux_heads.") for key in incompatible.missing_keys):
            raise RuntimeError("父 checkpoint 与 Encoder 不完全兼容")
    wrapped = PretrainWrapper(model, {"mlm": 1.0, "aux": 1.0, "descriptor": 1.0})
    if distributed:
        wrapped = DistributedDataParallel(wrapped, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-5, weight_decay=0.01)
    scheduler = linear_scheduler(optimizer, 300, 15)
    trainer = Trainer(wrapped, optimizer, scheduler, run_dir, args.accumulation_steps,
                      300, device, rank, 10, 50, 1.0, True)
    if args.resume:
        trainer.resume(args.resume)
    samplers = [SourceSampler(dataset.spec.source) for dataset in datasets]
    sampler = RankSynchronizedWeightedSampler(samplers, source_weights, seed=43, stage=3,
                                              source_grouped=True, domain_weights=None)
    builder = BatchBuilder(cfg, tokenizers)
    executor = ThreadPoolExecutor(max_workers=args.prefetch) if args.prefetch > 1 else None
    by_name = {dataset.name: dataset for dataset in datasets}

    def prepare(micro_step):
        pointers = sampler.batch(micro_step, rank, world_size, args.micro_batch)
        started = time.monotonic()
        items = [by_name[pointer.source].get(pointer) for pointer in pointers]
        if time.monotonic() - started > 60:
            print(f"[watchdog] rank={rank} micro_step={micro_step} sources={[p.source for p in pointers]}",
                  flush=True)
        records, metadata = zip(*items)
        batch, targets = builder.build(
            list(records), microbatch_seed({"seed": 43, "stage": 3}, micro_step, rank),
            mlm=True, denoise=False, items=list(metadata))
        return move(batch, device), move(targets, device)

    def stream(start):
        stop = args.stop_after * args.accumulation_steps
        if executor is None:
            for micro_step in range(start, stop):
                yield prepare(micro_step)
            return
        for chunk_start in range(start, stop, args.prefetch):
            for value in executor.map(prepare, range(chunk_start, min(chunk_start + args.prefetch, stop))):
                yield value

    config = {
        "schema_version": "muse.stage3.polymer_electronic.v1",
        "run_name": "stage3-polymer-electronic-v1",
        "mode": args.mode, "seed": 43, "parent_checkpoint": str(PARENT),
        "model": cfg.to_dict(), "optimizer": "AdamW", "learning_rate": 1e-5,
        "weight_decay": 0.01, "warmup_steps": 15, "scheduler": "linear_decay",
        "effective_batch_size": 64, "micro_batch": args.micro_batch,
        "world_size": world_size, "accumulation_steps": args.accumulation_steps,
        "max_steps": 300, "stop_after": args.stop_after,
        "source_weights": source_weights, "external_data": False,
        "architecture_change": False, "tokenizer_change": False,
    }
    if rank == 0:
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        (run_dir / "data_manifest.jsonl").write_text(
            json.dumps(data_manifest, ensure_ascii=False) + "\n", encoding="utf-8")
        if incompatible is not None:
            (run_dir / "init_incompatible.json").write_text(json.dumps({
                "missing_keys": list(incompatible.missing_keys),
                "unexpected_keys": list(incompatible.unexpected_keys),
            }, indent=2), encoding="utf-8")
    try:
        trainer.fit(stream, config)
        if rank == 0:
            state = {"mode": args.mode, "step": trainer.state.step,
                     "samples": trainer.state.samples,
                     "checkpoint": str(run_dir / "checkpoints/final.pt")}
            if trainer.state.step >= 300:
                (run_dir / "result.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
            else:
                (run_dir / "paused.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
            if (run_dir / "train.jsonl").exists():
                shutil.copyfile(run_dir / "train.jsonl", run_dir / "train_log.jsonl")
            if (run_dir / "checkpoints/checkpoints.jsonl").exists():
                shutil.copyfile(run_dir / "checkpoints/checkpoints.jsonl", run_dir / "checkpoints.jsonl")
            print(json.dumps(state), flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        for dataset in datasets:
            dataset.close()
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
