from __future__ import annotations

import argparse
import datetime
import time
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np
import yaml
from torch.nn.parallel import DistributedDataParallel

from ..config import MuseConfig
from ..model import MuseEncoder
from ..tokenizers import load_tokenizers
from .batching import BatchBuilder
from .corpora import DescriptorNormalizer, PRETRAIN_ROOT
from .dataset import PretrainDataset
from .omc25_source import OMC25Source
from .sampling import HierarchicalSampler, ScaffoldIndex, SourceSampler
from .scaffolds import SUPPORTED_SOURCES as SCAFFOLD_SUPPORTED
from .scaffolds import load_scaffold_arrays
from .smoke_dcu import move
from .targets import TargetNormalizer
from .trainer import PretrainWrapper, Trainer, seed_everything, warmup_cosine_scheduler


def microbatch_seed(config, micro_step, rank):
    values = [int(config["seed"]), int(config["stage"]), int(micro_step), int(rank)]
    return int(np.random.SeedSequence(values).generate_state(1, dtype=np.uint32)[0])


def load_init_weights(model, path, rank=0):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(payload["model"], strict=False)
    if rank == 0:
        print(f"[init-weights] {path} missing={len(incompatible.missing_keys)} "
              f"unexpected={len(incompatible.unexpected_keys)}", flush=True)
    return incompatible


def load_normalizers(config):
    aux_path = config["data"].get("aux_targets")
    aux = TargetNormalizer.from_path(Path(aux_path)) if aux_path else None
    mtr_path = config["data"].get("mtr_norm")
    mtr = DescriptorNormalizer.from_path(Path(mtr_path)) if mtr_path else None
    return aux, mtr


def aux_model_specs(normalizer, sources):
    if normalizer is None or not any(dataset.kind == "aux_molecule" or dataset.kind == "aux_sequence"
                                     for dataset in sources):
        return {}
    source_names = {dataset.name for dataset in sources}
    specs = {}
    for key, spec in normalizer.specs.items():
        if spec.shadow or spec.source not in source_names or spec.task_type == "multiclass_classification":
            continue
        specs[key] = {"kind": spec.kind, "dim": spec.dim, "task_type": spec.task_type}
    return specs


def build_components(config, rank, world_size):
    tokenizers = load_tokenizers()
    aux_normalizer, descriptor_normalizer = load_normalizers(config)
    names = config["data"]["sources"]
    scaffold_sampling = config["data"].get("scaffold_sampling", False)
    datasets = []
    for name in names:
        if name == "omc25":
            dataset = OMC25Source(
                config["_model"], config["data"].get("crystal"), tokenizers)
        else:
            dataset = PretrainDataset(
                name, config["_model"], aux_normalizer,
                descriptor_normalizer, tokenizers, config["seed"],
                use_molecule3d=config["data"].get("molecule3d", False))
        datasets.append(dataset)
    samplers = []
    for dataset in datasets:
        scaffold = None
        if scaffold_sampling and dataset.name in SCAFFOLD_SUPPORTED:
            arrays = load_scaffold_arrays(dataset.name, required=True)
            scaffold = ScaffoldIndex(arrays[0], arrays[1])
        exclude = None
        if config["data"].get("exclude_unparseable", True):
            audit_path = PRETRAIN_ROOT / "audit" / f"smiles_unparseable.{dataset.name}.rows.txt"
            if audit_path.exists():
                exclude = frozenset(
                    int(line) for line in audit_path.read_text(encoding="utf-8").split())
        samplers.append(SourceSampler(dataset.spec.source, scaffold=scaffold, exclude=exclude))
    sampler = HierarchicalSampler(
        samplers, seed=config["seed"], stage=config["stage"],
        domain_weights=config["data"].get("domain_weights"),
        temperature=config["data"].get("temperature", 0.005),
        unique_scaffolds=config["data"].get("unique_scaffolds", False),
        source_grouped=config["data"].get("source_grouped", False),
    )
    return datasets, sampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-weights", type=Path,
                        help="仅加载本队 encoder 权重，使用新优化器从头开始当前阶段")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.init_weights is not None and args.resume is not None:
        parser.error("--init-weights 与 --resume 互斥")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["_raw"] = yaml.safe_dump(config, allow_unicode=True, sort_keys=True)
    training = config["training"]
    run_dir = args.run_dir
    if run_dir.exists() and (run_dir / "train.jsonl").exists() and args.resume is None:
        raise FileExistsError(f"正式运行目录已有训练日志，必须显式 resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=True), encoding="utf-8")

    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = env_world > 1
    if distributed:
        # 数据侧偶发慢读（Lustre/大 mmap 换页）可能让单 rank 掉队超过默认 10 分钟
        # NCCL 超时阈值放大到 1 小时，避免误杀；真正的挂死由看板/监控暴露。
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=3600))
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.device)))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    seed_everything(config["seed"], config.get("deterministic", True))

    config["_model"] = MuseConfig(**config.get("model", {}))
    if config["_model"].conservative_force and training.get("denoise", False):
        raise ValueError("保守力阶段必须使用干净坐标，配置不得启用 denoise")
    probe_datasets = [PretrainDataset(name, config["_model"], seed=config["seed"])
                      for name in config["data"]["sources"]]
    config["_model"].aux_specs = aux_model_specs(
        TargetNormalizer.from_path(Path(config["data"]["aux_targets"]))
        if config["data"].get("aux_targets") else None,
        probe_datasets,
    )
    for dataset in probe_datasets:
        dataset.close()
    model = MuseEncoder(config["_model"]).to(device)
    if args.init_weights:
        load_init_weights(model, args.init_weights, rank)
    wrapped = PretrainWrapper(model, training.get("weights", {}))
    if distributed:
        wrapped = DistributedDataParallel(wrapped, device_ids=[local_rank],
                                          find_unused_parameters=True)
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=training["lr"])
    total_steps = training["total_steps"]
    scheduler = warmup_cosine_scheduler(optimizer, total_steps, training["warmup_steps"])
    trainer = Trainer(
        wrapped, optimizer, scheduler, run_dir,
        training["accumulation_steps"], total_steps, device, rank,
        training.get("log_interval", 20), training.get("checkpoint_interval", 1000),
        training.get("grad_clip", 1.0), training.get("bf16", True),
    )
    if args.resume:
        trainer.resume(args.resume)
    datasets, sampler = build_components(config, rank, world_size)
    builder = BatchBuilder(config["_model"], load_tokenizers())
    prefetch = training.get("prefetch", 1)
    executor = ThreadPoolExecutor(max_workers=prefetch) if prefetch > 1 else None

    def prepare(micro_step):
        pointers = sampler.batch(micro_step, rank, world_size, training["micro_batch"])
        by_name = {dataset.name: dataset for dataset in datasets}
        t0 = time.monotonic()
        items = [by_name[pointer.source].get(pointer) for pointer in pointers]
        fetch_seconds = time.monotonic() - t0
        if fetch_seconds > 60.0:
            # 数据读取超过 60s：记录来源与行号，供 NCCL 超时类故障定位。
            print(f"[watchdog] rank={rank} micro_step={micro_step} "
                  f"fetch={fetch_seconds:.1f}s sources={[p.source for p in pointers]} "
                  f"rows={[p.row for p in pointers]}", flush=True)
        records, metadata = zip(*items)
        records = list(records)
        metadata = list(metadata)
        align_pairs = []
        if training.get("align_2d3d", False):
            original_count = len(records)
            for index, (pointer, record) in enumerate(zip(pointers, records)):
                if (by_name[pointer.source].kind == "aux_molecule"
                        and record.coord_mask is not None and np.any(record.coord_mask)):
                    view_record, view_item = by_name[pointer.source].get_2d_view(pointer)
                    records.append(view_record)
                    metadata.append(view_item)
                    align_pairs.append([index, original_count + len(align_pairs)])
        batch, targets = builder.build(
            records, microbatch_seed(config, micro_step, rank),
            mlm=training.get("mlm", True),
            denoise=training.get("denoise", False),
            items=metadata)
        if align_pairs:
            targets["align_idx"] = torch.tensor(align_pairs, dtype=torch.long)
        return move(batch, device), move(targets, device)

    def stream(start):
        stop = total_steps * training["accumulation_steps"]
        if executor is None:
            for micro_step in range(start, stop):
                yield prepare(micro_step)
            return
        for chunk_start in range(start, stop, prefetch):
            for value in executor.map(prepare, range(chunk_start, min(chunk_start + prefetch, stop))):
                yield value

    try:
        trainer.fit(stream, config)
        if rank == 0:
            result = {"stage": config["stage"], "step": trainer.state.step,
                      "samples": trainer.state.samples,
                      "checkpoint": str(run_dir / "checkpoints" / "final.pt")}
            (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result), flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        for dataset in datasets:
            dataset.close()
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
