"""Single S5 entry point for continuous four-domain training."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel

from muse_encoder.config import MuseConfig
from muse_encoder.model import MuseEncoder
from muse_encoder.pretrain.batching import BatchBuilder
from muse_encoder.pretrain.corpora import PRETRAIN_ROOT
from muse_encoder.pretrain.sampling import SourceSampler
from muse_encoder.pretrain.smoke_dcu import move
from muse_encoder.pretrain.trainer import seed_everything
from muse_encoder.tokenizers import load_tokenizers

from .curriculum import CourseLogger, microbatch_seed
from .data import S5Data
from .provenance import atomic_json, sha256_file, verify_data_manifest
from .runtime import (
    S5Trainer,
    S5Wrapper,
    build_optimizer,
    load_yaml,
    strict_initialize,
    warmup_stable_scheduler,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--init-weights", type=Path)
    parser.add_argument("--resume", type=Path, help="step checkpoint 或 run_dir/latest.json")
    parser.add_argument("--stop-after", type=int, required=True,
                        help="本段结束时的绝对 optimizer step；续跑时增大，不改变 LR 轨迹")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--prefetch", type=int)
    parser.add_argument("--full-hash", action="store_true")
    args = parser.parse_args()
    if args.stop_after < 1:
        parser.error("--stop-after 必须为正数")
    if bool(args.init_weights) == bool(args.resume):
        parser.error("新段必须且只能指定 --init-weights；续跑必须且只能指定 --resume")

    config = load_yaml(args.config)
    training = config["training"]
    micro_batch = args.micro_batch or int(training["micro_batch"])
    accumulation = args.accumulation_steps or int(training["accumulation_steps"])
    prefetch = args.prefetch if args.prefetch is not None else int(training.get("prefetch", 2))
    if min(micro_batch, accumulation, prefetch) < 1:
        raise ValueError("micro_batch/accumulation_steps/prefetch 必须为正数")
    verification = verify_data_manifest(config["data"]["data_manifest"], full_hash=args.full_hash)

    run_dir = args.run_dir
    config_text = yaml.safe_dump(config, allow_unicode=True, sort_keys=True)
    resolved_path = run_dir / "config.resolved.yaml"
    if run_dir.exists() and (run_dir / "train.jsonl").exists() and not args.resume:
        raise FileExistsError(f"正式运行目录已有训练日志，必须显式 resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and resolved_path.exists():
        if resolved_path.read_text(encoding="utf-8") != config_text:
            raise ValueError("resume 配置与原段不一致；请使用原配置，只增大 --stop-after")
    elif rank0_environment():
        resolved_path.write_text(config_text, encoding="utf-8")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=3600))
    rank = dist.get_rank() if distributed else 0
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.device)))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    seed_everything(int(config["seed"]), bool(config.get("deterministic", True)))

    cfg = MuseConfig(**config.get("model", {}))
    tokenizers = load_tokenizers()
    # prepare_data normally builds all indexes.  Rank 0 remains a fail-safe for
    # any lazily created asset; other ranks never concurrently create it.
    data = S5Data(config, cfg, tokenizers) if (not distributed or rank == 0) else None
    if distributed:
        dist.barrier()
    if data is None:
        data = S5Data(config, cfg, tokenizers)
    cfg.aux_specs = data.aux_specs
    model = MuseEncoder(cfg).to(device)

    if args.resume:
        from .runtime import resolve_resume_path
        resume_path = resolve_resume_path(args.resume)
        previous = torch.load(resume_path, map_location="cpu", weights_only=False)
        init_record = (previous.get("metadata") or {}).get("s5_init")
        if not init_record:
            raise ValueError("S5 resume checkpoint 缺少 s5_init 血缘；不能猜测新头参数组")
    else:
        resume_path = None
        init_record = strict_initialize(model, args.init_weights, rank)
        if rank == 0:
            atomic_json(run_dir / "init.json", init_record)

    wrapped = S5Wrapper(model, training.get("weights", {}))
    optimizer = build_optimizer(
        wrapped, init_record, float(training["lr"]), float(training["new_head_lr"]),
        float(training.get("weight_decay", 0.01)),
    )
    scheduler = warmup_stable_scheduler(optimizer, int(training.get("warmup_steps", 50)))
    if distributed:
        wrapped = DistributedDataParallel(wrapped, device_ids=[local_rank],
                                          find_unused_parameters=True)
    trainer = S5Trainer(
        wrapped, optimizer, scheduler, run_dir, accumulation, args.stop_after, device, rank,
        int(training.get("log_interval", 20)), int(training.get("checkpoint_interval", 100)),
        float(training.get("grad_clip", 1.0)), bool(training.get("bf16", True)),
    )
    if args.resume:
        trainer.resume(resume_path)
        if trainer.state.step >= args.stop_after:
            raise ValueError(f"resume step={trainer.state.step} 已达到 stop_after={args.stop_after}")

    samplers = {}
    for name, dataset in data.datasets.items():
        exclude = None
        audit_path = PRETRAIN_ROOT / "audit" / f"smiles_unparseable.{name}.rows.txt"
        if training.get("exclude_unparseable", True) and audit_path.exists():
            exclude = frozenset(int(line) for line in audit_path.read_text().split())
        samplers[name] = SourceSampler(dataset.spec.source, exclude=exclude)
    builder = BatchBuilder(cfg, tokenizers)
    executor = ThreadPoolExecutor(max_workers=prefetch) if prefetch > 1 else None
    course_logger = CourseLogger(run_dir, rank, bool(training.get("log_course_rows", True)))
    dataset_locks = {name: threading.Lock() for name in data.datasets}

    def prepare(micro_step: int):
        optimizer_step = micro_step // accumulation
        choice = data.curriculum.resolve(optimizer_step)
        rng = np.random.default_rng(microbatch_seed(config["seed"], int(config["stage"]),
                                                    micro_step, rank))
        sampler = samplers[choice.source]
        pointers = [sampler.sample(rng) for _ in range(micro_batch)]
        started = time.monotonic()
        dataset = data.datasets[choice.source]
        with dataset_locks[choice.source]:
            items = [dataset.get(pointer) for pointer in pointers]
        fetch_seconds = time.monotonic() - started
        records, metadata = zip(*items)
        records, metadata = list(records), list(metadata)
        for record in records:
            record.meta["s5_source"] = choice.source
            record.meta["s5_step"] = choice.step
        align_pairs = []
        if choice.weights.get("align", 0.0) > 0:
            original_count = len(records)
            for index, record in enumerate(list(records)):
                if dataset.kind == "aux_molecule" and record.coord_mask is not None \
                        and np.any(record.coord_mask):
                    with dataset_locks[choice.source]:
                        view_record, view_item = dataset.get_2d_view(pointers[index])
                    records.append(view_record)
                    metadata.append(view_item)
                    align_pairs.append([index, original_count + len(align_pairs)])
        batch, targets = builder.build(
            records, microbatch_seed(config["seed"], int(config["stage"]), micro_step, rank),
            mlm=choice.weights.get("mlm", 0.0) > 0,
            denoise=choice.weights.get("denoise", 0.0) > 0,
            items=metadata,
        )
        if align_pairs:
            targets["align_idx"] = torch.tensor(align_pairs, dtype=torch.long)
        targets["_s5_loss_weights"] = choice.weights
        course_logger.log(choice, micro_step, [p.row for p in pointers],
                          [p.view for p in pointers], fetch_seconds)
        if fetch_seconds > 60:
            print(f"[watchdog] rank={rank} micro_step={micro_step} source={choice.source} "
                  f"rows={[p.row for p in pointers]} fetch={fetch_seconds:.1f}s", flush=True)
        return move(batch, device), move(targets, device)

    def stream(start):
        stop = args.stop_after * accumulation
        if executor is None:
            for micro_step in range(start, stop):
                yield prepare(micro_step)
            return
        for chunk_start in range(start, stop, prefetch):
            for value in executor.map(prepare, range(chunk_start, min(chunk_start + prefetch, stop))):
                yield value

    run_config = {
        "schema_version": "muse_encoder.s5_train.v1",
        "config": config,
        "resolved_curriculum": data.resolved_config["curriculum"],
        "model": cfg.to_dict(),
        "micro_batch": micro_batch,
        "accumulation_steps": accumulation,
        "world_size": world_size,
        "stop_after": args.stop_after,
        "data_verification": verification,
        "data_manifest_sha256": sha256_file(config["data"]["data_manifest"]),
        "validation_test_labels_used": False,
    }
    metadata = {"s5_init": init_record, "data_manifest_sha256": run_config["data_manifest_sha256"]}
    original_save = trainer.save

    def save_with_lineage(name, save_config, save_metadata=None):
        merged = dict(metadata)
        merged.update(save_metadata or {})
        return original_save(name, save_config, merged)

    trainer.save = save_with_lineage
    if not args.resume:
        # A recoverable lineage point exists even if the process dies before the
        # first periodic checkpoint.  This is not an evaluated/best candidate.
        trainer.save("step_00000000", run_config, metadata)
    try:
        trainer.fit(stream, run_config)
        if rank == 0:
            result = {
                "stage": config["stage"], "step": trainer.state.step,
                "micro_step": trainer.state.micro_step, "samples": trainer.state.samples,
                "checkpoint": str(run_dir / "checkpoints" / "final.pt"),
                "latest": str(run_dir / "latest.json"),
                "note": "final/latest 仅表示最近训练状态，不表示最佳或已验收候选",
            }
            atomic_json(run_dir / "result.json", result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        data.close()
        if distributed:
            dist.destroy_process_group()


def rank0_environment():
    return int(os.environ.get("RANK", "0")) == 0


if __name__ == "__main__":
    main()
