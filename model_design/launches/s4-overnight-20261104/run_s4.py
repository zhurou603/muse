from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel

from muse_encoder.model import MuseEncoder
from muse_encoder.pretrain.batching import BatchBuilder
from muse_encoder.pretrain.run_pretrain import load_init_weights, microbatch_seed
from muse_encoder.pretrain.smoke_dcu import move
from muse_encoder.pretrain.trainer import PretrainWrapper, seed_everything, sha256_file
from muse_encoder.tokenizers import load_tokenizers

from s4_common import (
    MAX_STEPS,
    PARENT,
    PARENT_SHA256,
    RUN_NAME,
    SEED,
    S4Trainer,
    build_datasets,
    domain_source_weights,
    linear_scheduler,
    load_config_from_checkpoint,
    make_sampler,
    verify_data_manifest,
)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def code_files():
    root = Path(__file__).resolve().parents[3]
    model_root = root / "model_design"
    return [
        Path(__file__).resolve(), Path(__file__).with_name("s4_common.py").resolve(),
        model_root / "muse_encoder/model.py",
        model_root / "muse_encoder/pretrain/trainer.py",
        model_root / "muse_encoder/pretrain/a_train.py",
        model_root / "muse_encoder/pretrain/dataset.py",
        model_root / "muse_encoder/pretrain/sampling.py",
        model_root / "muse_encoder/pretrain/batching.py",
        model_root / "launches/stage2-20261031/dbaasp.py",
    ]


def snapshot_code(run_dir, config_path, stop_after):
    root = Path(__file__).resolve().parents[3]
    destination = run_dir / "code_snapshot" / f"step_{stop_after:08d}"
    for source in [*code_files(), config_path.resolve()]:
        target = destination / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def optimizer_groups(model, base_lr, aux_lr, weight_decay):
    regular, auxiliary = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (auxiliary if name.startswith("heads.aux_heads.") else regular).append(parameter)
    if not regular or not auxiliary:
        raise RuntimeError("S4 optimizer 参数组为空")
    return [
        {"params": regular, "lr": base_lr, "role": "encoder_and_frontends"},
        {"params": auxiliary, "lr": aux_lr, "role": "auxiliary_heads"},
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("treatment", "control"), required=True)
    parser.add_argument("--stop-after", type=int, choices=(1, 50, 100, 200, 300), required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--prefetch", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["run_name"] != RUN_NAME or config["seed"] != SEED or config["max_steps"] != MAX_STEPS:
        raise ValueError("配置不是固定的 S4 主线")
    if args.resume and args.stop_after <= 1:
        raise ValueError("resume 的 stop-after 必须是正式门控步数")
    data_manifest = verify_data_manifest()
    print(f"[s4] data manifest verified: mode={args.mode}", flush=True)
    parent = PARENT.resolve()
    if sha256_file(parent) != PARENT_SHA256:
        raise RuntimeError("父 checkpoint 哈希变化")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=3600))
    rank = dist.get_rank() if distributed else 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    seed_everything(SEED, True)
    micro_batch = args.micro_batch or config["micro_batch"]
    accumulation = args.accumulation_steps or 1
    effective_batch = micro_batch * world_size * accumulation
    if effective_batch != config["effective_batch_size"]:
        raise ValueError(f"有效 batch={effective_batch}，必须为 {config['effective_batch_size']}")
    run_dir = args.run_dir.resolve()
    resolved_path = run_dir / "config.resolved.json"
    if not args.resume and (run_dir / "train.jsonl").exists():
        raise FileExistsError(f"已有训练日志，必须显式 resume: {run_dir}")
    if args.resume and not resolved_path.is_file():
        raise FileNotFoundError(f"resume 缺少 resolved config: {resolved_path}")
    run_dir.mkdir(parents=True, exist_ok=True)
    source_checkpoint = args.resume.resolve() if args.resume else parent
    print(f"[s4] loading config: {source_checkpoint}", flush=True)
    cfg = load_config_from_checkpoint(source_checkpoint)
    tokenizers = load_tokenizers()
    print("[s4] building datasets", flush=True)
    datasets, source_weights, aux_specs = build_datasets(args.mode, cfg, tokenizers)
    print(f"[s4] datasets={len(datasets)} aux_heads={len(aux_specs)}", flush=True)
    cfg.aux_specs = aux_specs
    model = MuseEncoder(cfg).to(device)
    print("[s4] model constructed", flush=True)
    incompatible = None
    if not args.resume:
        incompatible = load_init_weights(model, parent, rank)
        if incompatible.unexpected_keys or any(
                not key.startswith("heads.aux_heads.") for key in incompatible.missing_keys):
            raise RuntimeError("父 checkpoint 与 S4 Encoder 不完全兼容")
    groups = optimizer_groups(
        model, config["learning_rate"], config["auxiliary_head_learning_rate"], config["weight_decay"])
    optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
    scheduler = linear_scheduler(optimizer, MAX_STEPS, config["warmup_steps"])
    wrapped = PretrainWrapper(model, {"mlm": 1.0, "aux": 1.0, "descriptor": 1.0, "energy": 1.0})
    if distributed:
        wrapped = DistributedDataParallel(wrapped, device_ids=[local_rank], find_unused_parameters=True)
    trainer = S4Trainer(
        wrapped, optimizer, scheduler, run_dir, accumulation, MAX_STEPS, device, rank,
        config["log_interval"], config["checkpoint_interval"], config["grad_clip_norm"], config["bf16"])
    if args.resume:
        trainer.resume(args.resume)
    if trainer.state.step >= args.stop_after:
        raise ValueError(f"checkpoint step={trainer.state.step} 已超过目标 {args.stop_after}")
    sampler = make_sampler(datasets, source_weights)
    builder = BatchBuilder(cfg, tokenizers)
    prefetch = args.prefetch if args.prefetch is not None else config["prefetch"]
    executor = ThreadPoolExecutor(max_workers=prefetch) if prefetch > 1 else None
    by_name = {dataset.name: dataset for dataset in datasets}
    log_lock = threading.Lock()
    segment_started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    segment = {
        "run_name": RUN_NAME,
        "mode": args.mode,
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "resume": str(args.resume.resolve()) if args.resume else None,
        "start_step": trainer.state.step,
        "stop_after": args.stop_after,
        "started_at": segment_started,
        "world_size": world_size,
        "micro_batch": micro_batch,
        "accumulation_steps": accumulation,
        "effective_batch_size": effective_batch,
        "status": "running",
    }
    if rank == 0:
        snapshot_code(run_dir, args.config, args.stop_after)
        write_json(run_dir / "segments" / f"step_{args.stop_after:08d}.json", segment)

    resolved = {
        **config,
        "mode": args.mode,
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": PARENT_SHA256,
        "model": cfg.to_dict(),
        "source_weights": source_weights,
        "realized_domain_weights": domain_source_weights(source_weights),
        "world_size": world_size,
        "micro_batch": micro_batch,
        "accumulation_steps": accumulation,
        "data_manifest": data_manifest,
        "code_sha256": {str(path): sha256_file(path) for path in code_files()},
        "code_snapshot": f"code_snapshot/step_{args.stop_after:08d}",
        "aux_head_count": len(aux_specs),
        "train_log": "train.jsonl",
        "optimizer_log": "optimizer.jsonl",
        "data_stream_logs": "data_stream.rank*.jsonl",
    }
    if rank == 0:
        if resolved_path.exists():
            old = json.loads(resolved_path.read_text(encoding="utf-8"))
            for key in ("run_name", "mode", "parent_checkpoint_sha256", "source_weights",
                        "learning_rate", "auxiliary_head_learning_rate", "effective_batch_size",
                        "world_size", "micro_batch", "accumulation_steps", "code_sha256"): 
                if old[key] != resolved[key]:
                    raise RuntimeError(f"resume 配置变化: {key}")
        else:
            write_json(resolved_path, resolved)
            write_json(run_dir / "data_manifest.json", data_manifest)
            if incompatible is not None:
                write_json(run_dir / "init_incompatible.json", {
                    "missing_keys": list(incompatible.missing_keys),
                    "unexpected_keys": list(incompatible.unexpected_keys),
                    "parent_aux_heads_loaded": True,
                })

    def prepare(micro_step):
        pointers = sampler.batch(micro_step, rank, world_size, micro_batch)
        started = time.monotonic()
        print(f"[s4] fetch start rank={rank} micro_step={micro_step} "
              f"sources={sorted({pointer.source for pointer in pointers})}", flush=True)
        items = [by_name[pointer.source].get(pointer) for pointer in pointers]
        fetch_seconds = time.monotonic() - started
        counts = Counter(pointer.source for pointer in pointers)
        record = {
            "micro_step": micro_step,
            "rank": rank,
            "world_size": world_size,
            "fetch_seconds": round(fetch_seconds, 6),
            "source_counts": dict(sorted(counts.items())),
            "rows": [{"source": p.source, "row": p.row, "view": p.view} for p in pointers],
        }
        with log_lock:
            with (run_dir / f"data_stream.rank{rank}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        if fetch_seconds > 60:
            print(f"[watchdog] rank={rank} micro_step={micro_step} fetch={fetch_seconds:.1f}s "
                  f"sources={list(counts)}", flush=True)
        records, metadata = zip(*items)
        batch, targets = builder.build(
            list(records), microbatch_seed({"seed": SEED, "stage": 4}, micro_step, rank),
            mlm=True, denoise=False, items=list(metadata))
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

    try:
        trainer.fit(stream, resolved)
        if rank == 0:
            state = {
                "run_name": RUN_NAME,
                "mode": args.mode,
                "step": trainer.state.step,
                "samples": trainer.state.samples,
                "samples_rank0": trainer.state.samples,
                "samples_global": trainer.state.samples * world_size,
                "checkpoint": str(run_dir / "checkpoints/final.pt"),
                "checkpoint_sha256": sha256_file(run_dir / "checkpoints/final.pt"),
                "stop_after": args.stop_after,
            }
            write_json(run_dir / ("result.json" if trainer.state.step == MAX_STEPS else "paused.json"), state)
            segment.update(status="completed", finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                           end_step=trainer.state.step, samples=trainer.state.samples,
                           final_checkpoint_sha256=state["checkpoint_sha256"])
            write_json(run_dir / "segments" / f"step_{args.stop_after:08d}.json", segment)
            print(json.dumps(state), flush=True)
    except BaseException as exc:
        if rank == 0:
            segment.update(status="failed", finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                           error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            write_json(run_dir / "segments" / f"step_{args.stop_after:08d}.json", segment)
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        for dataset in datasets:
            dataset.close()
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
