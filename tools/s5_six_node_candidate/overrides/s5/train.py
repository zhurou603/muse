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
    parser.add_argument("--expansion-contract", type=Path, required=True)
    parser.add_argument("--resume-sha256", required=True)
    args = parser.parse_args()
    if args.stop_after < 1:
        parser.error("--stop-after 必须为正数")
    if bool(args.init_weights) == bool(args.resume):
        parser.error("新段必须且只能指定 --init-weights；续跑必须且只能指定 --resume")

    config = load_yaml(args.config)
    if not args.resume or args.init_weights:
        parser.error("六机候选只允许显式恢复，不允许重新初始化模型")
    contract = json.loads(args.expansion_contract.read_text(encoding="utf-8"))
    contract_sha256 = sha256_file(args.expansion_contract)
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
        temporary = resolved_path.with_suffix(resolved_path.suffix + ".tmp")
        temporary.write_text(config_text, encoding="utf-8")
        os.replace(temporary, resolved_path)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 48:
        raise ValueError("六机候选要求48个训练进程")
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.device)))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if distributed:
        nccl_timeout = int(os.environ.get("S5_NCCL_TIMEOUT_SECONDS", "3600"))
        if nccl_timeout < 1:
            raise ValueError("S5_NCCL_TIMEOUT_SECONDS 必须为正数")
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=nccl_timeout))
    rank = dist.get_rank() if distributed else 0
    launch_identity = (contract_sha256, args.resume_sha256, str(args.run_dir.resolve()),
                       args.stop_after, micro_batch, accumulation, prefetch)
    launch_identities = [None] * world_size
    dist.all_gather_object(launch_identities, launch_identity)
    if any(value != launch_identity for value in launch_identities):
        raise ValueError("各rank启动参数不一致：须使用同一份session")
    seed_everything(int(config["seed"]), bool(config.get("deterministic", True)))
    from .health import RankEventLogger, enable_stack_dump_signal
    enable_stack_dump_signal()
    rank_logger = RankEventLogger(
        run_dir, rank, world_size, str(device),
        float(os.environ.get("S5_HEALTH_HEARTBEAT_SECONDS", "15")),
    )
    rank_logger.start()

    cfg = MuseConfig(**config.get("model", {}))
    tokenizers = load_tokenizers()
    # prepare_data has already built every index, so all ranks can initialize
    # concurrently without an additional NCCL barrier.
    data = S5Data(config, cfg, tokenizers)
    cfg.aux_specs = data.aux_specs
    model = MuseEncoder(cfg).to(device)

    if args.resume:
        from .runtime import resolve_resume_path
        resume_path = resolve_resume_path(args.resume)
        if sha256_file(resume_path) != args.resume_sha256:
            raise ValueError("恢复checkpoint SHA256与本次启动固定值不符")
        previous = torch.load(resume_path, map_location="cpu", weights_only=False)
        from .expansion import resume_plan
        plan = resume_plan(previous, config, contract, contract_sha256, args.resume_sha256,
                           world_size, rank, micro_batch, accumulation)
        if previous["config"]["model"] != cfg.to_dict():
            raise ValueError("模型配置或辅助任务定义与checkpoint不一致")
        if sha256_file(config["data"]["data_manifest"]) != contract["data_manifest_sha256"]:
            raise ValueError("数据manifest已变化")
        init_record = (previous.get("metadata") or {}).get("s5_init")
        if not init_record:
            raise ValueError("S5 resume checkpoint 缺少 s5_init 血缘；不能猜测新头参数组")
    else:
        resume_path = None
        print(f"[s5] rank={rank} strict-init-start", flush=True)
        init_record = strict_initialize(model, args.init_weights, rank)
        print(f"[s5] rank={rank} strict-init-done", flush=True)
        if rank == 0:
            atomic_json(run_dir / "init.json", init_record)

    wrapped = S5Wrapper(model, training.get("weights", {}))
    optimizer = build_optimizer(
        wrapped, init_record, float(training["lr"]), float(training["new_head_lr"]),
        float(training.get("weight_decay", 0.01)),
    )
    scheduler = warmup_stable_scheduler(optimizer, int(training.get("warmup_steps", 50)))
    print(f"[s5] rank={rank} optimizer-ready", flush=True)
    if distributed:
        print(f"[s5] rank={rank} ddp-start", flush=True)
        wrapped = DistributedDataParallel(
            wrapped, device_ids=[local_rank], broadcast_buffers=False,
            find_unused_parameters=True)
        print(f"[s5] rank={rank} ddp-ready", flush=True)
    trainer = S5Trainer(
        wrapped, optimizer, scheduler, run_dir, accumulation, args.stop_after, device, rank,
        int(training.get("log_interval", 20)), int(training.get("checkpoint_interval", 100)),
        float(training.get("grad_clip", 1.0)), bool(training.get("bf16", True)),
    )
    print(f"[s5] rank={rank} trainer-ready", flush=True)
    if args.resume:
        trainer.resume(resume_path, payload=previous, expansion_seed=plan["expansion_seed"])
        del previous
        rank_logger.event("resume_verified", **plan, checkpoint=str(resume_path),
                          checkpoint_sha256=args.resume_sha256)
        print(f"[s5] rank={rank} RESUME VERIFIED step={trainer.state.step} "
              f"rng={plan['rng_policy']} nominal_global_batch=384", flush=True)
        if trainer.state.step >= args.stop_after:
            raise ValueError(f"resume step={trainer.state.step} 已达到 stop_after={args.stop_after}")

    from .data_skip import DataSkipLogger, fetch_valid_batch, load_skipped_rows
    skipped_rows = load_skipped_rows(run_dir)
    samplers = {}
    exclusions = {}
    for name, dataset in data.datasets.items():
        excluded = set(skipped_rows.get(name, ()))
        if training.get("exclude_unparseable", True):
            audit_path = PRETRAIN_ROOT / "audit" / f"smiles_unparseable.{name}.rows.txt"
            if audit_path.exists():
                excluded.update(int(line) for line in audit_path.read_text().split())
            if name == "dbaasp_v4/target_activity":
                unsupported = Path(config["data"]["dbaasp_replay_dir"]) / "unsupported_train.rows.txt"
                if unsupported.exists():
                    excluded.update(int(line) for line in unsupported.read_text().split())
            if name == "QM9/qm9":
                failures = PRETRAIN_ROOT / "index" / "a_train" / "QM9__qm9.registered_3d_failures.rows.txt"
                if failures.exists():
                    excluded.update(int(line) for line in failures.read_text().split())
            excluded.update(registered_structure_failure_rows(dataset))
        exclude = frozenset(excluded) or None
        exclusions[name] = exclude or frozenset()
        samplers[name] = SourceSampler(dataset.spec.source, exclude=exclude)
    builder = BatchBuilder(cfg, tokenizers)
    process_prefetch = os.environ.get("S5_PROCESS_PREFETCH", "0") == "1"
    if process_prefetch:
        from .prefetch import create_process_pool
        executor = create_process_pool(
            config, cfg.to_dict(), rank, exclusions, prefetch, micro_batch, accumulation,
            run_dir=run_dir)
    else:
        executor = ThreadPoolExecutor(max_workers=prefetch) if prefetch > 1 else None
    rank_logger.event("prefetch_backend", backend="process" if process_prefetch else "thread",
                      workers=prefetch, persisted_skip_sources=len(skipped_rows),
                      persisted_skip_rows=sum(map(len, skipped_rows.values())))
    course_logger = CourseLogger(run_dir, rank, bool(training.get("log_course_rows", True)))
    dataset_locks = {name: threading.Lock() for name in data.datasets}
    skip_logger = DataSkipLogger(run_dir, rank)

    def prepare(micro_step: int):
        optimizer_step = micro_step // accumulation
        choice = data.curriculum.resolve(optimizer_step)
        rng = np.random.default_rng(microbatch_seed(config["seed"], int(config["stage"]),
                                                    micro_step, rank))
        sampler = samplers[choice.source]
        pointers = [sampler.sample(rng) for _ in range(micro_batch)]
        started = time.monotonic()
        dataset = data.datasets[choice.source]
        with rank_logger.stage("fetch", micro_step, optimizer_step):
            records, metadata, pointers, align_pairs, skipped = fetch_valid_batch(
                dataset, sampler, rng, pointers,
                align=choice.weights.get("align", 0.0) > 0,
                skip_logger=skip_logger, micro_step=micro_step,
                dataset_lock=dataset_locks[choice.source], cfg=cfg,
            )
        fetch_seconds = time.monotonic() - started
        for record in records:
            record.meta["s5_source"] = choice.source
            record.meta["s5_step"] = choice.step
        with rank_logger.stage("build", micro_step, optimizer_step):
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
        rank_logger.event("course_logged", source=choice.source, batch_samples=len(records),
                          skipped_samples=len(skipped))
        if skipped:
            rank_logger.event("data_items_skipped", source=choice.source,
                              count=len(skipped), rows=[item["row"] for item in skipped])
        if fetch_seconds > 60:
            print(f"[watchdog] rank={rank} micro_step={micro_step} source={choice.source} "
                  f"rows={[p.row for p in pointers]} fetch={fetch_seconds:.1f}s", flush=True)
        with rank_logger.stage("h2d", micro_step, optimizer_step):
            return move(batch, device), move(targets, device)

    def compute_and_progress(value, micro_step):
        with rank_logger.stage("compute", micro_step, micro_step // accumulation):
            yield value
        rank_logger.progress(
            trainer.state.micro_step, trainer.state.step,
            batch_samples=int(value[0]["token_ids"].shape[0]),
        )

    def stream(start):
        if process_prefetch:
            from .prefetch import prepare_worker
        stop = args.stop_after * accumulation
        if executor is None:
            for micro_step in range(start, stop):
                yield from compute_and_progress(prepare(micro_step), micro_step)
            return
        for chunk_start in range(start, stop, prefetch):
            prepare_fn = prepare_worker if process_prefetch else prepare
            values = executor.map(prepare_fn, range(chunk_start, min(chunk_start + prefetch, stop)))
            for micro_step in range(chunk_start, min(chunk_start + prefetch, stop)):
                if process_prefetch:
                    with rank_logger.stage("process_fetch", micro_step, micro_step // accumulation):
                        batch, targets, audit = next(values)
                        course_logger.log(audit["choice"], micro_step, audit["rows"],
                                          audit["views"], audit["fetch_seconds"])
                        rank_logger.event("course_logged", source=audit["choice"].source,
                                          batch_samples=audit["batch_samples"],
                                          worker_pid=audit["worker_pid"],
                                          skipped_samples=len(audit["skipped"]))
                        if audit["skipped"]:
                            rank_logger.event(
                                "data_items_skipped", source=audit["choice"].source,
                                count=len(audit["skipped"]),
                                rows=[item["row"] for item in audit["skipped"]],
                                worker_pid=audit["worker_pid"])
                        if audit["fetch_seconds"] > 60:
                            print(f"[watchdog] rank={rank} micro_step={micro_step} "
                                  f"source={audit['choice'].source} rows={audit['rows']} "
                                  f"fetch={audit['fetch_seconds']:.1f}s "
                                  f"worker_pid={audit['worker_pid']}", flush=True)
                    with rank_logger.stage("h2d", micro_step, micro_step // accumulation):
                        value = move(batch, device), move(targets, device)
                else:
                    value = next(values)
                yield from compute_and_progress(value, micro_step)

    run_config = {
        "schema_version": "muse_encoder.s5_train.v1",
        "config": config,
        "resolved_curriculum": data.resolved_config["curriculum"],
        "model": cfg.to_dict(),
        "micro_batch": micro_batch,
        "accumulation_steps": accumulation,
        "world_size": world_size,
        "prefetch_per_rank": prefetch,
        "nominal_global_batch": world_size * micro_batch * accumulation,
        "stop_after": args.stop_after,
        "data_verification": verification,
        "data_manifest_sha256": sha256_file(config["data"]["data_manifest"]),
        "validation_test_labels_used": False,
    }
    metadata = {"s5_init": init_record, "data_manifest_sha256": run_config["data_manifest_sha256"],
                "expansion_contract_sha256": contract_sha256,
                "expansion_source_sha256": contract["checkpoint_sha256"]}
    original_save = trainer.save

    def save_with_lineage(name, save_config, save_metadata=None):
        merged = dict(metadata)
        merged.update(save_metadata or {})
        with rank_logger.stage("checkpoint", trainer.state.micro_step, trainer.state.step):
            return original_save(name, save_config, merged)

    trainer.save = save_with_lineage
    # Preserve a named end checkpoint across later runs overwriting final.pt.
    if args.resume:
        end_step = args.stop_after
        fit_save = trainer.save

        def save_candidate_end(name, save_config, save_metadata=None):
            if name == "final":
                fit_save(f"candidate_end_{end_step:08d}", save_config, save_metadata)
            return fit_save(name, save_config, save_metadata)

        trainer.save = save_candidate_end
    if not args.resume:
        # A recoverable lineage point exists even if the process dies before the
        # first periodic checkpoint.  This is not an evaluated/best candidate.
        print(f"[s5] rank={rank} step0-save-start", flush=True)
        trainer.save("step_00000000", run_config, metadata)
        print(f"[s5] rank={rank} step0-save-done", flush=True)
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
    except Exception as exc:
        rank_logger.error(exc)
        raise
    finally:
        try:
            if executor is not None:
                executor.shutdown(wait=True)
            data.close()
            if distributed:
                dist.destroy_process_group()
        finally:
            rank_logger.stop(
                "completed" if trainer.state.step >= args.stop_after else "stopped",
                "fit_finished",
            )


def registered_structure_failure_rows(dataset):
    if dataset.kind != "aux_molecule" or not dataset.use_molecule3d:
        return frozenset()
    store = dataset._open_structure()
    failure_ids = sorted(dataset._structure_failures or ())
    if not failure_ids:
        return frozenset()
    encoded = np.asarray([value.encode("ascii") for value in failure_ids])
    positions = np.searchsorted(store.records["id"], encoded)
    valid = (positions < len(store.records))
    if not valid.all() or not np.array_equal(store.records["id"][positions], encoded):
        raise ValueError(f"{dataset.name}: 3D 失败清单含未登记 ID，拒绝猜测排除行")
    return frozenset(int(value) for value in store.records["aux_row"][positions])


def rank0_environment():
    return int(os.environ.get("RANK", "0")) == 0


if __name__ == "__main__":
    main()
