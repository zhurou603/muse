"""Process-based S5 batch prefetch so CPU featurization does not hold the trainer GIL."""
from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from muse_encoder.pretrain.sampling import SourceSampler
from muse_encoder.pretrain.trainer import seed_everything
from .curriculum import microbatch_seed


_CONTEXT = None


def initialize_worker(config: dict[str, Any], model_config: dict[str, Any], rank: int,
                      exclusions: dict[str, frozenset[int]], micro_batch: int,
                      accumulation_steps: int, run_dir=None) -> None:
    global _CONTEXT
    from muse_encoder.config import MuseConfig
    from muse_encoder.pretrain.batching import BatchBuilder
    from muse_encoder.tokenizers import load_tokenizers
    from .data import S5Data
    from .data_skip import DataSkipLogger

    seed_everything(int(config["seed"]), bool(config.get("deterministic", True)))
    cfg = MuseConfig(**model_config)
    data = S5Data(config, cfg, load_tokenizers())
    cfg.aux_specs = data.aux_specs
    samplers = {
        name: SourceSampler(dataset.spec.source,
                            exclude=frozenset(exclusions.get(name, ())) or None)
        for name, dataset in data.datasets.items()
    }
    _CONTEXT = (data, cfg, BatchBuilder(cfg, data.tokenizers), samplers, int(rank),
                int(micro_batch), int(accumulation_steps), DataSkipLogger(run_dir, rank))
    atexit.register(data.close)


def prepare_worker(micro_step: int):
    if _CONTEXT is None:
        raise RuntimeError("prefetch worker 尚未初始化")
    from .data_skip import fetch_valid_batch
    data, cfg, builder, samplers, rank, micro_batch, accumulation_steps, skip_logger = _CONTEXT
    optimizer_step = micro_step // accumulation_steps
    choice = data.curriculum.resolve(optimizer_step)
    seed = microbatch_seed(int(data.config["seed"]), int(data.config["stage"]),
                           micro_step, rank)
    rng = np.random.default_rng(seed)
    sampler = samplers[choice.source]
    pointers = [sampler.sample(rng) for _ in range(micro_batch)]
    dataset = data.datasets[choice.source]
    started = time.monotonic()
    records, metadata, pointers, align_pairs, skipped = fetch_valid_batch(
        dataset, sampler, rng, pointers,
        align=choice.weights.get("align", 0.0) > 0,
        skip_logger=skip_logger, micro_step=micro_step, cfg=cfg,
    )
    fetch_seconds = time.monotonic() - started
    for record in records:
        record.meta["s5_source"] = choice.source
        record.meta["s5_step"] = choice.step
    batch, targets = builder.build(
        records, seed,
        mlm=choice.weights.get("mlm", 0.0) > 0,
        denoise=choice.weights.get("denoise", 0.0) > 0,
        items=metadata,
    )
    if align_pairs:
        import torch
        targets["align_idx"] = torch.tensor(align_pairs, dtype=torch.long)
    targets["_s5_loss_weights"] = choice.weights
    return batch, targets, {
        "choice": choice,
        "rows": [pointer.row for pointer in pointers],
        "views": [pointer.view for pointer in pointers],
        "fetch_seconds": fetch_seconds,
        "batch_samples": len(records),
        "worker_pid": os.getpid(),
        "skipped": skipped,
    }


def create_process_pool(config: dict[str, Any], model_config: dict[str, Any], rank: int,
                        exclusions: dict[str, frozenset[int]], workers: int,
                        micro_batch: int, accumulation_steps: int,
                        run_dir=None) -> ProcessPoolExecutor:
    if workers < 1:
        raise ValueError("prefetch 进程数必须为正数")
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=initialize_worker,
        initargs=(config, model_config, rank, exclusions, micro_batch, accumulation_steps, run_dir),
    )


def main() -> None:
    import argparse
    import json
    from pathlib import Path
    from muse_encoder.config import MuseConfig
    from muse_encoder.pretrain.corpora import PRETRAIN_ROOT
    from muse_encoder.tokenizers import load_tokenizers
    from .data import S5Data
    from .provenance import atomic_json
    from .runtime import load_yaml
    from .train import registered_structure_failure_rows

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--micro-step", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--expected-course", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    cfg = MuseConfig(**config.get("model", {}))
    data = S5Data(config, cfg, load_tokenizers())
    cfg.aux_specs = data.aux_specs
    exclusions = {}
    for name, dataset in data.datasets.items():
        excluded = set()
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
        exclusions[name] = frozenset(excluded)
    pool = create_process_pool(
        config, cfg.to_dict(), args.rank, exclusions, args.workers,
        int(config["training"]["micro_batch"]), int(config["training"]["accumulation_steps"]),
    )
    try:
        batch, _targets, audit = pool.submit(prepare_worker, args.micro_step).result(timeout=300)
    finally:
        pool.shutdown(wait=True)
        data.close()
    summary = {
        "source": audit["choice"].source, "micro_step": args.micro_step, "rank": args.rank,
        "rows": audit["rows"], "views": audit["views"],
        "batch_samples": audit["batch_samples"], "worker_pid": audit["worker_pid"],
        "batch_tokens": list(batch["token_ids"].shape), "process_prefetch": True,
    }
    if args.expected_course is not None:
        for line in args.expected_course.read_text(encoding="utf-8").splitlines():
            old = json.loads(line)
            if int(old["micro_step"]) == args.micro_step:
                if old["source"] != summary["source"] or old["rows"] != summary["rows"] \
                        or old["views"] != summary["views"]:
                    raise RuntimeError("进程 prefetch 改变了课程 source/row/view")
                summary["course_matches_existing"] = True
                break
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, summary)


if __name__ == "__main__":
    main()
