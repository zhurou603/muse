"""Run the DBAASP treatment or matched replay-only control pilot."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[3]
for path in (str(ROOT), str(ROOT / "model_design")):
    if path not in sys.path:
        sys.path.insert(0, path)

from dbaasp import DESCRIPTOR_KEY, DESCRIPTOR_NAMES, LOG_MIC_KEY, DBAASPDataset, load_dbaasp_normalizer
from muse_encoder.config import MuseConfig
from muse_encoder.model import MuseEncoder
from muse_encoder.pretrain.a_train import ATrainTargetNormalizer
from muse_encoder.pretrain.batching import BatchBuilder
from muse_encoder.pretrain.dataset import PretrainDataset, PretrainSourceSpec
from muse_encoder.pretrain.run_pretrain import aux_model_specs, load_init_weights, microbatch_seed
from muse_encoder.pretrain.sampling import HierarchicalSampler, SourceSampler, SourceSpec, normalize_weights
from muse_encoder.pretrain.smoke_dcu import move
from muse_encoder.pretrain.targets import TargetNormalizer
from muse_encoder.pretrain.trainer import PretrainWrapper, Trainer, seed_everything, warmup_cosine_scheduler
from muse_encoder.tokenizers import load_tokenizers

DATA = ROOT / "data/pretrain/stage2/dbaasp"
DEFAULT_INIT = ROOT / "model_design/runs/s3-residual-pilot-02/checkpoints/final.pt"


class WeightedSampler(HierarchicalSampler):
    def __init__(self, sources, source_weights, **kwargs):
        super().__init__(sources, **kwargs)
        names = [source.spec.name for source in self.sources]
        if set(names) != set(source_weights):
            raise ValueError("source_weights 必须精确覆盖 pilot sources")
        self.source_weights = normalize_weights(source_weights)

    def _sample_source(self, rng):
        probabilities = [self.source_weights[source.spec.name] for source in self.sources]
        return self.sources[int(rng.choice(len(self.sources), p=probabilities))]


def _replay_dataset(name, rows_path, cfg, tokenizers):
    dataset = PretrainDataset(name, cfg, tokenizers=tokenizers, seed=42)
    rows = np.load(rows_path)
    dataset._rows = rows
    dataset.spec = PretrainSourceSpec(
        SourceSpec(name, "bio", len(rows), 1), dataset.kind)
    return dataset


def build_datasets(mode, cfg, tokenizers):
    aux_normalizer = TargetNormalizer.from_path(ROOT / "data/pretrain/targets/aux_targets.json")
    dbaasp_normalizer = load_dbaasp_normalizer(DATA / "targets.json")
    main = DBAASPDataset(
        "dbaasp_v4/target_activity", cfg, dbaasp_normalizer,
        tokenizers=tokenizers, seed=42,
    )
    hemo = PretrainDataset("dbaasp_hemo", cfg, aux_normalizer, tokenizers=tokenizers, seed=42)
    nonmic = PretrainDataset("dbaasp_nonmic", cfg, aux_normalizer, tokenizers=tokenizers, seed=42)
    short_uniref = _replay_dataset(
        "natural_uniref50", DATA / "natural_uniref50.rows.npy", cfg, tokenizers)
    short_pdb = _replay_dataset(
        "natural_pdb", DATA / "natural_pdb.rows.npy", cfg, tokenizers)
    long_pdb = _replay_dataset("pdb", DATA / "pdb.rows.npy", cfg, tokenizers)
    if mode == "treatment":
        datasets = [main, hemo, nonmic, short_uniref, short_pdb, long_pdb]
        weights = {
            main.name: 0.45, hemo.name: 0.10, nonmic.name: 0.10,
            short_uniref.name: 0.15, short_pdb.name: 0.05, long_pdb.name: 0.15,
        }
    elif mode == "control":
        datasets = [short_uniref, short_pdb, long_pdb]
        weights = {short_uniref.name: 3 / 7, short_pdb.name: 1 / 7, long_pdb.name: 3 / 7}
    else:
        raise ValueError(mode)
    aux_specs = aux_model_specs(aux_normalizer, [main, hemo, nonmic])
    aux_specs.update(ATrainTargetNormalizer.model_specs(dbaasp_normalizer))
    aux_specs[LOG_MIC_KEY] = {"kind": "graph", "dim": 1, "task_type": "regression"}
    aux_specs[DESCRIPTOR_KEY] = {
        "kind": "graph", "dim": len(DESCRIPTOR_NAMES), "task_type": "regression",
    }
    if mode == "control":
        for unused in (main, hemo, nonmic):
            unused.close()
    return datasets, weights, aux_specs


def verify_data_manifest():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    from muse_encoder.pretrain.trainer import sha256_file
    for name, item in manifest["sources"].items():
        actual = sha256_file(DATA / f"{name}.rows.npy")
        if actual != item["rows_sha256"]:
            raise ValueError(f"{name}: replay rows hash 不匹配")
    if sha256_file(DATA / "targets.json") != manifest["targets"]["targets_sha256"]:
        raise ValueError("DBAASP targets hash 不匹配")
    if sha256_file(DATA / "unsupported_train.rows.txt") != manifest["targets"]["unsupported_train_rows_sha256"]:
        raise ValueError("DBAASP unsupported train rows hash 不匹配")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--init-weights", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--mode", choices=("treatment", "control"), required=True)
    parser.add_argument("--total-steps", type=int, default=100)
    parser.add_argument("--micro-batch", type=int, default=32)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--prefetch", type=int, default=4)
    args = parser.parse_args()
    if args.total_steps < 1 or args.micro_batch < 1 or args.accumulation_steps < 1:
        raise ValueError("训练步数和 batch 必须为正数")
    data_manifest = verify_data_manifest()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=3600))
    rank = dist.get_rank() if distributed else 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    seed_everything(42, True)
    run_dir = args.run_dir
    if run_dir.exists() and (run_dir / "train.jsonl").exists():
        raise FileExistsError(f"pilot 目录已有训练日志: {run_dir}")
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
    incompatible = load_init_weights(model, args.init_weights, rank)
    if incompatible.unexpected_keys or any(
            not key.startswith("heads.aux_heads.") for key in incompatible.missing_keys):
        raise RuntimeError("init checkpoint 与 Encoder 不完全兼容，禁止静默续训")
    if rank == 0:
        (run_dir / "init_incompatible.json").write_text(json.dumps({
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }, indent=2), encoding="utf-8")
    wrapped = PretrainWrapper(model, {"mlm": 1.0, "aux": 1.0})
    if distributed:
        wrapped = DistributedDataParallel(wrapped, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-4)
    scheduler = warmup_cosine_scheduler(optimizer, args.total_steps, min(20, args.total_steps))
    trainer = Trainer(
        wrapped, optimizer, scheduler, run_dir, args.accumulation_steps,
        args.total_steps, device, rank, 10, 50, 1.0, True,
    )
    unsupported = frozenset(
        int(line) for line in (DATA / "unsupported_train.rows.txt").read_text(
            encoding="utf-8").split())
    samplers = [
        SourceSampler(dataset.spec.source,
                      exclude=unsupported if dataset.name == "dbaasp_v4/target_activity" else None)
        for dataset in datasets
    ]
    sampler = WeightedSampler(
        samplers, source_weights, seed=42, stage=2,
        source_grouped=True, domain_weights=None,
    )
    builder = BatchBuilder(cfg, tokenizers)
    executor = ThreadPoolExecutor(max_workers=args.prefetch) if args.prefetch > 1 else None
    by_name = {dataset.name: dataset for dataset in datasets}

    def prepare(micro_step):
        pointers = sampler.batch(micro_step, rank, world_size, args.micro_batch)
        started = time.monotonic()
        items = [by_name[pointer.source].get(pointer) for pointer in pointers]
        elapsed = time.monotonic() - started
        if elapsed > 60:
            print(f"[watchdog] rank={rank} micro_step={micro_step} fetch={elapsed:.1f}s "
                  f"sources={[p.source for p in pointers]}", flush=True)
        records, metadata = zip(*items)
        batch, targets = builder.build(
            list(records), microbatch_seed({"seed": 42, "stage": 2}, micro_step, rank),
            mlm=True, denoise=False, items=list(metadata),
        )
        return move(batch, device), move(targets, device)

    def stream(start):
        stop = args.total_steps * args.accumulation_steps
        if executor is None:
            for micro_step in range(start, stop):
                yield prepare(micro_step)
            return
        for chunk_start in range(start, stop, args.prefetch):
            for value in executor.map(prepare, range(chunk_start, min(chunk_start + args.prefetch, stop))):
                yield value

    run_config = {
        "schema_version": "muse.stage2.dbaasp_pilot.v1",
        "mode": args.mode, "seed": 42, "init_weights": str(args.init_weights),
        "model": cfg.to_dict(),
        "total_steps": args.total_steps, "micro_batch": args.micro_batch,
        "accumulation_steps": args.accumulation_steps, "world_size": world_size,
        "source_weights": source_weights, "data_manifest": data_manifest,
        "aux_objectives": [LOG_MIC_KEY, DESCRIPTOR_KEY],
        "validation_test_labels_used": False,
    }
    if rank == 0:
        (run_dir / "config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        trainer.fit(stream, run_config)
        if rank == 0:
            result = {"mode": args.mode, "step": trainer.state.step,
                      "samples": trainer.state.samples,
                      "checkpoint": str(run_dir / "checkpoints/final.pt")}
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
