from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def warmup_cosine_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def factor(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class PretrainWrapper(nn.Module):
    def __init__(self, encoder, weights):
        super().__init__()
        self.encoder = encoder
        self.weights = weights

    def forward(self, batch, targets):
        return self.encoder.pretraining_losses(batch, targets, self.weights)


@dataclass
class TrainState:
    step: int = 0
    micro_step: int = 0
    samples: int = 0


class CheckpointManager:
    def __init__(self, directory: Path, rank: int = 0):
        self.directory = directory
        self.rank = rank
        directory.mkdir(parents=True, exist_ok=True)

    def _atomic_torch_save(self, value, path: Path):
        tmp = path.with_suffix(path.suffix + f".tmp-rank{self.rank}")
        torch.save(value, tmp)
        os.replace(tmp, path)

    def save(self, name: str, model, optimizer, scheduler, state: TrainState,
             config: dict, metadata: dict | None = None):
        path = self.directory / f"{name}.pt"
        rng_path = self.directory / f"{name}.rng_rank{self.rank}.pt"
        self._atomic_torch_save(rng_state(), rng_path)
        if self.rank == 0:
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            encoder = raw_model.encoder if isinstance(raw_model, PretrainWrapper) else raw_model
            payload = {
                "model": encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "state": asdict(state),
                "config": config,
                "metadata": metadata or {},
            }
            self._atomic_torch_save(payload, path)
            record = {
                "path": str(path),
                "sha256": sha256_file(path),
                "step": state.step,
                "micro_step": state.micro_step,
                "samples": state.samples,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                **(metadata or {}),
            }
            with (self.directory / "checkpoints.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path

    def load(self, path: Path, model, optimizer=None, scheduler=None,
             map_location="cpu") -> TrainState:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        encoder = raw_model.encoder if isinstance(raw_model, PretrainWrapper) else raw_model
        encoder.load_state_dict(payload["model"])
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(payload["scheduler"])
        rng_path = path.with_suffix(f".rng_rank{self.rank}.pt")
        if not rng_path.exists():
            raise FileNotFoundError(f"缺少 rank{self.rank} RNG checkpoint: {rng_path}")
        restore_rng_state(torch.load(rng_path, map_location="cpu", weights_only=False))
        return TrainState(**payload["state"])


class Trainer:
    def __init__(self, model, optimizer, scheduler, run_dir: Path,
                 accumulation_steps: int, total_steps: int,
                 device: torch.device, rank: int = 0,
                 log_interval: int = 20, checkpoint_interval: int = 1000,
                 grad_clip: float = 1.0, bf16: bool = True):
        if accumulation_steps < 1 or total_steps < 1:
            raise ValueError("accumulation_steps/total_steps 必须为正数")
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.run_dir = run_dir
        self.accumulation_steps = accumulation_steps
        self.total_steps = total_steps
        self.device = device
        self.rank = rank
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.grad_clip = grad_clip
        self.bf16 = bf16
        self.checkpoints = CheckpointManager(run_dir / "checkpoints", rank)
        self.state = TrainState()
        self.allow_nonfinite_skip = os.environ.get("S5_ALLOW_NONFINITE_SKIP") == "1"
        self.max_nonfinite_skips = int(os.environ.get("S5_MAX_NONFINITE_SKIPS", "20"))
        self.max_consecutive_nonfinite_skips = int(
            os.environ.get("S5_MAX_CONSECUTIVE_NONFINITE_SKIPS", "3"))
        if min(self.max_nonfinite_skips, self.max_consecutive_nonfinite_skips) < 1:
            raise ValueError("非有限步跳过上限必须为正数")
        self.nonfinite_skip_count = 0
        self.consecutive_nonfinite_skips = 0
        self.course_log_path = None

    def should_stop(self):
        return False

    def resume(self, path: Path):
        self.state = self.checkpoints.load(path, self.model, self.optimizer, self.scheduler)

    def _mole_metrics(self, batch):
        raw_model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        encoder = raw_model.encoder if isinstance(raw_model, PretrainWrapper) else raw_model
        if not hasattr(encoder, "domain_emb") or not hasattr(encoder, "blocks"):
            return {}
        domain = batch["domain"].long()
        cond = encoder.domain_emb(domain)
        domains = int(encoder.cfg.num_domains)
        output = {}
        with torch.no_grad():
            for index, block in enumerate(encoder.blocks):
                if not getattr(block, "is_mole", False):
                    continue
                stats = block.ffn.gate_stats(cond)
                weights = stats["weights"].float()
                counts = torch.bincount(domain, minlength=domains).to(
                    device=weights.device, dtype=weights.dtype)
                parts = [weights.sum(dim=0), stats["entropy"].float().sum().reshape(1), counts]
                for domain_id in range(domains):
                    selected = (domain == domain_id).to(dtype=weights.dtype).unsqueeze(-1)
                    parts.append((weights * selected).sum(dim=0))
                packed = torch.cat(parts)
                if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                experts = weights.shape[1]
                total = packed[experts + 1:experts + 1 + domains].sum().clamp(min=1.0)
                domain_counts = packed[experts + 1:experts + 1 + domains]
                offset = experts + 1 + domains
                domain_weights = []
                for domain_id in range(domains):
                    values = packed[offset + domain_id * experts:offset + (domain_id + 1) * experts]
                    domain_weights.append(
                        (values / domain_counts[domain_id].clamp(min=1.0)).cpu().tolist())
                output[f"layer_{index:02d}"] = {
                    "entropy": float((packed[experts] / total).cpu()),
                    "weights": (packed[:experts] / total).cpu().tolist(),
                    "domain_counts": domain_counts.cpu().tolist(),
                    "domain_weights": domain_weights,
                }
        return output

    def _log(self, losses: dict, lr: float, elapsed: float, batch_samples: int, mole=None,
             grad_norm: float | None = None, skipped: bool = False):
        if self.rank != 0:
            return
        record = {
            "step": self.state.step,
            "micro_step": self.state.micro_step,
            "samples": self.state.samples,
            "lr": lr,
            "elapsed_seconds": round(elapsed, 3),
            "batch_samples": batch_samples,
            "losses": {k: float(v) for k, v in losses.items()},
        }
        if grad_norm is not None:
            record["grad_norm"] = float(grad_norm)
        if skipped:
            record["skipped"] = True
            record["skip_reason"] = "nonfinite_gradient"
        if mole:
            record["mole"] = mole
        with (self.run_dir / "train.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

    def _gradient_is_nonfinite(self, grad_norm) -> bool:
        nonfinite = not bool(torch.isfinite(grad_norm))
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            flag = torch.tensor(int(nonfinite), dtype=torch.int32, device=self.device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            nonfinite = bool(flag.item())
        return nonfinite

    def _record_nonfinite_skip(self, grad_norm, total, config):
        if not self.allow_nonfinite_skip:
            raise FloatingPointError(f"非有限梯度: step={self.state.step}")
        consecutive = self.consecutive_nonfinite_skips + 1
        total_skips = self.nonfinite_skip_count + 1
        if consecutive > self.max_consecutive_nonfinite_skips:
            raise FloatingPointError(
                f"连续非有限梯度跳过超过{self.max_consecutive_nonfinite_skips}次，停止")
        if total_skips > self.max_nonfinite_skips:
            raise FloatingPointError(f"累计非有限梯度跳过超过{self.max_nonfinite_skips}次，停止")
        self.consecutive_nonfinite_skips = consecutive
        self.nonfinite_skip_count = total_skips
        import fcntl
        import socket
        course = {}
        if self.course_log_path is not None and self.course_log_path.exists():
            rows = self.course_log_path.read_text(encoding="utf-8").splitlines()
            if rows:
                latest = json.loads(rows[-1])
                if latest.get("micro_step") == self.state.micro_step - 1:
                    course = {key: latest.get(key) for key in (
                        "source", "domain", "rows", "views")}
        record = {
            "event": "optimization_step_skipped",
            "reason": "nonfinite_gradient",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "step": self.state.step,
            "micro_step": self.state.micro_step - 1,
            "rank": self.rank,
            "world_size": dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "session": os.environ.get("S5_SESSION_ID"),
            "grad_norm": str(float(grad_norm)),
            "loss_total": float(total.detach()),
            "optimizer_update": False,
            "scheduler_policy": "advance_with_course_step",
            "consecutive_skips": self.consecutive_nonfinite_skips,
            "nonfinite_skip_count": self.nonfinite_skip_count,
            "max_consecutive_skips": self.max_consecutive_nonfinite_skips,
            "max_nonfinite_skips": self.max_nonfinite_skips,
            **course,
        }
        path = self.run_dir / "optimization_skips.jsonl"
        with path.open("ab") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        config["nonfinite_skip_count"] = self.nonfinite_skip_count

    def save(self, name: str, config: dict, metadata: dict | None = None):
        return self.checkpoints.save(name, self.model, self.optimizer, self.scheduler,
                                     self.state, config, metadata)

    def fit(self, stream_factory: Callable[[int], Iterable], config: dict):
        if self.state.step >= self.total_steps:
            return self.state
        config.setdefault("nonfinite_skip_count", self.nonfinite_skip_count)
        started = time.monotonic()
        self.optimizer.zero_grad(set_to_none=True)
        aggregated = {}
        aggregated_count = 0
        batch_samples = 0
        for batch, targets in stream_factory(self.state.micro_step):
            if self.should_stop():
                break
            is_last = (self.state.micro_step + 1) % self.accumulation_steps == 0
            sync_context = self.model.no_sync() if (
                isinstance(self.model, DistributedDataParallel) and not is_last
            ) else _NullContext()
            with sync_context:
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                    enabled=self.bf16):
                    losses = self.model(batch, targets)
                total = losses["total"]
                if not torch.isfinite(total):
                    raise FloatingPointError(f"非有限 loss: micro_step={self.state.micro_step}")
                (total / self.accumulation_steps).backward()
            for key, value in losses.items():
                aggregated[key] = aggregated.get(key, 0.0) + float(value.detach())
            aggregated_count += 1
            self.state.micro_step += 1
            batch_samples += int(batch["token_ids"].shape[0])
            if not is_last:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            skipped = self._gradient_is_nonfinite(grad_norm)
            if skipped:
                self._record_nonfinite_skip(grad_norm, total, config)
            else:
                self.consecutive_nonfinite_skips = 0
                self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.state.step += 1
            self.state.samples += batch_samples
            if self.state.step % self.log_interval == 0:
                values = {k: v / max(1, aggregated_count) for k, v in aggregated.items()}
                self._log(values, self.scheduler.get_last_lr()[0],
                          time.monotonic() - started, batch_samples,
                          grad_norm=None if skipped else float(grad_norm), skipped=skipped)
            aggregated = {}
            aggregated_count = 0
            batch_samples = 0
            if self.state.step % self.checkpoint_interval == 0:
                self.save(f"step_{self.state.step:08d}", config,
                          {"loss": float(total.detach())})
            if self.state.step >= self.total_steps:
                break
        if self.state.micro_step % self.accumulation_steps:
            raise RuntimeError("数据流在梯度累积边界前结束，不能安全保存续跑状态")
        self.save("final", config)
        return self.state


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False
