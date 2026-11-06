"""Continuous S5 runtime: strict initialization, stable LR, resume and lineage."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel

from muse_encoder.pretrain.trainer import CheckpointManager, PretrainWrapper, Trainer

from .provenance import atomic_json, sha256_file


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read YAML and expand explicit environment variables such as CODE_ROOT."""
    text = os.path.expandvars(Path(path).read_text(encoding="utf-8"))
    if "${" in text:
        # expandvars leaves unknown variables untouched; silently training with a
        # literal '${...}' path is much worse than a clear configuration error.
        import re
        missing = sorted(set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", text)))
        if missing:
            raise KeyError(f"未设置配置环境变量: {missing}")
    value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("S5 配置根节点必须是 mapping")
    return value


def warmup_stable_scheduler(optimizer, warmup_steps: int):
    warmup_steps = int(warmup_steps)
    if warmup_steps < 0:
        raise ValueError("warmup_steps 必须非负")

    def factor(step):
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class S5Wrapper(PretrainWrapper):
    def forward(self, batch, targets):
        weights = dict(self.weights)
        source_weights = targets.get("_s5_loss_weights", {})
        if not isinstance(source_weights, Mapping):
            raise TypeError("_s5_loss_weights 必须是 mapping")
        unknown = set(source_weights) - {
            "mlm", "denoise", "coordinate", "distance", "energy", "force",
            "stress", "descriptor", "align", "aux",
        }
        if unknown:
            raise KeyError(f"未知损失权重: {sorted(unknown)}")
        weights.update({key: float(value) for key, value in source_weights.items()})
        clean_targets = dict(targets)
        clean_targets.pop("_s5_loss_weights", None)
        return self.encoder.pretraining_losses(batch, clean_targets, weights)


def strict_initialize(model, checkpoint_path: Path, rank: int = 0):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("父 checkpoint 缺少 model")
    incompatible = model.load_state_dict(payload["model"], strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"父 checkpoint 含未知参数: {incompatible.unexpected_keys[:10]}")
    illegal = [key for key in incompatible.missing_keys
               if not key.startswith("heads.aux_heads.")]
    if illegal:
        raise RuntimeError(f"父 checkpoint 缺少非新增 aux 参数: {illegal[:10]}")
    new_prefixes = sorted({key.rsplit(".", 1)[0] for key in incompatible.missing_keys})
    record = {
        "parent_checkpoint": str(Path(checkpoint_path).resolve()),
        "parent_sha256": sha256_file(checkpoint_path),
        "new_aux_parameter_count": len(incompatible.missing_keys),
        "new_aux_prefixes": new_prefixes,
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
    if rank == 0:
        print("[strict-init] " + json.dumps(record, ensure_ascii=False), flush=True)
    return record


def build_optimizer(model, init_record: Mapping[str, Any], base_lr: float,
                    new_head_lr: float, weight_decay: float):
    prefixes = tuple("encoder." + prefix for prefix in init_record.get("new_aux_prefixes", []))
    old, new = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (new if prefixes and name.startswith(prefixes) else old).append(parameter)
    if not old:
        raise RuntimeError("旧参数组为空")
    if new and not prefixes:
        raise RuntimeError("新头参数组状态不一致")
    groups = [{"params": old, "lr": float(base_lr), "weight_decay": float(weight_decay),
               "group_name": "existing"}]
    if new:
        groups.append({"params": new, "lr": float(new_head_lr),
                       "weight_decay": float(weight_decay), "group_name": "new_aux_heads"})
    optimizer = torch.optim.AdamW(groups, lr=float(base_lr), weight_decay=float(weight_decay))
    return optimizer


def resolve_resume_path(path: str | Path) -> Path:
    path = Path(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        path = Path(payload["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


class S5Trainer(Trainer):
    """Trainer with a latest.json pointer and atomic checkpoint metadata."""

    def save(self, name: str, config: dict, metadata: dict | None = None):
        metadata = dict(metadata or {})
        metadata.setdefault("curriculum_schema", "muse_encoder.s5_curriculum.v1")
        path = super().save(name, config, metadata)
        # Every rank writes its own RNG sidecar.  Do not publish latest.json until
        # all sidecars exist, otherwise an immediate crash/resume can lose a rank.
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if self.rank == 0:
            atomic_json(self.run_dir / "latest.json", {
                "path": str(path), "step": int(self.state.step),
                "micro_step": int(self.state.micro_step), "sha256": sha256_file(path),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })
        return path

    def should_stop(self):
        control = os.environ.get("S5_CONTROL_FILE")
        if not control:
            return False
        message = [None]
        if not dist.is_initialized() or self.rank == 0:
            path = Path(control)
            if path.is_file():
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                    if record.get("action") != "stop" or not record.get("request_id"):
                        raise ValueError("Invalid coordinated-stop request")
                    message[0] = record
                except Exception as exc:
                    message[0] = {"error": f"{type(exc).__name__}: {exc}"}
        if dist.is_initialized():
            dist.broadcast_object_list(message, src=0)
        if message[0] and "error" in message[0]:
            raise ValueError(message[0]["error"])
        return bool(message[0])

    def resume(self, path: Path, *, payload=None, expansion_seed=None):
        path = resolve_resume_path(path)
        if payload is None:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        expected = sha256_file(path)
        self.state = self.checkpoints.load(
            path, self.model, self.optimizer, self.scheduler, payload=payload,
            restore_rng=expansion_seed is None)
        if expansion_seed is not None:
            from muse_encoder.pretrain.trainer import seed_everything
            seed_everything(expansion_seed)
        state = self.state
        if self.rank == 0:
            atomic_json(self.run_dir / "resumed_from.json", {
                "path": str(path.resolve()), "sha256": expected,
                "step": int(state.step), "micro_step": int(state.micro_step),
                "rng_policy": "rank_reseed" if expansion_seed is not None else "restore_sidecar",
                "rank0_expansion_seed": expansion_seed,
            })
        return state


def checkpoint_encoder_payload(path: str | Path) -> dict[str, Any]:
    """Return a read-only evaluation copy description; never mutate training files."""
    path = resolve_resume_path(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}
