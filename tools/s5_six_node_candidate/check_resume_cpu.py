"""Exercise candidate resume with tiny CPU models; no real checkpoint training or GPU work."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""

from prepare_candidate import digest, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    root = args.candidate.resolve()
    for path in (root / "model_design/muse_encoder").rglob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    sys.path.insert(0, str(root / "model_design"))
    import torch
    from muse_encoder.pretrain.trainer import CheckpointManager, TrainState
    from muse_encoder.s5.runtime import S5Trainer, warmup_stable_scheduler
    from muse_encoder.s5.expansion import resume_plan, rank_seed

    require(not torch.cuda.is_available(), "CPU verification unexpectedly sees a GPU")

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            require(torch.equal(a, b), "Tensor state differs")
        elif isinstance(a, dict):
            require(a.keys() == b.keys(), "State keys differ")
            for key in a:
                equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            require(len(a) == len(b), "State lengths differ")
            for left, right in zip(a, b):
                equal(left, right)
        else:
            require(a == b, f"State differs: {a!r} vs {b!r}")

    def make(run, rank):
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        scheduler = warmup_stable_scheduler(optimizer, 50)
        trainer = S5Trainer(model, optimizer, scheduler, run, 1, 20, torch.device("cpu"), rank, bf16=False)
        return trainer

    def update(trainer):
        trainer.optimizer.zero_grad()
        (trainer.model(torch.ones(2, 3)) - 2).square().mean().backward()
        trainer.optimizer.step()
        trainer.scheduler.step()

    with tempfile.TemporaryDirectory(prefix="s5_resume_cpu_") as temporary:
        temp = Path(temporary)
        source = make(temp / "source", 0)
        for _ in range(3):
            update(source)
        source.state = TrainState(3, 3, 24)
        config = {"seed": 42}
        run_config = {"schema_version": "muse_encoder.s5_train.v1", "config": config,
                      "world_size": 15, "micro_batch": 8, "accumulation_steps": 1,
                      "data_manifest_sha256": "data"}
        source_path = source.checkpoints.save("source", source.model, source.optimizer, source.scheduler,
                                              source.state, run_config, {"s5_init": {"parent": "test"}})
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        source_hash = digest(source_path)
        contract = {"source_world_size": 15, "target_world_size": 48,
                    "source_state": payload["state"], "checkpoint_sha256": source_hash,
                    "data_manifest_sha256": "data"}
        plan = resume_plan(payload, config, contract, "contract", source_hash, 48, 47, 8, 1)
        expanded = make(temp / "expanded", 47)
        expanded.resume(source_path, payload=payload, expansion_seed=plan["expansion_seed"])
        equal(source.model.state_dict(), expanded.model.state_dict())
        equal(source.optimizer.state_dict(), expanded.optimizer.state_dict())
        equal(source.scheduler.state_dict(), expanded.scheduler.state_dict())
        equal(source.state, expanded.state)
        require(len({rank_seed(42, 3, rank) for rank in range(48)}) == 48, "Rank seed collision")
        print("PASS: rank47 restores exact model, optimizer moments/LR, scheduler and counters without old RNG47")

        config48 = {**run_config, "world_size": 48}
        path48 = expanded.checkpoints.save("next", expanded.model, expanded.optimizer, expanded.scheduler,
                                          expanded.state, config48,
                                          {"s5_init": {"parent": "test"}, "expansion_contract_sha256": "contract"})
        require(not path48.exists() and path48.with_suffix(".rng_rank47.pt").is_file(),
                "Nonzero rank must write its RNG sidecar only")
        rank0_checkpoints = CheckpointManager(path48.parent, rank=0)
        rank0_checkpoints.save("next", expanded.model, expanded.optimizer, expanded.scheduler,
                               expanded.state, config48,
                               {"s5_init": {"parent": "test"}, "expansion_contract_sha256": "contract"})
        require(path48.is_file() and path48.with_suffix(".rng_rank0.pt").is_file(),
                "Rank0 must write the shared checkpoint and its RNG sidecar")
        payload48 = torch.load(path48, map_location="cpu", weights_only=False)
        plan48 = resume_plan(payload48, config, contract, "contract", digest(path48), 48, 47, 8, 1)
        require(plan48["expansion_seed"] is None, "Ordinary resume must not reseed")
        expected_random = torch.rand(20)
        restored = make(temp / "restored", 47)
        restored.resume(path48, payload=payload48)
        equal(expected_random, torch.rand(20))
        update(expanded)
        update(restored)
        equal(expanded.model.state_dict(), restored.model.state_dict())
        equal(expanded.optimizer.state_dict(), restored.optimizer.state_dict())
        equal(expanded.scheduler.state_dict(), restored.scheduler.state_dict())
        print("PASS: 48->48 restores RNG; next deterministic optimizer update exactly matches")

        cases = [("world", lambda p: p["config"].update(world_size=16)),
                 ("batch", lambda p: p["config"].update(micro_batch=3)),
                 ("boundary", lambda p: p["state"].update(micro_step=4)),
                 ("scheduler", lambda p: p["scheduler"].update(last_epoch=1)),
                 ("optimizer", lambda p: p["optimizer"].update(state={})),
                 ("config", lambda p: p["config"].update(config={"seed": 1}))]
        for name, mutate in cases:
            bad = copy.deepcopy(payload)
            mutate(bad)
            try:
                resume_plan(bad, config, contract, "contract", source_hash, 48, 47, 8, 1)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid {name} accepted")
        for state_payload, contract_hash, cp_hash in (
                (payload, "contract", "wrong-checkpoint"),
                (payload48, "wrong-contract", digest(path48))):
            try:
                resume_plan(state_payload, config, contract, contract_hash, cp_hash, 48, 47, 8, 1)
            except ValueError:
                pass
            else:
                raise AssertionError("Mismatched checkpoint/contract accepted")
        path48.with_suffix(".rng_rank47.pt").unlink()
        try:
            restored.resume(path48, payload=payload48)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Missing same-world RNG accepted")
        require(digest(source_path) == source_hash, "Source checkpoint changed")
    print("CPU RESUME CHECK PASS; synthetic states only. Real S5/DDP/save/restart still pending.")


if __name__ == "__main__":
    main()
