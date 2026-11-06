"""Restore the actual pinned S5 model/optimizer on CPU, including expanded rank 47."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""

from launch import check_candidate
from prepare_candidate import digest


def equal(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    elif hasattr(left, "shape"):
        assert left.shape == right.shape and left.dtype == right.dtype
        assert bool((left == right).all())
    else:
        assert left == right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    root = args.candidate.resolve()
    contract = check_candidate(root)
    sys.path.insert(0, str(root / "model_design"))
    import torch
    from muse_encoder.config import MuseConfig
    from muse_encoder.model import MuseEncoder
    from muse_encoder.s5.expansion import rank_seed, resume_plan
    from muse_encoder.s5.runtime import S5Trainer, S5Wrapper, build_optimizer, warmup_stable_scheduler

    checkpoint = root / "handoff/source.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]["config"]
    plan = resume_plan(payload, config, contract, digest(root / "contract.json"),
                       contract["checkpoint_sha256"], 48, 47, 8, 1)
    model = MuseEncoder(MuseConfig(**payload["config"]["model"]))
    wrapped = S5Wrapper(model, {})
    optimizer = build_optimizer(wrapped, payload["metadata"]["s5_init"],
                                config["training"]["lr"], config["training"]["new_head_lr"],
                                config["training"].get("weight_decay", 0.01))
    scheduler = warmup_stable_scheduler(optimizer, config["training"]["warmup_steps"])
    with tempfile.TemporaryDirectory(prefix="s5_actual_cpu_resume_") as temporary:
        trainer = S5Trainer(wrapped, optimizer, scheduler, Path(temporary), 1,
                            contract["source_state"]["step"] + 1, torch.device("cpu"), rank=47)
        trainer.resume(checkpoint, payload=payload, expansion_seed=plan["expansion_seed"])
        equal(trainer.state.__dict__, payload["state"])
        equal(model.state_dict(), payload["model"])
        equal(optimizer.state_dict(), payload["optimizer"])
        equal(scheduler.state_dict(), payload["scheduler"])
        finite = all(bool(torch.isfinite(value).all()) for value in model.state_dict().values())
        assert finite
        report = {"status": "pass", "checkpoint": str(checkpoint),
                  "checkpoint_sha256": digest(checkpoint), "source_state": payload["state"],
                  "rank": 47, "rng_policy": plan["rng_policy"], "model_exact": True,
                  "optimizer_exact": True, "scheduler_exact": True, "finite_parameters": finite,
                  "optimizer_param_groups": [{key: group[key] for key in ("lr", "weight_decay", "group_name")}
                                             for group in optimizer.param_groups],
                  "scope": "actual weights/optimizer/scheduler loaded on CPU; no GPU/DDP update or quality claim",
                  "third_party_weights_loaded": False, "validation_test_labels_used": False}
        output = root / "actual_checkpoint_cpu_verification.json"
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
