"""Explicit, guarded S5 15->48 expansion; ordinary 48->48 restores RNG sidecars."""
from __future__ import annotations

import hashlib


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rank_seed(seed, step, rank):
    value = f"s5-expand-15-to-48:{seed}:{step}:{rank}".encode("ascii")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little")


def resume_plan(payload, config, contract, contract_sha256, checkpoint_sha256,
                world_size, rank, micro_batch, accumulation):
    require(world_size == 48 and 0 <= rank < world_size, "Candidate requires 48 ranks")
    require((micro_batch, accumulation) == (8, 1), "Candidate batch must be 8 x 1 per rank")
    require(contract["target_world_size"] == 48 and contract["source_world_size"] == 15,
            "Unexpected expansion contract")
    old = payload["config"]
    require(old["schema_version"] == "muse_encoder.s5_train.v1", "Not an S5 checkpoint")
    require(old["config"] == config, "Checkpoint/config mismatch; no silent config migration")
    require(old["micro_batch"] == 8 and old["accumulation_steps"] == 1,
            "Source runtime batch differs from audited 8 x 1")
    require(old["data_manifest_sha256"] == contract["data_manifest_sha256"],
            "Data manifest identity changed")
    state = payload["state"]
    require(state["step"] >= 0 and state["micro_step"] == state["step"],
            "Checkpoint is not at the expected optimizer/accumulation boundary")
    require(payload["scheduler"]["last_epoch"] == state["step"], "Scheduler step mismatch")
    require(bool(payload["optimizer"]["state"]), "Missing optimizer moments")
    require(bool(payload["metadata"].get("s5_init")), "Missing S5 initialization lineage")
    source = old["world_size"]
    if source == 15:
        require(checkpoint_sha256 == contract["checkpoint_sha256"],
                "Expansion must use the pinned checkpoint")
        require(state == contract["source_state"], "Pinned training state changed")
        seed = rank_seed(config["seed"], state["step"], rank)
    elif source == 48:
        require(payload["metadata"].get("expansion_contract_sha256") == contract_sha256,
                "48-rank resume belongs to another expansion")
        require(state["step"] >= contract["source_state"]["step"], "Resume predates expansion")
        seed = None
    else:
        raise ValueError(f"Unsupported source world size: {source}")
    return {"source_world_size": source, "target_world_size": 48,
            "old_nominal_batch": source * 8, "new_nominal_batch": 384,
            "expansion_seed": seed, "step": state["step"],
            "rng_policy": "rank_reseed" if seed is not None else "restore_sidecar",
            "sampling": "existing stateless (seed, stage, micro_step, rank); with replacement",
            "samples_counter": "historical rank0 local expanded rows; NOT global unique records"}
