"""Read all 47 smoke logs and both checkpoint generations; never train or use a GPU."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from launch import check_candidate
from prepare_candidate import digest, require, write_json
from topology import WORLD_SIZE, local_worker_count, rank_location


def records(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    root = args.candidate.resolve()
    contract = check_candidate(root)
    base = contract["source_state"]["step"]
    run = root / "runs/smoke"
    for rank in range(WORLD_SIZE):
        host, local_rank = rank_location(rank)
        rows = records(run / "health" / f"rank_{rank:02d}.jsonl")
        sessions = {}
        for row in rows:
            sessions.setdefault(row["session"], []).append(row)
        require(len(sessions) == 2, f"rank{rank}: expected exactly smoke + resume sessions; review retries/errors")
        ordered = sorted(sessions.values(), key=lambda group: group[0]["timestamp"])
        for number, group in enumerate(ordered):
            require(all(r["world_size"] == WORLD_SIZE and r["device"] == f"cuda:{local_rank}" and
                        r["host"] == host for r in group), "Rank/device/host changed")
            require(not any(r["event"] == "error" or r.get("status") == "error" for r in group),
                    f"rank{rank}: error recorded")
            require(group[-1]["event"] == "stopped" and group[-1]["status"] == "completed",
                    f"rank{rank}: no clean completion")
            resumed = [r for r in group if r["event"] == "resume_verified"]
            require(len(resumed) == 1, f"rank{rank}: missing/duplicate resume")
            require(resumed[0]["rng_policy"] == ("rank_reseed" if number == 0 else "restore_sidecar"),
                    f"rank{rank}: wrong RNG resume policy")
            progress = [r["optimizer_step"] for r in group if r["event"] == "progress"]
            expected = list(range(base + 1, base + 21)) if number == 0 else [base + 21]
            require(progress == expected[:-1], f"rank{rank}: intermediate step progression mismatch")
            require(group[-1]["optimizer_step"] == expected[-1], f"rank{rank}: wrong final step")
    logs = records(run / "train.jsonl")
    require(bool(logs) and all(math.isfinite(v) for r in logs for v in r["losses"].values()),
            "Missing/nonfinite training loss log")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    import torch
    checkpoints = []
    for step in (base + 20, base + 21):
        path = run / "checkpoints" / f"candidate_end_{step:08d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require(payload["config"]["world_size"] == WORLD_SIZE and payload["config"]["micro_batch"] == 8 and
                payload["config"]["accumulation_steps"] == 1, "Saved batch/world size mismatch")
        require(payload["state"]["step"] == step == payload["state"]["micro_step"] ==
                payload["scheduler"]["last_epoch"], "Saved training/scheduler step mismatch")
        require(payload["metadata"]["expansion_contract_sha256"] == digest(root / "contract.json"),
                "Saved checkpoint lineage mismatch")
        require(bool(payload["optimizer"]["state"]), "Saved optimizer moments absent")
        del payload
        for rank in range(WORLD_SIZE):
            host, _ = rank_location(rank)
            rng = torch.load(path.with_suffix(f".rng_rank{rank}.pt"), map_location="cpu", weights_only=False)
            require(all(key in rng for key in ("python", "numpy", "torch", "cuda")) and
                    len(rng["cuda"]) == local_worker_count(host),
                    f"Missing/incomplete rank{rank} RNG")
        checkpoints.append({"path": str(path), "sha256": digest(path), "step": step})
    old_skip = (root / "handoff/data_skips.jsonl").read_bytes()
    all_skip = (run / "data_skips.jsonl").read_bytes()
    require(all_skip.startswith(old_skip), "Inherited skip log changed")
    new_skips = [json.loads(line) for line in all_skip[len(old_skip):].splitlines() if line.strip()]
    report = {"status": "review_new_data_skips" if new_skips else "smoke_and_resume_pass",
              "contract_sha256": digest(root / "contract.json"), "ranks": WORLD_SIZE,
              "new_data_skip_events": len(new_skips), "checkpoints": checkpoints,
              "scope": "20 optimizer steps + save + 47-rank resume + 1 step; not quality/performance acceptance",
              "reserved_device": "physical card 7 on 171 remains available for A-test",
              "full_curriculum_note": "20 source slots tested; DBAASP 200-step option cycle not fully covered"}
    write_json(root / "smoke_review.json", report)
    require(not new_skips, "Compute completed but NEW data skips require review; see smoke_review.json/data_skips.jsonl")
    write_json(root / "acceptance.json", report)
    print("47-RANK S5 SMOKE + RESUME PASS; no new data skips; not a throughput or model-quality claim.")


if __name__ == "__main__":
    main()
