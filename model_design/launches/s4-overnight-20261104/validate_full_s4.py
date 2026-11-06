from __future__ import annotations

import argparse
import json
from pathlib import Path

from aggregate_s4 import aggregate


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-scores", type=Path, required=True)
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--candidate-dbaasp-rescue", type=float, required=True)
    parser.add_argument("--parent-shadow", type=Path, required=True)
    parser.add_argument("--candidate-shadow", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    parent_raw = read_json(args.parent_scores)
    candidate_raw = read_json(args.candidate_scores)
    parent = aggregate([args.parent_scores], "full")
    candidate = aggregate([args.candidate_scores], "full")
    if not parent_raw["complete"] or not candidate_raw["complete"]:
        raise ValueError("父子模型都必须是 complete=true 的90项结果")
    if abs(parent["J_val"] - parent_raw["final_score"]) > 1e-8 \
            or abs(candidate["J_val"] - candidate_raw["final_score"]) > 1e-8:
        raise ValueError("J_val 与官方 final_score 不一致")
    results = read_json(args.candidate_results)
    missing_heads = [task_id for task_id, row in results["tasks"].items()
                     if row.get("status") != "ok"]
    dataset_delta = {name: candidate["datasets"][name] - parent["datasets"][name]
                     for name in parent["datasets"]}
    task_delta = {name: candidate["tasks"][name] - parent["tasks"][name]
                  for name in parent["tasks"]}
    parent_shadow = read_json(args.parent_shadow)
    candidate_shadow = read_json(args.candidate_shadow)
    parent_shadow_score = parent_shadow["summary"]["mean_normalized_score"]
    candidate_shadow_score = candidate_shadow["summary"]["mean_normalized_score"]
    checks = {
        "all_90_heads_ok": not missing_heads and candidate_raw["scored_tasks"] == 90,
        "J_val_improvement_at_least_0_20": candidate["J_val"] >= parent["J_val"] + 0.20,
        "improvement_not_single_task": sum(value > 0 for value in task_delta.values()) >= 2,
        "dbaasp_default_drop_at_most_5": dataset_delta["dbaasp_v4"] >= -5.0,
        "dbaasp_rescue_drop_at_most_5": args.candidate_dbaasp_rescue >= 34.14737327267912 - 5.0,
        "peer_drop_at_most_1": dataset_delta["PEER"] >= -1.0,
        "cath_drop_at_most_1": dataset_delta["CATH"] >= -1.0,
        "omc25_drop_at_most_0_2": dataset_delta["OMC25"] >= -0.2,
        "shadow_all_ok": parent_shadow["summary"]["all_ok"] and candidate_shadow["summary"]["all_ok"],
        "shadow_not_decreased": candidate_shadow_score >= parent_shadow_score,
    }
    result = {
        "schema_version": "muse.s4.full_acceptance.v1",
        "parent": parent,
        "candidate": candidate,
        "J_val_delta": candidate["J_val"] - parent["J_val"],
        "dataset_delta": dataset_delta,
        "task_delta": task_delta,
        "candidate_dbaasp_rescue": args.candidate_dbaasp_rescue,
        "parent_dbaasp_rescue": 34.14737327267912,
        "missing_heads": missing_heads,
        "shadow": {"parent": parent_shadow_score, "candidate": candidate_shadow_score,
                   "delta": candidate_shadow_score - parent_shadow_score},
        "checks": checks,
        "accepted": all(checks.values()),
        "test_used": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"accepted": result["accepted"], "J_val_delta": result["J_val_delta"],
                      "checks": checks}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
