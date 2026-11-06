"""Deterministic 20-optimizer-step curriculum for S5.

A source is selected from the optimizer step, never from a random draw.  All DDP
ranks therefore see the same source for a step and for every accumulation micro
step.  Only row/view sampling differs between ranks.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DBAASP_MICRO_PATTERN = (
    ("main",) * 9
    + ("hemo",) * 2
    + ("nonmic",) * 2
    + ("short_uniref",) * 3
    + ("short_pdb",)
    + ("long_pdb",) * 3
)


@dataclass(frozen=True)
class Slot:
    domain: str
    options: tuple[str, ...]
    weights: Mapping[str, float]
    group: str = "default"
    label: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Slot":
        options = value.get("options") or ([value["source"]] if value.get("source") else [])
        if not options:
            raise ValueError(f"课程槽位没有 source/options: {value}")
        weights = {str(k): float(v) for k, v in (value.get("weights") or {}).items()}
        if any(v < 0 or not np.isfinite(v) for v in weights.values()):
            raise ValueError(f"非法损失权重: {weights}")
        return cls(
            domain=str(value["domain"]),
            options=tuple(str(item) for item in options),
            weights=weights,
            group=str(value.get("group", "default")),
            label=str(value.get("label", "")),
        )


@dataclass(frozen=True)
class Choice:
    step: int
    slot_index: int
    cycle: int
    domain: str
    source: str
    weights: Mapping[str, float]
    group: str
    label: str
    dbaasp_micro_index: int | None = None


class Curriculum:
    def __init__(self, slots: Sequence[Slot | Mapping[str, Any]], dbaasp_sources: Mapping[str, str]):
        self.slots = tuple(item if isinstance(item, Slot) else Slot.from_dict(item) for item in slots)
        if len(self.slots) != 20:
            raise ValueError(f"S5 课程必须恰好 20 个 optimizer-step 槽位，实际 {len(self.slots)}")
        counts = {domain: sum(slot.domain == domain for slot in self.slots)
                  for domain in ("small_molecule", "polymer", "bio", "crystal")}
        expected = {"small_molecule": 6, "polymer": 5, "bio": 6, "crystal": 3}
        if counts != expected:
            raise ValueError(f"领域槽位数 {counts} != {expected}")
        dbaasp_slots = [index for index, slot in enumerate(self.slots) if slot.group == "dbaasp"]
        if dbaasp_slots != [11, 12]:
            raise ValueError("DBAASP 必须固定在槽位 11、12，才能复用 20 项微课程")
        self.dbaasp_sources = dict(dbaasp_sources)
        if set(self.dbaasp_sources) != set(DBAASP_MICRO_PATTERN):
            raise ValueError("DBAASP 微课程 source 映射不完整")
        for slot in self.slots:
            if slot.group != "dbaasp" and not slot.options:
                raise ValueError("空课程槽位")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Curriculum":
        return cls(config["curriculum"]["slots"], config["data"]["dbaasp_sources"])

    def resolve(self, step: int) -> Choice:
        if isinstance(step, bool) or int(step) != step or step < 0:
            raise ValueError(f"step 必须是非负整数: {step!r}")
        step = int(step)
        slot_index = step % 20
        cycle = step // 20
        slot = self.slots[slot_index]
        if slot.group == "dbaasp":
            # Two DBAASP slots per large cycle.  Ten large cycles traverse the
            # exact 9/2/2/3/1/3 pattern (45/10/10/15/5/15 percent).
            micro = (2 * cycle + (slot_index - 11)) % 20
            source = self.dbaasp_sources[DBAASP_MICRO_PATTERN[micro]]
        else:
            micro = None
            source = slot.options[cycle % len(slot.options)]
        return Choice(step, slot_index, cycle, slot.domain, source, dict(slot.weights),
                      slot.group, slot.label, micro)


def microbatch_seed(seed: int, stage: int, micro_step: int, rank: int) -> int:
    values = [int(seed), int(stage), int(micro_step), int(rank)]
    return int(np.random.SeedSequence(values).generate_state(1, dtype=np.uint32)[0])


class CourseLogger:
    """Per-rank append-only course/row audit; never used to alter sampling."""

    def __init__(self, run_dir: Path, rank: int, log_rows: bool = True):
        self.path = Path(run_dir) / f"course_rank{rank}.jsonl"
        self.rank = int(rank)
        self.log_rows = bool(log_rows)

    def log(self, choice: Choice, micro_step: int, rows: Sequence[int], views: Sequence[int],
            fetch_seconds: float) -> None:
        record = asdict(choice)
        record.update({
            "rank": self.rank,
            "micro_step": int(micro_step),
            "fetch_seconds": round(float(fetch_seconds), 6),
        })
        if self.log_rows:
            record["rows"] = [int(row) for row in rows]
            record["views"] = [int(view) for view in views]
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
