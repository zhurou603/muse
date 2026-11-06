from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SamplePointer:
    source: str
    domain: str
    row: int
    view: int = 0


@dataclass(frozen=True)
class SourceSpec:
    name: str
    domain: str
    records: int
    views: int = 1

    @property
    def size(self):
        return self.records * self.views


class ScaffoldIndex:
    def __init__(self, row_groups: np.ndarray, group_hashes: np.ndarray | None = None):
        row_groups = np.asarray(row_groups, dtype=np.int64)
        if row_groups.ndim != 1 or len(row_groups) == 0 or (row_groups < 0).any():
            raise ValueError("row_groups 必须是非空非负一维数组")
        self.row_groups = row_groups
        self.group_sizes = np.bincount(row_groups)
        if group_hashes is not None:
            group_hashes = np.asarray(group_hashes, dtype=np.uint64)
            if group_hashes.shape != (len(self.group_sizes), 2):
                raise ValueError("group_hashes 必须为 [n_groups,2] uint64")
        self.group_hashes = group_hashes
        order = np.argsort(row_groups, kind="stable")
        self.sorted_rows = order.astype(np.int64)
        self.offsets = np.zeros(len(self.group_sizes) + 1, dtype=np.int64)
        np.cumsum(self.group_sizes, out=self.offsets[1:])

    def group_row(self, group: int, offset: int) -> int:
        start = self.offsets[group]
        return int(self.sorted_rows[start + offset])

    def group_hash(self, row: int):
        group = int(self.row_groups[row])
        if self.group_hashes is None:
            return group
        return tuple(int(value) for value in self.group_hashes[group])


class SourceSampler:
    def __init__(self, spec: SourceSpec, scaffold: ScaffoldIndex | None = None,
                 exclude: frozenset[int] | None = None, max_exclude_attempts: int = 64):
        if spec.records < 1 or spec.views < 1:
            raise ValueError("records/views 必须为正数")
        if scaffold is not None and len(scaffold.row_groups) != spec.records:
            raise ValueError(f"{spec.name}: scaffold 行数与 records 不一致")
        self.spec = spec
        self.scaffold = scaffold
        self.scaffold_weights = (
            temperature_weights(scaffold.group_sizes, 0.005) if scaffold is not None else None
        )
        self.exclude = exclude or frozenset()
        if len(self.exclude) >= spec.records:
            raise ValueError(f"{spec.name}: 排除行数覆盖全部记录")
        self.max_exclude_attempts = int(max_exclude_attempts)

    def sample(self, rng) -> SamplePointer:
        for _ in range(self.max_exclude_attempts):
            if self.scaffold is None:
                row = int(rng.integers(self.spec.records))
            else:
                weights = self.scaffold_weights
                group = int(rng.choice(len(weights), p=weights))
                row = self.scaffold.group_row(group, int(rng.integers(self.scaffold.group_sizes[group])))
            if row not in self.exclude:
                break
        else:
            raise RuntimeError(
                f"{self.spec.name}: 连续 {self.max_exclude_attempts} 次命中排除行")
        view = int(rng.integers(self.spec.views)) if self.spec.views > 1 else 0
        return SamplePointer(self.spec.name, self.spec.domain, row, view)

    def fingerprint(self, pointer: SamplePointer):
        if self.scaffold is None:
            return None
        value = self.scaffold.group_hash(pointer.row)
        return value if self.scaffold.group_hashes is not None else (self.spec.name, value)


def temperature_weights(counts, temperature: float = 0.005) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or (values <= 0).any():
        raise ValueError("采样计数必须为正数一维数组")
    if not 0 < temperature <= 1:
        raise ValueError("temperature 必须在 (0,1] 内")
    weights = np.power(values, temperature)
    return weights / weights.sum()


class HierarchicalSampler:
    def __init__(self, sources, seed: int = 42, stage: int = 0,
                 domain_weights: dict[str, float] | None = None,
                 temperature: float = 0.005, unique_scaffolds: bool = False,
                 max_attempts: int = 1000, source_grouped: bool = False):
        if not sources:
            raise ValueError("sources 不能为空")
        if max_attempts < 1:
            raise ValueError("max_attempts 必须为正数")
        self.sources = list(sources)
        self.by_name = {source.spec.name: source for source in self.sources}
        self.seed = seed
        self.stage = stage
        self.temperature = temperature
        self.unique_scaffolds = bool(unique_scaffolds)
        self.source_grouped = bool(source_grouped)
        self.max_attempts = int(max_attempts)
        self.domains = sorted({s.spec.domain for s in sources})
        self.by_domain = {
            domain: [s for s in self.sources if s.spec.domain == domain]
            for domain in self.domains
        }
        if domain_weights is None:
            totals = {d: sum(s.spec.size for s in self.by_domain[d]) for d in self.domains}
            self.domain_weights = temperature_weights_dict(totals, temperature)
        else:
            if set(domain_weights) != set(self.domains):
                raise ValueError("domain_weights 必须精确覆盖本阶段全部 domain")
            self.domain_weights = normalize_weights(domain_weights)
        if self.unique_scaffolds and self.source_grouped:
            raise ValueError("unique_scaffolds 与 source_grouped 暂不支持同时开启")

    def _rng(self, index: int, attempt: int = 0):
        values = [self.seed, self.stage, index]
        if attempt:
            values.append(attempt)
        return np.random.default_rng(np.random.SeedSequence(values))

    def _sample_source(self, rng):
        domain = str(rng.choice(self.domains, p=[self.domain_weights[d] for d in self.domains]))
        candidates = self.by_domain[domain]
        weights = temperature_weights([s.spec.size for s in candidates], self.temperature)
        return candidates[int(rng.choice(len(candidates), p=weights))]

    def sample(self, index: int, attempt: int = 0) -> SamplePointer:
        if index < 0 or attempt < 0:
            raise ValueError("index/attempt 必须非负")
        rng = self._rng(index, attempt)
        return self._sample_source(rng).sample(rng)

    def batch(self, micro_step: int, rank: int, world_size: int,
              micro_batch: int) -> list[SamplePointer]:
        if min(micro_step, rank, micro_batch) < 0 or world_size < 1 or rank >= world_size:
            raise ValueError("非法 DDP batch 参数")
        start = (micro_step * world_size + rank) * micro_batch
        if self.source_grouped:
            # 每个 micro-batch 只取一个源：同源样本长度相近，消灭跨域 padding 浪费。
            rng = self._rng(start)
            source = self._sample_source(rng)
            return [source.sample(rng) for _ in range(micro_batch)]
        if not self.unique_scaffolds:
            return [self.sample(start + i) for i in range(micro_batch)]
        pointers = []
        seen = set()
        for offset in range(micro_batch):
            index = start + offset
            for attempt in range(self.max_attempts):
                pointer = self.sample(index, attempt)
                fingerprint = self.by_name[pointer.source].fingerprint(pointer)
                if fingerprint is None or fingerprint not in seen:
                    pointers.append(pointer)
                    if fingerprint is not None:
                        seen.add(fingerprint)
                    break
            else:
                raise RuntimeError(
                    f"批内骨架去重超过 {self.max_attempts} 次: micro_step={micro_step}, rank={rank}")
        return pointers


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    if not weights or any(v < 0 for v in weights.values()):
        raise ValueError("权重必须非空且非负")
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("权重和必须为正数")
    return {key: float(value) / total for key, value in weights.items()}


def temperature_weights_dict(counts: dict[str, int], temperature: float) -> dict[str, float]:
    keys = sorted(counts)
    values = temperature_weights([counts[key] for key in keys], temperature)
    return dict(zip(keys, values))
