from __future__ import annotations

import json
from pathlib import Path

from ..featurize import RecordFeaturizer
from .corpora import PRETRAIN_ROOT, PROJECT_ROOT
from .crystal_dataset import (
    FrameTargetStats,
    OMC25CrystalDataset,
    load_exclusion_spec,
)
from .crystal_references import ElementReferences
from .dataset import PretrainSourceSpec
from .sampling import SamplePointer, SourceSpec


DEFAULT_EXTERNAL_ROOT = (
    PROJECT_ROOT / "data/muse-a/MUSE-A/external_inputs/A_datasets/organic_crystal/OMC25"
)


def _sample_heaps(path):
    count = 0
    heaps = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("id") or not record.get("starting_heap"):
                raise ValueError(f"{path}:{line_number}: 抽样记录缺少 id/starting_heap")
            count += 1
            heaps.add(record["starting_heap"])
    if count < 1:
        raise ValueError(f"{path}: 抽样为空")
    return count, heaps


def _spacegroups(path, required_heaps):
    values = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            heap = record.get("starting_heap")
            if heap not in required_heaps or record.get("missing"):
                continue
            spacegroup = record.get("spacegroup")
            if isinstance(spacegroup, bool) or not isinstance(spacegroup, int) \
                    or not 1 <= spacegroup <= 230:
                raise ValueError(f"{path}:{line_number}: 非法空间群 {spacegroup!r}")
            previous = values.get(heap)
            if previous is not None and previous != spacegroup:
                raise ValueError(f"{heap}: 空间群记录不一致")
            values[heap] = spacegroup
    missing = required_heaps - values.keys()
    if missing:
        examples = sorted(missing)[:5]
        raise ValueError(f"入选晶体缺少有效空间群: {len(missing)} heaps, {examples}")
    return values


class OMC25Source:
    name = "omc25"
    kind = "crystal"

    def __init__(self, cfg, paths, tokenizers=None):
        paths = paths or {}
        cache_dir = Path(paths.get(
            "cache_dir", PRETRAIN_ROOT / "cache" / "omc25_frames_7m"))
        stats_path = Path(paths.get(
            "stats", PRETRAIN_ROOT / "targets" / "omc25_frame_stats.json"))
        references_path = paths.get("element_references")
        references = ElementReferences.from_json(references_path) if references_path else None
        sample_path = Path(paths.get(
            "sample_jsonl", PRETRAIN_ROOT / "index" / "omc25.sample7m.jsonl"))
        exclusions_path = Path(paths.get(
            "suggested_exclude_heaps",
            PRETRAIN_ROOT / "audit" / "omc25-parent-recovery" /
            "suggested_exclude_heaps.jsonl"))
        recovery_path = Path(paths.get(
            "parent_recovery",
            PRETRAIN_ROOT / "audit" / "omc25-parent-recovery" /
            "train.heap_recovery.jsonl"))
        records, heaps = _sample_heaps(sample_path)
        self.spec = PretrainSourceSpec(
            SourceSpec(self.name, "crystal", records), self.kind)
        spacegroups = _spacegroups(recovery_path, heaps)
        source = OMC25CrystalDataset(
            cache_dir=cache_dir,
            cfg=cfg,
            stats=FrameTargetStats.from_json(stats_path),
            external_root=Path(paths.get("external_root", DEFAULT_EXTERNAL_ROOT)),
            lmdb_cache_root=Path(paths.get(
                "lmdb_cache_root", PRETRAIN_ROOT / "cache" / "omc25")),
            sample_jsonl=sample_path,
            exclusions=load_exclusion_spec(exclusions_path),
            tokenizers=tokenizers,
            featurizer=RecordFeaturizer(cfg, tokenizers, strict=True),
            spacegroups=spacegroups,
            strict_targets=True,
            verify_digests=True,
            references=references,
        )
        if len(source) != records:
            raise ValueError(f"OMC25 cache/master {len(source)} != sample {records}")
        self.source = source

    def __len__(self):
        return len(self.source)

    def get(self, pointer: SamplePointer):
        return self.source.get(pointer.row)

    def get_2d_view(self, pointer):
        raise ValueError("OMC25 没有 2D/3D InfoNCE 视图")

    def close(self):
        self.source._views.clear()
