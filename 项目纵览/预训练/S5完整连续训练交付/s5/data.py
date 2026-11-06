"""S5 dataset factory: official train/aux + reviewed unlabeled replay only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from muse_encoder.pretrain.a_train import ATrainDataset, load_source_specs
from muse_encoder.pretrain.corpora import DescriptorNormalizer
from muse_encoder.pretrain.dataset import (
    MOLECULE3D_SOURCES,
    PretrainDataset,
    PretrainSourceSpec,
)
from muse_encoder.pretrain.omc25_source import OMC25Source
from muse_encoder.pretrain.run_pretrain import aux_model_specs
from muse_encoder.pretrain.sampling import SamplePointer, SourceSpec
from muse_encoder.pretrain.targets import TargetNormalizer

from .curriculum import Curriculum
from .dbaasp import S5DBAASPNormalizer
from .long_sequence import TrainingRecordFeaturizer
from .robust_targets import S5TargetNormalizer


POLYMER_GENOME_TASKS = (
    "Polymer-Genome-2016/DFT-HSE06/electronic_band_gap",
    "Polymer-Genome-2016/DFT-rPW86/electronic_band_gap",
)
POLYMER_ELECTRONIC_TASKS = (
    "TransPolymer/Egb/egb",
    "TransPolymer/Egc/egc",
    "TransPolymer/Eea/eea",
    "TransPolymer/Ei/ei",
    "Aldeghi-Coley-Copolymer/vipea/electron_affinity",
    "Aldeghi-Coley-Copolymer/vipea/ionization_potential",
    "PolyGraphMT/BANDGAP_DFT/bandgap_dft",
    "PolyGraphMT/HOMO_DFT/homo_dft",
    "PolyGraphMT/LUMO_DFT/lumo_dft",
    "PolyGraphMT/MU_DFT/mu_dft",
)


def expand_slots(config: Mapping[str, Any], registered: Mapping[str, Any]) -> list[dict[str, Any]]:
    def dataset_sources(dataset: str) -> list[str]:
        return sorted(name for name, spec in registered.items() if spec.dataset == dataset)

    groups = {
        "@small_quantum_train": ["QMugs/qmugs", "QM9/qm9"],
        "@small_physical": ["moleculenet/freesolv"] + dataset_sources("tdc_admet"),
        "@polymer_genome_gap": list(POLYMER_GENOME_TASKS),
        "@polymer_electronic": list(POLYMER_ELECTRONIC_TASKS),
        "@peer": dataset_sources("PEER"),
        "@cath": dataset_sources("CATH"),
    }
    slots = []
    for raw in config["curriculum"]["slots"]:
        slot = dict(raw)
        options = slot.get("options") or ([slot["source"]] if slot.get("source") else [])
        expanded = []
        for option in options:
            if option.startswith("@"):
                if option not in groups or not groups[option]:
                    raise ValueError(f"课程数据组为空或未知: {option}")
                expanded.extend(groups[option])
            else:
                expanded.append(option)
        slot.pop("source", None)
        slot["options"] = expanded
        slots.append(slot)
    return slots


class FilteredTaskDataset:
    """Expose exactly one official train task while retaining base train checks."""

    def __init__(self, task_id, cfg, normalizer, data_root, tokenizers, seed):
        self.task_id = task_id
        source_name = task_id.rsplit("/", 1)[0]
        self.base = ATrainDataset(source_name, cfg, normalizer, data_root=data_root,
                                  tokenizers=tokenizers, seed=seed)
        self.name = task_id
        self.kind = "a_train"
        self.spec = PretrainSourceSpec(
            SourceSpec(task_id, self.base.spec.source.domain, len(self.base), 1), self.kind)
        self.allowed = {key for key, spec in normalizer.specs.items() if spec.task_id == task_id}
        if len(self.allowed) != 1:
            raise ValueError(f"{task_id}: 任务头映射异常")
        self.base.featurizer = TrainingRecordFeaturizer(cfg, tokenizers, crop_seed=seed)

    def __len__(self):
        return len(self.base)

    def get(self, pointer):
        translated = SamplePointer(self.base.name, pointer.domain, pointer.row, 0)
        record, item = self.base.get(translated)
        item["aux_records"] = [aux for aux in item["aux_records"] if aux["name"] in self.allowed]
        if not item["aux_records"]:
            raise ValueError(f"{self.task_id}: 过滤后没有监督目标")
        record.meta["s5_task"] = self.task_id
        return record, item

    def close(self):
        self.base.close()


class S5Data:
    def __init__(self, config: Mapping[str, Any], cfg, tokenizers):
        self.config = dict(config)
        self.cfg = cfg
        self.tokenizers = tokenizers
        paths = config["data"]
        self.data_root = Path(paths["data_root"])
        self.registered = load_source_specs(self.data_root)
        resolved = dict(config)
        resolved["curriculum"] = {"slots": expand_slots(config, self.registered)}
        self.curriculum = Curriculum.from_config(resolved)
        self.resolved_config = resolved
        self.generic = S5TargetNormalizer.from_path(Path(paths["a_train_targets"]))
        self.dbaasp = S5DBAASPNormalizer(json.loads(
            Path(paths["dbaasp_targets"]).read_text(encoding="utf-8")))
        self.aux = TargetNormalizer.from_path(Path(paths["aux_targets"]))
        self.descriptor = DescriptorNormalizer.from_path(Path(paths["mtr_norm"]))
        self.datasets: dict[str, Any] = {}
        names = self.required_sources()
        for name in sorted(names):
            self.datasets[name] = self._build(name)
        self.aux_specs = self._model_specs()

    def required_sources(self) -> set[str]:
        names = {option for slot in self.curriculum.slots for option in slot.options}
        names.update(self.curriculum.dbaasp_sources.values())
        return names

    def _training_featurizer(self, dataset):
        if not isinstance(dataset, OMC25Source):
            dataset.featurizer = TrainingRecordFeaturizer(
                self.cfg, self.tokenizers, crop_seed=int(self.config["seed"]))

    def _build(self, name: str):
        seed = int(self.config["seed"])
        if name in self.generic.specs:
            raise ValueError(f"不能把 aux head key 当作 source: {name}")
        generic_tasks = {spec.task_id for spec in self.generic.specs.values()}
        if name in generic_tasks and name not in self.registered:
            return FilteredTaskDataset(name, self.cfg, self.generic, self.data_root,
                                       self.tokenizers, seed)
        if name in self.registered:
            normalizer = self.dbaasp if name == "dbaasp_v4/target_activity" else self.generic
            dataset = ATrainDataset(name, self.cfg, normalizer, data_root=self.data_root,
                                    tokenizers=self.tokenizers, seed=seed)
            self._training_featurizer(dataset)
            return dataset
        if name == "omc25":
            return OMC25Source(self.cfg, self.config["data"].get("crystal"), self.tokenizers)
        molecule3d = name in set(self.config["data"].get("molecule3d_sources", []))
        if molecule3d and name not in MOLECULE3D_SOURCES:
            raise ValueError(f"{name}: 无官方 3D 结构索引，禁止启用 molecule3d")
        dataset = PretrainDataset(name, self.cfg, self.aux, self.descriptor,
                                  self.tokenizers, seed, use_molecule3d=molecule3d)
        replay_dir = self.config["data"].get("dbaasp_replay_dir")
        replay_path = Path(replay_dir) / f"{name}.rows.npy" if replay_dir else None
        if replay_path is not None and replay_path.exists() and name in (
                "natural_uniref50", "natural_pdb", "pdb"):
            rows = np.load(replay_path)
            if rows.ndim != 1 or not len(rows):
                raise ValueError(f"{replay_path}: replay 行索引为空或形状错误")
            dataset._rows = rows
            dataset.spec = PretrainSourceSpec(SourceSpec(name, "bio", len(rows), 1), dataset.kind)
        self._training_featurizer(dataset)
        return dataset

    def _model_specs(self):
        datasets = list(self.datasets.values())
        output = self.generic.model_specs()
        for key, value in self.dbaasp.model_specs().items():
            if key in output and output[key] != value:
                raise ValueError(f"DBAASP/generic head 冲突: {key}")
            output[key] = value
        for key, value in aux_model_specs(self.aux, datasets).items():
            if key in output and output[key] != value:
                raise ValueError(f"aux/generic head 冲突: {key}")
            output[key] = value
        return output

    def close(self):
        for dataset in self.datasets.values():
            dataset.close()
