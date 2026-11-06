from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from muse_encoder.config import MuseConfig
from muse_encoder.pretrain.a_train import ATrainDataset, ATrainTargetNormalizer, load_source_specs
from muse_encoder.pretrain.corpora import DescriptorNormalizer
from muse_encoder.pretrain.dataset import PretrainDataset, PretrainSourceSpec
from muse_encoder.pretrain.run_pretrain import aux_model_specs
from muse_encoder.pretrain.sampling import SamplePointer, SourceSampler, SourceSpec
from muse_encoder.pretrain.targets import TargetNormalizer
from muse_encoder.pretrain.trainer import Trainer, sha256_file

from dbaasp import (
    DESCRIPTOR_KEY,
    DESCRIPTOR_NAMES,
    LOG_MIC_KEY,
    DBAASPDataset,
    load_dbaasp_normalizer,
)
from run_dbaasp_pilot import WeightedSampler

ROOT = Path(__file__).resolve().parents[3]
RUN_NAME = "s4-overnight-polymer-electronic-consolidation-v1"
RUN_ROOT = ROOT / "model_design/runs" / RUN_NAME
DATA = ROOT / "data/pretrain/s4-polymer-consolidation"
PARENT = ROOT / "model_design/runs/stage2-dbaasp-expanded-300/checkpoints/final.pt"
PARENT_SHA256 = "d6254aec6941d19aa1bae0a9d19d5a3ca4b39f1aa9854f4ac9a269f14fbe1046"
SEED = 42
MAX_STEPS = 300

POLYMER_INTERNAL_TASKS = {
    "Polymer-Genome-2016/DFT-HSE06/electronic_band_gap": 0.18,
    "Polymer-Genome-2016/DFT-rPW86/electronic_band_gap": 0.18,
    "Polymer-Genome-2016/DFT-rPW86/atomization_energy": 0.04,
    "TransPolymer/Egb/egb": 0.0625,
    "TransPolymer/Egc/egc": 0.0625,
    "TransPolymer/Eea/eea": 0.0625,
    "TransPolymer/Ei/ei": 0.0625,
    "Aldeghi-Coley-Copolymer/vipea/electron_affinity": 0.075,
    "Aldeghi-Coley-Copolymer/vipea/ionization_potential": 0.075,
    "PolyGraphMT/BANDGAP_DFT/bandgap_dft": 0.025,
    "PolyGraphMT/HOMO_DFT/homo_dft": 0.025,
    "PolyGraphMT/LUMO_DFT/lumo_dft": 0.025,
    "PolyGraphMT/MU_DFT/mu_dft": 0.025,
}
POLYMER_TASKS = {name: weight * 0.75 for name, weight in POLYMER_INTERNAL_TASKS.items()}
DBAASP_INTERNAL_WEIGHTS = {
    "dbaasp_v4/target_activity": 0.45,
    "dbaasp_hemo": 0.10,
    "dbaasp_nonmic": 0.10,
    "natural_uniref50": 0.15,
    "natural_pdb": 0.05,
    "pdb": 0.15,
}
DBAASP_WEIGHTS = {name: weight * 0.09 for name, weight in DBAASP_INTERNAL_WEIGHTS.items()}
SMALL_WEIGHTS = {"qmugs": 0.025, "qm9": 0.025, "geom_drugs": 0.03, "pubchem": 0.02}
CONTROL_WEIGHTS = {
    "pi1m": 0.75,
    "natural_uniref50": 0.09,
    "natural_pdb": 0.03,
    "pdb": 0.03,
    **SMALL_WEIGHTS,
}

DATASET_TASK_COUNTS = {
    "moleculenet": 1,
    "tdc_admet": 22,
    "GEOM-Drugs": 1,
    "QMugs": 5,
    "QM9": 4,
    "PolyGraphMT": 20,
    "Aldeghi-Coley-Copolymer": 2,
    "TransPolymer": 10,
    "Polymer-Genome-2016": 3,
    "OMC25": 1,
    "dbaasp_v4": 1,
    "PEER": 16,
    "CATH": 4,
}
DATASET_DOMAINS = {
    "small_molecule": ("moleculenet", "tdc_admet", "GEOM-Drugs", "QMugs", "QM9"),
    "polymer": ("PolyGraphMT", "Aldeghi-Coley-Copolymer", "TransPolymer", "Polymer-Genome-2016"),
    "organic_crystal": ("OMC25",),
    "biomacromolecule": ("dbaasp_v4", "PEER", "CATH"),
}
GAPS = (
    "Polymer-Genome-2016/DFT-HSE06/electronic_band_gap",
    "Polymer-Genome-2016/DFT-rPW86/electronic_band_gap",
)


def peer_cath_tasks() -> dict[str, float]:
    specs = load_source_specs()
    peer = [task.task_id for spec in specs.values() if spec.dataset == "PEER" for task in spec.tasks]
    cath = [task.task_id for spec in specs.values() if spec.dataset == "CATH" for task in spec.tasks]
    if len(peer) != 16 or len(cath) != 4:
        raise RuntimeError(f"PEER/CATH 任务数异常: {len(peer)}/{len(cath)}")
    return {**{name: 0.03 / 16 for name in peer}, **{name: 0.03 / 4 for name in cath}}


def treatment_weights() -> dict[str, float]:
    weights = {
        **POLYMER_TASKS,
        "pi1m": 0.075,
        **DBAASP_WEIGHTS,
        **peer_cath_tasks(),
        **SMALL_WEIGHTS,
    }
    if len(weights) != 44 or not np.isclose(sum(weights.values()), 1.0):
        raise RuntimeError(f"S4 treatment 来源或权重异常: {len(weights)}/{sum(weights.values())}")
    return weights


def load_config_from_checkpoint(path: Path) -> MuseConfig:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw = payload["config"]["model"]
    if "model" in raw and isinstance(raw["model"], dict) and "d_model" not in raw:
        raw = raw["model"]
    valid = set(MuseConfig.__dataclass_fields__)
    return MuseConfig(**{key: value for key, value in raw.items() if key in valid})


class FilteredTaskDataset:
    def __init__(self, task_id, cfg, normalizer, tokenizers):
        self.task_id = task_id
        source_name = task_id.rsplit("/", 1)[0]
        self.base = ATrainDataset(source_name, cfg, normalizer, tokenizers=tokenizers, seed=SEED)
        self.name = task_id
        self.kind = "a_train"
        self.spec = PretrainSourceSpec(
            SourceSpec(task_id, self.base.spec.source.domain, len(self.base), 1), self.kind)
        self.allowed = {key for key, spec in normalizer.specs.items() if spec.task_id == task_id}
        if len(self.allowed) != 1:
            raise ValueError(f"{task_id}: 任务头映射异常")

    def get(self, pointer):
        translated = SamplePointer(self.base.name, pointer.domain, pointer.row, pointer.view)
        record, item = self.base.get(translated)
        item["aux_records"] = [aux for aux in item["aux_records"] if aux["name"] in self.allowed]
        if not item["aux_records"]:
            raise ValueError(f"{self.task_id}: 过滤后没有监督目标")
        record.meta["s4_task"] = self.task_id
        return record, item

    def close(self):
        self.base.close()


class ReplayOnlyDataset:
    def __init__(self, base):
        self.base = base
        self.name = base.name
        self.kind = base.kind
        self.spec = base.spec

    def get(self, pointer):
        record, item = self.base.get(pointer)
        item = dict(item)
        item.pop("aux_records", None)
        item.pop("energy", None)
        record.meta["s4_replay_only"] = True
        return record, item

    def close(self):
        self.base.close()


class RankSynchronizedWeightedSampler(WeightedSampler):
    def batch(self, micro_step, rank, world_size, micro_batch):
        if min(micro_step, rank, micro_batch) < 0 or world_size < 1 or rank >= world_size:
            raise ValueError("非法 DDP batch 参数")
        source = self._sample_source(self._rng(micro_step))
        row_rng = self._rng(micro_step * world_size + rank, attempt=1)
        return [source.sample(row_rng) for _ in range(micro_batch)]


def linear_scheduler(optimizer, total_steps=MAX_STEPS, warmup_steps=15):
    def factor(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class S4Trainer(Trainer):
    def _log(self, losses, lr, elapsed, batch_samples, mole=None, grad_norm=None):
        super()._log(losses, lr, elapsed, batch_samples, mole=mole, grad_norm=grad_norm)
        if self.rank != 0:
            return
        record = {
            "step": self.state.step,
            "micro_step": self.state.micro_step,
            "learning_rates": [float(value) for value in self.scheduler.get_last_lr()],
            "grad_norm": None if grad_norm is None else float(grad_norm),
            "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
            "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
            "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
        }
        with (self.run_dir / "optimizer.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")


def _normalizers():
    return {
        "polymer": ATrainTargetNormalizer.from_path(ROOT / "data/pretrain/stage3-polymer/targets.json"),
        "bio": ATrainTargetNormalizer.from_path(DATA / "bio_targets.json"),
        "dbaasp": load_dbaasp_normalizer(ROOT / "data/pretrain/stage2/dbaasp/targets.json"),
        "aux": TargetNormalizer.from_path(ROOT / "data/pretrain/targets/aux_targets.json"),
        "descriptor": DescriptorNormalizer.from_path(ROOT / "data/pretrain/targets/mtr_norm.json"),
    }


def _generic(name, cfg, normalizers, tokenizers, molecule3d=False):
    return PretrainDataset(
        name, cfg, normalizers["aux"], normalizers["descriptor"], tokenizers, SEED,
        use_molecule3d=molecule3d,
    )


def build_datasets(mode, cfg, tokenizers):
    if mode not in ("treatment", "control"):
        raise ValueError(mode)
    normalizers = _normalizers()
    polymer = [FilteredTaskDataset(name, cfg, normalizers["polymer"], tokenizers)
               for name in POLYMER_TASKS]
    bio = [FilteredTaskDataset(name, cfg, normalizers["bio"], tokenizers)
           for name in peer_cath_tasks()]
    dbaasp_main = DBAASPDataset(
        "dbaasp_v4/target_activity", cfg, normalizers["dbaasp"], tokenizers=tokenizers, seed=SEED)
    treatment = polymer + bio + [
        dbaasp_main,
        _generic("dbaasp_hemo", cfg, normalizers, tokenizers),
        _generic("dbaasp_nonmic", cfg, normalizers, tokenizers),
        _generic("natural_uniref50", cfg, normalizers, tokenizers),
        _generic("natural_pdb", cfg, normalizers, tokenizers),
        _generic("pdb", cfg, normalizers, tokenizers),
        _generic("pi1m", cfg, normalizers, tokenizers),
        _generic("qmugs", cfg, normalizers, tokenizers),
        _generic("qm9", cfg, normalizers, tokenizers),
        _generic("geom_drugs", cfg, normalizers, tokenizers),
        _generic("pubchem", cfg, normalizers, tokenizers),
    ]
    aux_specs = {}
    aux_specs.update(normalizers["polymer"].model_specs())
    aux_specs.update(normalizers["bio"].model_specs())
    aux_specs.update(normalizers["dbaasp"].model_specs())
    supported_aux = aux_model_specs(normalizers["aux"], treatment)
    aux_specs.update({key: spec for key, spec in supported_aux.items()
                      if spec["task_type"] in ("regression", "binary_classification",
                                               "multiclass_classification", "sequence_classification")})
    aux_specs[LOG_MIC_KEY] = {"kind": "graph", "dim": 1, "task_type": "regression"}
    aux_specs[DESCRIPTOR_KEY] = {
        "kind": "graph", "dim": len(DESCRIPTOR_NAMES), "task_type": "regression"}
    if mode == "treatment":
        return treatment, treatment_weights(), aux_specs
    for dataset in treatment:
        dataset.close()
    control = [
        _generic("pi1m", cfg, normalizers, tokenizers),
        _generic("natural_uniref50", cfg, normalizers, tokenizers),
        _generic("natural_pdb", cfg, normalizers, tokenizers),
        _generic("pdb", cfg, normalizers, tokenizers),
    ]
    control.extend(ReplayOnlyDataset(_generic(name, cfg, normalizers, tokenizers))
                   for name in SMALL_WEIGHTS)
    return control, dict(CONTROL_WEIGHTS), aux_specs


def make_sampler(datasets, weights):
    unsupported = frozenset(
        int(line) for line in (ROOT / "data/pretrain/stage2/dbaasp/unsupported_train.rows.txt")
        .read_text(encoding="utf-8").split())
    samplers = []
    for dataset in datasets:
        exclude = unsupported if dataset.name == "dbaasp_v4/target_activity" else None
        samplers.append(SourceSampler(dataset.spec.source, exclude=exclude))
    return RankSynchronizedWeightedSampler(
        samplers, weights, seed=SEED, stage=4, source_grouped=True, domain_weights=None)


def verify_data_manifest():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["inputs"].values():
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"输入哈希不匹配: {path}")
    if sha256_file(DATA / "bio_targets.json") != manifest["bio_targets_sha256"]:
        raise ValueError("S4 bio targets 哈希不匹配")
    if manifest["new_external_data"] or manifest["validation_test_labels_used"]:
        raise ValueError("S4 数据合规标记异常")
    return manifest


def domain_source_weights(weights):
    specs = load_source_specs()
    output = {"polymer": 0.0, "bio": 0.0, "small_molecule": 0.0, "crystal": 0.0}
    for name, weight in weights.items():
        if name in ("pi1m",):
            domain = "polymer"
        elif name in DBAASP_WEIGHTS or name in ("natural_uniref50", "natural_pdb", "pdb"):
            domain = "bio"
        elif name in SMALL_WEIGHTS:
            domain = "small_molecule"
        elif name in POLYMER_TASKS or name in peer_cath_tasks():
            domain = specs[name.rsplit("/", 1)[0]].source.domain
        else:
            raise KeyError(name)
        output[domain] += weight
    return output
