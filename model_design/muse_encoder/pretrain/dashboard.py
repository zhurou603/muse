"""MUSE-A validation 固定探针看板。

只从用户提供的 MuseEncoder checkpoint 取表征；不读取旧 ``emb.npy``，也不加载
``models/`` 下的公开权重。所有 probe 超参数固定在 ``ProbeConfig``；结果通过
官方 ``MUSE-eval/score.py`` 的 validation 口径计算。

常用命令::

    PYTHONPATH=model_design python -m muse_encoder.pretrain.dashboard \
      --checkpoint RUN/checkpoints/final.pt --output-dir RUN/a_board \
      --device cuda:0 --batch-size 128

``--limit-per-split`` 只用于 smoke：看板仍列出 A 榜全部 90 个子任务，但结果会
明确标记 ``smoke=true``，不会把截断结果写成全量通过。
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import math
import pickle
import random
import re
import sys
import tarfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DESIGN_ROOT = PROJECT_ROOT / "model_design"
for _path in (str(PROJECT_ROOT), str(MODEL_DESIGN_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from muse import tasks as T  # noqa: E402
from muse_encoder.config import MuseConfig  # noqa: E402
from muse_encoder.featurize import RecordFeaturizer  # noqa: E402
from muse_encoder.inference import (  # noqa: E402
    _can_window,
    _encode_compatible,
    _encode_long_record,
    _expected_tokens,
    _featurize_record,
    _token_output,
    _torch_batch,
)
from muse_encoder.model import MuseEncoder, check_probe_params  # noqa: E402
from muse_encoder.pretrain.omc25 import (  # noqa: E402
    extract_lmdb_member,
    read_ase_lmdb_row,
)
from muse_encoder.pretrain.trainer import seed_everything, sha256_file  # noqa: E402
from muse_encoder.tokenizers import load_tokenizers  # noqa: E402

DATA_ROOT = T.DATA_DIR
EXTERNAL_ROOT = T.EXTERNAL_DIR
OFFICIAL_SCORER_PATH = T.ROOT / "data/muse-a/eval/MUSE-eval/score.py"
SHADOW_MANIFEST = T.ROOT / "data/pretrain/manifest/shadow_properties.json"
PUBLIC_MODEL_ROOT = (T.ROOT / "models").resolve()
SPLITS = ("train", "validation")


@dataclass(frozen=True)
class ProbeConfig:
    """固定的 validation probe 配方；CLI 不暴露任何一项。"""

    hidden: int = 48
    seeds: Tuple[int, ...] = (0, 1, 2, 3)
    max_epochs: int = 300
    patience: int = 30
    lr: float = 1e-3
    batch_size: int = 256
    weight_decay: float = 1e-4
    early_stop_fraction: float = 0.10
    early_stop_seed: int = 12345
    max_params: int = 100_000

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["seeds"] = list(self.seeds)
        out["architecture"] = "Linear(D,hidden)->ReLU->Linear(hidden,1/K)"
        out["budget_rule"] = (
            "hidden=48 unless D=1024 with K=1195 makes the requested head exceed "
            "100K; then hidden is floored to the largest budget-compliant value and flagged"
        )
        return out


PROBE_CONFIG = ProbeConfig()


@dataclass
class EncodedSplit:
    ids: List[str]
    embeddings: np.ndarray
    tokens: Optional[np.ndarray]
    offsets: Optional[np.ndarray]
    failed: List[dict[str, str]]


class FixedProbe(nn.Module):
    def __init__(self, d_in: int, hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


def probe_parameter_count(d_in: int, hidden: int, d_out: int) -> int:
    return check_probe_params(d_in, hidden, d_out, PROBE_CONFIG.max_params)


def probe_hidden_dim(d_in: int, d_out: int, config: ProbeConfig = PROBE_CONFIG) -> Tuple[int, bool, int]:
    """返回 (hidden, budget_adapted, params)。

    A 榜 PEER/fold 有 K=1195。D=1024 时 D×48→1195 需要 107,755 个参数，和
    ≤100K 硬上限冲突；只有这一种情况使用预算允许的最大 hidden，并在结果中标记。
    """
    requested = config.hidden
    params = requested * (d_in + 1 + d_out) + d_out
    if params <= config.max_params:
        return requested, False, params
    hidden = (config.max_params - d_out) // (d_in + 1 + d_out)
    if hidden < 1:
        raise ValueError(f"D={d_in}, K={d_out} 在 {config.max_params} 参数预算内无可行 MLP")
    params = check_probe_params(d_in, hidden, d_out, config.max_params)
    return int(hidden), True, params


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _predict_probe(model: nn.Module, X: np.ndarray, task_type: str,
                   device: torch.device, batch_size: int = 4096) -> np.ndarray:
    outputs = []
    tensor = torch.from_numpy(np.asarray(X, dtype=np.float32))
    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch_size):
            value = model(tensor[start:start + batch_size].to(device))
            if task_type in ("regression", "binary_classification"):
                value = value.squeeze(-1)
            if task_type == "binary_classification":
                value = torch.sigmoid(value)
            elif task_type in ("multiclass_classification", "sequence_classification"):
                value = torch.softmax(value, dim=-1)
            outputs.append(value.float().cpu().numpy())
    if not outputs:
        width = 1 if task_type in ("regression", "binary_classification") else model.net[-1].out_features
        return np.empty((0, width), dtype=np.float32)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def _train_probe_seed(X_train: np.ndarray, y_train: np.ndarray,
                      X_stop: np.ndarray, y_stop: np.ndarray,
                      task_type: str, n_class: Optional[int], seed: int,
                      device: torch.device, config: ProbeConfig) -> nn.Module:
    _set_seed(seed)
    d_out = 1 if task_type in ("regression", "binary_classification") else int(n_class)
    hidden, _, _ = probe_hidden_dim(X_train.shape[1], d_out, config)
    model = FixedProbe(X_train.shape[1], hidden, d_out).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    if task_type == "regression":
        criterion = nn.HuberLoss(delta=1.0)
    elif task_type == "binary_classification":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    x_train = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    x_stop = torch.from_numpy(X_stop).float().to(device)
    y_stop_t = torch.from_numpy(y_stop).float().to(device)
    if task_type in ("multiclass_classification", "sequence_classification"):
        y_train_t = y_train_t.long()
        y_stop_t = y_stop_t.long()

    rng = np.random.default_rng(seed)
    best_loss, best_state, stale = math.inf, None, 0
    for _epoch in range(config.max_epochs):
        model.train()
        order = rng.permutation(x_train.shape[0])
        for start in range(0, x_train.shape[0], config.batch_size):
            index = order[start:start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            output = model(x_train[index])
            if task_type in ("regression", "binary_classification"):
                output = output.squeeze(-1)
            loss = criterion(output, y_train_t[index])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            stop_output = model(x_stop)
            if task_type in ("regression", "binary_classification"):
                stop_output = stop_output.squeeze(-1)
            stop_loss = float(criterion(stop_output, y_stop_t).cpu())
        if not math.isfinite(stop_loss):
            raise FloatingPointError("probe early-stop loss 非有限")
        if stop_loss < best_loss - 1e-6:
            best_loss, stale = stop_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def fit_probe(X_train: np.ndarray, y_train_raw: Sequence[Any], X_validation: np.ndarray,
              task_type: str, class_labels: Optional[Sequence[Any]] = None,
              positive_label: Optional[str] = None, device: str | torch.device = "cpu",
              config: ProbeConfig = PROBE_CONFIG) -> Tuple[np.ndarray, dict[str, Any]]:
    """训练固定 probe，并只返回官方 validation 记录的预测。"""
    X_train = np.asarray(X_train, dtype=np.float32)
    X_validation = np.asarray(X_validation, dtype=np.float32)
    if X_train.ndim != 2 or X_validation.ndim != 2 or X_train.shape[1] != X_validation.shape[1]:
        raise ValueError("probe 输入必须是特征维度一致的二维矩阵")
    if len(X_train) != len(y_train_raw):
        raise ValueError("训练特征与标签数量不一致")
    if len(X_train) < 2:
        raise ValueError("至少需要 2 条训练样本")

    device = torch.device(device)
    classes = None
    if task_type == "regression":
        y_raw = np.asarray(y_train_raw, dtype=np.float64)
        if not np.isfinite(y_raw).all():
            raise ValueError("回归训练标签含 NaN/Inf")
        y_mean, y_std = float(y_raw.mean()), float(y_raw.std())
        if y_std <= 0:
            raise ValueError("回归训练标签方差为 0")
        y_train = ((y_raw - y_mean) / y_std).astype(np.float32)
        d_out = 1
    elif task_type == "binary_classification":
        if positive_label is None:
            raise ValueError("二分类 probe 需要官方 positive_label")
        y_train = np.asarray([
            1.0 if official_scorer().encode_class_label(value) == positive_label else 0.0
            for value in y_train_raw
        ], dtype=np.float32)
        if len(set(y_train.tolist())) < 2:
            raise ValueError("二分类训练集只有一个类别")
        y_mean = y_std = None
        d_out = 1
    elif task_type in ("multiclass_classification", "sequence_classification"):
        if not class_labels:
            raise ValueError("多分类/逐残基 probe 需要官方 class_labels")
        classes = [str(value) for value in class_labels]
        lookup = {value: index for index, value in enumerate(classes)}
        encoded = [official_scorer().encode_class_label(value) for value in y_train_raw]
        unknown = sorted(set(encoded) - set(lookup))
        if unknown:
            raise ValueError(f"训练标签不在官方类别表: {unknown[:5]}")
        y_train = np.asarray([lookup[value] for value in encoded], dtype=np.int64)
        if len(set(y_train.tolist())) < 2:
            raise ValueError("分类训练集少于两个类别")
        y_mean = y_std = None
        d_out = len(classes)
    else:
        raise NotImplementedError(task_type)

    hidden, adapted, params = probe_hidden_dim(X_train.shape[1], d_out, config)
    rng = np.random.default_rng(config.early_stop_seed)
    order = rng.permutation(len(X_train))
    n_stop = max(1, int(config.early_stop_fraction * len(X_train)))
    stop_index, train_index = order[:n_stop], order[n_stop:]
    if len(train_index) < 1:
        raise ValueError("early-stop 切分后没有训练样本")

    seed_predictions = []
    for seed in config.seeds:
        model = _train_probe_seed(
            X_train[train_index], y_train[train_index],
            X_train[stop_index], y_train[stop_index],
            task_type, len(classes) if classes else None, seed, device, config,
        )
        seed_predictions.append(_predict_probe(model, X_validation, task_type, device))
    prediction = np.mean(seed_predictions, axis=0) if seed_predictions else np.empty((0,))
    if task_type == "regression":
        prediction = prediction * y_std + y_mean
    meta = {
        "d_in": int(X_train.shape[1]),
        "hidden_requested": config.hidden,
        "hidden": int(hidden),
        "budget_adapted": bool(adapted),
        "params_per_head": int(params),
        "seeds": list(config.seeds),
        "max_epochs": config.max_epochs,
        "patience": config.patience,
        "lr": config.lr,
        "early_stop_fraction": config.early_stop_fraction,
        "early_stop_seed": config.early_stop_seed,
        "n_train": int(len(X_train)),
        "classes": classes,
    }
    return prediction.astype(np.float32), meta


_OFFICIAL_SCORER = None


def official_scorer(path: Path = OFFICIAL_SCORER_PATH):
    """加载官方 scorer；dataclass 模块必须先注册进 sys.modules。"""
    global _OFFICIAL_SCORER
    if _OFFICIAL_SCORER is not None and Path(path) == OFFICIAL_SCORER_PATH:
        return _OFFICIAL_SCORER
    path = Path(path)
    spec = importlib.util.spec_from_file_location("muse_official_scorer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载官方 scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if Path(path) == OFFICIAL_SCORER_PATH:
        _OFFICIAL_SCORER = module
    return module


@dataclass(frozen=True)
class BoardTask:
    """官方 Task 的只读适配器；官方 scorer 使用 kind，看板内部沿用 task_type。"""

    official: Any

    def __getattr__(self, name: str):
        return getattr(self.official, name)

    @property
    def task_type(self) -> str:
        return self.official.kind

    @property
    def prediction_column(self) -> str:
        return official_scorer().PREDICTION_FIELD[self.official.kind]


@dataclass(frozen=True)
class BoardData:
    subsets: Dict[Tuple[str, str], T.Subset]
    tasks: List[BoardTask]
    manifest: dict[str, Any]
    data_root: Path


def load_board(data_root: Path = DATA_ROOT, board: str = "A") -> BoardData:
    data_root = Path(data_root)
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    subsets: Dict[Tuple[str, str], T.Subset] = {}
    for dataset, dataset_meta in manifest["datasets"].items():
        for subset_name, subset_meta in dataset_meta["subsets"].items():
            parts = PurePosixPath(subset_meta["directory"]).parts
            if len(parts) != 4 or parts[0] != "A_datasets":
                continue
            subsets[(dataset, subset_name)] = T._parse_subset(  # noqa: SLF001
                parts[1], dataset, subset_name, subset_meta
            )
    tasks, _hashes = official_scorer().load_tasks(data_root, board)
    tasks = [BoardTask(task) for task in tasks]
    if data_root.resolve() == DATA_ROOT.resolve():
        if len(manifest["datasets"]) != 13 or len(tasks) != 90:
            raise RuntimeError(
                f"A 榜清单异常: dataset_groups={len(manifest['datasets'])}, "
                f"subsets={len(subsets)}, tasks={len(tasks)}"
            )
    return BoardData(subsets, tasks, manifest, data_root)


def _read_jsonl(path: Path, limit: Optional[int] = None) -> List[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
                if limit is not None and len(records) >= limit:
                    break
    return records


def _with_normalized_targets(record: Mapping[str, Any]) -> dict[str, Any]:
    targets = {}
    for task_name, observations in (record.get("labels") or {}).items():
        if observations:
            targets[task_name] = (
                observations[0]["task_type"],
                [observation.get("value") for observation in observations],
            )
    return {
        "id": record["id"],
        "sample_type": record.get("sample_type"),
        "inputs": record.get("inputs", {}),
        "external_inputs": record.get("external_inputs", []),
        "targets": targets,
    }


def _split_file(data_root: Path, subset: T.Subset, split: str) -> Path:
    return data_root / subset.directory / f"{split}.jsonl"


def _external_base(data_root: Path, subset: T.Subset) -> Path:
    return data_root.parent / "external_inputs" / "A_datasets" / subset.domain / subset.dataset


def _read_sdf_selected(path: Path, selected: set[str]) -> tuple[dict[str, Any], dict[str, str]]:
    from rdkit import Chem

    found = {}
    failed = {}
    seen = set()
    with Path(path).open("rb") as stream:
        block = []
        for line_number, line in enumerate(stream, start=1):
            block.append(line)
            if line.strip() != b"$$$$":
                continue
            if not block or not block[0].strip():
                raise ValueError(f"{path}:{line_number}: SDF 记录没有名称")
            key = block[0].decode("utf-8").strip()
            if key in seen:
                raise ValueError(f"SDF 结构编号重复: {key}")
            seen.add(key)
            if key in selected:
                try:
                    supplier = Chem.ForwardSDMolSupplier(
                        io.BytesIO(b"".join(block)), removeHs=False, sanitize=True
                    )
                    mol = next(supplier, None)
                    if mol is None:
                        raise ValueError(f"SDF 严格解析失败: {key}")
                    found[key] = mol
                except Exception as exc:
                    failed[key] = str(exc)[:500]
                if len(found) + len(failed) == len(selected):
                    return found, failed
            block = []
    missing = selected - set(found) - set(failed)
    if missing:
        raise KeyError(f"{path}: 找不到 SDF 结构 {sorted(missing)[:5]}")
    return found, failed


def _read_geom_pickles(path: Path, selected: Mapping[Tuple[str, int], str]) -> dict[Tuple[str, int], Any]:
    """读取 GEOM rdkit_pickle tar 中选中的成员；输入为本机赛题数据。"""
    output = {}
    by_member: dict[str, list[Tuple[int, str]]] = defaultdict(list)
    for (member, conformer_index), record_id in selected.items():
        by_member[member].append((conformer_index, record_id))
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            if member.name not in by_member:
                continue
            if not member.isfile() or member.linkname:
                raise ValueError(f"拒绝读取非普通 tar member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"无法读取 tar member: {member.name}")
            payload = pickle.loads(stream.read())
            conformers = payload["conformers"]
            for conformer_index, _record_id in by_member[member.name]:
                if conformer_index >= len(conformers):
                    raise ValueError(f"{member.name}: conformer_index 越界")
                mol = conformers[conformer_index]["rd_mol"]
                if mol.GetNumConformers() != 1:
                    raise ValueError(f"{member.name}: 需要恰好一个构象")
                output[(member.name, conformer_index)] = mol
    missing = set(selected) - set(output)
    if missing:
        raise KeyError(f"{path}: 找不到 GEOM member {sorted(missing)[:3]}")
    return output


def _cif_spacegroup(path: Path) -> int:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for key in ("_space_group_IT_number", "_symmetry_Int_Tables_number"):
        match = re.search(rf"^{re.escape(key)}\s+['\"]?(\d+)['\"]?\s*$", text, re.M)
        if match:
            value = int(match.group(1))
            if 1 <= value <= 230:
                return value
    return 0


def _read_cif_structure(path: Path) -> dict[str, Any]:
    from ase.io import read as ase_read

    atoms = ase_read(path, format="cif")
    if len(atoms) == 0:
        raise ValueError(f"空 CIF: {path}")
    if not np.all(atoms.pbc):
        raise ValueError(f"CIF 必须是三维周期结构: {path}")
    return {
        "lattice": np.asarray(atoms.cell.array, dtype=np.float64).tolist(),
        "frac_coords": np.asarray(atoms.get_scaled_positions(wrap=False), dtype=np.float64).tolist(),
        "elements": list(atoms.get_chemical_symbols()),
        "spacegroup_id": _cif_spacegroup(path),
    }


class ExternalStructureResolver:
    """真实解析 SDF / GEOM rdkit_pickle / CIF / ASE-LMDB，不读取旧表征。"""

    def __init__(self, external_base: Path, cache_root: Path):
        self.external_base = Path(external_base)
        self.cache_root = Path(cache_root)
        self.sdf_mols: dict[Tuple[str, str], Any] = {}
        self.sdf_errors: dict[Tuple[str, str], str] = {}
        self.geom_mols: dict[Tuple[str, str, int], Any] = {}
        self.cif_cache: dict[str, dict[str, Any]] = {}
        self.lmdb_paths: dict[Tuple[str, str], Path] = {}

    def prepare(self, records: Sequence[Mapping[str, Any]]) -> None:
        sdf_needed: dict[str, set[str]] = defaultdict(set)
        geom_needed: dict[str, dict[Tuple[str, int], str]] = defaultdict(dict)
        for record in records:
            for ref in record.get("external_inputs") or []:
                fmt = ref.get("format")
                selector = ref.get("selector") or {}
                source_path = ref.get("source_path")
                if fmt == "sdf":
                    sdf_needed[source_path].add(selector["record_id"])
                elif fmt == "rdkit_pickle":
                    geom_needed[source_path][(selector["archive_member"], int(selector["conformer_index"]))] = record["id"]
        for source_path, keys in sdf_needed.items():
            missing = [key for key in keys if (source_path, key) not in self.sdf_mols
                       and (source_path, key) not in self.sdf_errors]
            if missing:
                parsed, failed = _read_sdf_selected(self.external_base / source_path, set(missing))
                for key, mol in parsed.items():
                    self.sdf_mols[(source_path, key)] = mol
                for key, error in failed.items():
                    self.sdf_errors[(source_path, key)] = error
        for source_path, selected in geom_needed.items():
            missing = {
                key: record_id for key, record_id in selected.items()
                if (source_path, key[0], key[1]) not in self.geom_mols
            }
            if missing:
                parsed = _read_geom_pickles(self.external_base / source_path, missing)
                for (member, index), mol in parsed.items():
                    self.geom_mols[(source_path, member, index)] = mol

    @staticmethod
    def _one_ref(record: Mapping[str, Any], roles: set[str], formats: set[str]):
        refs = [ref for ref in record.get("external_inputs") or []
                if ref.get("role") in roles and ref.get("format") in formats]
        if len(refs) != 1:
            raise ValueError(f"{record['id']}: 结构引用数={len(refs)}，期望 1")
        return refs[0]

    def molecular_mol(self, record: Mapping[str, Any]):
        ref = self._one_ref(record, {"molecular_structure", "molecular_conformer"}, {"sdf", "rdkit_pickle"})
        selector = ref.get("selector") or {}
        if ref["format"] == "sdf":
            key = (ref["source_path"], selector["record_id"])
            if key in self.sdf_errors:
                raise ValueError(self.sdf_errors[key])
            if key not in self.sdf_mols:
                raise KeyError(f"SDF 结构未预取: {key}")
            return self.sdf_mols[key]
        key = (ref["source_path"], selector["archive_member"], int(selector["conformer_index"]))
        if key not in self.geom_mols:
            raise KeyError(f"GEOM 构象未预取: {key}")
        return self.geom_mols[key]

    def periodic_inputs(self, record: Mapping[str, Any]) -> dict[str, Any]:
        ref = self._one_ref(record, {"periodic_structure"}, {"cif", "ase_lmdb"})
        if ref["format"] == "cif":
            path = str(self.external_base / ref["source_path"])
            if path not in self.cif_cache:
                self.cif_cache[path] = _read_cif_structure(Path(path))
            return dict(self.cif_cache[path])
        selector = ref.get("selector") or {}
        key = (ref["source_path"], selector["archive_member"])
        if key not in self.lmdb_paths:
            self.lmdb_paths[key] = extract_lmdb_member(
                self.external_base / ref["source_path"], selector["archive_member"],
                self.cache_root / "ase_lmdb",
            )
        row = read_ase_lmdb_row(self.lmdb_paths[key], int(selector["lmdb_row_id"]))
        atoms = row.atoms
        if not np.all(atoms.pbc):
            raise ValueError(f"ASE-LMDB 结构必须是三维周期结构: {record['id']}")
        inputs = {
            "lattice": np.asarray(atoms.cell.array, dtype=np.float64).tolist(),
            "frac_coords": np.asarray(atoms.get_scaled_positions(wrap=False), dtype=np.float64).tolist(),
            "elements": list(atoms.get_chemical_symbols()),
            "spacegroup_id": 0,
        }
        expected = record.get("inputs", {}).get("atom_count")
        if expected is not None and int(expected) != len(atoms):
            raise ValueError(f"{record['id']}: atom_count={expected} != ASE atoms={len(atoms)}")
        return inputs


def _stable_unit_scalar(value: Any) -> float:
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return 2.0 * int.from_bytes(digest, "little") / 2**64 - 1.0


def measurement_context_vector(context: Optional[Mapping[str, Any]]) -> np.ndarray:
    """与预训练 aux_sequence 相同的 DBAASP context 哈希条件，绝不丢弃 context。"""
    output = np.zeros(4, dtype=np.float32)
    if not context:
        return output
    keys = ("target", "target_cell", "medium", "measurement_group", "measurement_value")
    for index, key in enumerate(keys[:4]):
        if key in context:
            output[index] = _stable_unit_scalar(context[key])
    return output


_ALLOWED_RESIDUES = set("ACDEFGHIKLMNPQRSTVWYBZUXO")
_ALLOWED_LOWER = set("acdefghiklmnpqrstvwy")


def _peptide_sequence_stats(sequence: str) -> dict[str, Any]:
    chars = [char for char in sequence.strip() if not char.isspace()]
    lower = sum(char in _ALLOWED_LOWER for char in chars)
    ambiguous = sum(char in "XxBUZO" for char in chars)
    unsupported = sorted({
        char for char in chars
        if char not in _ALLOWED_RESIDUES and char not in _ALLOWED_LOWER and char != "x"
    })
    if unsupported:
        raise ValueError(f"肽序列含当前残基/原子路由不支持的修饰记号: {unsupported}")
    return {
        "lowercase_d_residues": int(lower),
        "ambiguous_residues": int(ambiguous),
        "case_handling": "lowercase retained as D-residue chirality feature; no silent upper fallback",
    }


def _numeric_tail(fields: Sequence[Any]) -> List[float]:
    values = []
    for value in fields:
        try:
            number = float(value)
        except (TypeError, ValueError):
            break
        if not math.isfinite(number):
            break
        values.append(number)
    return values


def _normalize_polymer_electrolyte(record: Mapping[str, Any]) -> dict[str, Any]:
    out = _with_normalized_targets(record)
    inputs = dict(out["inputs"])
    if "electrolyte_string" in inputs:
        parts = str(inputs["electrolyte_string"]).split("|")
        components = [part.split("$", 1)[0] for part in (parts[:-1] if len(parts) > 1 else parts)]
        if not all(components):
            raise ValueError("PE_I electrolyte_string 含空组分")
        ratios = _numeric_tail(parts[-1].split("$")) if len(parts) > 1 else []
        joined = ".".join(components)
        dummy_count = joined.count("[^]")
        if dummy_count:
            joined = joined.replace("[^]", "*")
        if ratios:
            logs = np.log1p(np.asarray(ratios, dtype=np.float64)) / math.log1p(100.0)
            scalar = [len(components) / 8.0, float(logs.mean()), float(logs.min()), float(logs.max())]
        else:
            scalar = [len(components) / 8.0, 0.0, 0.0, 0.0]
        normalization = "PE_I all components joined as one PSMILES sequence; ratio summary in scalar_cond"
        if not ratios:
            normalization += "; no pipe-delimited numeric ratio"
        if dummy_count:
            normalization += f"; normalized {dummy_count} [^] dummy atoms to *"
    elif "context" in inputs:
        context = inputs["context"]
        components = [context.get("SMILES descriptor 1"), context.get("SMILES descriptor 2"),
                      context.get("Amion SMILES")]
        if not all(components):
            raise ValueError("PE_II context 缺少两个聚合物/阴离子 SMILES")
        joined = ".".join(str(value) for value in components)
        mw = max(float(context.get("approxMW(kDa)", 0.0)), 0.0)
        tg = float(context.get("approxTg", 0.0))
        scalar = [
            float(context.get("log Li:functional group", 0.0)),
            float(context.get("Comonomer percentage", 0.0)) / 100.0,
            math.log1p(mw) / math.log1p(1_000_000.0),
            (tg + 150.0) / 300.0,
        ]
        normalization = (
            "PE_II three chemical components joined as PSMILES; four numeric context fields in scalar_cond; "
            "chain_architecture has no scalar_cond slot and is recorded as dropped"
        )
    else:
        raise ValueError("polymer_electrolyte 缺少 electrolyte_string/context")
    out.update({
        "sample_type": "polymer",
        "inputs": {"psmiles": joined, "sequence_frontend": True},
    })
    out["_scalar_cond"] = np.asarray(scalar, dtype=np.float32)
    out["_normalization"] = normalization
    return out


_INPUT_REPAIRS: dict[str, dict[str, Any]] = {}
_INPUT_REPAIR_ROOT: Optional[Path] = None


def _input_repair_fingerprint(record: Mapping[str, Any]) -> str:
    if not isinstance(record.get("id"), str) or not isinstance(record.get("inputs"), Mapping):
        raise ValueError("输入修复需要 record id 和 inputs")
    return _config_sha({
        "id": record["id"], "sample_type": record.get("sample_type"),
        "inputs": record["inputs"], "external_inputs": record.get("external_inputs", []),
    })


def _official_graph_input_fingerprint(record: Mapping[str, Any]) -> str:
    sample_type = record.get("sample_type")
    keys = set(record.get("inputs") or {})
    allowed = {"smiles", "parent_molecule_id", "charge"} if sample_type == "conformer" else {"smiles"}
    if sample_type not in ("molecule", "conformer") or "smiles" not in keys \
            or not keys <= allowed:
        raise ValueError("官方图修复仅支持登记的 molecule/conformer 输入")
    return _input_repair_fingerprint(record)


def _validate_graph_repair(entry, payload, artifact_root: Path, source_hashes: dict) -> None:
    source = entry.get("source")
    if source not in ("qm9", "geom_drugs"):
        raise ValueError("官方图修复来源必须是 qm9 或 geom_drugs")
    source_path = entry.get("source_path")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("官方图修复 source_path 非法")
    source_file = _AUX_EXTERNAL_BASES[source] / source_path
    if not source_file.resolve().is_relative_to(_AUX_EXTERNAL_BASES[source].resolve()):
        raise ValueError("官方图修复 source_path 超出官方目录")
    expected = entry.get("source_sha256")
    if source_file not in source_hashes:
        source_hashes[source_file] = sha256_file(source_file)
    if source_hashes[source_file] != expected:
        raise RuntimeError("官方图修复源文件 SHA256 不一致")
    if payload.get("schema") == "muse_encoder.official_graph_repairs.v1" and (
            source != payload.get("source") or source_path != payload.get("source_path")
            or expected != payload.get("source_sha256")):
        raise ValueError("旧版官方图修复记录来源不一致")
    block = entry.get("mol_block")
    if not isinstance(block, str) or "M  END" not in block:
        raise ValueError("官方图修复结构块非法")
    if hashlib.sha256(block.encode("utf-8")).hexdigest() != entry.get("structure_block_sha256"):
        raise RuntimeError("官方图修复结构块 SHA256 不一致")
    from rdkit import Chem
    mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
    if mol is None or mol.GetNumAtoms() < 1 or mol.GetNumConformers() != 1:
        raise ValueError("官方图修复结构块无法读取")
    artifact = entry.get("artifact")
    if not isinstance(artifact, str) or Path(artifact).name != artifact:
        raise ValueError("官方图修复数组文件名非法")
    if sha256_file(artifact_root / artifact) != entry.get("artifact_sha256"):
        raise RuntimeError("官方图修复数组 SHA256 不一致")


def _validate_sequence_repair(entry, source_hashes: dict) -> None:
    if entry.get("representation") not in (
            "smiles_sequence", "raw_peptide_sequence", "missing_polymer_sequence"):
        raise ValueError("未知输入修复表示")
    source_jsonl = entry.get("source_jsonl")
    if not isinstance(source_jsonl, str):
        raise ValueError("序列修复缺少原始 JSONL 来源")
    source_file = Path(source_jsonl).resolve()
    if not source_file.is_relative_to(DATA_ROOT.resolve()):
        raise ValueError("序列修复来源超出官方数据目录")
    if source_file not in source_hashes:
        source_hashes[source_file] = sha256_file(source_file)
    if source_hashes[source_file] != entry.get("source_jsonl_sha256"):
        raise RuntimeError("序列修复原始 JSONL SHA256 不一致")
    if entry.get("artifact") is not None or entry.get("mol_block") is not None:
        raise ValueError("序列修复不得携带官方图数组")


def configure_official_graph_repairs(path: Optional[Path]) -> None:
    global _INPUT_REPAIRS, _INPUT_REPAIR_ROOT
    if path is None:
        _INPUT_REPAIRS = {}
        _INPUT_REPAIR_ROOT = None
        return
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    schemas = {"muse_encoder.official_graph_repairs.v1", "muse_encoder.input_repairs.v1"}
    if payload.get("schema") not in schemas:
        raise ValueError("未知输入修复清单")
    if payload.get("schema") == "muse_encoder.input_repairs.v1" \
            and payload.get("contains_labels") is not False:
        raise ValueError("输入修复清单必须明确不含标签")
    artifact_dir = payload.get("artifact_dir")
    if not isinstance(artifact_dir, str) or Path(artifact_dir).name != artifact_dir:
        raise ValueError("输入修复 artifact_dir 非法")
    artifact_root = path.parent / artifact_dir
    repairs = {}
    source_hashes = {}
    for entry in payload.get("repairs", []):
        record_id = entry.get("record_id")
        if not record_id or record_id in repairs:
            raise ValueError("输入修复 record_id 缺失或重复")
        if not isinstance(entry.get("input_fingerprint"), str):
            raise ValueError("输入修复缺少原始输入指纹")
        if entry.get("representation", "official_graph") == "official_graph":
            _validate_graph_repair(entry, payload, artifact_root, source_hashes)
        else:
            _validate_sequence_repair(entry, source_hashes)
        repairs[record_id] = entry
    _INPUT_REPAIRS = repairs
    _INPUT_REPAIR_ROOT = artifact_root


def _registered_input_repair(record: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    entry = _INPUT_REPAIRS.get(record.get("id"))
    if entry is None:
        return None
    if _input_repair_fingerprint(record) != entry.get("input_fingerprint"):
        raise RuntimeError("输入修复的原始输入指纹不一致")
    if entry.get("representation", "official_graph") == "official_graph":
        refs = [ref for ref in record.get("external_inputs", [])
                if ref.get("role") in ("molecular_structure", "molecular_conformer")
                and ref.get("format") == entry.get("source_format", "sdf")]
        if len(refs) != 1 or refs[0].get("source_path") != entry.get("source_path") \
                or refs[0].get("selector") != entry.get("selector"):
            raise RuntimeError("官方图修复的结构引用不一致")
    return entry


def _registered_official_graph(record: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    entry = _registered_input_repair(record)
    if entry is not None and entry.get("representation", "official_graph") != "official_graph":
        return None
    return entry


def _load_official_graph_record(repair: Mapping[str, Any]):
    from muse_encoder.featurize import FeaturizedRecord

    if _INPUT_REPAIR_ROOT is None:
        raise RuntimeError("官方图修复数组目录未配置")
    path = _INPUT_REPAIR_ROOT / repair["artifact"]
    if sha256_file(path) != repair.get("artifact_sha256"):
        raise RuntimeError("官方图修复数组 SHA256 不一致")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files if key != "meta_json"}
        metadata = json.loads(str(data["meta_json"]))
    metadata["official_graph_repair"] = dict(repair)
    return FeaturizedRecord(
        domain=int(arrays.pop("domain").item()),
        modality=int(arrays.pop("modality").item()),
        use_atom_features=bool(arrays.pop("use_atom_features").item()),
        n_real_tokens=int(arrays.pop("n_real_tokens").item()),
        meta=metadata,
        **arrays,
    )


def normalize_record(record: Mapping[str, Any], resolver: Optional[ExternalStructureResolver] = None) -> dict[str, Any]:
    """把官方 sample_type/外部结构规范化为当前 inference 可路由的记录。"""
    sample_type = record.get("sample_type")
    registered = _registered_input_repair(record)
    if sample_type == "polymer_electrolyte":
        out = _normalize_polymer_electrolyte(record)
    else:
        out = _with_normalized_targets(record)
        out["inputs"] = dict(out["inputs"])
        if sample_type == "polymer" and "csmiles" in out["inputs"] and "psmiles" not in out["inputs"]:
            out["inputs"]["psmiles"] = out["inputs"].pop("csmiles")
            out["_normalization"] = "csmiles -> psmiles"
    if registered is not None and registered.get("representation", "official_graph") != "official_graph":
        out["_input_repair"] = dict(registered)
        out["_normalization"] = f"registered {registered['representation']}; no fabricated 3D graph"

    if sample_type in ("molecule", "conformer", "protein_ligand") and record.get("external_inputs"):
        repair = _registered_official_graph(record)
        if repair is not None:
            from rdkit import Chem
            mol = Chem.MolFromMolBlock(repair["mol_block"], sanitize=False, removeHs=False)
            if mol is None:
                raise ValueError(f"{record['id']}: 注册官方图无法读取")
            out["inputs"]["sdf_mol"] = mol
            out["_official_graph_repair"] = {
                key: value for key, value in repair.items() if key != "mol_block"
            }
            out["_normalization"] = "registered official graph repair"
        else:
            if resolver is None:
                raise ValueError(f"{record['id']}: 分子外部结构需要 resolver")
            out["inputs"]["sdf_mol"] = resolver.molecular_mol(record)
            out["_normalization"] = f"external {resolver.molecular_mol.__name__}"

    if sample_type in ("periodic_structure", "periodic_frame"):
        if record.get("external_inputs"):
            if resolver is None:
                raise ValueError(f"{record['id']}: 周期结构需要 resolver")
            out["inputs"].update(resolver.periodic_inputs(record))
            out["sample_type"] = "periodic_frame"
            out["_normalization"] = "external periodic structure -> lattice/frac_coords/elements"
        elif not {"lattice", "frac_coords", "elements"} <= set(out["inputs"]):
            raise ValueError(f"{record['id']}: periodic_frame 缺少结构字段或 external_inputs")

    if sample_type == "peptide_measurement":
        sequence = out["inputs"].get("sequence")
        if not isinstance(sequence, str) or not sequence.strip():
            raise ValueError(f"{record['id']}: peptide_measurement 缺少 sequence")
        if registered is not None and registered.get("representation") == "raw_peptide_sequence":
            stats = {
                "uninterpreted_modification_notation": True,
                "policy": "preserve raw characters; unsupported characters become unknown residue tokens",
            }
        else:
            stats = _peptide_sequence_stats(sequence)
        if "x" in sequence:
            out["inputs"]["sequence"] = sequence.replace("x", "X")
            stats["unknown_x_handling"] = "lowercase x -> unknown X without D chirality"
        out["_peptide_stats"] = stats
        out["_context"] = out["inputs"].get("context")
    if sample_type == "protein_construct":
        sequence = out["inputs"].get("sequence")
        if not isinstance(sequence, str) or not sequence.strip():
            raise ValueError(f"{record['id']}: protein_construct 缺少 sequence")
        out["sample_type"] = "protein"
        out["_normalization"] = "protein_construct -> protein sequence inference"
    return out


def _prepare_registered_sequence(featurizer: RecordFeaturizer, raw: Mapping[str, Any], repair):
    inputs = raw["inputs"]
    sample_type = raw.get("sample_type")
    representation = repair["representation"]
    if representation == "smiles_sequence":
        if sample_type in ("molecule", "conformer"):
            record = featurizer.featurize_molecule(
                inputs["smiles"], sequence=True, strict=False)
        elif sample_type == "protein_ligand":
            record = featurizer.featurize_protein_ligand(
                inputs["protein_sequence"], inputs["ligand_smiles"],
                strict=False, ligand_sequence=True)
        elif sample_type == "polymer":
            record = featurizer.featurize_polymer(
                psmiles=inputs["psmiles"], sequence=True, strict=False)
        else:
            raise ValueError(f"{sample_type} 不支持注册的 SMILES 序列表示")
    elif representation == "raw_peptide_sequence" and sample_type == "peptide_measurement":
        record = _featurize_record(featurizer, raw)
    elif representation == "missing_polymer_sequence" and sample_type == "polymer":
        value = inputs.get("psmiles")
        if value is not None and (not isinstance(value, str)
                                  or value.strip().lower() not in ("", "nan", "none", "null")):
            raise ValueError("缺失聚合物表示只能用于明确缺失的 PSMILES")
        record = featurizer.featurize_polymer(psmiles="", sequence=True, strict=False)
        record.meta["original_psmiles"] = value
    else:
        raise ValueError("输入修复表示与 sample_type 不匹配")
    record.meta["input_repair"] = dict(repair)
    record.meta["degraded"] = True
    return record


def _prepare_featurized(featurizer: RecordFeaturizer, raw: Mapping[str, Any]):
    graph_repair = raw.get("_official_graph_repair")
    input_repair = raw.get("_input_repair")
    if graph_repair is not None:
        record = _load_official_graph_record(graph_repair)
    elif input_repair is not None:
        record = _prepare_registered_sequence(featurizer, raw, input_repair)
    else:
        try:
            record = _featurize_record(featurizer, raw)
        except ValueError:
            inputs = raw.get("inputs") or {}
            mol = inputs.get("sdf_mol")
            smiles = inputs.get("smiles")
            if raw.get("sample_type") not in ("molecule", "conformer", "protein_ligand") \
                    or mol is None or not smiles:
                raise
            from .molecule3d import _official_graph_variant, require_rdkit
            variant = _official_graph_variant(smiles, mol)
            if variant is None:
                raise
            graph, _coords, _mapping = variant
            chem = require_rdkit()
            raw = dict(raw)
            raw["inputs"] = dict(
                inputs, smiles=chem.MolToSmiles(graph, isomericSmiles=True), sdf_mol=graph)
            raw["_normalization"] = "official conformer graph; formula/connectivity verified"
            record = _featurize_record(featurizer, raw)
    scalar = None
    if raw.get("_scalar_cond") is not None:
        scalar = np.asarray(raw["_scalar_cond"], dtype=np.float32)
    elif raw.get("_context") is not None:
        scalar = measurement_context_vector(raw.get("_context"))
    elif any(key in raw.get("inputs", {}) for key in ("charge", "multiplicity")):
        scalar = np.asarray(record.scalar_cond if record.scalar_cond is not None else np.zeros(4), dtype=np.float32).copy()
        if "charge" in raw["inputs"]:
            scalar[0] = float(raw["inputs"]["charge"]) / 10.0
        if "multiplicity" in raw["inputs"]:
            scalar[1] = float(raw["inputs"]["multiplicity"]) / 10.0
    if scalar is not None:
        record.scalar_cond = scalar
    for key in ("_normalization", "_peptide_stats"):
        if key in raw:
            record.meta[key.lstrip("_")] = raw[key]
    repair = raw.get("_official_graph_repair")
    if repair is not None:
        actual = featurizer.official_graph_fingerprint(record)
        if actual != repair.get("model_input_sha256"):
            raise RuntimeError("注册官方图修复后的模型输入 SHA256 不一致")
    return record


@torch.no_grad()
def _encode_prepared_chunk(encoder: MuseEncoder, cfg: MuseConfig, prepared, device: str):
    batch = _torch_batch(prepared, cfg, device)
    return _encode_compatible(encoder, batch)


@torch.no_grad()
def encode_records(encoder: MuseEncoder, featurizer: RecordFeaturizer, cfg: MuseConfig,
                   records: Sequence[Mapping[str, Any]], batch_size: int = 128,
                   token_level: bool = False, device: str = "cpu") -> EncodedSplit:
    """带 context/电荷条件的 inference 路径；长序列与逐 token 逻辑复用 inference。"""
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正数")
    encoder.eval()
    ok_ids: List[str] = []
    embeddings: List[np.ndarray] = []
    token_chunks: List[np.ndarray] = []
    failed: List[dict[str, str]] = []

    def encode_chunk(chunk: Sequence[Mapping[str, Any]]) -> None:
        if not chunk:
            return
        try:
            prepared = [_prepare_featurized(featurizer, raw) for raw in chunk]
            regular = []
            for raw, record in zip(chunk, prepared):
                if _can_window(record, cfg):
                    embedding, tokens = _encode_long_record(
                        encoder, featurizer, cfg, record, batch_size, token_level, device
                    )
                    expected = _expected_tokens(raw, record)
                    if token_level and len(tokens) != expected:
                        raise RuntimeError(f"{raw['id']}: 长序列 token 数不符")
                    ok_ids.append(raw["id"])
                    embeddings.append(np.asarray(embedding, dtype=np.float32))
                    if token_level:
                        token_chunks.append(tokens)
                else:
                    regular.append((raw, record))
            if regular:
                from muse_encoder.inference import _token_bounded_ranges
                ranges = _token_bounded_ranges([len(record.token_ids) for _, record in regular], cfg)
                for start, stop in ranges:
                    bounded = regular[start:stop]
                    output = _encode_prepared_chunk(encoder, cfg, [r for _, r in bounded], device)
                    graph = output["embedding"].float().cpu().numpy()
                    tokens = output["tokens"].float().cpu().numpy()
                    for index, (raw, record) in enumerate(bounded):
                        token_out = _token_output(record, tokens[index]) if token_level else None
                        if token_level:
                            expected = _expected_tokens(raw, record)
                            if len(token_out) != expected:
                                raise RuntimeError(f"{raw['id']}: token 数 {len(token_out)} != {expected}")
                        ok_ids.append(raw["id"])
                        embeddings.append(np.asarray(graph[index], dtype=np.float32))
                        if token_level:
                            token_chunks.append(token_out)
        except Exception as exc:
            if len(chunk) == 1:
                failed.append({"id": str(chunk[0]["id"]), "error": str(exc)[:500]})
                return
            middle = len(chunk) // 2
            encode_chunk(chunk[:middle])
            encode_chunk(chunk[middle:])

    for start in range(0, len(records), batch_size):
        encode_chunk(records[start:start + batch_size])
    embedding_array = (np.stack(embeddings).astype(np.float32)
                       if embeddings else np.empty((0, cfg.embed_dim), dtype=np.float32))
    token_array = None
    offsets = None
    if token_level:
        token_array = (np.concatenate(token_chunks, axis=0).astype(np.float32)
                       if token_chunks else np.empty((0, cfg.embed_dim), dtype=np.float32))
        offsets = np.zeros(len(token_chunks) + 1, dtype=np.int64)
        for index, chunk in enumerate(token_chunks):
            offsets[index + 1] = offsets[index] + len(chunk)
    if not np.isfinite(embedding_array).all():
        raise ValueError("表征含 NaN/Inf")
    if token_array is not None and not np.isfinite(token_array).all():
        raise ValueError("逐 token 表征含 NaN/Inf")
    return EncodedSplit(ok_ids, embedding_array, token_array, offsets, failed)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def _subset_cache(root: Path, subset: T.Subset) -> Path:
    return root / _safe_name(subset.dataset) / _safe_name(subset.subset)


def _config_sha(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def load_checkpoint(path: Path, device: str = "cpu") -> Tuple[MuseEncoder, MuseConfig, dict[str, Any]]:
    """加载自有 checkpoint，显式拒绝 models/ 下公开权重。"""
    path = Path(path).resolve()
    if path == PUBLIC_MODEL_ROOT or PUBLIC_MODEL_ROOT in path.parents:
        raise ValueError(f"禁止从公开模型目录加载 checkpoint: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload or "config" not in payload:
        raise ValueError("checkpoint 必须包含 config/model（并可包含 state）")
    raw_config = payload["config"]
    if not isinstance(raw_config, Mapping):
        raise ValueError("checkpoint config 必须是 mapping")
    if "model" in raw_config and isinstance(raw_config.get("model"), Mapping) and "d_model" not in raw_config:
        raw_config = raw_config["model"]
    valid = {field.name for field in fields(MuseConfig)}
    cfg = MuseConfig(**{key: value for key, value in raw_config.items() if key in valid})
    model = MuseEncoder(cfg)
    model.load_state_dict(payload["model"], strict=True)
    model.to(torch.device(device))
    model.eval()
    info = {
        "path": str(path),
        "sha256": digest,
        "state": payload.get("state", {}),
        "config_sha256": _config_sha(cfg.to_dict()),
    }
    return model, cfg, info


def encode_split_cached(encoder: MuseEncoder, featurizer: RecordFeaturizer, cfg: MuseConfig,
                        subset: T.Subset, data_root: Path, split: str, run_root: Path,
                        checkpoint_sha: str, limit: Optional[int], token_level: bool,
                        batch_size: int, device: str,
                        probe_config: ProbeConfig = PROBE_CONFIG) -> Tuple[EncodedSplit, List[dict[str, Any]], bool]:
    cache = _subset_cache(run_root / "embeddings", subset)
    meta_path = cache / f"{split}.encoding.json"
    wanted_ids = [record["id"] for record in _read_jsonl(_split_file(data_root, subset, split), limit)]
    metadata = {
        "checkpoint_sha256": checkpoint_sha,
        "model_config_sha256": _config_sha(cfg.to_dict()),
        "split": split,
        "limit_per_split": limit,
        "token_level": token_level,
        "ids": wanted_ids,
        "probe_config_sha256": _config_sha(probe_config.to_json()),
    }
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in old.items() if key not in ("failed", "elapsed_seconds")}
        if comparable == metadata:
            ids = json.loads((cache / f"{split}.ids.json").read_text(encoding="utf-8"))
            embeddings = np.load(cache / f"{split}.embeddings.npy")
            tokens = offsets = None
            if token_level:
                tokens = np.load(cache / f"{split}.tokens.npy")
                offsets = np.load(cache / f"{split}.token_offsets.npy")
            raw = _read_jsonl(_split_file(data_root, subset, split), limit)
            return EncodedSplit(ids, embeddings, tokens, offsets, old.get("failed", [])), raw, True

    started = time.monotonic()
    raw = _read_jsonl(_split_file(data_root, subset, split), limit)
    normalized_errors = []
    normalized = []
    resolver = ExternalStructureResolver(_external_base(data_root, subset), run_root / "external_cache")
    try:
        resolver.prepare(raw)
    except Exception as exc:
        # 外部结构整体不可读时，本 split 全部走显式 fallback，而不是混入旧表征。
        normalized_errors = [{"id": record["id"], "error": str(exc)[:500]} for record in raw]
    else:
        for record in raw:
            try:
                normalized.append(normalize_record(record, resolver))
            except Exception as exc:
                normalized_errors.append({"id": record["id"], "error": str(exc)[:500]})
    encoded = encode_records(encoder, featurizer, cfg, normalized, batch_size, token_level, device)
    encoded.failed.extend(normalized_errors)
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / f"{split}.embeddings.npy", encoded.embeddings)
    if token_level:
        np.save(cache / f"{split}.tokens.npy", encoded.tokens)
        np.save(cache / f"{split}.token_offsets.npy", encoded.offsets)
    (cache / f"{split}.ids.json").write_text(json.dumps(encoded.ids, ensure_ascii=False), encoding="utf-8")
    metadata.update({"failed": encoded.failed, "elapsed_seconds": round(time.monotonic() - started, 3)})
    _write_json(meta_path, metadata)
    return encoded, raw, False


def _records_by_id(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        record["id"]: (record if "targets" in record else _with_normalized_targets(record))
        for record in records
    }


def _task_values(records: Sequence[Mapping[str, Any]], encoded: EncodedSplit,
                 task, sequence: bool = False):
    by_id = _records_by_id(records)
    encoded_rows = {record_id: index for index, record_id in enumerate(encoded.ids)}
    ids, features, labels = [], [], []
    token_counts = []
    for record_id, row in encoded_rows.items():
        record = by_id.get(record_id)
        if record is None or task.task not in record.get("targets", {}):
            continue
        _task_type, values = record["targets"][task.task]
        if not values:
            continue
        value = values[0]
        if task.task_type == "regression":
            observations = [float(item) for item in values]
            label = sum(observations) / len(observations)
        elif sequence:
            label = list(value)
            count = int(encoded.offsets[row + 1] - encoded.offsets[row])
            if len(label) != count:
                raise ValueError(f"{record_id}: 残基标签 {len(label)} != token {count}")
            token_counts.append(count)
        else:
            label = value
        ids.append(record_id)
        labels.append(label)
        if not sequence:
            features.append(encoded.embeddings[row])
    if sequence:
        selected = [encoded.tokens[int(encoded.offsets[encoded_rows[rid]]):int(encoded.offsets[encoded_rows[rid] + 1])]
                    for rid in ids]
        X = np.concatenate(selected, axis=0) if selected else np.empty((0, encoded.embeddings.shape[1]), dtype=np.float32)
        y = [item for label in labels for item in label]
        return ids, X, y, token_counts
    X = np.stack(features).astype(np.float32) if features else np.empty((0, encoded.embeddings.shape[1]), dtype=np.float32)
    return ids, X, labels, None


def _train_positive_rate(records: Sequence[Mapping[str, Any]], task) -> float:
    positives = total = 0
    for raw in records:
        record = raw if "targets" in raw else _with_normalized_targets(raw)
        target = record.get("targets", {}).get(task.task)
        if not target:
            continue
        value = target[1][0]
        positives += int(official_scorer().encode_class_label(value) == task.positive_label)
        total += 1
    return positives / total if total else 0.5


def _sequence_length(record: Mapping[str, Any]) -> int:
    inputs = record.get("inputs", {})
    sequence = inputs.get("sequence") or inputs.get("protein_sequence") or inputs.get("sequence_1")
    if not sequence:
        return 1
    return sum(not char.isspace() for char in str(sequence).strip())


def _fallback_value(task, records: Sequence[Mapping[str, Any]], record: Optional[Mapping[str, Any]] = None):
    if task.task_type == "regression":
        return float(task.train_median)
    if task.task_type == "binary_classification":
        return float(_train_positive_rate(records, task))
    width = len(task.class_labels)
    value = [1.0 / width] * width
    if task.task_type == "sequence_classification":
        return [value] * _sequence_length(record or {})
    return value


def write_task_predictions(path: Path, task, validation_records: Sequence[Mapping[str, Any]],
                           train_records: Sequence[Mapping[str, Any]],
                           predictions: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = 0
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", task.prediction_column])
        for record in validation_records:
            record_id = record["id"]
            value = predictions.get(record_id)
            if value is None:
                value = _fallback_value(task, train_records, record)
                missing += 1
            if task.task_type in ("regression", "binary_classification"):
                writer.writerow([record_id, float(value)])
            else:
                writer.writerow([record_id, json.dumps(value, separators=(",", ":"))])
    return {"rows": len(validation_records), "fallback_predictions": missing}


def run_task(task, subset: T.Subset, train_raw, validation_raw,
             train_encoded: EncodedSplit, validation_encoded: EncodedSplit,
             prediction_root: Path, cache_root: Path, run_key: str,
             device: str, probe_config: ProbeConfig = PROBE_CONFIG) -> dict[str, Any]:
    safe_task = _safe_name(task.task_id.replace("/", "__"))
    cache_path = cache_root / f"{safe_task}.json"
    prediction_path = prediction_root / task.prediction_file
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("run_key") == run_key and (cached.get("status") != "ok" or prediction_path.exists()):
            return cached

    sequence = task.task_type == "sequence_classification"
    try:
        train_ids, X_train, y_train, _counts = _task_values(train_raw, train_encoded, task, sequence)
        validation_ids, X_validation, _y_validation, token_counts = _task_values(
            validation_raw, validation_encoded, task, sequence
        )
        predictions_array, probe_meta = fit_probe(
            X_train, y_train, X_validation, task.task_type,
            class_labels=task.class_labels if task.task_type != "regression" else None,
            positive_label=task.positive_label if task.task_type == "binary_classification" else None,
            device=device,
            config=probe_config,
        )
        if sequence:
            predictions = {}
            offset = 0
            for record_id, count in zip(validation_ids, token_counts):
                predictions[record_id] = predictions_array[offset:offset + count].tolist()
                offset += count
        else:
            predictions = {record_id: value for record_id, value in zip(validation_ids, predictions_array.tolist())}
        write_info = write_task_predictions(
            prediction_path, task, validation_raw, train_raw, predictions
        )
        status = "ok"
        detail = ""
    except Exception as exc:
        probe_meta = {}
        write_info = {"rows": 0, "fallback_predictions": 0}
        status = "missing_head"
        detail = str(exc)[:1000]
    result = {
        "run_key": run_key,
        "task_id": task.task_id,
        "dataset": task.dataset,
        "subset": task.subset,
        "task": task.task,
        "domain": task.domain,
        "task_type": task.task_type,
        "status": status,
        "detail": detail,
        "prediction_file": task.prediction_file,
        "prediction_path": str(prediction_path),
        "n_train_encoded": len(train_ids) if "train_ids" in locals() else 0,
        "n_validation_encoded": len(validation_ids) if "validation_ids" in locals() else 0,
        "n_validation_total": len(validation_raw),
        "encoding_missing_train": len(train_raw) - (len(train_ids) if "train_ids" in locals() else 0),
        "encoding_missing_validation": len(validation_raw) - (len(validation_ids) if "validation_ids" in locals() else 0),
        "probe": probe_meta,
        "predictions": write_info,
    }
    _write_json(cache_path, result)
    return result


def _task_has_smoke_truth(records: Sequence[Mapping[str, Any]], task) -> bool:
    observations = []
    scorer = official_scorer()
    for record in records:
        for observation in (record.get("labels") or {}).get(task.task, []):
            observations.append(observation.get("value"))
    if not observations:
        return False
    if task.task_type == "regression":
        return True
    values = set()
    for value in observations:
        if task.task_type == "sequence_classification":
            values.update(scorer.encode_class_label(item) for item in value)
        else:
            values.add(scorer.encode_class_label(value))
    return len(values) >= 2


def build_smoke_data_root(data_root: Path, manifest: Mapping[str, Any], selected: Sequence[T.Subset],
                          raw_by_subset: Mapping[Tuple[str, str], Sequence[Mapping[str, Any]]],
                          tasks: Sequence[Any], out_root: Path) -> Path:
    """生成仅 validation 的临时官方数据根；只写入 run 目录，不改原始数据。"""
    smoke_root = Path(out_root)
    selected_keys = {(subset.dataset, subset.subset) for subset in selected}
    task_by_key: dict[Tuple[str, str], list[Any]] = defaultdict(list)
    for task in tasks:
        task_by_key[(task.dataset, task.subset)].append(task)
    new_manifest = copy.deepcopy(dict(manifest))
    new_manifest["datasets"] = {}
    for dataset, subset_name in selected_keys:
        subset = next(s for s in selected if (s.dataset, s.subset) == (dataset, subset_name))
        records = list(raw_by_subset[(dataset, subset_name)])
        source_meta = manifest["datasets"][dataset]["subsets"][subset_name]
        meta = copy.deepcopy(source_meta)
        relative = Path(subset.directory) / "validation.jsonl"
        target = smoke_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        meta["files"] = {"validation": {"file": relative.as_posix()}}
        labels = {}
        for task in task_by_key[(dataset, subset_name)]:
            if not _task_has_smoke_truth(records, task):
                continue
            label_meta = copy.deepcopy(source_meta["labels"][task.task])
            count = sum(len((record.get("labels") or {}).get(task.task, [])) for record in records)
            label_meta["split"] = {"validation": count}
            labels[task.task] = label_meta
        meta["labels"] = labels
        new_manifest["datasets"].setdefault(dataset, {"subsets": {}})["subsets"][subset_name] = meta
    _write_json(smoke_root / "manifest.json", new_manifest)
    return smoke_root


def _merge_official_scores(all_tasks: Sequence[Any], official_payload: Mapping[str, Any],
                           task_runs: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    rows = {}
    official_rows = official_payload.get("tasks", {})
    for task in all_tasks:
        run = task_runs.get(task.task_id, {})
        official = official_rows.get(task.task_id)
        row = {
            "task_id": task.task_id,
            "dataset": task.dataset,
            "subset": task.subset,
            "task": task.task,
            "domain": task.domain,
            "task_type": task.task_type,
            "status": run.get("status", "not_run"),
            "detail": run.get("detail", ""),
            "probe": run.get("probe", {}),
            "n_train": run.get("n_train_encoded"),
            "n_validation_encoded": run.get("n_validation_encoded"),
            "n_validation_total": run.get("n_validation_total"),
            "missing_predictions": run.get("predictions", {}).get("fallback_predictions"),
        }
        if official is not None:
            row.update({
                "official_status": official.get("status"),
                "raw_metric": official.get("value"),
                "baseline": official.get("baseline"),
                "normalized_score": official.get("score"),
                "n": official.get("n"),
                "official_extra": official.get("extra", {}),
            })
        else:
            row.update({
                "official_status": "not_scored",
                "raw_metric": None,
                "baseline": None,
                "normalized_score": None,
                "n": None,
                "official_extra": {},
            })
        rows[task.task_id] = row
    return rows


def _write_tables(out_dir: Path, rows: Mapping[str, Mapping[str, Any]],
                  summary: Mapping[str, Any]) -> None:
    table_path = out_dir / "validation_tasks.csv"
    fields = ["task_id", "domain", "dataset", "subset", "task_type", "status", "official_status",
              "raw_metric", "baseline", "normalized_score", "n", "n_train", "n_validation_total",
              "missing_predictions"]
    with table_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for task_id in sorted(rows):
            writer.writerow({key: rows[task_id].get(key) for key in fields})
    lines = ["# MUSE-A fixed validation probe dashboard", "",
             f"- final_score: `{summary.get('final_score')}`",
             f"- complete: `{summary.get('complete')}`",
             f"- scored_tasks: `{summary.get('scored_tasks')}` / `{summary.get('num_tasks')}`",
             "", "## Domain summary", "", "| domain | datasets | score |", "|---|---:|---:|"]
    for domain, info in (summary.get("domains") or {}).items():
        lines.append(f"| {domain} | {info.get('num_datasets')} | {info.get('score'):.6f} |")
    lines.extend(["", "## Dataset summary", "", "| dataset | domain | tasks | score |", "|---|---|---:|---:|"])
    for dataset, info in (summary.get("datasets") or {}).items():
        lines.append(f"| {dataset} | {info.get('domain')} | {info.get('num_tasks')} | {info.get('score'):.6f} |")
    (out_dir / "validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


_AUX_EXTERNAL_BASES = {
    "qm9": EXTERNAL_ROOT / "A_datasets/small_molecule/QM9",
    "qmugs": EXTERNAL_ROOT / "A_datasets/small_molecule/QMugs",
    "geom_drugs": EXTERNAL_ROOT / "A_datasets/small_molecule/GEOM-Drugs",
    "pubchemqc": EXTERNAL_ROOT / "A_datasets/small_molecule/PubChemQC",
    "pcqm4v2": EXTERNAL_ROOT / "A_datasets/small_molecule/pcqm4v2",
    "megascale_d1": EXTERNAL_ROOT / "A_datasets/biomacromolecule/MegaScale",
    "megascale_d23": EXTERNAL_ROOT / "A_datasets/biomacromolecule/MegaScale",
    "dbaasp_hemo": EXTERNAL_ROOT / "A_datasets/biomacromolecule/dbaasp_v4",
    "dbaasp_nonmic": EXTERNAL_ROOT / "A_datasets/biomacromolecule/dbaasp_v4",
}


def _shadow_split(record_id: str) -> str:
    value = int.from_bytes(hashlib.blake2b(record_id.encode(), digest_size=8).digest(), "little")
    return "validation" if value % 5 == 0 else "train"


def _shadow_regression_value(property_name: str, observation: Mapping[str, Any]) -> Optional[float]:
    value = observation.get("value")
    task_type = observation.get("task_type")
    if value is None:
        return None
    if task_type == "regression":
        return float(value)
    if property_name == "gfn2_dipole" and task_type == "vector_regression":
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (4,) or not np.isfinite(vector).all():
            raise ValueError("gfn2_dipole shadow 需要有限的 [x,y,z,norm] 四分量")
        expected = float(np.linalg.norm(vector[:3]))
        if not np.isclose(float(vector[3]), expected, rtol=0.002, atol=0.002):
            raise ValueError("gfn2_dipole 第四分量与三分量模长不一致")
        return float(vector[3])
    return None


def evaluate_shadow_records(encoder: MuseEncoder, featurizer: RecordFeaturizer, cfg: MuseConfig,
                            property_name: str, records: Sequence[Mapping[str, Any]],
                            batch_size: int, device: str,
                            probe_config: ProbeConfig = PROBE_CONFIG,
                            record_encoder=None, namespace: Optional[str] = None,
                            train_cap: Optional[int] = None) -> dict[str, Any]:
    """通用 aux 回归留出探针接口；调用方提供不属于 A 榜训练任务的记录。"""
    train, validation = [], []
    for record in records:
        (validation if _shadow_split(record["id"]) == "validation" else train).append(record)
    train_available = len(train)
    if train_cap is not None:
        if train_cap < 2:
            raise ValueError("shadow train_cap 至少为 2")
        train = train[:train_cap]
    if len(train) < 2 or len(validation) < 1:
        raise ValueError(f"{property_name}: shadow train/validation 样本不足")

    def encode_part(records_part, split):
        if record_encoder is None:
            return encode_records(encoder, featurizer, cfg, records_part, batch_size, False, device)
        return record_encoder(
            encoder, featurizer, cfg, records_part, batch_size, False, device,
            namespace=f"{namespace or property_name}/{split}")

    train_encoded = encode_part(train, "train")
    validation_encoded = encode_part(validation, "validation")

    def values(records_part, encoded):
        by_id = {record["id"]: record for record in records_part}
        X, y, ids = [], [], []
        for row, record_id in enumerate(encoded.ids):
            record = by_id[record_id]
            if property_name in (record.get("labels") or {}):
                observations = record["labels"][property_name]
            elif property_name in (record.get("targets") or {}):
                task_type, target_values = record["targets"][property_name]
                observations = [{"task_type": task_type, "value": value} for value in target_values]
            else:
                continue
            for observation in observations:
                value = _shadow_regression_value(property_name, observation)
                if value is not None:
                    X.append(encoded.embeddings[row])
                    y.append(value)
                    ids.append(record_id)
                    break
        return ids, (np.stack(X) if X else np.empty((0, cfg.embed_dim), dtype=np.float32)), y

    train_ids, X_train, y_train = values(train, train_encoded)
    validation_ids, X_validation, y_validation = values(validation, validation_encoded)
    predictions, meta = fit_probe(
        X_train, y_train, X_validation, "regression", device=device,
        config=probe_config,
    )
    mae = float(np.mean(np.abs(predictions - np.asarray(y_validation))))
    median = float(np.median(y_train))
    baseline = float(np.mean(np.abs(np.asarray(y_validation) - median)))
    return {
        "property": property_name,
        "status": "ok",
        "metric": "MAE",
        "raw_metric": mae,
        "baseline": baseline,
        "normalized_score": float(np.clip(1.0 - mae / baseline, 0.0, 1.0)) if baseline > 0 else None,
        "n_train": len(train_ids),
        "n_validation": len(validation_ids),
        "n_train_available": train_available,
        "train_cap": train_cap,
        "label_projection": ("gfn2_dipole[3], checked against norm(gfn2_dipole[:3])"
                             if property_name == "gfn2_dipole" else "scalar"),
        "encoding_failed": train_encoded.failed + validation_encoded.failed,
        "probe": meta,
    }


def _collect_shadow_records(source: str, property_name: str, limit: Optional[int]) -> List[dict[str, Any]]:
    from muse_encoder.pretrain.aux import iter_source

    records = []
    for namespaced in iter_source(source):
        observations = (namespaced.payload.get("labels") or {}).get(property_name, [])
        if any(_shadow_regression_value(property_name, obs) is not None for obs in observations):
            records.append(dict(namespaced.payload, labels={property_name: observations}))
            if limit is not None and len(records) >= limit:
                break
    return records


def run_shadow_probes(encoder: MuseEncoder, featurizer: RecordFeaturizer, cfg: MuseConfig,
                      output_dir: Path, checkpoint_sha: str, limit_per_property: Optional[int],
                      batch_size: int, device: str, record_encoder=None,
                      train_cap: Optional[int] = None) -> dict[str, Any]:
    manifest_path = SHADOW_MANIFEST
    if not manifest_path.exists():
        return {"status": "missing_manifest", "path": str(manifest_path), "properties": {}}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = {
        "status": "ok",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "limit_per_property": limit_per_property,
        "train_cap": train_cap,
        "properties": {},
    }
    cache_root = output_dir / "shadow"
    by_source: dict[str, list[str]] = defaultdict(list)
    for key in manifest.get("properties", []):
        source, name = key.split("::", 1)
        by_source[source].append(name)
    for source, names in sorted(by_source.items()):
        for name in names:
            key = f"{source}::{name}"
            cache = cache_root / f"{_safe_name(key)}.json"
            wanted = {"checkpoint_sha256": checkpoint_sha, "limit_per_property": limit_per_property,
                      "train_cap": train_cap,
                      "probe_config_sha256": _config_sha(PROBE_CONFIG.to_json())}
            if cache.exists():
                old = json.loads(cache.read_text(encoding="utf-8"))
                if old.get("cache_key") == wanted:
                    output["properties"][key] = old["result"]
                    continue
            try:
                raw = _collect_shadow_records(source, name, limit_per_property)
                normalization_errors = []
                if record_encoder is None:
                    external_base = _AUX_EXTERNAL_BASES.get(source)
                    resolver = (ExternalStructureResolver(external_base, output_dir / "shadow_external_cache")
                                if external_base is not None and external_base.exists() else None)
                    if resolver is not None:
                        resolver.prepare(raw)
                    normalized = []
                    for record in raw:
                        try:
                            normalized.append(normalize_record(record, resolver))
                        except Exception as exc:
                            normalization_errors.append({"id": record.get("id"), "error": str(exc)[:500]})
                else:
                    normalized = raw
                result = evaluate_shadow_records(
                    encoder, featurizer, cfg, name, normalized, batch_size, device,
                    record_encoder=record_encoder, namespace=key, train_cap=train_cap,
                )
                encoded_count = result["n_train"] + result["n_validation"]
                result.update({
                    "n_input": len(raw),
                    "n_normalized": len(normalized) if record_encoder is None else encoded_count,
                    "normalization_delegated": record_encoder is not None,
                    "missing": normalization_errors + result["encoding_failed"],
                })
            except Exception as exc:
                result = {"property": key, "status": "error", "detail": str(exc)[:1000]}
            _write_json(cache, {"cache_key": wanted, "result": result})
            output["properties"][key] = result
    return output


def run_dashboard(checkpoint_path: Path, output_dir: Path, device: str = "cpu",
                  batch_size: int = 128, limit_per_split: Optional[int] = None,
                  selected_subsets: Optional[Sequence[str]] = None,
                  data_root: Path = DATA_ROOT, team: str = "zhaoyao",
                  name: str = "muse-encoder-fixed-probe", run_shadow: bool = True,
                  shadow_limit: Optional[int] = None,
                  probe_config: ProbeConfig = PROBE_CONFIG) -> dict[str, Any]:
    if limit_per_split is not None and limit_per_split <= 0:
        raise ValueError("limit_per_split 必须为正数")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    board = load_board(data_root, "A")
    selected = list(board.subsets.values())
    if selected_subsets:
        wanted = set(selected_subsets)
        selected = [subset for subset in selected if f"{subset.dataset}/{subset.subset}" in wanted]
        missing = wanted - {f"{subset.dataset}/{subset.subset}" for subset in selected}
        if missing:
            raise KeyError(f"未知子集: {sorted(missing)}")
    encoder, cfg, checkpoint_info = load_checkpoint(checkpoint_path, device)
    tokenizers = load_tokenizers()
    featurizer = RecordFeaturizer(cfg, tokenizers)
    run_key = _config_sha({
        "checkpoint": checkpoint_info["sha256"],
        "model_config": cfg.to_dict(),
        "probe": probe_config.to_json(),
        "limit_per_split": limit_per_split,
        "subsets": [f"{s.dataset}/{s.subset}" for s in selected],
        "data_manifest": sha256_file(data_root / "manifest.json"),
    })
    state_path = output_dir / "run_state.json"
    if state_path.exists():
        old = json.loads(state_path.read_text(encoding="utf-8"))
        if old.get("run_key") != run_key:
            raise RuntimeError("输出目录已有不同 checkpoint/配置/limit 的看板；请换新目录以保留审计")
    _write_json(state_path, {
        "schema_version": "muse_encoder.dashboard_state.v1",
        "run_key": run_key,
        "checkpoint": checkpoint_info,
        "model_config": cfg.to_dict(),
        "probe_config": probe_config.to_json(),
        "limit_per_split": limit_per_split,
        "smoke": limit_per_split is not None,
        "selected_subsets": [f"{s.dataset}/{s.subset}" for s in selected],
    })

    prediction_root = output_dir / "predictions"
    task_cache = output_dir / "task_cache"
    task_runs: dict[str, dict[str, Any]] = {}
    raw_validation: dict[Tuple[str, str], List[dict[str, Any]]] = {}
    cached_splits = 0
    for subset in selected:
        token_level = any(task.task_type == "sequence_classification" for task in subset.tasks)
        encoded = {}
        raw = {}
        for split in SPLITS:
            encoded[split], raw[split], from_cache = encode_split_cached(
                encoder, featurizer, cfg, subset, board.data_root, split, output_dir,
                checkpoint_info["sha256"], limit_per_split, token_level, batch_size, device,
                probe_config,
            )
            cached_splits += int(from_cache)
        raw_validation[(subset.dataset, subset.subset)] = raw["validation"]
        for task in board.tasks:
            if task.dataset != subset.dataset or task.subset != subset.subset:
                continue
            task_runs[task.task_id] = run_task(
                task, subset, raw["train"], raw["validation"], encoded["train"], encoded["validation"],
                prediction_root, task_cache, run_key, device, probe_config,
            )

    heads = {}
    for task in board.tasks:
        run = task_runs.get(task.task_id)
        if run and run.get("status") == "missing_head":
            heads[task.task_id] = {"status": "missing_head", "detail": run.get("detail", "")}
    _write_json(prediction_root / "manifest.json", {
        "team": team,
        "name": name,
        "board": "A",
        "split": "validation",
        "heads": heads,
        "smoke": limit_per_split is not None,
        "limit_per_split": limit_per_split,
    })

    score_root = board.data_root
    official_scope = "full_validation"
    if limit_per_split is not None:
        official_scope = "smoke_limited_validation"
        score_root = build_smoke_data_root(
            board.data_root, board.manifest, selected, raw_validation, board.tasks,
            output_dir / "official_smoke_data",
        )
    official_scores = official_scorer().score_predictions(
        prediction_root, score_root, "A", split="validation"
    )
    _write_json(output_dir / "official_scores.json", official_scores)
    rows = _merge_official_scores(board.tasks, official_scores, task_runs)
    summary = dict(official_scores)
    summary["scope"] = official_scope
    summary["is_full_a_board"] = limit_per_split is None and not selected_subsets
    if limit_per_split is not None or selected_subsets:
        summary["complete"] = False
        summary["note"] = "smoke/selected-subset 结果只用于调试，不能标成 13×90 全量通过"
    _write_tables(output_dir, rows, summary)

    if run_shadow:
        effective_shadow_limit = shadow_limit if shadow_limit is not None else limit_per_split
        shadow = run_shadow_probes(
            encoder, featurizer, cfg, output_dir, checkpoint_info["sha256"],
            effective_shadow_limit, batch_size, device,
        )
    else:
        shadow = {"status": "skipped"}
    result = {
        "schema_version": "muse_encoder.dashboard.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoint": checkpoint_info,
        "model_config": cfg.to_dict(),
        "probe_config": probe_config.to_json(),
        "board": "A",
        "split": "validation",
        "smoke": limit_per_split is not None,
        "limit_per_split": limit_per_split,
        "selected_subsets": [f"{s.dataset}/{s.subset}" for s in selected],
        "cached_splits": cached_splits,
        "summary": summary,
        "tasks": rows,
        "shadow": shadow,
        "files": {
            "predictions": str(prediction_root),
            "official_scores": str(output_dir / "official_scores.json"),
            "task_table": str(output_dir / "validation_tasks.csv"),
            "summary_table": str(output_dir / "validation_summary.md"),
        },
    }
    _write_json(output_dir / "results.json", result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="自有 MuseEncoder checkpoint（禁止 models/ 下公开权重）")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit-per-split", type=int, default=None,
                        help="仅 smoke；正式全量不得设置")
    parser.add_argument("--subsets", default=None,
                        help="仅 smoke/调试：逗号分隔 dataset/subset；正式默认 13 个全量")
    parser.add_argument("--team", default="zhaoyao")
    parser.add_argument("--name", default="muse-encoder-fixed-probe")
    parser.add_argument("--skip-shadow", action="store_true")
    parser.add_argument("--shadow-limit", type=int, default=None,
                        help="shadow smoke 限制；正式默认不截断")
    args = parser.parse_args(argv)
    result = run_dashboard(
        args.checkpoint, args.output_dir, args.device, args.batch_size,
        args.limit_per_split,
        args.subsets.split(",") if args.subsets else None,
        DATA_ROOT, args.team, args.name, not args.skip_shadow, args.shadow_limit,
    )
    print(json.dumps({
        "results": str(args.output_dir / "results.json"),
        "official_scores": str(args.output_dir / "official_scores.json"),
        "scope": result["summary"].get("scope"),
        "smoke": result["smoke"],
        "complete": result["summary"].get("complete"),
        "final_score": result["summary"].get("final_score"),
        "scored_tasks": result["summary"].get("scored_tasks"),
        "num_tasks": result["summary"].get("num_tasks"),
    }, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
