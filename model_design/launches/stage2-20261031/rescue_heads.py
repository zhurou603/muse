"""Validation-selected robust heads on frozen MUSE train/validation embeddings."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
for path in (str(ROOT), str(ROOT / "model_design")):
    if path not in sys.path:
        sys.path.insert(0, path)

from muse_encoder.model import check_probe_params
from muse_encoder.pretrain import dashboard as D

DEFAULT_TASKS = (
    "dbaasp_v4/target_activity/target_activity",
    "GEOM-Drugs/rdkit_folder/energy_class",
    "tdc_admet/clearance_hepatocyte_az/clearance_hepatocyte_az",
    "tdc_admet/clearance_microsome_az/clearance_microsome_az",
    "tdc_admet/half_life_obach/half_life_obach",
    "tdc_admet/ppbr_az/ppbr_az",
    "tdc_admet/vdss_lombardo/vdss_lombardo",
    "PolyGraphMT/DIF_MD/dif_md",
    "TransPolymer/Nc/nc",
    "TransPolymer/OPV/pce_ave",
    "PEER/beta_lactamase/fitness",
    "Polymer-Genome-2016/DFT-HSE06/electronic_band_gap",
    "Polymer-Genome-2016/DFT-rPW86/electronic_band_gap",
    "TransPolymer/EPS/eps",
    "TransPolymer/Eea/eea",
    "TransPolymer/Ei/ei",
    "TransPolymer/PE_II/logcond60",
)


@dataclass(frozen=True)
class HeadSpec:
    hidden: int
    activation: str
    loss: str
    lr: float
    weight_decay: float


@dataclass
class TaskData:
    task: Any
    subset: Any
    train_raw: list[dict[str, Any]]
    validation_raw: list[dict[str, Any]]
    train_ids: list[str]
    x_train: np.ndarray
    y_train: np.ndarray
    validation_ids: list[str]
    x_validation: np.ndarray
    y_validation: np.ndarray
    validation_encoded: np.ndarray
    train_median: float
    train_mad: float
    positive_rate: float
    old_validation: dict[str, Any]


class RobustHead(nn.Module):
    def __init__(self, d_in: int, spec: HeadSpec):
        super().__init__()
        if spec.hidden == 0:
            self.net = nn.Linear(d_in, 1)
        else:
            activation = nn.GELU() if spec.activation == "gelu" else nn.Tanh()
            self.net = nn.Sequential(nn.Linear(d_in, spec.hidden), activation,
                                     nn.Linear(spec.hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def parameter_count(d_in: int, spec: HeadSpec) -> int:
    if spec.hidden == 0:
        count = d_in + 1
        if count > 100_000:
            raise ValueError("Linear head 超过 100K 参数预算")
        return count
    return check_probe_params(d_in, spec.hidden, 1, 100_000)


def robust_location_scale(values: np.ndarray) -> tuple[float, float, float, str]:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 1e-8:
        return median, mad, mad, "mad"
    std = float(values.std())
    if std > 1e-8:
        return median, mad, std, "std_fallback"
    return median, mad, 1.0, "constant_fallback"


def _load_embedding_cache(run_root: Path, task: Any, split: str):
    cache = run_root / "embeddings" / task.dataset / task.subset
    ids = json.loads((cache / f"{split}.ids.json").read_text(encoding="utf-8"))
    embeddings = np.load(cache / f"{split}.embeddings.npy")
    if len(ids) != len(embeddings) or len(set(ids)) != len(ids):
        raise ValueError(f"{task.task_id} {split}: 缓存 ID 重复或行数不符")
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError(f"{task.task_id} {split}: 表征维度或数值异常")
    return {record_id: embeddings[index] for index, record_id in enumerate(ids)}


def _old_validation(run_root: Path, task_id: str) -> dict[str, Any]:
    path = run_root / "official_scores.json"
    if not path.is_file():
        return {}
    scores = json.loads(path.read_text(encoding="utf-8"))
    return scores.get("tasks", {}).get(task_id, {})


def load_task_data(board: Any, run_root: Path, task: Any) -> TaskData:
    subset = board.subsets[(task.dataset, task.subset)]
    metadata_path = run_root / "embeddings" / task.dataset / task.subset / "train.encoding.json"
    train_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    train_limit = train_metadata.get("limit_per_split")
    train_raw = D._read_jsonl(D._split_file(board.data_root, subset, "train"), train_limit)
    validation_raw = D._read_jsonl(D._split_file(board.data_root, subset, "validation"))
    train_embeddings = _load_embedding_cache(run_root, task, "train")
    validation_embeddings = _load_embedding_cache(run_root, task, "validation")

    def rows(raw_records, embeddings):
        ids, features, labels, encoded = [], [], [], []
        for record in raw_records:
            observations = (record.get("labels") or {}).get(task.task, [])
            for observation in observations:
                value = observation["value"]
                if task.task_type == "regression":
                    label = float(value)
                elif task.task_type == "binary_classification":
                    label = float(D.official_scorer().encode_class_label(value) == task.positive_label)
                else:
                    raise ValueError(f"阶段2 P0项不支持 {task.task_type}: {task.task_id}")
                ids.append(record["id"])
                labels.append(label)
                feature = embeddings.get(record["id"])
                encoded.append(feature is not None)
                features.append(np.zeros(next(iter(embeddings.values())).shape, dtype=np.float32)
                                if feature is None else feature)
        return ids, np.stack(features), np.asarray(labels, dtype=np.float32), np.asarray(encoded, dtype=bool)

    train_ids, x_train, y_train, train_encoded = rows(train_raw, train_embeddings)
    validation_ids, x_validation, y_validation, validation_encoded = rows(validation_raw, validation_embeddings)
    x_train = x_train[train_encoded]
    train_ids = [rid for rid, keep in zip(train_ids, train_encoded) if keep]
    y_train = y_train[train_encoded]
    if len(y_train) < 2 or len(y_validation) < 2:
        raise ValueError(f"{task.task_id}: 有效训练或 validation 观测不足")
    if not np.isfinite(y_train).all():
        raise ValueError(f"{task.task_id}: 训练标签含 NaN/Inf")
    median, mad, _, _ = robust_location_scale(y_train)
    positive_rate = float(y_train.mean()) if task.task_type == "binary_classification" else 0.0
    return TaskData(task, subset, train_raw, validation_raw, train_ids, x_train, y_train,
                    validation_ids, x_validation, y_validation, validation_encoded,
                    median, mad, positive_rate, _old_validation(run_root, task.task_id))


def _record_split(ids: Sequence[str], seed: int = 12345, fraction: float = 0.1):
    unique = list(dict.fromkeys(ids))
    order = np.random.default_rng(seed).permutation(len(unique))
    n_stop = max(1, int(fraction * len(unique)))
    if n_stop >= len(unique):
        n_stop = len(unique) - 1
    stop_records = {unique[index] for index in order[:n_stop]}
    stop = np.asarray([rid in stop_records for rid in ids], dtype=bool)
    if not stop.any() or stop.all():
        raise ValueError("记录级内部早停切分为空")
    return ~stop, stop


def _predict(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), 4096):
            x = torch.from_numpy(features[start:start + 4096]).float().to(device)
            outputs.append(model(x).float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def fit_candidate(data: TaskData, spec: HeadSpec, device: str | torch.device,
                  max_epochs: int = 150, patience: int = 15, seed: int = 0):
    if parameter_count(data.x_train.shape[1], spec) > 100_000:
        raise ValueError("head 超过 100K 参数预算")
    device = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    fit_index, stop_index = _record_split(data.train_ids)
    x_fit = data.x_train[fit_index]
    y_fit_raw = data.y_train[fit_index].astype(np.float64)
    x_stop = data.x_train[stop_index]
    y_stop = data.y_train[stop_index]
    median, mad, scale, scale_kind = robust_location_scale(y_fit_raw)
    if data.task.task_type == "regression":
        y_fit = ((y_fit_raw - median) / scale).astype(np.float32)
    else:
        y_fit = y_fit_raw.astype(np.float32)
    model = RobustHead(x_fit.shape[1], spec).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    criterion = nn.L1Loss() if spec.loss == "mae" else (nn.HuberLoss() if spec.loss == "huber" else nn.BCEWithLogitsLoss())
    x_fit_t = torch.from_numpy(x_fit).float().to(device)
    y_fit_t = torch.from_numpy(y_fit).float().to(device)
    x_stop_t = torch.from_numpy(x_stop).float().to(device)
    y_stop_t = torch.from_numpy(y_stop).float().to(device)
    rng = np.random.default_rng(seed)
    best_loss, best_state, stale, best_epoch = math.inf, None, 0, -1
    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(len(x_fit))
        for start in range(0, len(order), 512):
            index = order[start:start + 512]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_fit_t[index]), y_fit_t[index])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            stop_output = model(x_stop_t)
            if data.task.task_type == "regression":
                stop_prediction = stop_output.double() * scale + median
                stop_loss = float((stop_prediction - y_stop_t.double()).abs().mean().cpu())
            else:
                stop_loss = float(nn.functional.binary_cross_entropy_with_logits(stop_output, y_stop_t).cpu())
        if not math.isfinite(stop_loss):
            raise FloatingPointError(f"{data.task.task_id}: 早停指标非有限")
        if stop_loss < best_loss - 1e-7:
            best_loss, stale, best_epoch = stop_loss, 0, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("没有得到有效 head")
    model.load_state_dict(best_state)
    model.eval()
    prediction = _predict(model, data.x_validation, device)
    if data.task.task_type == "regression":
        prediction = prediction.astype(np.float64) * scale + median
        prediction[~data.validation_encoded] = data.task.train_median
    else:
        prediction = 1.0 / (1.0 + np.exp(-np.clip(prediction, -80, 80)))
        prediction[~data.validation_encoded] = data.positive_rate
    metrics = D.official_scorer().compute_metrics(data.task.official, data.y_validation, prediction)
    training = {
        "epochs_run": epoch + 1, "best_epoch": best_epoch, "early_stop_loss": best_loss,
        "fit_observations": int(len(x_fit)), "stop_observations": int(len(x_stop)),
        "target_median": median, "target_mad": mad, "target_scale": scale,
        "target_scale_kind": scale_kind, "seed": seed,
    }
    return model, prediction.astype(np.float32), metrics, training


def screen_specs(task_type: str) -> list[HeadSpec]:
    specs = []
    for hidden in (0, 8, 16, 32, 64):
        activations = ("linear",) if hidden == 0 else ("gelu", "tanh")
        for activation in activations:
            for loss in (("mae", "huber") if task_type == "regression" else ("bce",)):
                specs.append(HeadSpec(hidden, activation, loss, 3e-4, 1e-4))
    return specs


def refinement_specs(task_type: str, screened: list[dict[str, Any]]) -> list[HeadSpec]:
    ranked = sorted(screened, key=lambda row: row["metrics"]["score"], reverse=True)
    count = 4 if task_type == "regression" else 3
    specs = []
    for row in ranked[:count]:
        base = HeadSpec(**row["spec"])
        for lr in (1e-4, 3e-4, 1e-3):
            for weight_decay in (1e-6, 1e-4, 1e-2):
                spec = HeadSpec(base.hidden, base.activation, base.loss, lr, weight_decay)
                if spec != base and spec not in specs:
                    specs.append(spec)
    return specs


def _write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def rescue_task(board: Any, run_root: Path, task: Any, output_root: Path,
                device: str, max_epochs: int, patience: int) -> dict[str, Any]:
    started = time.monotonic()
    data = load_task_data(board, run_root, task)
    task_root = output_root / D._safe_name(task.task_id.replace("/", "__"))
    task_root.mkdir(parents=True, exist_ok=True)
    trials = []
    best = None

    def evaluate(spec: HeadSpec):
        nonlocal best
        model, prediction, metrics, training = fit_candidate(
            data, spec, device, max_epochs=max_epochs, patience=patience)
        row = {"spec": asdict(spec), "metrics": metrics, "training": training,
               "params": parameter_count(data.x_train.shape[1], spec)}
        trials.append(row)
        with (task_root / "trials.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[rescue] {task.task_id} {asdict(spec)} score={100 * metrics['score']:.4f}", flush=True)
        if best is None or metrics["score"] > best[2]["score"]:
            best = (model, prediction, metrics, row)

    for spec in screen_specs(task.task_type):
        evaluate(spec)
    for spec in refinement_specs(task.task_type, list(trials)):
        evaluate(spec)
    model, prediction, metrics, best_row = best
    prediction_by_id = {rid: float(value) for rid, value in zip(data.validation_ids, prediction)}
    write_info = D.write_task_predictions(
        output_root / "predictions" / task.prediction_file, task,
        data.validation_raw, data.train_raw, prediction_by_id)
    torch.save({
        "schema_version": "muse.stage2.robust_head.v1",
        "task_id": task.task_id, "input_dim": int(data.x_train.shape[1]),
        "spec": asdict(HeadSpec(**best_row["spec"])), "state_dict": model.state_dict(),
        "training": best_row["training"],
    }, task_root / "head.pt")
    baseline_prediction = np.where(
        data.validation_encoded, data.task.train_median, data.task.train_median
    ) if task.task_type == "regression" else np.full(len(data.y_validation), data.positive_rate)
    baseline_metrics = D.official_scorer().compute_metrics(task.official, data.y_validation, baseline_prediction)
    result = {
        "schema_version": "muse.stage2.rescue_result.v1",
        "task_id": task.task_id, "task_type": task.task_type,
        "run_root": str(run_root), "split_used_for_selection": "validation",
        "test_used": False, "train_observations": len(data.y_train),
        "train_records": len(set(data.train_ids)), "validation_observations": len(data.y_validation),
        "validation_missing_embeddings": int((~data.validation_encoded).sum()),
        "full_train_median": data.train_median, "full_train_mad": data.train_mad,
        "positive_class": (json.loads(task.positive_label) if task.task_type == "binary_classification" else None),
        "baseline_metrics": baseline_metrics, "old_validation": data.old_validation,
        "best": best_row, "validation_metrics": metrics,
        "num_trials": len(trials), "predictions": write_info,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    _write_json(task_root / "result.json", result)
    return result


def locate_run(cache_root: Path, task: Any) -> Path:
    candidates = []
    state_paths = list(cache_root.glob("s*/run_state.json"))
    if (cache_root / "run_state.json").is_file():
        state_paths.append(cache_root / "run_state.json")
    for state_path in state_paths:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        key = f"{task.dataset}/{task.subset}"
        if key in state.get("selected_subsets", []):
            candidates.append(state_path.parent)
    if len(candidates) != 1:
        raise ValueError(f"{task.task_id}: 预期一个缓存分片，实际 {len(candidates)}")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path,
                        default=ROOT / "model_design/runs/s3-residual-pilot-02/stage2_validation_cache")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "model_design/runs/s3-residual-pilot-02/stage2_rescue")
    parser.add_argument("--task-id", action="append", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    board = D.load_board()
    wanted = args.task_id or list(DEFAULT_TASKS)
    tasks = [task for task in board.tasks if task.task_id in wanted]
    missing = set(wanted) - {task.task_id for task in tasks}
    if missing:
        raise KeyError(f"未知任务: {sorted(missing)}")
    results = []
    for task in tasks:
        run_root = locate_run(args.cache_root, task)
        results.append(rescue_task(board, run_root, task, args.output_dir,
                                   args.device, args.max_epochs, args.patience))
    _write_json(args.output_dir / f"summary_{args.device.replace(':', '_')}.json", results)


if __name__ == "__main__":
    main()
