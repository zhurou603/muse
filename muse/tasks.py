"""MUSE-A 任务注册表：解析 manifest.json，提供子集元信息与记录读取。

所有路径解析、标签结构归一化集中在这里，其余组件不直接读 manifest。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path("/yinghepool/tech-dcu/zhaoyao")
BASE = ROOT / "data/muse-a/MUSE-A"
DATA_DIR = BASE / "data"
EXTERNAL_DIR = BASE / "external_inputs"
MANIFEST_PATH = DATA_DIR / "manifest.json"
PROCESSED_DIR = ROOT / "data/muse-a/processed"

SPLITS = ("train", "validation", "test")


@dataclass
class LabelTask:
    name: str
    task_type: str            # regression / binary_classification / multiclass_classification / sequence_classification
    train_median: float | None = None
    class_labels: list | None = None
    class_stratified: bool = False


@dataclass
class Subset:
    domain: str
    dataset: str
    subset: str
    directory: str           # 相对 DATA_DIR，如 A_datasets/small_molecule/QM9/qm9
    input_keys: list
    external_formats: dict
    tasks: list = field(default_factory=list)   # list[LabelTask]

    def split_file(self, split: str) -> Path:
        return DATA_DIR / self.directory / f"{split}.jsonl"

    def external_base(self) -> Path:
        # manifest.layout: external_input_base = <root>/A_datasets/<domain>/<dataset>/
        return EXTERNAL_DIR / "A_datasets" / self.domain / self.dataset

    def cache_dir(self, tag: str) -> Path:
        return PROCESSED_DIR / f"{self.dataset}-{self.subset}-{tag}"


def _parse_subset(domain, dataset, sname, s):
    tasks = []
    for tname, t in s.get("labels", {}).items():
        ttypes = list(t["task_types"].keys())
        if len(ttypes) != 1:
            raise ValueError(f"{dataset}/{sname}/{tname}: 混合任务类型 {ttypes}")
        tasks.append(LabelTask(
            name=tname,
            task_type=ttypes[0],
            train_median=t.get("train_median"),
            class_labels=t.get("class_labels"),
            class_stratified=t.get("class_stratified", False),
        ))
    return Subset(
        domain=domain, dataset=dataset, subset=sname,
        directory=s["directory"], input_keys=s["input_keys"],
        external_formats=s.get("external_formats", {}), tasks=tasks,
    )


def load_subsets():
    """返回 {(dataset, subset): Subset}，按 manifest 顺序。"""
    m = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    out = {}
    for dname, d in m["datasets"].items():
        for sname, s in d["subsets"].items():
            out[(dname, sname)] = _parse_subset(d["subsets"][sname]["domain"], dname, sname, s)
    return out


def normalize_label_value(value):
    """分类标签在 jsonl 里可能是带引号的字符串（如 '"1"'），统一成 stripped str。"""
    if isinstance(value, str):
        v = value.strip()
        if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
            v = v[1:-1]
        return v
    return str(value)


def read_records(split_file, task_names=None):
    """读取一个 split 的 jsonl。

    返回 list[dict]，每条:
      id, sample_type, inputs, external_inputs,
      targets: {task: (task_type, values list[float|str])}
    values 保留全部观测（同一样本可有多条测量），调用方决定聚合方式。
    """
    records = []
    with Path(split_file).open(encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            targets = {}
            for tname, obs in (r.get("labels") or {}).items():
                if task_names is not None and tname not in task_names:
                    continue
                if not obs:
                    continue
                ttype = obs[0]["task_type"]
                vals = [o["value"] for o in obs]
                targets[tname] = (ttype, vals)
            records.append({
                "id": r["id"],
                "sample_type": r.get("sample_type"),
                "inputs": r.get("inputs", {}),
                "external_inputs": r.get("external_inputs", []),
                "targets": targets,
            })
    return records


def regression_targets(records, task):
    """回归训练目标：每条记录取观测均值。返回 (ids, y[float])。跳过无标签记录。"""
    ids, y = [], []
    for r in records:
        if task not in r["targets"]:
            continue
        ttype, vals = r["targets"][task]
        if ttype != "regression":
            raise ValueError(f"{task} 不是回归任务: {ttype}")
        vals = [float(v) for v in vals]
        ids.append(r["id"])
        y.append(sum(vals) / len(vals))
    return ids, y


def classification_targets(records, task):
    """分类训练目标：每条记录取首个观测的类别（观测间类别必然一致）。"""
    ids, y = [], []
    for r in records:
        if task not in r["targets"]:
            continue
        ttype, vals = r["targets"][task]
        ids.append(r["id"])
        y.append(normalize_label_value(vals[0]))
    return ids, y


def regression_observations(records, task):
    """评分用：每条记录的全部观测值列表。返回 {id: [float,...]}。"""
    out = {}
    for r in records:
        if task in r["targets"]:
            out[r["id"]] = [float(v) for v in r["targets"][task][1]]
    return out
