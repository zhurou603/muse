"""晶体 v0 组成基线编码器：chemical_formula → 元素计数向量 + atom_count。

不读 CIF/LMDB，只用 JSONL 里的化学式。对所有晶体子集（OMC25、Polymer-Genome）
提供统一的保底表征；正经的周期图神经网络前端属后续阶段。
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem

PT = Chem.GetPeriodicTable()
N_ELEMENTS = 100  # H..Fm 足够覆盖本赛题全部体系


def formula_features(formula, atom_count=None):
    counts = Counter()
    for sym, num in re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", str(formula)):
        try:
            z = PT.GetAtomicNumber(sym)
        except Exception:
            continue
        if 1 <= z <= N_ELEMENTS:
            counts[z] += float(num) if num else 1.0
    vec = np.zeros(N_ELEMENTS + 3, dtype=np.float32)
    for z, c in counts.items():
        vec[z - 1] = c
    total = float(sum(counts.values()))
    vec[N_ELEMENTS] = float(atom_count) if atom_count else total
    vec[N_ELEMENTS + 1] = np.log1p(vec[N_ELEMENTS])
    vec[N_ELEMENTS + 2] = total / max(vec[N_ELEMENTS], 1.0)  # 平均化学计量
    return vec


def encode_subset(subset, cache_dir, limit_train=None):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        # 大训练集（OMC25 15GB JSONL）：流式读取，到限量即停
        records = []
        with subset.split_file(split).open(encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
                if split == "train" and limit_train and len(records) >= limit_train:
                    break
        ids = [r["id"] for r in records]
        X = np.stack([formula_features(r["inputs"]["chemical_formula"],
                                       r["inputs"].get("atom_count"))
                      for r in records]).astype(np.float32)
        np.save(cache_dir / f"{split}.emb.npy", X)
        (cache_dir / f"{split}.ids.txt").write_text(
            "\n".join(ids) + "\n", encoding="utf-8")
        with (cache_dir / f"{split}.jsonl").open("x", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps({"id": r["id"], "labels": r.get("labels", {})},
                                   ensure_ascii=False, allow_nan=False) + "\n")
        print(f"COMPOSITION-ENCODED {subset.dataset}/{subset.subset}/{split}: "
              f"{X.shape}", flush=True)
