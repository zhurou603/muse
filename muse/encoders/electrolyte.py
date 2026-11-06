"""聚合物电解质编码器（TransPolymer PE_I / PE_II，input_key=electrolyte_string）。

格式（$ 分隔字段、| 分隔组分）：
  聚合物SMILES$...$S_4$Tg | 盐SMILES$...$S_3$Tg | mole_ratio$...$35
v0 提取：组分 1、2 的 SMILES 分别过 Uni-Mol2 编码，拼接摩尔比数值。
表征：[emb_polymer(1024), emb_salt(1024), mole_ratio, log1p(mole_ratio)] → 2050 维。
"""

import json
from pathlib import Path

import numpy as np

from muse.encoders import unimol2_mol as U


def parse_electrolyte(s):
    parts = str(s).split("|")
    smi_poly = parts[0].split("$")[0] if len(parts) > 0 else ""
    smi_salt = parts[1].split("$")[0] if len(parts) > 1 else ""
    ratio = np.nan
    if len(parts) > 2:
        try:
            ratio = float(parts[2].split("$")[0])
        except ValueError:
            ratio = np.nan
    if not smi_poly or not smi_salt:
        raise ValueError("electrolyte_string 解析失败")
    return smi_poly, smi_salt, ratio


def encode_subset(subset, cache_dir, device_id=0, batch_size=128):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    first = json.loads(subset.split_file("train").open(encoding="utf-8").readline())
    if "electrolyte_string" in first["inputs"]:
        return _encode_pe1(subset, cache_dir, device_id, batch_size)
    if "context" in first["inputs"]:
        return _encode_pe2(subset, cache_dir, device_id, batch_size)
    raise ValueError(f"未识别的电解质输入: {list(first['inputs'].keys())}")


def _collect_and_encode(splits_data, smiles_of, device_id, batch_size):
    """通用骨架：收集唯一 SMILES → Uni-Mol2 编码 → emb_map。"""
    all_smiles = []
    for records in splits_data.values():
        for r in records:
            all_smiles += smiles_of(r)
    uniq = sorted(set(all_smiles))
    print(f"  {len(uniq)} 个唯一组分分子", flush=True)
    rows, bad = U.smiles_to_rows(uniq)
    if bad:
        print(f"  构象失败 {len(bad)}: {[uniq[i] for i in bad][:5]}", flush=True)
    ok = [uniq[i] for i in range(len(uniq)) if i not in set(bad)]
    model, device = U.load_model(device_id=device_id)
    embs = U.encode_rows(rows, model, device, batch_size)
    return dict(zip(ok, embs))


def _write_splits(subset, cache_dir, splits_data, build_feat):
    n_skip = 0
    for split, records in splits_data.items():
        ids, feats, kept = [], [], []
        for r in records:
            x = build_feat(r)
            if x is None:
                n_skip += 1
                continue
            feats.append(x)
            ids.append(r["id"])
            kept.append(r)
        X = np.stack(feats).astype(np.float32)
        np.save(cache_dir / f"{split}.emb.npy", X)
        (cache_dir / f"{split}.ids.txt").write_text("\n".join(ids) + "\n",
                                                   encoding="utf-8")
        with (cache_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps({"id": r["id"], "labels": r.get("labels", {})},
                                   ensure_ascii=False, allow_nan=False) + "\n")
        print(f"ELECTROLYTE-ENCODED {split}: {X.shape} (skip={n_skip})", flush=True)


def _encode_pe1(subset, cache_dir, device_id, batch_size):
    """PE_I：electrolyte_string 格式。"""
    splits_data = {}
    for split in ("train", "validation", "test"):
        splits_data[split] = [json.loads(l) for l in
                              subset.split_file(split).open(encoding="utf-8")]

    def smiles_of(r):
        try:
            a, b, _ = parse_electrolyte(r["inputs"]["electrolyte_string"])
            return [a, b]
        except ValueError:
            return []

    emb_map = _collect_and_encode(splits_data, smiles_of, device_id, batch_size)

    def build(r):
        try:
            a, b, ratio = parse_electrolyte(r["inputs"]["electrolyte_string"])
        except ValueError:
            return None
        ea, eb = emb_map.get(a), emb_map.get(b)
        if ea is None or eb is None:
            return None
        if not np.isfinite(ratio):
            ratio = 0.0
        return np.concatenate([ea, eb, np.array([ratio, np.log1p(ratio)],
                                                dtype=np.float32)])

    _write_splits(subset, cache_dir, splits_data, build)


def _encode_pe2(subset, cache_dir, device_id, batch_size):
    """PE_II：context 字典格式（双 PSMILES + 阴离子 + 数值/类别特征）。"""
    splits_data, archs = {}, set()
    for split in ("train", "validation", "test"):
        records = [json.loads(l) for l in
                   subset.split_file(split).open(encoding="utf-8")]
        splits_data[split] = records
        for r in records:
            archs.add(str(r["inputs"]["context"].get("chain_architecture", "")))
    arch_list = sorted(archs)

    def smiles_of(r):
        c = r["inputs"]["context"]
        return [c.get("SMILES descriptor 1", ""), c.get("SMILES descriptor 2", ""),
                c.get("Amion SMILES", "")]

    emb_map = _collect_and_encode(splits_data, smiles_of, device_id, batch_size)

    def num(c, key):
        try:
            v = float(c.get(key, 0.0))
            return v if np.isfinite(v) else 0.0
        except (TypeError, ValueError):
            return 0.0

    def build(r):
        c = r["inputs"]["context"]
        e1, e2, e3 = (emb_map.get(s) for s in smiles_of(r))
        if e1 is None or e2 is None or e3 is None:
            return None
        tail = [num(c, "log Li:functional group"), num(c, "Comonomer percentage"),
                num(c, "approxMW(kDa)"), num(c, "approxTg")]
        tail += [1.0 if str(c.get("chain_architecture", "")) == a else 0.0
                 for a in arch_list]
        return np.concatenate([e1, e2, e3, np.array(tail, dtype=np.float32)])

    _write_splits(subset, cache_dir, splits_data, build)

