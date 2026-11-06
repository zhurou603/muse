"""共聚物编码器（Aldeghi-Coley vipea）：双单体 Uni-Mol2 表征 + 组成数值特征拼接。

表征布局：[emb_a(1024), emb_b(1024), fraction_a, fraction_b,
          comp_a, comp_b（composition 如 '4A_4B' 解析出的计量数）,
          polymer_type one-hot]
总维度 = 2052 + n_polymer_types。
"""

import json
import re
from pathlib import Path

import numpy as np

from muse.encoders import unimol2_mol as U


def parse_composition(comp):
    """'4A_4B' → (4.0, 4.0)；解析失败给 (1,1)。"""
    m = re.findall(r"(\d+(?:\.\d+)?)\s*([AB])", str(comp))
    a = b = 1.0
    for num, which in m:
        if which == "A":
            a = float(num)
        else:
            b = float(num)
    return a, b


def encode_subset(subset, cache_dir, device_id=0, batch_size=128):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    splits_data = {}
    all_smiles = []
    ptypes = set()
    for split in ("train", "validation", "test"):
        records = [json.loads(l) for l in
                   subset.split_file(split).open(encoding="utf-8")]
        splits_data[split] = records
        for r in records:
            all_smiles += [r["inputs"]["monomer_a_smiles"],
                           r["inputs"]["monomer_b_smiles"]]
            ptypes.add(str(r["inputs"].get("polymer_type", "")))
    ptype_list = sorted(ptypes)

    uniq = sorted(set(all_smiles))
    print(f"共聚物 {subset.subset}: {len(uniq)} 个唯一单体", flush=True)
    rows, bad = U.smiles_to_rows(uniq)
    if bad:
        print(f"  构象生成失败单体 {len(bad)} 个: {[uniq[i] for i in bad][:5]}", flush=True)
    ok_smiles = [uniq[i] for i in range(len(uniq)) if i not in set(bad)]
    model, device = U.load_model(device_id=device_id)
    embs = U.encode_rows(rows, model, device, batch_size)
    emb_map = dict(zip(ok_smiles, embs))

    for split, records in splits_data.items():
        ids, feats = [], []
        kept_records = []
        for r in records:
            ea = emb_map.get(r["inputs"]["monomer_a_smiles"])
            eb = emb_map.get(r["inputs"]["monomer_b_smiles"])
            if ea is None or eb is None:
                continue  # 单体构象失败，隔离（official.py 会用回退补齐）
            ca, cb = parse_composition(r["inputs"].get("composition", ""))
            pt_onehot = [1.0 if str(r["inputs"].get("polymer_type", "")) == p else 0.0
                         for p in ptype_list]
            num = np.array([float(r["inputs"]["fraction_a"]),
                            float(r["inputs"]["fraction_b"]), ca, cb] + pt_onehot,
                           dtype=np.float32)
            feats.append(np.concatenate([ea, eb, num]))
            ids.append(r["id"])
            kept_records.append(r)
        X = np.stack(feats).astype(np.float32)
        np.save(cache_dir / f"{split}.emb.npy", X)
        (cache_dir / f"{split}.ids.txt").write_text(
            "\n".join(ids) + "\n", encoding="utf-8")
        with (cache_dir / f"{split}.jsonl").open("x", encoding="utf-8") as f:
            for r in kept_records:
                f.write(json.dumps({"id": r["id"], "labels": r.get("labels", {})},
                                   ensure_ascii=False, allow_nan=False) + "\n")
        print(f"COPOLY-ENCODED {split}: {X.shape}", flush=True)
    (cache_dir / "encode_stats.json").write_text(json.dumps(
        {"unique_monomers": len(uniq), "failed_monomers": len(bad),
         "polymer_types": ptype_list}, ensure_ascii=False, indent=2),
        encoding="utf-8")
