"""Uni-Mol2-310M 分子编码器：SMILES(+SDF 坐标) → CLS 1024 维表征。

转换与特征化复用 data/installers/ 中已验证的逻辑：
- check_qm9_conversion.align_coordinates / read_structures / graph_from_smiles
- check_qm9_forward.load_310m
坐标约定：重原子、按 SMILES 图顺序对齐、中心化（与 pilot 缓存一致）。

两种输入模式：
- sdf 模式：SMILES 给图、官方 SDF 给坐标（QM9 / QMugs）；
- smiles 模式：RDKit ETKDG 生成单构象（freesolv / tdc_admet 等无结构文件数据集）。
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/yinghepool/tech-dcu/zhaoyao")
INSTALLERS = ROOT / "data/installers"
UNIMOL2_SRC = ROOT / "repos/Uni-Mol-main/unimol2"
WEIGHTS_310M = ROOT / "models/unimol2-310m/checkpoint.pt"

for p in (str(INSTALLERS), str(UNIMOL2_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from check_qm9_conversion import align_coordinates, read_structures, graph_from_smiles  # noqa: E402
from check_qm9_forward import load_310m  # noqa: E402


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _structure_refs(record):
    refs = [r for r in record["external_inputs"] if r["role"] == "molecular_structure"]
    if len(refs) != 1:
        raise ValueError(f"{record['id']}: 结构引用数={len(refs)}")
    return refs[0]


def get_smiles(inputs):
    """统一取分子串：普通 SMILES、聚合物 PSMILES 或 CSMILES。"""
    s = inputs.get("smiles") or inputs.get("psmiles") or inputs.get("csmiles")
    if not s:
        raise ValueError(f"输入缺少 smiles/psmiles/csmiles: {list(inputs.keys())}")
    return s


def encodable_smiles(smi):
    """PSMILES 的 '*' 端基 dummy 原子导致 ETKDG 距离几何失败；v0 替换为甲基。
    返回 (可编码 SMILES, 是否做了替换)。"""
    if "*" in smi:
        return smi.replace("*", "C"), True
    return smi, False


def _row_from_graph(record, graph, xyz, mapping, extra_source, smi=None):
    n = graph.GetNumAtoms()
    center = xyz.mean(axis=0, keepdims=True)
    centered = (xyz - center).astype(np.float32)
    if not np.isfinite(centered).all():
        raise ValueError("坐标非法")
    smi_raw = get_smiles(record["inputs"])
    row = {
        "id": record["id"],
        "smi": smi or smi_raw,
        "atoms": [a.GetSymbol() for a in graph.GetAtoms()],
        "coordinates": [centered.tolist()],
        "labels": record.get("labels", {}),
        "source": {**extra_source, "heavy_atom_mapping": list(map(int, mapping)),
                   "coordinate_center": center[0].tolist()},
    }
    if smi and smi != smi_raw:
        row["smi_raw"] = smi_raw
    return row


def convert_split_sdf(records, sdf_path, out_jsonl, rejected_jsonl):
    """SDF 模式转换一个 split。返回 (n_ok, n_rejected)。"""
    from rdkit import Chem  # noqa: F401  (read_structures 内部使用)

    selected, line_of = {}, {}
    for i, r in enumerate(records, start=1):
        ref = _structure_refs(r)
        if ref["format"] != "sdf":
            raise ValueError(f"{r['id']}: 非 SDF 结构 {ref['format']}")
        rid = ref["selector"]["record_id"]
        if rid in selected:
            raise ValueError(f"结构编号重复: {rid}")
        selected[rid] = r
        line_of[rid] = i
    rejected = {}
    structures = read_structures(Path(sdf_path), selected, rejected)
    n_ok = 0
    with Path(out_jsonl).open("x", encoding="utf-8") as out:
        for rid, r in selected.items():
            if rid in rejected:
                continue
            try:
                graph, xyz, mapping = align_coordinates(get_smiles(r["inputs"]), structures[rid])
                if graph.GetNumAtoms() == 0:
                    raise ValueError("去氢后无原子")
                row = _row_from_graph(r, graph, xyz, mapping,
                                      {"record_id": rid, "source_path": structures and str(sdf_path)})
            except (ValueError, RuntimeError) as exc:
                rejected[rid] = {"stage": "alignment", "reason": str(exc)[:200]}
                continue
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            n_ok += 1
    with Path(rejected_jsonl).open("x", encoding="utf-8") as rej:
        for rid, info in rejected.items():
            rej.write(json.dumps({"id": selected[rid]["id"], "record_id": rid,
                                  "line": line_of[rid], **info},
                                 ensure_ascii=False, allow_nan=False) + "\n")
    return n_ok, len(rejected)


def _gen_conformer(smiles, seed=42, max_attempts=3):
    """RDKit ETKDG 单构象；失败时 random coords 兜底。返回 (graph, xyz) 或抛 ValueError。"""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("SMILES 解析失败")
    mol = Chem.AddHs(mol, addCoords=True)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise ValueError("构象生成失败")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass  # 优化失败不致命，保留嵌入构象
    graph = Chem.RemoveAllHs(mol)
    if graph.GetNumAtoms() == 0:
        raise ValueError("去氢后无原子")
    xyz = np.asarray(graph.GetConformer().GetPositions(), dtype=np.float64)
    return graph, xyz


def convert_split_smiles(records, out_jsonl, rejected_jsonl):
    """SMILES-only 模式（自生成构象）。返回 (n_ok, n_rejected)。"""
    n_ok, n_rej = 0, 0
    with Path(out_jsonl).open("x", encoding="utf-8") as out, \
            Path(rejected_jsonl).open("x", encoding="utf-8") as rej:
        for i, r in enumerate(records, start=1):
            try:
                smi, replaced = encodable_smiles(get_smiles(r["inputs"]))
                graph, xyz = _gen_conformer(smi)
                row = _row_from_graph(r, graph, xyz, list(range(graph.GetNumAtoms())),
                                      {"conformer": "rdkit_etkdgv3",
                                       "psmiles_star_replaced": replaced}, smi=smi)
            except ValueError as exc:
                rej.write(json.dumps({"id": r["id"], "line": i, "stage": "conformer",
                                      "reason": str(exc)}, ensure_ascii=False) + "\n")
                n_rej += 1
                continue
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            n_ok += 1
    return n_ok, n_rej


def load_model(weights=WEIGHTS_310M, device_id=0):
    """加载 310M 模型到指定设备，供复用。"""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("无可用加速设备")
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    torch.manual_seed(42)
    return load_310m(weights).to(device), device


def encode_rows(rows, model, device, batch_size=128, max_atoms_per_batch=4096):
    """rows: [{smi, atoms, coordinates[[1,n,3]]}] → float32 [N, 1024]。

    按原子数排序后动态定批（大分子小批量），避免成对表征 OOM；返回顺序与输入一致。
    """
    import torch
    from unicore.data import RawArrayDataset
    from unimol2.data import IndexAtomDataset, MoleculeFeatureDataset
    from unimol2.data import Unimol2FinetuneFeatureDataset

    if not rows:
        raise ValueError("没有可编码的记录（转换阶段全部失败？）")
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["atoms"]))
    embs = [None] * len(rows)
    pos = 0
    while pos < len(order):
        last = min(pos + batch_size, len(order)) - 1
        n_max = max(len(rows[order[last]]["atoms"]), 1)  # 升序排序，chunk 末尾最大
        bs = max(1, min(batch_size, max_atoms_per_batch // n_max))
        chunk_idx = order[pos:pos + bs]
        chunk = [rows[j] for j in chunk_idx]
        smiles = RawArrayDataset([r["smi"] for r in chunk])
        atoms = RawArrayDataset([np.array(r["atoms"]) for r in chunk])
        coords = RawArrayDataset([np.asarray(r["coordinates"][0], dtype=np.float32)
                                  for r in chunk])
        raw = RawArrayDataset([{"smi": r["smi"], "atoms": np.array(r["atoms"])}
                               for r in chunk])
        tokens = IndexAtomDataset(smiles, atoms)
        graphs = MoleculeFeatureDataset(raw, drop_feat_prob=1.0, seed=42)
        feats = Unimol2FinetuneFeatureDataset(smiles, tokens, coords, graphs, seed=42)
        feats.set_epoch(1)
        batch = feats.collater([feats[i] for i in range(len(chunk))])
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.inference_mode():
            atom_repr, _ = model(batch, features_only=True)
            emb = atom_repr[:, 0, :].float().cpu().numpy()
        if not np.isfinite(emb).all():
            raise RuntimeError(f"第 {pos} 批出现非有限表征")
        for j, e in zip(chunk_idx, emb):
            embs[j] = e
        pos += bs
    return np.stack(embs, axis=0).astype(np.float32)


def smiles_to_rows(smiles_list):
    """SMILES/PSMILES 列表 → 编码 rows（自生成构象，'*' 自动替换）。失败的跳过并返回失败下标。"""
    rows, bad = [], []
    for i, smi_raw in enumerate(smiles_list):
        smi, _ = encodable_smiles(smi_raw)
        try:
            graph, xyz = _gen_conformer(smi)
        except ValueError:
            bad.append(i)
            continue
        n = graph.GetNumAtoms()
        center = xyz.mean(axis=0, keepdims=True)
        rows.append({
            "smi": smi,
            "atoms": [a.GetSymbol() for a in graph.GetAtoms()],
            "coordinates": [(xyz - center).astype(np.float32).tolist()],
        })
    return rows, bad


def encode_splits(cache_dir, splits, weights=WEIGHTS_310M, batch_size=128, device_id=0):
    """对 cache_dir 下每个 {split}.jsonl 产出 {split}.emb.npy 与 {split}.ids.txt。"""
    cache_dir = Path(cache_dir)
    model, device = load_model(weights, device_id)
    dim = None
    for split in splits:
        src = cache_dir / f"{split}.jsonl"
        emb_path = cache_dir / f"{split}.emb.npy"
        ids_path = cache_dir / f"{split}.ids.txt"
        if emb_path.exists() and ids_path.exists():
            prev = np.load(emb_path, mmap_mode="r")
            print(f"SKIP {cache_dir.name}/{split}: 表征已存在 {tuple(prev.shape)}", flush=True)
            if dim is None:
                dim = prev.shape[1]
            continue
        rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
        ids = [r["id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{split}: id 重复")
        emb = encode_rows(rows, model, device, batch_size)
        if dim is None:
            dim = emb.shape[1]
        elif emb.shape[1] != dim:
            raise RuntimeError("各 split 表征维度不一致")
        np.save(cache_dir / f"{split}.emb.npy", emb)
        (cache_dir / f"{split}.ids.txt").write_text(
            "\n".join(ids) + "\n", encoding="utf-8")
        print(f"ENCODED {cache_dir.name}/{split}: {emb.shape}", flush=True)
    return dim


def convert_split_pickle(records, external_base, out_jsonl, rejected_jsonl):
    """GEOM-Drugs 模式：rdkit_pickle（tar 分包）→ 对齐构象缓存。

    记录 selector: {archive_member, conformer_index, geom_id}；
    source_path: assets/rdkit_folder/part_XX.tar。
    pickle 内容: {'smiles', 'charge', 'conformers': [{'geom_id', 'rd_mol'}...]}。
    """
    import pickle
    import tarfile

    external_base = Path(external_base)
    # part -> member -> [record]
    need = {}
    for i, r in enumerate(records, start=1):
        refs = [x for x in r["external_inputs"] if x["role"] == "molecular_conformer"]
        if len(refs) != 1:
            raise ValueError(f"{r['id']}: 构象引用数={len(refs)}")
        ref = refs[0]
        if ref["format"] != "rdkit_pickle":
            raise ValueError(f"{r['id']}: 非 pickle 结构 {ref['format']}")
        need.setdefault(ref["source_path"], {}).setdefault(
            ref["selector"]["archive_member"], []).append((i, r, ref["selector"]))

    found_rows = {}
    rejected = {}
    for source_path, members in sorted(need.items()):
        tar_path = external_base / source_path
        remaining = set(members)
        with tarfile.open(tar_path, "r") as tf:
            for m in tf:
                if m.name not in remaining or not m.isfile():
                    continue
                remaining.discard(m.name)
                try:
                    obj = pickle.loads(tf.extractfile(m).read())
                    conformers = obj["conformers"]
                except Exception as exc:
                    for _, r, _ in members[m.name]:
                        rejected[r["id"]] = {"stage": "unpickle", "reason": str(exc)[:200]}
                    continue
                for line_no, r, sel in members[m.name]:
                    idx = sel["conformer_index"]
                    try:
                        if idx >= len(conformers):
                            raise ValueError(f"conformer_index {idx} 越界（共 {len(conformers)} 个）")
                        mol = conformers[idx]["rd_mol"]
                        if mol.GetNumConformers() != 1:
                            raise ValueError(f"构象数 {mol.GetNumConformers()} != 1")
                        graph, xyz, mapping = align_coordinates(
                            get_smiles(r["inputs"]), mol)
                        row = _row_from_graph(r, graph, xyz, mapping, {
                            "source_path": source_path,
                            "archive_member": m.name,
                            "conformer_index": idx,
                            "geom_id": sel.get("geom_id"),
                        })
                        found_rows[r["id"]] = row
                    except (ValueError, RuntimeError) as exc:
                        rejected[r["id"]] = {"stage": "alignment", "reason": str(exc)[:200]}
        for missing in remaining:
            for _, r, _ in members[missing]:
                rejected[r["id"]] = {"stage": "tar_member", "reason": "tar 中未找到成员"}
        print(f"  {source_path}: 累计 {len(found_rows)} ok / {len(rejected)} rejected",
              flush=True)

    n_ok = 0
    with Path(out_jsonl).open("x", encoding="utf-8") as out:
        for r in records:
            if r["id"] in found_rows:
                out.write(json.dumps(found_rows[r["id"]], ensure_ascii=False,
                                     allow_nan=False) + "\n")
                n_ok += 1
    with Path(rejected_jsonl).open("x", encoding="utf-8") as rej:
        for rid, info in rejected.items():
            rej.write(json.dumps({"id": rid, **info}, ensure_ascii=False,
                                 allow_nan=False) + "\n")
    return n_ok, len(rejected)
