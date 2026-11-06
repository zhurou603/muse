"""官方辅助标签 vs A 榜 val/test 泄漏校验（方案 §4.3：主办方已保证不含，此处哈希校验留证）。

覆盖：
- 分子类 aux：GEOM-Drugs / QMugs / QM9 / PubChemQC / PCQM4Mv2
  → 流式提取 inputs.smiles，并行算指纹，命中 val/test 指纹即记录。
- 序列类 aux：MegaScale ×2 / DBAASP ×2 → 同 dedup_sequences 判定。
- OMC25 aux：tar 内 ids.txt 的帧 id 与 A 榜 val/test 记录 id 比对；同时统计帧数。

输出：data/pretrain/audit/aux_<name>.hits.jsonl + aux_audit_summary.json
"""

import json
import re
import sys
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from muse import tasks as T
from muse.pretrain import deleak as D

AUDIT = D.PRETRAIN_DIR / "audit"
AUX = T.BASE / "aux_labels/A_datasets"

SMILES_RE = re.compile(r'"inputs":\{"smiles":"((?:[^"\\]|\\.)*)"')
SEQ_RE = re.compile(r'"inputs":\{"sequence":"((?:[^"\\]|\\.)*)"')
ID_RE = re.compile(r'"id":"([^"]+)"')

_G = {}


def _init_mol(iks, cans):
    _G["iks"], _G["cans"] = iks, cans


def _mol_worker(item):
    rid, smi = item
    can, ik = D.mol_fingerprint(smi.replace("\\\\", "\\"))
    if can is None:
        return rid, "unparseable", ik
    if (ik is not None and ik in _G["iks"]) or can in _G["cans"]:
        return rid, "valtest_fingerprint_hit", ik
    return rid, None, ik


def _init_seq(exact, kmers):
    _G["exact"], _G["kmers"] = exact, kmers


def _seq_worker(item):
    rid, seq = item
    seq = seq.strip().upper()
    if seq in _G["exact"]:
        return rid, "exact_match"
    n = len(seq)
    if n < D.KMER:
        return rid, "short_peptide_hit" if seq in _G["kmers"] else None
    hits = total = 0
    for i in range(0, n - D.KMER + 1, D.KMER_STRIDE):
        total += 1
        if seq[i:i + D.KMER] in _G["kmers"]:
            hits += 1
    frac = hits / total if total else 0.0
    return rid, f"kmer_cover_{frac:.2f}" if frac >= D.KMER_HIT_FRAC else None


def iter_smiles_file(path):
    with Path(path).open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = SMILES_RE.search(line)
            if not m:
                continue
            mid = ID_RE.search(line)
            yield (mid.group(1) if mid else "?"), m.group(1)


def iter_seq_file(path):
    with Path(path).open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = SEQ_RE.search(line)
            if not m:
                continue
            mid = ID_RE.search(line)
            yield (mid.group(1) if mid else "?"), m.group(1)


def check_files(name, files, kind, pool_init, worker, it, workers=96):
    t0 = time.monotonic()
    n = n_hit = n_bad = 0
    hits_path = AUDIT / f"aux_{name}.hits.jsonl"
    with Pool(workers, initializer=pool_init[0], initargs=pool_init[1]) as pool, \
            hits_path.open("w", encoding="utf-8") as hits:
        for fp in files:
            for rid, verdict, *extra in pool.imap(worker, it(fp), chunksize=2000):
                n += 1
                if verdict:
                    n_hit += 1
                    hits.write(json.dumps({"id": rid, "file": Path(fp).name,
                                           "reason": verdict,
                                           "detail": extra[0] if extra else None},
                                          ensure_ascii=False) + "\n")
            print(f"  {Path(fp).name}: 累计 {n:,} 条, 命中 {n_hit}", flush=True)
    return {"records": n, "hits": n_hit,
            "elapsed_seconds": round(time.monotonic() - t0, 1),
            "hits_file": hits_path.name}


def check_omc25():
    """OMC25 aux 帧 id vs A 榜 val/test 记录 id；同时统计总帧数。"""
    t0 = time.monotonic()
    valtest_ids = set()
    subset = T.load_subsets()[("OMC25", "omc25")]
    for split in ("validation", "test"):
        with subset.split_file(split).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    valtest_ids.add(json.loads(line)["id"])
    n_frames = 0
    hits = []
    for tar_path in sorted((AUX / "organic_crystal/OMC25/omc25").glob("part_*.tar")):
        with tarfile.open(tar_path, "r") as tf:
            for m in tf:
                if m.name.endswith("ids.txt") and m.isfile():
                    for raw in tf.extractfile(m):
                        rid = raw.decode().strip()
                        n_frames += 1
                        if rid in valtest_ids:
                            hits.append(rid)
        print(f"  {tar_path.name}: 累计 {n_frames:,} 帧", flush=True)
    hits_path = AUDIT / "aux_omc25.hits.jsonl"
    with hits_path.open("w", encoding="utf-8") as f:
        for rid in hits:
            f.write(json.dumps({"id": rid, "reason": "valtest_id_hit"}) + "\n")
    return {"frames": n_frames, "hits": len(hits),
            "valtest_ids": len(valtest_ids),
            "elapsed_seconds": round(time.monotonic() - t0, 1),
            "hits_file": hits_path.name}


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    iks, cans = D.load_molecule_fingerprints()
    exact, kmers = D.load_sequence_fingerprints()
    summary = {}

    sm = AUX / "small_molecule"
    mol_jobs = {
        "geom_drugs": sorted((sm / "GEOM-Drugs/rdkit_folder").glob("aux-*.jsonl")),
        "qmugs": sorted((sm / "QMugs/qmugs").glob("aux-*.jsonl")),
        "qm9": [sm / "QM9/qm9/aux.jsonl"],
        "pubchemqc": [sm / "PubChemQC/b3lyp/aux.jsonl"],
        "pcqm4v2": [sm / "pcqm4v2/pcqm4v2/aux.jsonl"],
    }
    for name, files in mol_jobs.items():
        print(f"== aux 分子校验 {name}", flush=True)
        summary[name] = check_files(
            name, files, "mol", (_init_mol, (iks, cans)),
            _mol_worker, iter_smiles_file)

    bio = AUX / "biomacromolecule"
    seq_jobs = {
        "megascale_d1": [bio / "MegaScale/dataset_1/aux.jsonl"],
        "megascale_d23": [bio / "MegaScale/dataset_2_3/aux.jsonl"],
        "dbaasp_hemo": [bio / "dbaasp_v4/hemolytic_cytotoxic/aux.jsonl"],
        "dbaasp_nonmic": [bio / "dbaasp_v4/target_activity_non_mic/aux.jsonl"],
    }
    for name, files in seq_jobs.items():
        print(f"== aux 序列校验 {name}", flush=True)
        summary[name] = check_files(
            name, files, "seq", (_init_seq, (exact, kmers)),
            _seq_worker, iter_seq_file)

    print("== aux 帧校验 omc25", flush=True)
    summary["omc25"] = check_omc25()

    D.write_json(AUDIT / "aux_audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
