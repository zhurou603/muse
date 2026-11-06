"""生成 data_manifest.jsonl 与审计报告。

汇总：外部语料（下载文件 + 清洗产物）与官方资产的登记信息，
字段：name/version/source_url/license/usage/num_records/sha256/local_path/
      dedup 统计引用。审计报告 audit/audit_report.md 汇总指纹库、
      各语料剔除统计、官方 aux 校验结果与处置决定。

用法：python -m muse.pretrain.write_manifest
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from muse.pretrain import deleak as D

CORPORA = D.PRETRAIN_DIR / "corpora"
CLEAN = D.PRETRAIN_DIR / "corpora_clean"
AUDIT = D.PRETRAIN_DIR / "audit"
MANIFEST = D.PRETRAIN_DIR / "manifest/data_manifest.jsonl"

# 外部语料登记信息（许可条款以其官网当前版本为准）
SOURCES = {
    "uniparc_peptides": {
        "name": "UniParc 短肽序列（<=50 残基）",
        "version": "UniProt release 2026_03（API release date 2026-09-02）",
        "source_url": "https://rest.uniprot.org/uniparc/stream?compressed=true&format=fasta&query=length%3A%5B1%20TO%2050%5D",
        "license": "CC-BY-4.0 (UniProt)",
        "usage": "天然肽第一阶段 MLM 预训练；仅无标签序列，去泄漏后 seed=42 抽样 800 万条",
        "raw": "uniparc_le50_2026_03.fasta.gz", "clean": "uniparc_peptides.fasta",
    },
    "uniref50": {
        "name": "UniRef50",
        "version": "zpn/uniref50 快照（2023-02，HF 镜像）；官方源 release 2026-09 续传校验中",
        "source_url": "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz （官方源）；实际字节经 https://hf-mirror.com/datasets/zpn/uniref50 获取（106+2 个 parquet 分卷）",
        "license": "CC-BY-4.0 (UniProt)",
        "usage": "蛋白/肽域 MLM 预训练语料（无标签序列）",
        "raw": None, "clean": "uniref50.fasta",
        "raw_dir": "uniref50_hf",
    },
    "pdb": {
        "name": "RCSB PDB 链序列 (pdb_seqres)",
        "version": "下载日快照",
        "source_url": "https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz",
        "license": "CC0-1.0 (wwPDB)",
        "usage": "结构相关蛋白序列补充（无标签序列）",
        "raw": "pdb_seqres.txt.gz", "clean": "pdb.fasta",
    },
    "chembl": {
        "name": "ChEMBL 37 chemreps（仅结构字段）",
        "version": "ChEMBL 37",
        "source_url": "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_37_chemreps.txt.gz",
        "license": "CC-BY-SA-3.0",
        "usage": "小分子 SMILES 语料（仅用 canonical_smiles，不用任何活性标签）",
        "raw": "chembl_37_chemreps.txt.gz", "clean": "chembl.tsv",
    },
    "pubchem": {
        "name": "PubChem CID-SMILES（抽样子集）",
        "version": "CURRENT-Full 快照（下载日）",
        "source_url": "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz",
        "license": "PubChem 公开数据 (NCBI)，可自由使用",
        "usage": "小分子 SMILES MLM/MTR 预训练语料（去泄漏后固定种子抽样）",
        "raw": "pubchem_cid_smiles.gz", "clean": "pubchem.tsv",
    },
    "pi1m": {
        "name": "PI1M v2",
        "version": "v2 (master 分支快照)",
        "source_url": "https://raw.githubusercontent.com/RUIMINMA1996/PI1M/master/PI1M_v2.csv",
        "license": "MIT (仓库 LICENSE)",
        "usage": "聚合物 PSMILES MLM 预训练语料",
        "raw": "PI1M_v2.csv", "clean": "pi1m.tsv",
    },
}

NOT_FETCHED = [
    {"name": "ZINC20 类药分子", "planned": "10-20M SMILES",
     "decision": "不获取",
     "rationale": "下载入口为数百个小分卷，装配成本高；PubChem 抽样子集已覆盖"
                  "类药化学空间（GEOM-Drugs/QMugs/PubChemQC 同源），边际收益低"},
    {"name": "OE62 母分子", "planned": "6.2 万分子",
     "decision": "不获取",
     "rationale": "OMC25 官方数据（帧+力/应力）已覆盖其化学空间；"
                  "规模（6.2 万）对预训练贡献边际"},
]


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def count_lines(path):
    r = subprocess.run(["wc", "-l", str(path)], capture_output=True, text=True)
    return int(r.stdout.split()[0])


def main():
    entries = []
    for key, meta in SOURCES.items():
        raw_name = meta.get("raw")
        raw = CORPORA / raw_name if raw_name else None
        clean = CLEAN / meta["clean"]
        stats_file = AUDIT / f"{key}.stats.json"
        e = {k: meta[k] for k in
             ("name", "version", "source_url", "license", "usage")}
        if raw is not None and raw.exists():
            e["raw_file"] = {"path": str(raw), "bytes": raw.stat().st_size,
                             "sha256": sha256_file(raw)}
        if meta.get("raw_dir"):
            shards = sorted((CORPORA / meta["raw_dir"]).glob("*.parquet"))
            e["raw_file"] = {"path": str(CORPORA / meta["raw_dir"]),
                             "n_shards": len(shards),
                             "bytes": sum(p.stat().st_size for p in shards)}
        if clean.exists():
            n_lines = count_lines(clean)
            n_records = n_lines // 2 if clean.suffix == ".fasta" else n_lines
            e["clean_file"] = {"path": str(clean), "bytes": clean.stat().st_size,
                               "sha256": sha256_file(clean),
                               "num_records": n_records}
        if stats_file.exists():
            e["dedup_audit"] = json.loads(
                stats_file.read_text(encoding="utf-8"))
        entries.append(e)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
        for nf in NOT_FETCHED:
            f.write(json.dumps({**nf, "type": "not_fetched"},
                               ensure_ascii=False) + "\n")
    print(f"WROTE {MANIFEST} ({len(entries) + len(NOT_FETCHED)} entries)")

    # 审计报告
    fp_summary = json.loads(
        (D.FP_DIR / "summary.json").read_text(encoding="utf-8"))
    aux_summary = {}
    aux_file = AUDIT / "aux_audit_summary.json"
    if aux_file.exists():
        aux_summary = json.loads(aux_file.read_text(encoding="utf-8"))

    lines = [
        "# 预训练数据去泄漏审计报告",
        "",
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 指纹库（A 榜 validation/test）",
        "",
        f"- 覆盖记录: {fp_summary['records']:,} 条（78 子集 val+test 全量）",
        f"- 唯一分子指纹: {fp_summary['unique_molecules']:,}（InChIKey/标准异构 SMILES）",
        f"- 唯一蛋白/肽序列: {fp_summary['unique_sequences']:,}",
        f"- 唯一晶体组成: {fp_summary['unique_crystals']:,}",
        f"- 分子解析失败: {fp_summary['mol_fail']:,}（多为 electrolyte_string 非 SMILES 段）",
        "",
        "## 2. 外部语料去泄漏",
        "",
        "| 语料 | 输入 | 剔除(泄漏) | 无法解析 | 清洗后 | 说明 |",
        "|---|---|---|---|---|---|",
    ]
    for key in SOURCES:
        stats_file = AUDIT / f"{key}.stats.json"
        if not stats_file.exists():
            lines.append(f"| {key} | - | - | - | - | 未处理 |")
            continue
        s = json.loads(stats_file.read_text(encoding="utf-8"))
        kept = s.get("kept", 0)
        out_n = s.get("sampled_out", kept)
        lines.append(
            f"| {key} | {s.get('input_records', 0):,} | "
            f"{s.get('leak_removed', s.get('removed', 0)):,} | "
            f"{s.get('unparseable', 0):,} | {out_n:,} | "
            f"{json.dumps(s.get('removal_reasons', {}), ensure_ascii=False)[:60]} |")
    lines += [
        "",
        "分子判定口径：RDKit 标准异构 SMILES 或 InChIKey 命中 val/test 指纹；",
        "序列判定口径：完全匹配，或 10-mer 估计覆盖率 ≥ 0.5（保守阈值）。"
        "环境无 MMseqs2，同一性复核由该保守 k-mer 预筛替代（会多剔远缘同源，"
        "语料规模足够，代价可接受）；若后续补装 MMseqs2，可对剔除清单复核放行。",
        "",
        "## 3. 官方辅助标签校验（主办方保证之外的实证核查）",
        "",
        "| 辅助标签 | 记录数 | 命中 val/test 指纹 | 处置 |",
        "|---|---|---|---|",
    ]
    for name, s in aux_summary.items():
        n = s.get("records", s.get("frames", 0))
        hits_file = AUDIT / s.get("hits_file", f"aux_{name}.hits.jsonl")
        n_fp = n_bad = 0
        if hits_file.exists():
            with hits_file.open(encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)["reason"]
                    if r == "valtest_fingerprint_hit" or r.startswith(
                            ("exact_match", "kmer_cover", "short_peptide",
                             "valtest_id_hit")):
                        n_fp += 1
                    else:
                        n_bad += 1
        detail = f"指纹命中 {n_fp:,}"
        if n_bad:
            detail += f"；另有无法解析 {n_bad:,}（不计入泄漏）"
        lines.append(f"| {name} | {n:,} | {detail} | "
                     f"命中记录列入排除清单，预训练不使用 |")
    lines += [
        "",
        "**重要发现**：官方辅助标签与 A 榜 val/test 存在**跨数据集**物质重叠"
        "（例如 GEOM-Drugs aux 中的分子命中 tdc_admet solubility 测试集）。"
        "主办方的保证口径为同数据集不重叠；跨数据集重叠由本审计发现并处置："
        "全部命中记录写入 `audit/aux_*.hits.jsonl`，预训练数据装载时必须按 id 排除。",
        "",
        "## 4. 显式排除项（方案 §4.2）",
        "",
        "- 不使用任何公开预训练权重；不用其输出做蒸馏标签。",
        "- 不爬取与 A 榜 90 个性质或 B 榜潜在性质等价的标注数据库。",
    ]
    for nf in NOT_FETCHED:
        lines.append(f"- {nf['name']}：{nf['decision']}。{nf['rationale']}。")
    report = "\n".join(lines) + "\n"
    (AUDIT / "audit_report.md").write_text(report, encoding="utf-8")
    print(f"WROTE {AUDIT / 'audit_report.md'}")


if __name__ == "__main__":
    main()
