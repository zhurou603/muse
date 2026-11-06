"""去泄漏审计共享工具：加载指纹库、分子/序列指纹计算与命中判定。

判定口径（与方案 §4.3 一致）：
- 分子：InChIKey 或标准异构 SMILES 命中 A 榜 val/test 指纹 → 泄漏。
- 聚合物：含 '*' 的标准 PSMILES 命中 → 泄漏。
- 序列：完全匹配命中；或 10-mer 覆盖率 ≥ 阈值（默认 0.5，保守）→ 泄漏。
- 晶体：约化化学式+atom_count(+sid) 命中 → 泄漏（本方案暂无外部晶体语料）。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from muse import tasks as T

FP_DIR = T.ROOT / "data/pretrain/fingerprints"
PRETRAIN_DIR = T.ROOT / "data/pretrain"

KMER = 10
KMER_STRIDE = 5          # 语料侧抽样步长，覆盖率为估计值
KMER_HIT_FRAC = 0.5      # 命中比例阈值（保守）


def load_molecule_fingerprints():
    """返回 (inchikey_set, canonical_smiles_set)。"""
    iks, cans = set(), set()
    with (FP_DIR / "molecules.tsv").open(encoding="utf-8") as f:
        for line in f:
            ik, can, _src = line.rstrip("\n").split("\t")
            cans.add(can)
            if ik:
                iks.add(ik)
    return iks, cans


def load_sequence_fingerprints(kmer=KMER):
    """返回 (exact_set, kmer_set)。"""
    exact, kmers = set(), set()
    with (FP_DIR / "sequences.tsv").open(encoding="utf-8") as f:
        for line in f:
            seq, _src = line.rstrip("\n").split("\t")
            exact.add(seq)
            if len(seq) >= kmer:
                for i in range(0, len(seq) - kmer + 1):
                    kmers.add(seq[i:i + kmer])
            else:
                kmers.add(seq)  # 短肽整串作为" k-mer "，子串命中靠精确包含另查
    return exact, kmers


def mol_fingerprint(smi):
    """(canonical, inchikey|None)；失败返回 (None, None)。"""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None or mol.GetNumAtoms() == 0:
        return None, None
    can = Chem.MolToSmiles(mol, isomericSmiles=True)
    ik = None
    if "*" not in can:
        try:
            ik = Chem.MolToInchiKey(mol)
        except Exception:
            ik = None
    return can, ik


def mol_is_leak(smi, iks, cans):
    can, ik = mol_fingerprint(smi)
    if can is None:
        return None          # 无法解析，调用方另行计数
    return (ik is not None and ik in iks) or can in cans


def seq_is_leak(seq, exact, kmers, kmer=KMER, stride=KMER_STRIDE,
                thresh=KMER_HIT_FRAC):
    """完全匹配或估计 k-mer 覆盖率 ≥ thresh。"""
    if seq in exact:
        return True
    n = len(seq)
    if n < kmer:
        return seq in kmers
    hits = total = 0
    for i in range(0, n - kmer + 1, stride):
        total += 1
        if seq[i:i + kmer] in kmers:
            hits += 1
    # 短肽整串情形：语料序列包含某个短肽（其整串进了 kmers 集合）
    if hits == 0 and n >= kmer:
        pass
    return total > 0 and hits / total >= thresh


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                          encoding="utf-8")
