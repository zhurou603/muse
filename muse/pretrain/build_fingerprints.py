"""构建 A 榜 validation/test 指纹库（去泄漏审计的基准集合）。

遍历 manifest 全部子集的 validation/test JSONL，按输入类型提取指纹：
- 分子类输入（smiles/psmiles/csmiles/monomer_*_smiles/ligand_smiles 及
  electrolyte_string 中可解析的 SMILES 段）→ RDKit 标准异构 SMILES + InChIKey；
  含 '*' 的 PSMILES 只留标准 SMILES（InChI 不支持 dummy 原子）。
- 序列类输入（sequence/sequence_1/sequence_2/protein_sequence）→ 大写原序列。
- 晶体输入（chemical_formula + atom_count，另记录 OMC25 的 sid）→ 组成指纹。

输出（data/pretrain/fingerprints/）：
  molecules.tsv    inchikey<TAB>canonical_smiles<TAB>source
  sequences.tsv    sequence<TAB>source
  crystals.tsv     formula<TAB>atom_count<TAB>sid<TAB>source
  summary.json     各类数量与解析失败统计
指纹按 key 去重，source 保留首个来源（dataset/subset/split/id）。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from muse import tasks as T

OUT = T.ROOT / "data/pretrain/fingerprints"
SMILES_KEYS = ("smiles", "psmiles", "csmiles", "monomer_a_smiles",
               "monomer_b_smiles", "ligand_smiles")
SEQ_KEYS = ("sequence", "sequence_1", "sequence_2", "protein_sequence")


def canon_smiles(smi):
    """返回 (canonical_smiles, inchikey_or_None)。解析失败抛 ValueError。"""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("unparseable")
    can = Chem.MolToSmiles(mol, isomericSmiles=True)
    ik = None
    if "*" not in can:
        try:
            ik = Chem.MolToInchiKey(mol)
        except Exception:
            ik = None
    return can, ik


def electrolyte_smiles_parts(s):
    """electrolyte_string: 段以 '|' 分隔组分，组分内以 '$' 分隔字段，
    首字段为 SMILES。返回可解析的 SMILES 段列表。"""
    parts = []
    for comp in str(s).split("|"):
        first = comp.split("$", 1)[0].strip()
        if first and "NAN" not in first:
            parts.append(first)
    return parts


def iter_molecule_strings(inputs):
    for k in SMILES_KEYS:
        v = inputs.get(k)
        if v:
            yield k, str(v)
    v = inputs.get("electrolyte_string")
    if v:
        for part in electrolyte_smiles_parts(v):
            yield "electrolyte_string", part


def main():
    from rdkit import Chem  # noqa: F401  确认环境可用

    OUT.mkdir(parents=True, exist_ok=True)
    subsets = T.load_subsets()
    mols, seqs, crys = {}, {}, {}
    stats = {"records": 0, "mol_ok": 0, "mol_fail": 0, "seq": 0, "crystal": 0,
             "files_missing": []}

    for (dname, sname), subset in sorted(subsets.items()):
        for split in ("validation", "test"):
            fp = subset.split_file(split)
            if not fp.exists():
                stats["files_missing"].append(str(fp))
                continue
            with fp.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    stats["records"] += 1
                    src = f"{dname}/{sname}/{split}/{r['id']}"
                    inputs = r.get("inputs", {})
                    for key, smi in iter_molecule_strings(inputs):
                        try:
                            can, ik = canon_smiles(smi)
                        except ValueError:
                            stats["mol_fail"] += 1
                            continue
                        fp_key = ik if ik else f"CAN:{can}"
                        if fp_key not in mols:
                            mols[fp_key] = (ik or "", can, src)
                        stats["mol_ok"] += 1
                    for key in SEQ_KEYS:
                        v = inputs.get(key)
                        if v:
                            s = str(v).strip().upper()
                            if s and s not in seqs:
                                seqs[s] = src
                            stats["seq"] += 1
                    if "chemical_formula" in inputs:
                        sid = ""
                        for ext in r.get("external_inputs", []):
                            sid = ext.get("selector", {}).get("sid", "") or sid
                        key = (str(inputs["chemical_formula"]),
                               str(inputs.get("atom_count", "")), sid)
                        if key not in crys:
                            crys[key] = src
                        stats["crystal"] += 1

    with (OUT / "molecules.tsv").open("w", encoding="utf-8") as f:
        for ik, can, src in mols.values():
            f.write(f"{ik}\t{can}\t{src}\n")
    with (OUT / "sequences.tsv").open("w", encoding="utf-8") as f:
        for s, src in seqs.items():
            f.write(f"{s}\t{src}\n")
    with (OUT / "crystals.tsv").open("w", encoding="utf-8") as f:
        for (formula, ac, sid), src in crys.items():
            f.write(f"{formula}\t{ac}\t{sid}\t{src}\n")

    stats.update({"unique_molecules": len(mols), "unique_sequences": len(seqs),
                  "unique_crystals": len(crys)})
    (OUT / "summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
