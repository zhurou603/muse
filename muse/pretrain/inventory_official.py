"""官方数据资产索引 → data/pretrain/manifest/official_assets.json。

盘点 MUSE-A 三个层级（data / external_inputs / aux_labels）：
- aux_labels：每个叶目录的文件、字节数、fields.json 字段表、JSONL 行数；
- data：每个子集 train/validation/test 的字节数与行数；
- external_inputs：每数据集的文件数与字节数（内容量大，不逐一哈希；
  原始压缩包 SHA256 见 data/muse-a/archives/ 校验清单）。
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from muse import tasks as T

OUT = T.ROOT / "data/pretrain/manifest/official_assets.json"


def wc_l(path):
    r = subprocess.run(["wc", "-l", str(path)], capture_output=True, text=True)
    return int(r.stdout.split()[0])


def dir_size(path):
    total = 0
    n_files = 0
    for p in Path(path).rglob("*"):
        if p.is_file():
            total += p.stat().st_size
            n_files += 1
    return total, n_files


def main():
    inv = {"aux_labels": {}, "data": {}, "external_inputs": {}}

    aux_root = T.BASE / "aux_labels/A_datasets"
    for leaf in sorted(p for p in aux_root.rglob("*") if p.is_dir()
                       and any(c.is_file() for c in p.iterdir())):
        rel = str(leaf.relative_to(T.BASE))
        entry = {"files": {}, "n_records": 0}
        fields_file = leaf / "fields.json"
        if fields_file.exists():
            entry["fields"] = list(json.loads(
                fields_file.read_text(encoding="utf-8")).keys())
        for f in sorted(leaf.iterdir()):
            if not f.is_file():
                continue
            info = {"bytes": f.stat().st_size}
            if f.suffix == ".jsonl":
                info["lines"] = wc_l(f)
                entry["n_records"] += info["lines"]
            entry["files"][f.name] = info
        inv["aux_labels"][rel] = entry
        print(f"AUX {rel}: {entry['n_records']} records", flush=True)

    for (dname, sname), subset in sorted(T.load_subsets().items()):
        entry = {"domain": subset.domain, "directory": subset.directory,
                 "input_keys": subset.input_keys,
                 "tasks": {t.name: t.task_type for t in subset.tasks},
                 "splits": {}}
        for split in T.SPLITS:
            fp = subset.split_file(split)
            if fp.exists():
                entry["splits"][split] = {"bytes": fp.stat().st_size,
                                          "lines": wc_l(fp)}
        inv["data"][f"{dname}/{sname}"] = entry
        print(f"DATA {dname}/{sname}: "
              f"{ {k: v['lines'] for k, v in entry['splits'].items()} }", flush=True)

    ext_root = T.BASE / "external_inputs/A_datasets"
    for ds in sorted(p for p in ext_root.rglob("*") if p.is_dir()
                     and (p / "assets").exists()):
        rel = str(ds.relative_to(T.BASE))
        total, n_files = dir_size(ds / "assets")
        inv["external_inputs"][rel] = {"bytes": total, "n_files": n_files}
        print(f"EXT {rel}: {n_files} files, {total/1e9:.1f} GB", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(inv, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {OUT}", flush=True)


if __name__ == "__main__":
    main()
