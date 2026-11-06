"""按 manifest 行数把 78 个子集 LPT 装箱成 16 片，输出 shards.json。

成本估计 = min(train, cap) + validation；不同模态吞吐不同，运行中再按实测调整。
"""
import json
from pathlib import Path

ROOT = Path("/yinghepool/tech-dcu/zhaoyao")
MANIFEST = ROOT / "data/muse-a/MUSE-A/data/manifest.json"
CAP = 200_000
N_SHARDS = 16


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    subsets = []
    for dataset, dv in manifest["datasets"].items():
        for name, sv in dv["subsets"].items():
            counts = {}
            for split in ("train", "validation"):
                entry = sv.get("files", {}).get(split, {}).get("file")
                if entry:
                    with (MANIFEST.parent / entry).open("rb") as stream:
                        counts[split] = sum(1 for _ in stream)
            train = min(counts.get("train", 0), CAP)
            valid = counts.get("validation", 0)
            subsets.append({"key": f"{dataset}/{name}", "cost": train + valid,
                            "train_raw": counts.get("train", 0), "validation": valid})
    subsets.sort(key=lambda item: -item["cost"])
    shards = [{"name": f"s{i:02d}", "subsets": [], "cost": 0} for i in range(N_SHARDS)]
    for item in subsets:
        target = min(shards, key=lambda shard: shard["cost"])
        target["subsets"].append(item["key"])
        target["cost"] += item["cost"]
    out = {"train_cap": CAP, "subsets": subsets, "shards": shards}
    path = Path(__file__).resolve().parent / "shards.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for shard in shards:
        print(f"{shard['name']}: cost={shard['cost']:>8d} n={len(shard['subsets']):2d} "
              f"{shard['subsets'][:3]}{'...' if len(shard['subsets']) > 3 else ''}")
    print(f"total cost={sum(s['cost'] for s in shards)}")


if __name__ == "__main__":
    main()
