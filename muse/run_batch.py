"""批量驱动：对某数据集的全部子集（或指定清单）顺序执行管线，单个子集失败不中断。

用法：
  python -m muse.run_batch --dataset tdc_admet --mode smiles --device 0
  python -m muse.run_batch --dataset moleculenet --mode smiles --device 0
"""

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from muse import tasks as T
from muse.run import run_subset, append_scoreboard


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mode", choices=["sdf", "smiles", "pickle", "bio", "copolymer", "composition"], required=True)
    ap.add_argument("--only", default=None, help="逗号分隔的子集清单，默认全部")
    ap.add_argument("--stages", default="convert,encode,train,score")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-eval", type=int, default=None)
    ap.add_argument("--tag", default="v0")
    ap.add_argument("--seeds", default="0,1,2,3")
    args = ap.parse_args()

    subsets = T.load_subsets()
    only = set(args.only.split(",")) if args.only else None
    targets = [s for (d, _), s in subsets.items()
               if d == args.dataset and (only is None or s.subset in only)]
    if not targets:
        raise SystemExit(f"没有匹配的子集: {args.dataset} {args.only}")

    # 小子集优先：中断时能保住更多完成品
    def train_size(s):
        try:
            return s.split_file("train").stat().st_size
        except OSError:
            return 0
    targets.sort(key=train_size)

    stages = set(args.stages.split(","))
    seeds = tuple(int(s) for s in args.seeds.split(","))
    ok, failed = 0, []
    for subset in targets:
        cache_dir = subset.cache_dir(args.tag)
        print(f"\n===== {subset.dataset}/{subset.subset} =====", flush=True)
        try:
            rows = run_subset(subset, args.mode, cache_dir, stages, args.device,
                              args.batch_size, args.limit_train, args.limit_eval, seeds)
            append_scoreboard(rows)
            ok += 1
        except Exception:
            failed.append(subset.subset)
            print(f"FAILED {subset.dataset}/{subset.subset}:", flush=True)
            traceback.print_exc()
    print(f"\n批量完成: 成功 {ok}/{len(targets)}", flush=True)
    if failed:
        print("失败子集:", ", ".join(failed), flush=True)


if __name__ == "__main__":
    main()
