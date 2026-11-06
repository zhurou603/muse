#!/usr/bin/env bash
set -euo pipefail
DEST=${1:?用法：bash scripts/download_official.sh /空目录/MUSE-A-download}
BASE=https://scicomp26.oss-cn-beijing.aliyuncs.com
mkdir -p "$DEST"
files=(
  MUSE-A_README.txt MUSE-A_SHA256SUMS.txt MUSE-eval_20261019.zip
  MUSE-A_jsonl.tar.zst
  MUSE-A_structures_GEOM-Drugs_part1of2.tar.zst
  MUSE-A_structures_GEOM-Drugs_part2of2.tar.zst
  MUSE-A_structures_QMugs_QM9_Polymer-Genome-2016.tar.zst
  MUSE-A_structures_OMC25_val.tar.zst
  MUSE-A_aux_OMC25_forces_part1of2.tar.zst
  MUSE-A_aux_OMC25_forces_part2of2.tar.zst
)
for i in {1..8}; do files+=("MUSE-A_structures_OMC25_train_part${i}of8.tar.zst"); done
for name in "${files[@]}"; do
  if [[ ! -s "$DEST/$name" ]]; then
    curl --fail --location --retry 5 --retry-delay 10 --continue-at - \
      --output "$DEST/$name" "$BASE/$name"
  fi
done
(cd "$DEST" && sha256sum -c MUSE-A_SHA256SUMS.txt)
printf '官方包下载与 SHA256 校验完成。下一步逐个安全解包，不要覆盖现役数据目录。\n'
