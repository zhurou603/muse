#!/usr/bin/env bash
set -euo pipefail
REPO=${1:?用法：bash scripts/install.sh /path/to/项目根目录}
PACKAGE=$(cd "$(dirname "$0")/.." && pwd)
TARGET="$REPO/code/model_design/muse_encoder/s5"
BACKUP="$REPO/code/model_design/muse_encoder/featurize.py.pre_s5"
TARGET_CONFIG="$REPO/code/model_design/configs/s5_continuous.yaml"
[[ -f "$REPO/code/model_design/muse_encoder/model.py" ]] || { echo '项目根目录不正确' >&2; exit 2; }
[[ ! -e "$TARGET" ]] || { echo "拒绝覆盖 $TARGET" >&2; exit 2; }
[[ ! -e "$BACKUP" ]] || { echo "拒绝覆盖已有备份 $BACKUP" >&2; exit 2; }
[[ ! -e "$TARGET_CONFIG" ]] || { echo "拒绝覆盖已有配置 $TARGET_CONFIG" >&2; exit 2; }
EXPECTED=3ef39081114d976826da142b06bc0c9d3c13d2e4e6476d8aa7b4e349c3dd729c
ACTUAL=$(sha256sum "$REPO/code/model_design/muse_encoder/featurize.py" | awk '{print $1}')
[[ "$ACTUAL" == "$EXPECTED" ]] || { echo 'featurize.py 不是已审查基线；请先人工核对补丁' >&2; exit 2; }
# Complete every fail-closed check before changing the source tree.
(cd "$REPO" && git apply --check "$PACKAGE/patches/long_sequence.patch")
cp "$REPO/code/model_design/muse_encoder/featurize.py" "$BACKUP"
(cd "$REPO" && git apply "$PACKAGE/patches/long_sequence.patch")
cp -R "$PACKAGE/s5" "$TARGET"
mkdir -p "$REPO/code/model_design/configs" "$REPO/data/pretrain/manifest"
cp "$PACKAGE/config/s5.yaml" "$TARGET_CONFIG"
if [[ ! -e "$REPO/data/pretrain/manifest/external_provenance.yaml" ]]; then
  cp "$PACKAGE/config/external_provenance.yaml" "$REPO/data/pretrain/manifest/external_provenance.yaml"
fi
cat <<EOF
S5 已安装。设置：export REPO_ROOT='$REPO'
先运行单元测试和数据准备，不要直接覆盖旧 run 目录。
EOF
