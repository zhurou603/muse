#!/usr/bin/env bash
set -euo pipefail
DEST=${1:?用法：bash scripts/download_external_reference.sh /空目录/external-reference}
mkdir -p "$DEST"
fetch() {
  local url=$1 name=$2
  if [[ -e "$DEST/$name" ]]; then echo "拒绝覆盖 $DEST/$name" >&2; return 1; fi
  curl --fail --location --retry 5 --continue-at - --output "$DEST/$name" "$url"
  sha256sum "$DEST/$name" > "$DEST/$name.sha256"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$DEST/$name.downloaded_utc"
}
fetch https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz CID-SMILES.gz
fetch https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/chembl_37_chemreps.txt.gz chembl_37_chemreps.txt.gz
fetch https://raw.githubusercontent.com/RUIMINMA1996/PI1M/master/PI1M_v2.csv PI1M_v2.csv
fetch https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz uniref50.fasta.gz
fetch https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.release_note uniref50.release_note
fetch https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz pdb_seqres.txt.gz
cat <<'EOF'
这些是参考副本，不会自动成为 S5 训练语料。PubChem、UniRef50、PDB 和 PI1M master
均可能变化；请固定 release/commit，并核对原仓库既有文件哈希。只有清洗、去泄漏、
行数/键对齐、许可证和血缘审计全部通过后，才能在新的独立数据目录使用。
EOF
