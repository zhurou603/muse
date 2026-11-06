"""Fail-closed external-corpus provenance audit for final reproduction materials."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime import load_yaml

REQUIRED = ("name", "version_or_release", "source_url", "license", "purpose",
            "records", "raw_sha256", "download_date", "cleaned_sha256")
UNKNOWN = {"", "unknown", "todo", "required", "待核验", None}


def audit(path: Path) -> list[str]:
    payload = load_yaml(path)
    errors = []
    for index, item in enumerate(payload.get("external_corpora", [])):
        name = item.get("name", f"index_{index}")
        for key in REQUIRED:
            value = item.get(key)
            if value in UNKNOWN or (isinstance(value, str) and value.strip().lower() in UNKNOWN):
                errors.append(f"{name}: 缺少 {key}")
        if item.get("labels_used_for_encoder", True) is not False:
            errors.append(f"{name}: 外部语料只能使用无标签输入")
        if item.get("decontamination_audit") in UNKNOWN:
            errors.append(f"{name}: 缺少 decontamination_audit")
    if not payload.get("external_corpora"):
        errors.append("external_corpora 为空")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    errors = audit(args.manifest)
    report = {"manifest": str(args.manifest), "complete": not errors, "errors": errors}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
