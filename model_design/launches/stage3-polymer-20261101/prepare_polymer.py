"""Prepare train-only robust targets for the polymer electronic pilot."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for path in (str(ROOT), str(ROOT / "model_design")):
    if path not in sys.path:
        sys.path.insert(0, path)

from dbaasp import robust_stats
from muse_encoder.pretrain.a_train import (
    _aggregate_label, compute_a_train_target_stats, iter_train_records, load_source_specs,
)
from muse_encoder.pretrain.dashboard import DATA_ROOT, sha256_file

OUTPUT = ROOT / "data/pretrain/stage3-polymer"
SOURCES = (
    "Polymer-Genome-2016/DFT-HSE06",
    "Polymer-Genome-2016/DFT-rPW86",
    "TransPolymer/Egb",
    "TransPolymer/Egc",
    "TransPolymer/Eea",
    "TransPolymer/Ei",
    "Aldeghi-Coley-Copolymer/vipea",
    "PolyGraphMT/BANDGAP_DFT",
    "PolyGraphMT/HOMO_DFT",
    "PolyGraphMT/LUMO_DFT",
    "PolyGraphMT/MU_DFT",
)
TASKS = {
    "Polymer-Genome-2016/DFT-HSE06/electronic_band_gap": 0.18,
    "Polymer-Genome-2016/DFT-rPW86/electronic_band_gap": 0.18,
    "Polymer-Genome-2016/DFT-rPW86/atomization_energy": 0.04,
    "TransPolymer/Egb/egb": 0.0625,
    "TransPolymer/Egc/egc": 0.0625,
    "TransPolymer/Eea/eea": 0.0625,
    "TransPolymer/Ei/ei": 0.0625,
    "Aldeghi-Coley-Copolymer/vipea/electron_affinity": 0.075,
    "Aldeghi-Coley-Copolymer/vipea/ionization_potential": 0.075,
    "PolyGraphMT/BANDGAP_DFT/bandgap_dft": 0.025,
    "PolyGraphMT/HOMO_DFT/homo_dft": 0.025,
    "PolyGraphMT/LUMO_DFT/lumo_dft": 0.025,
    "PolyGraphMT/MU_DFT/mu_dft": 0.025,
}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    registered = load_source_specs()
    specs = [registered[name] for name in SOURCES]
    payload = compute_a_train_target_stats(specs, OUTPUT / "targets.base.json")
    for source in specs:
        values = {task.name: [] for task in source.tasks}
        for record in iter_train_records(source):
            for task in source.tasks:
                value, _, _ = _aggregate_label(record, task)
                if value is not None:
                    values[task.name].append(value)
        for task in source.tasks:
            median, scale = robust_stats(np.asarray(values[task.name], dtype=np.float32))
            payload["stats"][task.key]["mean"] = [float(median)]
            payload["stats"][task.key]["std"] = [float(scale)]
            payload["stats"][task.key]["robust_scaling"] = "median/MAD with std-or-1 fallback"
    payload["label_policy"]["regression"] = "train-only per-task median/MAD; Huber beta=1"
    targets = OUTPUT / "targets.json"
    targets.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                       encoding="utf-8")
    manifest = {
        "schema_version": "muse.stage3.polymer_data.v1",
        "data_manifest_sha256": sha256_file(DATA_ROOT / "manifest.json"),
        "targets_sha256": sha256_file(targets),
        "task_weights": TASKS,
        "replay": {"source": "pi1m", "weight": 0.10, "labels": False},
        "external_data": False,
        "validation_test_labels_used": False,
    }
    text = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    manifest["manifest_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
