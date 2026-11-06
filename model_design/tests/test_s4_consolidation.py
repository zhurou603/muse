from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "model_design/launches/s4-overnight-20261104"
for path in (str(ROOT / "model_design/launches/stage2-20261031"), str(STAGE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import s4_common as S4
from run_s4 import optimizer_groups


class S4ConsolidationTests(unittest.TestCase):
    def test_treatment_weights_and_no_crystal(self):
        weights = S4.treatment_weights()
        self.assertEqual(len(weights), 44)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        domains = S4.domain_source_weights(weights)
        self.assertAlmostEqual(domains["polymer"], 0.75)
        self.assertAlmostEqual(domains["bio"], 0.15)
        self.assertAlmostEqual(domains["small_molecule"], 0.10)
        self.assertEqual(domains["crystal"], 0.0)
        self.assertNotIn("omc25", weights)
        self.assertAlmostEqual(weights["Polymer-Genome-2016/DFT-HSE06/electronic_band_gap"], 0.135)
        self.assertAlmostEqual(weights["Polymer-Genome-2016/DFT-rPW86/electronic_band_gap"], 0.135)
        self.assertAlmostEqual(weights["Polymer-Genome-2016/DFT-rPW86/atomization_energy"], 0.03)
        self.assertAlmostEqual(weights["pi1m"], 0.075)

    def test_manifest_class_labels_are_encoded_idempotently(self):
        from muse_encoder.pretrain.a_train import _encoded_classes
        self.assertEqual(_encoded_classes(("1", "0")), ("0", "1"))
        self.assertEqual(_encoded_classes(("blue", "red")), ('"blue"', '"red"'))

    def test_aux_energy_keys_exist_in_normalizer(self):
        from muse_encoder.pretrain.dataset import ENERGY_KEYS
        from muse_encoder.pretrain.targets import TargetNormalizer
        normalizer = TargetNormalizer.from_path(S4.ROOT / "data/pretrain/targets/aux_targets.json")
        for source in ("geom_drugs", "qmugs", "qm9"):
            self.assertIn(ENERGY_KEYS[source], normalizer.specs)

    def test_control_is_fixed_domain_matched_replay(self):
        self.assertAlmostEqual(sum(S4.CONTROL_WEIGHTS.values()), 1.0)
        self.assertEqual(S4.CONTROL_WEIGHTS["pi1m"], 0.75)
        self.assertFalse(any(name.startswith(("PEER/", "CATH/")) for name in S4.CONTROL_WEIGHTS))
        self.assertFalse(any(name.startswith("dbaasp") for name in S4.CONTROL_WEIGHTS))

    def test_replay_wrapper_removes_task_supervision(self):
        record = SimpleNamespace(meta={})
        base = SimpleNamespace(
            name="qm9", kind="aux_molecule", spec="spec",
            get=lambda pointer: (record, {"aux_records": [1], "energy": 2, "other": 3}),
        )
        dataset = S4.ReplayOnlyDataset(base)
        actual_record, item = dataset.get("pointer")
        self.assertNotIn("aux_records", item)
        self.assertNotIn("energy", item)
        self.assertEqual(item["other"], 3)
        self.assertTrue(actual_record.meta["s4_replay_only"])

    def test_sampler_shares_source_but_not_rows_across_ranks(self):
        sources = [
            S4.SourceSampler(S4.SourceSpec("a", "polymer", 1000)),
            S4.SourceSampler(S4.SourceSpec("b", "polymer", 1000)),
        ]
        sampler = S4.RankSynchronizedWeightedSampler(
            sources, {"a": 0.5, "b": 0.5}, seed=42, stage=4)
        first = sampler.batch(7, 0, 4, 16)
        second = sampler.batch(7, 1, 4, 16)
        self.assertEqual({pointer.source for pointer in first},
                         {pointer.source for pointer in second})
        self.assertNotEqual([pointer.row for pointer in first],
                            [pointer.row for pointer in second])

    def test_optimizer_has_separate_auxiliary_head_lr(self):
        from muse_encoder.config import MuseConfig
        from muse_encoder.model import MuseEncoder
        cfg = MuseConfig(d_model=8, n_layers=1, n_heads=2, d_ffn=16, embed_dim=8,
                         contrastive_dim=4, descriptor_dim=4, num_experts=2, mole_every=1,
                         aux_specs={"task": {"kind": "graph", "dim": 1, "task_type": "regression"}})
        model = MuseEncoder(cfg)
        groups = optimizer_groups(model, 1e-5, 1e-4, 0.01)
        self.assertEqual([group["lr"] for group in groups], [1e-5, 1e-4])
        self.assertEqual(len(groups[0]["params"]) + len(groups[1]["params"]),
                         sum(parameter.requires_grad for parameter in model.parameters()))

    def test_official_weighted_objective(self):
        scores = {name: float(index + 1) for index, name in enumerate(S4.DATASET_TASK_COUNTS)}
        expected = (
            sum(scores[name] for name in S4.DATASET_DOMAINS["small_molecule"]) / 20
            + sum(scores[name] for name in S4.DATASET_DOMAINS["polymer"]) / 16
            + scores["OMC25"] / 4
            + sum(scores[name] for name in S4.DATASET_DOMAINS["biomacromolecule"]) / 12
        )
        from aggregate_s4 import official_weighted_score
        self.assertAlmostEqual(official_weighted_score(scores), expected)


if __name__ == "__main__":
    unittest.main()
