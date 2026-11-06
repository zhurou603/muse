from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from muse_encoder.config import DOMAIN_POLYMER, MODALITY_1D, MuseConfig
from muse_encoder.featurize import FeaturizedRecord, RecordFeaturizer
from muse_encoder.s5.long_sequence import TrainingRecordFeaturizer


def full_record(count):
    length = count + 1
    return FeaturizedRecord(
        domain=DOMAIN_POLYMER, modality=MODALITY_1D,
        token_ids=np.arange(length), use_atom_features=False, n_real_tokens=count,
        meta={"output_token_indices": np.arange(1, length), "expected_token_count": count},
        token_kind=np.ones(length, dtype=np.int64),
        atom_features=np.zeros((length, 9), dtype=np.int64),
        segment_ids=np.zeros(length, dtype=np.int64),
        scalar_cond=np.zeros(4, dtype=np.float32),
        pool_weights=np.r_[0.0, np.ones(count, dtype=np.float32)],
    )


class LongSequenceTest(unittest.TestCase):
    def test_training_crop_is_deterministic_and_bounded(self):
        cfg = MuseConfig(max_seq_len=16)
        fz = TrainingRecordFeaturizer(cfg, {}, crop_seed=42)
        with patch.object(RecordFeaturizer, "_chemical_sequence", return_value=full_record(60)):
            first = fz._chemical_sequence("dummy", DOMAIN_POLYMER, 1, "psmiles", False)
            second = fz._chemical_sequence("dummy", DOMAIN_POLYMER, 1, "psmiles", False)
        self.assertEqual(len(first.token_ids), 16)
        self.assertEqual(first.n_real_tokens, 15)
        self.assertEqual(first.meta["training_original_tokens"], 60)
        self.assertEqual(first.meta["training_crop_start"], second.meta["training_crop_start"])
        np.testing.assert_array_equal(first.token_ids, second.token_ids)
        self.assertTrue(first.meta["truncated"])

    def test_short_sequence_is_unchanged(self):
        cfg = MuseConfig(max_seq_len=16)
        fz = TrainingRecordFeaturizer(cfg, {}, crop_seed=42)
        with patch.object(RecordFeaturizer, "_chemical_sequence", return_value=full_record(10)):
            result = fz._chemical_sequence("dummy", DOMAIN_POLYMER, 1, "psmiles", False)
        self.assertEqual(result.n_real_tokens, 10)
        self.assertNotIn("training_crop", result.meta)

    def test_residue_crop_is_bounded_deterministic_and_preserves_d_flags(self):
        cfg = MuseConfig(max_seq_len=16)
        fz = TrainingRecordFeaturizer(cfg, crop_seed=42)
        sequence = "ACd" * 20
        first = fz.featurize_sequence(sequence)
        second = fz.featurize_sequence(sequence)
        self.assertEqual(len(first.token_ids), 16)
        self.assertEqual(first.n_real_tokens, 15)
        self.assertEqual(first.meta["training_original_tokens"], 60)
        self.assertEqual(first.meta["training_crop_start"], second.meta["training_crop_start"])
        np.testing.assert_array_equal(first.token_ids, second.token_ids)
        start = first.meta["training_crop_start"]
        stop = first.meta["training_crop_stop"]
        np.testing.assert_array_equal(
            first.meta["lowercase_mask"],
            np.asarray([char.islower() for char in sequence[start:stop]]),
        )
        self.assertEqual(first.meta["sequence"], sequence[start:stop])

    def test_residue_pair_shares_the_token_budget(self):
        fz = TrainingRecordFeaturizer(MuseConfig(max_seq_len=16), crop_seed=42)
        record = fz.featurize_sequence("ACd" * 20, "dE" * 20)
        self.assertLessEqual(len(record.token_ids), 16)
        self.assertEqual(record.n_real_tokens, len(record.token_ids) - 2)
        self.assertTrue(record.meta["training_crop"])
        self.assertEqual(record.meta["training_original_tokens"], 100)
        short = fz.featurize_sequence("ACd")
        self.assertEqual(short.meta["sequence"], "ACd")
        self.assertNotIn("training_crop", short.meta)


if __name__ == "__main__":
    unittest.main()
