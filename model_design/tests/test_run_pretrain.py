import tempfile
import unittest
from pathlib import Path

import torch

from muse_encoder.config import MuseConfig
from muse_encoder.model import MuseEncoder
from muse_encoder.pretrain.run_pretrain import load_init_weights, microbatch_seed


class MicrobatchSeedTests(unittest.TestCase):
    def test_seed_is_unique_per_step_rank_and_reproducible(self):
        config = {"seed": 42, "stage": 2}
        values = [microbatch_seed(config, step, rank)
                  for step in range(16) for rank in range(8)]
        self.assertEqual(len(values), len(set(values)))
        self.assertEqual(values, [microbatch_seed(config, step, rank)
                                  for step in range(16) for rank in range(8)])
        self.assertTrue(all(0 <= value < 2**32 for value in values))
        self.assertNotEqual(microbatch_seed(config, 3, 0),
                            microbatch_seed({"seed": 43, "stage": 2}, 3, 0))
        self.assertNotEqual(microbatch_seed(config, 3, 0),
                            microbatch_seed({"seed": 42, "stage": 3}, 3, 0))


class InitWeightsTests(unittest.TestCase):
    def test_legacy_checkpoint_only_adds_composition_projection(self):
        values = dict(
            d_model=8, n_layers=1, n_heads=2, d_ffn=16, embed_dim=8,
            num_experts=1, dropout=0.0, smiles_vocab_size=32,
            psmiles_vocab_size=40, residue_vocab_size=24,
        )
        legacy = MuseEncoder(MuseConfig(**values))
        upgraded = MuseEncoder(MuseConfig(**values, crystal_composition_features=True))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"model": legacy.state_dict()}, path)
            incompatible = load_init_weights(upgraded, path, rank=1)
        self.assertEqual(incompatible.missing_keys, ["composition_readout.weight"])
        self.assertEqual(incompatible.unexpected_keys, [])


if __name__ == "__main__":
    unittest.main()
