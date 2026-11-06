from __future__ import annotations

import unittest
from collections import Counter

import numpy as np
import torch
from rdkit import Chem

from muse_encoder.masking import (IGNORE, apply_masking, peptide_aware_mask,
                                  random_atom_mask, random_mlm_mask,
                                  sample_coordinate_noise)
from muse_encoder.peptide import (annotate_peptide, check_fragment_recovery,
                                  fragment_recovery_rate)


class SequenceAndAtomMaskTests(unittest.TestCase):
    def test_sequence_rate_replacements_labels_and_specials(self):
        generator = torch.Generator().manual_seed(2026)
        vocab_size = 506
        token_ids = torch.randint(6, vocab_size, (64, 48), generator=generator)
        token_ids[:, 0] = 3
        token_ids[:, 1] = 0
        token_ids[:, 2] = 1
        token_ids[:, 3] = 2
        token_ids[:, 4] = 4
        token_ids[:, 5] = 5
        token_ids[0, 0] = 9
        pad_mask = torch.ones_like(token_ids, dtype=torch.bool)
        pad_mask[:, 40:] = False
        eligible = pad_mask.clone()
        eligible[:, :6] = False
        eligible[0, 0] = False

        total_selected = 0
        total_mask = 0
        total_random = 0
        total_unchanged = 0
        repetitions = 120
        for _ in range(repetitions):
            masked, labels = random_mlm_mask(token_ids, pad_mask, 2, vocab_size,
                                             0.15, generator)
            selected = labels != IGNORE
            self.assertFalse((selected & ~eligible).any())
            self.assertTrue(torch.equal(labels[selected], token_ids[selected]))
            self.assertTrue((labels[~selected] == IGNORE).all())
            total_selected += int(selected.sum())
            total_mask += int((selected & (masked == 2)).sum())
            random_replaced = selected & (masked != 2) & (masked != token_ids)
            total_random += int(random_replaced.sum())
            total_unchanged += int((selected & (masked == token_ids)).sum())
            if random_replaced.any():
                self.assertTrue((masked[random_replaced] >= 6).all())
                self.assertTrue((masked[random_replaced] < vocab_size).all())

        rate = total_selected / (repetitions * int(eligible.sum()))
        self.assertAlmostEqual(rate, 0.15, delta=0.01)
        self.assertAlmostEqual(total_mask / total_selected, 0.80, delta=0.01)
        normal_count = vocab_size - 6
        expected_random_changed = 0.10 * (1.0 - 1.0 / normal_count)
        self.assertAlmostEqual(total_random / total_selected,
                               expected_random_changed, delta=0.01)
        expected_unchanged = 0.10 + 0.10 / normal_count
        self.assertAlmostEqual(total_unchanged / total_selected,
                               expected_unchanged, delta=0.01)

    def test_custom_special_ids_are_not_selected_or_sampled(self):
        generator = torch.Generator().manual_seed(9)
        token_ids = torch.full((16, 24), 9, dtype=torch.long)
        token_ids[:, 8:] = 10
        pad_mask = torch.ones_like(token_ids, dtype=torch.bool)
        for _ in range(20):
            masked, labels = random_mlm_mask(
                token_ids, pad_mask, 2, 32, generator=generator,
                special_token_ids=(0, 1, 2, 3, 4, 5, 9))
            self.assertTrue((labels[token_ids == 9] == IGNORE).all())
            random_values = masked[(labels != IGNORE) & (masked != 2)
                                   & (masked != token_ids)]
            self.assertFalse((random_values == 9).any())

    def test_atom_rate_and_dummy_mask_are_distinct(self):
        generator = torch.Generator().manual_seed(73)
        element_ids = torch.zeros((128, 32), dtype=torch.long)
        pad_mask = torch.ones_like(element_ids, dtype=torch.bool)
        pad_mask[:, -6:] = False
        eligible = pad_mask.clone()
        eligible[:, 0] = False
        selected_count = 0
        repetitions = 30
        for _ in range(repetitions):
            masked, labels = random_atom_mask(element_ids, pad_mask, 0.15, generator)
            selected = labels != IGNORE
            self.assertTrue((masked[selected] == 119).all())
            self.assertTrue((labels[selected] == 0).all())
            self.assertTrue((masked[~selected] == 0).all())
            self.assertFalse((selected & ~eligible).any())
            selected_count += int(selected.sum())
        rate = selected_count / (repetitions * int(eligible.sum()))
        self.assertAlmostEqual(rate, 0.15, delta=0.01)

        masked, labels = apply_masking('atom', element_ids, pad_mask,
                                       cfg_mask_prob=0.15, generator=generator,
                                       atom_mask_id=118)
        self.assertTrue((masked[labels != IGNORE] == 118).all())


class CoordinateNoiseTests(unittest.TestCase):
    def test_sigma_distribution_noise_equation_and_invalid_positions(self):
        generator = torch.Generator().manual_seed(419)
        coords = torch.randn((4096, 5, 3), generator=generator)
        coord_mask = torch.ones((4096, 5), dtype=torch.bool)
        coord_mask[:, -1] = False
        coords[0, 2, 0] = float('nan')
        coords[1, 3, 1] = float('inf')
        coords[2, 4] = 7.0
        noisy, true_noise, sigma, noise_mask = sample_coordinate_noise(
            coords, coord_mask, generator=generator, return_noise_mask=True)

        expected_mask = coord_mask & torch.isfinite(coords).all(dim=-1)
        expected_mask[:, 0] = False
        self.assertTrue(torch.equal(noise_mask, expected_mask))
        self.assertTrue(((sigma >= 0.04) & (sigma <= 1.0)).all())
        self.assertGreater(sigma.unique().numel(), 100)
        expected = coords + sigma.view(-1, 1, 1) * true_noise
        torch.testing.assert_close(noisy[noise_mask], expected[noise_mask])
        recovered_noise = (noisy - coords) / sigma.view(-1, 1, 1)
        torch.testing.assert_close(recovered_noise[noise_mask], true_noise[noise_mask])
        self.assertTrue((true_noise[~noise_mask] == 0).all())
        finite_invalid = ~noise_mask & torch.isfinite(coords).all(dim=-1)
        torch.testing.assert_close(noisy[finite_invalid], coords[finite_invalid])
        self.assertTrue(torch.isfinite(noisy).all())
        self.assertTrue((noisy[0, 2] == 0).all())
        self.assertTrue((noisy[1, 3] == 0).all())

        normalized = (sigma.log() - np.log(0.04)) / np.log(1.0 / 0.04)
        self.assertAlmostEqual(float(normalized.mean()), 0.5, delta=0.02)
        for lower, upper in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
            fraction = ((normalized >= lower) & (normalized < upper)).float().mean()
            self.assertAlmostEqual(float(fraction), 0.25, delta=0.025)
        epsilon = true_noise[noise_mask]
        self.assertAlmostEqual(float(epsilon.mean()), 0.0, delta=0.02)
        self.assertAlmostEqual(float(epsilon.std()), 1.0, delta=0.02)

    def test_default_three_return_values_and_unbatched_input(self):
        coords = torch.randn(4, 3)
        coord_mask = torch.ones(4, dtype=torch.bool)
        result = sample_coordinate_noise(coords, coord_mask,
                                         generator=torch.Generator().manual_seed(1))
        self.assertEqual(len(result), 3)
        noisy, noise, sigma = result
        self.assertEqual(noisy.shape, coords.shape)
        self.assertEqual(noise.shape, coords.shape)
        self.assertEqual(sigma.ndim, 0)
        self.assertTrue((noisy[0] == coords[0]).all())
        self.assertTrue((noise[0] == 0).all())


class PeptideMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = Chem.MolFromSequence('ACDEFGHIKLMNPQRSTVWY')
        arrays = annotate_peptide(cls.mol)
        cls.group = np.concatenate(([-1], arrays[0]))
        cls.sidechain = np.concatenate(([False], arrays[1]))
        cls.fragment = np.concatenate(([-1], arrays[2]))
        cls.element = np.array([0] + [atom.GetAtomicNum() for atom in cls.mol.GetAtoms()])

    def inputs(self):
        element = torch.from_numpy(self.element).long().unsqueeze(0)
        pad = torch.ones_like(element, dtype=torch.bool)
        group = torch.from_numpy(self.group).long().unsqueeze(0)
        sidechain = torch.from_numpy(self.sidechain).bool().unsqueeze(0)
        fragment = torch.from_numpy(self.fragment).long().unsqueeze(0)
        return element, pad, group, sidechain, fragment

    def test_all_three_real_strategies_execute(self):
        element, pad, group, sidechain, fragment = self.inputs()
        for strategy in ('bulk', 'sidechain', 'fragment'):
            with self.subTest(strategy=strategy):
                masked, labels = peptide_aware_mask(
                    element, pad, group, sidechain, fragment,
                    generator=torch.Generator().manual_seed(17), strict=True,
                    strategy=strategy)
                selected = labels != IGNORE
                self.assertTrue(selected.any())
                self.assertEqual(int(labels[0, 0]), IGNORE)
                self.assertTrue((masked[selected] == 119).all())
                if strategy == 'sidechain':
                    self.assertTrue(sidechain[selected].all())
                    self.assertFalse((selected & ~sidechain).any())
                else:
                    annotation = group if strategy == 'bulk' else fragment
                    for unit in annotation[selected].unique():
                        self.assertTrue(selected[annotation == unit].all())

    def test_strict_missing_or_unusable_annotation_raises_without_fallback(self):
        element, pad, group, sidechain, fragment = self.inputs()
        required = {'bulk': group, 'sidechain': sidechain, 'fragment': fragment}
        for strategy in required:
            with self.subTest(strategy=strategy, case='missing'):
                kwargs = {'group_ids': group, 'sidechain_flags': sidechain,
                          'fragment_ids': fragment}
                name = {'bulk': 'group_ids', 'sidechain': 'sidechain_flags',
                        'fragment': 'fragment_ids'}[strategy]
                kwargs[name] = None
                with self.assertRaisesRegex(ValueError, 'strict'):
                    peptide_aware_mask(element, pad, strict=True,
                                       strategy=strategy, **kwargs)
            with self.subTest(strategy=strategy, case='unusable'):
                unusable = (torch.full_like(group, -1) if strategy != 'sidechain'
                            else torch.zeros_like(sidechain))
                kwargs = {'group_ids': group, 'sidechain_flags': sidechain,
                          'fragment_ids': fragment}
                kwargs[name] = unusable
                with self.assertRaisesRegex(ValueError, 'strict'):
                    peptide_aware_mask(element, pad, strict=True,
                                       strategy=strategy, **kwargs)

    def test_non_strict_smoke_fallback_still_works(self):
        element, pad, _, _, _ = self.inputs()
        masked, labels, strategies = peptide_aware_mask(
            element, pad, strategy='fragment', return_strategies=True,
            generator=torch.Generator().manual_seed(3))
        self.assertEqual(strategies, ['span'])
        self.assertTrue((labels != IGNORE).any())
        self.assertEqual(int(labels[0, 0]), IGNORE)
        self.assertTrue((masked[labels != IGNORE] == 119).all())

    def test_default_strategy_mix_is_30_40_30(self):
        batch_size, length = 3000, 12
        element = torch.arange(length).repeat(batch_size, 1) + 1
        element[:, 0] = 0
        pad = torch.ones_like(element, dtype=torch.bool)
        group = torch.tensor([-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
        group = group.repeat(batch_size, 1)
        sidechain = torch.tensor(
            [False, False, True, False, True, False, True,
             False, True, False, True, True]).repeat(batch_size, 1)
        fragment = torch.tensor([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        fragment = fragment.repeat(batch_size, 1)
        _, labels, strategies = peptide_aware_mask(
            element, pad, group, sidechain, fragment, strict=True,
            return_strategies=True, generator=torch.Generator().manual_seed(51))
        counts = Counter(strategies)
        self.assertEqual(set(counts), {'bulk', 'sidechain', 'fragment'})
        self.assertAlmostEqual(counts['bulk'] / batch_size, 0.30, delta=0.03)
        self.assertAlmostEqual(counts['sidechain'] / batch_size, 0.40, delta=0.03)
        self.assertAlmostEqual(counts['fragment'] / batch_size, 0.30, delta=0.03)
        self.assertTrue((labels != IGNORE).any(dim=1).all())


class PeptideAnnotationTests(unittest.TestCase):
    def test_natural_peptide_annotations_amides_and_recovery(self):
        mol = Chem.MolFromSequence('ACDEFGHIKLMNPQRSTVWY')
        group, sidechain, fragment = annotate_peptide(mol)
        self.assertEqual(group.shape, sidechain.shape, fragment.shape)
        self.assertEqual(group.shape, (mol.GetNumAtoms(),))
        self.assertEqual(group.dtype, np.int64)
        self.assertEqual(fragment.dtype, np.int64)
        self.assertEqual(sidechain.dtype, np.bool_)
        self.assertEqual(len(np.unique(group)), 20)
        self.assertTrue(sidechain.any())
        core = Chem.MolFromSmarts('[#7;X3,X4:1]-[#6;X4:2]-[#6;X3:3](=[#8;X1:4])')
        self.assertEqual(len(mol.GetSubstructMatches(core)), 20)
        for match in mol.GetSubstructMatches(core):
            self.assertFalse(sidechain[list(match)].any())
        amide = Chem.MolFromSmarts('[CX3:1](=[OX1:2])-[NX3,NX4:3]')
        for carbonyl_c, _, amide_n in mol.GetSubstructMatches(amide):
            self.assertEqual(fragment[carbonyl_c], fragment[amide_n])
        self.assertGreaterEqual(fragment_recovery_rate(mol, fragment), 0.97)
        self.assertGreaterEqual(check_fragment_recovery(mol, fragment), 0.97)

    def test_large_sidechain_uses_brics_without_cutting_amide(self):
        mol = Chem.MolFromSequence('WW')
        _, _, fragment = annotate_peptide(mol)
        benzylic = Chem.MolFromSmarts('[CH2:1]-[c:2]')
        matches = mol.GetSubstructMatches(benzylic)
        self.assertEqual(len(matches), 2)
        for begin, end in matches:
            self.assertNotEqual(fragment[begin], fragment[end])
        amide = Chem.MolFromSmarts('[CX3:1](=[OX1:2])-[NX3,NX4:3]')
        for carbonyl_c, _, amide_n in mol.GetSubstructMatches(amide):
            self.assertEqual(fragment[carbonyl_c], fragment[amide_n])

    def test_annotations_follow_graph_not_atom_order(self):
        mol = Chem.MolFromSequence('ACDHWY')
        group, sidechain, fragment = annotate_peptide(mol)
        new_order = list(reversed(range(mol.GetNumAtoms())))
        renumbered = Chem.RenumberAtoms(mol, new_order)
        new_group, new_sidechain, new_fragment = annotate_peptide(renumbered)
        group_map = {}
        fragment_map = {}
        for new_index, old_index in enumerate(new_order):
            self.assertEqual(new_sidechain[new_index], sidechain[old_index])
            old_group = int(group[old_index])
            old_fragment = int(fragment[old_index])
            self.assertEqual(group_map.setdefault(old_group,
                                                  int(new_group[new_index])),
                             int(new_group[new_index]))
            self.assertEqual(fragment_map.setdefault(old_fragment,
                                                     int(new_fragment[new_index])),
                             int(new_fragment[new_index]))
        self.assertGreaterEqual(fragment_recovery_rate(renumbered, new_fragment), 0.97)

    def test_d_chirality_isotope_and_atom_properties_are_preserved(self):
        mol = Chem.MolFromSequence('FW')
        mol.GetAtomWithIdx(0).SetProp('mixedCaseProperty', 'D-Peptide-13C')
        mol.GetAtomWithIdx(0).SetIsotope(13)
        d_mol = Chem.Mol(mol)
        inverted = 0
        for atom in d_mol.GetAtoms():
            if atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
                inverted += 1
            elif atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
                atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
                inverted += 1
        self.assertGreater(inverted, 0)
        before = Chem.MolToSmiles(d_mol, canonical=True)
        before_tags = [atom.GetChiralTag() for atom in d_mol.GetAtoms()]
        expected = annotate_peptide(mol)
        actual = annotate_peptide(d_mol)
        for expected_array, actual_array in zip(expected, actual):
            np.testing.assert_array_equal(actual_array, expected_array)
        self.assertEqual(Chem.MolToSmiles(d_mol, canonical=True), before)
        self.assertEqual([atom.GetChiralTag() for atom in d_mol.GetAtoms()], before_tags)
        self.assertEqual(d_mol.GetAtomWithIdx(0).GetIsotope(), 13)
        self.assertEqual(d_mol.GetAtomWithIdx(0).GetProp('mixedCaseProperty'),
                         'D-Peptide-13C')

    def test_non_alpha_terminal_group_is_separate_and_amide_protected(self):
        mol = Chem.MolFromSmiles('C[C@H](N)C(=O)N1CCCCC1')
        self.assertIsNotNone(mol)
        group, sidechain, fragment = annotate_peptide(mol)
        self.assertEqual(len(np.unique(group)), 2)
        amide = Chem.MolFromSmarts('[CX3:1](=[OX1:2])-[NX3,NX4:3]')
        for carbonyl_c, _, amide_n in mol.GetSubstructMatches(amide):
            self.assertFalse(sidechain[carbonyl_c])
            self.assertFalse(sidechain[amide_n])
            self.assertEqual(fragment[carbonyl_c], fragment[amide_n])
        self.assertTrue(sidechain.any())
        self.assertGreaterEqual(fragment_recovery_rate(mol, fragment), 0.97)

    def test_recovery_check_rejects_cut_amide_and_non_peptide_is_explicit(self):
        mol = Chem.MolFromSequence('AG')
        _, _, fragment = annotate_peptide(mol)
        amide = Chem.MolFromSmarts('[CX3:1](=[OX1:2])-[NX3,NX4:3]')
        carbonyl_c, _, amide_n = mol.GetSubstructMatches(amide)[0]
        broken = fragment.copy()
        broken[amide_n] = broken.max() + 1
        self.assertEqual(fragment_recovery_rate(mol, broken), 0.0)
        with self.assertRaises(ValueError):
            check_fragment_recovery(mol, broken)
        with self.assertRaises(ValueError):
            annotate_peptide(Chem.MolFromSmiles('CCO'))


if __name__ == '__main__':
    unittest.main()
