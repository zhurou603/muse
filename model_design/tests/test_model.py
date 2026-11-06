import unittest

import torch
import torch.nn.functional as F

from muse_encoder.config import MuseConfig
from muse_encoder.layers import MoleFFN
from muse_encoder.model import MuseEncoder


def small_cfg(**overrides):
    values = dict(
        d_model=16, n_layers=2, n_heads=4, d_ffn=24,
        num_experts=3, mole_every=1, embed_dim=12, contrastive_dim=8,
        descriptor_dim=208, max_seq_len=64, dropout=0.0,
        smiles_vocab_size=32, psmiles_vocab_size=40, residue_vocab_size=24,
    )
    values.update(overrides)
    return MuseConfig(**values)


def make_batch(batch_size=2, length=6, lengths=None):
    if lengths is None:
        lengths = [length] * batch_size
    lengths = torch.tensor(lengths, dtype=torch.long)
    token_ids = torch.randint(1, 20, (batch_size, length), dtype=torch.long)
    token_ids[:, 0] = 0
    pad_mask = torch.arange(length)[None, :] < lengths[:, None]
    pad_mask[:, 0] = True
    coord_mask = pad_mask.clone()
    coord_mask[:, 0] = False
    return {
        'token_ids': token_ids,
        'domain': torch.arange(batch_size, dtype=torch.long) % 4,
        'modality': torch.ones(batch_size, dtype=torch.long),
        'use_atom_features': torch.ones(batch_size, dtype=torch.bool),
        'pad_mask': pad_mask,
        'coords': torch.randn(batch_size, length, 3),
        'coord_mask': coord_mask,
        'bond_mat': torch.randint(0, 8, (batch_size, length, length)),
        'spd_mat': torch.randint(0, 16, (batch_size, length, length)),
    }


def slice_record(batch, index):
    n = int(batch['pad_mask'][index].sum())
    token_fields = {'token_ids', 'pad_mask', 'coord_mask', 'coords', 'token_kind',
                    'segment_ids', 'atom_features'}
    pair_fields = {'bond_mat', 'spd_mat', 'pair_disp', 'pbc_shift'}
    result = {}
    for key, value in batch.items():
        value = value[index:index + 1]
        if key in token_fields:
            value = value[:, :n]
        elif key in pair_fields:
            value = value[:, :n, :n]
        result[key] = value
    return result


class MuseModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_formal_defaults(self):
        cfg = MuseConfig()
        self.assertEqual(cfg.descriptor_dim, 208)
        self.assertEqual(cfg.atom_mask_id, 119)
        self.assertEqual(cfg.num_elements, 120)
        self.assertEqual((cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.d_ffn),
                         (24, 1024, 16, 2048))
        self.assertEqual((cfg.num_experts, cfg.embed_dim), (8, 1024))

    def test_mixed_token_kind_and_global_conditions(self):
        cfg = small_cfg()
        model = MuseEncoder(cfg).eval()
        batch = make_batch(batch_size=1, length=5)
        batch['token_ids'] = torch.tensor([[0, 6, 7, 8, 9]])
        batch['token_kind'] = torch.tensor([[0, 0, 1, 2, 3]])
        batch['use_atom_features'] = torch.tensor([False])
        batch['segment_ids'] = torch.tensor([[0, 0, 1, 1, 2]])
        batch['spacegroup_id'] = torch.tensor([25])
        batch['scalar_cond'] = torch.randn(1, cfg.scalar_cond_dim)
        with torch.no_grad():
            actual = model.embed_tokens(batch)
            tables = [model.element_emb, model.smiles_emb,
                      model.psmiles_emb, model.residue_emb]
            expected = torch.stack([
                tables[int(kind)](batch['token_ids'][0, i])
                for i, kind in enumerate(batch['token_kind'][0])
            ])
            expected[0] = model.cls_token[0, 0]
            expected = expected + model.domain_emb(batch['domain'])[0] \
                + model.modality_emb(batch['modality'])[0] \
                + model.segment_emb(batch['segment_ids'])[0] \
                + model.spacegroup_emb(batch['spacegroup_id'])[0] \
                + model.scalar_cond(batch['scalar_cond'])[0]
        self.assertTrue(torch.allclose(actual[0], expected, atol=1e-6, rtol=1e-6))

    def test_sigma_condition_changes_output(self):
        model = MuseEncoder(small_cfg()).eval()
        base = make_batch()
        low = dict(base)
        high = dict(base)
        low['denoise_sigma'] = torch.tensor([0.04, 0.08])
        high['denoise_sigma'] = torch.tensor([1.0, 0.7])
        with torch.no_grad():
            low_out = model.encode(low)
            high_out = model.encode(high)
        self.assertGreater((low_out['tokens'] - high_out['tokens']).abs().max().item(), 1e-5)
        self.assertGreater((low_out['embedding'] - high_out['embedding']).abs().max().item(), 1e-5)

    def test_drop_feature_mask_covers_atom_and_pair_features(self):
        model = MuseEncoder(small_cfg()).eval()
        batch = make_batch()
        batch['atom_features'] = torch.randint(1, 5, (2, 6, 9))
        keep = torch.zeros(2, dtype=torch.bool)
        drop = torch.ones(2, dtype=torch.bool)
        with torch.no_grad():
            kept_tokens = model.embed_tokens(batch, drop_feature_mask=keep)
            dropped_tokens = model.embed_tokens(batch, drop_feature_mask=drop)
            without_aux = dict(batch)
            without_aux.pop('atom_features')
            expected_dropped = model.embed_tokens(without_aux)
            pair = model.pair_encoder(
                batch['bond_mat'], batch['spd_mat'], batch['coords'], batch['coord_mask'],
                drop_feature_mask=keep)
            dropped_pair = model.pair_encoder(
                batch['bond_mat'], batch['spd_mat'], batch['coords'], batch['coord_mask'],
                drop_feature_mask=drop)
        self.assertGreater((kept_tokens - dropped_tokens).abs().max().item(), 1e-5)
        self.assertTrue(torch.allclose(dropped_tokens, expected_dropped, atol=1e-6, rtol=1e-6))
        self.assertGreater(pair.abs().max().item(), 0.0)
        self.assertEqual(dropped_pair.abs().max().item(), 0.0)

    def test_pair_disp_and_pbc_shift(self):
        model = MuseEncoder(small_cfg()).eval()
        batch = make_batch()
        virtual_coords = torch.randn_like(batch['coords'])
        pair_disp = virtual_coords[:, :, None, :] - virtual_coords[:, None, :, :]
        shift = torch.randn(2, 6, 6, 3)
        with torch.no_grad():
            expected = model.pair_encoder(
                batch['bond_mat'], batch['spd_mat'], virtual_coords, batch['coord_mask'],
                drop_feature_mask=torch.zeros(2, dtype=torch.bool))
            actual = model.pair_encoder(
                batch['bond_mat'], batch['spd_mat'], batch['coords'], batch['coord_mask'],
                pair_disp=pair_disp, drop_feature_mask=torch.zeros(2, dtype=torch.bool))
            periodic = model.pair_encoder(
                batch['bond_mat'], batch['spd_mat'], batch['coords'], batch['coord_mask'],
                pair_disp=pair_disp, pbc_shift=shift,
                drop_feature_mask=torch.zeros(2, dtype=torch.bool))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))
        self.assertGreater((periodic - actual).abs().max().item(), 1e-5)

    def test_descriptor_208_dual_readout_and_mask_backward(self):
        model = MuseEncoder(small_cfg())
        batch = make_batch()
        descriptor = torch.randn(2, 208)
        descriptor_mask = torch.rand(2, 208) > 0.25
        losses = model.pretraining_losses(
            batch,
            {'descriptor': descriptor, 'descriptor_mask': descriptor_mask,
             'drop_feature_mask': torch.zeros(2, dtype=torch.bool)},
            {'descriptor': 1.0})
        self.assertTrue(torch.isfinite(losses['descriptor']))
        losses['total'].backward()
        for head in (model.heads.descriptor_mid, model.heads.descriptor_out):
            self.assertIsNotNone(head.weight.grad)
            self.assertGreater(head.weight.grad.abs().sum().item(), 0.0)

    def test_aux_graph_atom_bond_regression_binary_and_missing_labels(self):
        aux_specs = {
            'g_reg': {'kind': 'graph', 'dim': 2, 'task_type': 'vector_regression'},
            'g_cls': {'kind': 'graph', 'dim': 1, 'task_type': 'binary_classification'},
            'a_reg': {'kind': 'atom', 'dim': 1, 'task_type': 'atom_regression'},
            'a_cls': {'kind': 'atom', 'dim': 2, 'task_type': 'binary_classification'},
            'b_reg': {'kind': 'bond', 'dim': 1, 'task_type': 'bond_regression'},
            'b_cls': {'kind': 'bond', 'dim': 2, 'task_type': 'binary_classification'},
            'unused': {'kind': 'graph', 'dim': 1, 'task_type': 'regression'},
        }
        model = MuseEncoder(small_cfg(aux_specs=aux_specs))
        batch = make_batch(length=4, lengths=[4, 3])
        records = [
            {'name': 'g_reg', 'value': torch.randn(2, 2),
             'mask': torch.tensor([[True, False], [True, True]])},
            {'name': 'g_cls', 'value': torch.zeros(2),
             'mask': torch.zeros(2, dtype=torch.bool)},
            {'name': 'a_reg', 'value': torch.randn(2, 4)},
            {'name': 'a_cls', 'value': torch.randint(0, 2, (2, 4, 2)).float(),
             'mask': torch.rand(2, 4) > 0.2},
            {'name': 'b_reg', 'value': torch.randn(2, 4, 4),
             'mask': torch.rand(2, 4, 4) > 0.2},
            {'name': 'b_cls', 'value': torch.randint(0, 2, (2, 4, 4, 2)).float(),
             'mask': torch.rand(2, 4, 4) > 0.2},
        ]
        losses = model.pretraining_losses(
            batch, {'aux_records': records,
                    'drop_feature_mask': torch.zeros(2, dtype=torch.bool)}, {'aux': 1.0})
        self.assertTrue(torch.isfinite(losses['aux']))
        losses['total'].backward()
        for name in ('g_reg', 'a_reg', 'a_cls', 'b_reg', 'b_cls'):
            grad = model.heads.aux_heads[name].weight.grad
            self.assertIsNotNone(grad, name)
            self.assertGreater(grad.abs().sum().item(), 0.0, name)
        for name in ('g_cls', 'unused'):
            grad = model.heads.aux_heads[name].weight.grad
            self.assertTrue(grad is None or grad.abs().sum().item() == 0.0, name)

    def test_aux_multiclass_token_ce_and_missing_mask(self):
        aux_specs = {
            'multi': {'kind': 'graph', 'dim': 3, 'task_type': 'multiclass_classification'},
            'token': {'kind': 'atom', 'dim': 3, 'task_type': 'sequence_classification'},
            'legacy_reg': {'kind': 'graph', 'dim': 1, 'task_type': 'regression'},
        }
        model = MuseEncoder(small_cfg(aux_specs=aux_specs))
        batch = make_batch(length=4, lengths=[4, 3])
        token_value = torch.tensor([[0, 0, 1, 0], [0, 2, 0, 0]])
        token_mask = torch.tensor([[False, True, True, False],
                                   [False, True, False, False]])
        records = [
            {'name': 'multi', 'value': torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
             'mask': torch.tensor([[True, True, True], [False, False, False]])},
            {'name': 'token', 'value': token_value, 'mask': token_mask},
            {'name': 'legacy_reg', 'value': torch.randn(2, 1),
             'mask': torch.ones(2, 1, dtype=torch.bool)},
        ]
        targets = {'aux_records': records,
                   'drop_feature_mask': torch.zeros(2, dtype=torch.bool)}
        losses = model.pretraining_losses(batch, targets, {'aux': 1.0})
        self.assertTrue(torch.isfinite(losses['aux']))
        losses['total'].backward()
        for name in aux_specs:
            grad = model.heads.aux_heads[name].weight.grad
            self.assertIsNotNone(grad, name)
            self.assertGreater(grad.abs().sum().item(), 0.0, name)

        changed = dict(targets)
        changed['aux_records'] = [dict(record) for record in records]
        changed['aux_records'][1]['value'] = token_value.clone()
        changed['aux_records'][1]['value'][~token_mask] = 99
        changed_losses = model.pretraining_losses(batch, changed, {'aux': 1.0})
        torch.testing.assert_close(losses['aux'], changed_losses['aux'])
        invalid = dict(changed)
        invalid['aux_records'] = [dict(record) for record in changed['aux_records']]
        invalid['aux_records'][1]['value'] = changed['aux_records'][1]['value'].clone()
        invalid['aux_records'][1]['value'][0, 1] = 3
        with self.assertRaisesRegex(ValueError, '类别索引越界'):
            model.pretraining_losses(batch, invalid, {'aux': 1.0})

    def test_legacy_aux_props_uses_independent_head(self):
        model = MuseEncoder(small_cfg())
        batch = make_batch()
        losses = model.pretraining_losses(
            batch, {'aux_props': torch.randn(2, 4),
                    'drop_feature_mask': torch.zeros(2, dtype=torch.bool)}, {})
        losses['total'].backward()
        self.assertGreater(model.heads.aux_legacy.weight.grad.abs().sum().item(), 0.0)
        self.assertTrue(model.heads.descriptor_out.weight.grad is None or
                        model.heads.descriptor_out.weight.grad.abs().sum().item() == 0.0)

    def test_noise_coordinate_distance_finite_backward_and_invalid_exclusion(self):
        model = MuseEncoder(small_cfg())
        batch = make_batch(length=6, lengths=[6, 4])
        clean = torch.randn(2, 6, 3)
        noise = torch.randn(2, 6, 3)
        batch['coords'] = clean + 0.2 * noise
        batch['coord_mask'][:, 0] = True
        base_targets = {
            'noise': noise,
            'noise_mask': batch['coord_mask'],
            'coords': clean,
            'coord_mask': batch['coord_mask'],
            'drop_feature_mask': torch.zeros(2, dtype=torch.bool),
        }
        losses = model.pretraining_losses(batch, base_targets, {})
        for key in ('denoise', 'coordinate', 'distance', 'total'):
            self.assertIn(key, losses)
            self.assertTrue(torch.isfinite(losses[key]), key)

        changed = dict(base_targets)
        changed['coords'] = clean.clone()
        changed['noise'] = noise.clone()
        changed['coords'][:, 0] = 1e5
        changed['coords'][1, 4:] = -1e5
        changed['noise'][:, 0] = 1e5
        changed['noise'][1, 4:] = -1e5
        changed_losses = model.pretraining_losses(batch, changed, {})
        for key in ('denoise', 'coordinate', 'distance'):
            self.assertTrue(torch.allclose(losses[key], changed_losses[key], atol=1e-6, rtol=1e-6),
                            key)
        losses['total'].backward()
        self.assertGreater(model.heads.coordinate[-1].weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.heads.denoise[-1].weight.grad.abs().sum().item(), 0.0)

    def test_pair_distance_l1_computes_bfloat16_inputs_in_float32(self):
        from muse_encoder.model import _pair_distance_l1
        pred = torch.randn(2, 4, 3, dtype=torch.bfloat16, requires_grad=True)
        target = torch.randn(2, 4, 3, dtype=torch.bfloat16)
        valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
        loss = _pair_distance_l1(pred, target, valid)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(pred.grad.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_energy_force_stress_masks_and_infonce_validation(self):
        model = MuseEncoder(small_cfg())
        batch = make_batch(batch_size=4, length=4)
        energy_mask = torch.tensor([True, False, True, False])
        stress_mask = torch.tensor([True, False, False, True])
        targets = {
            'energy': torch.randn(4), 'energy_mask': energy_mask,
            'force': torch.randn(4, 4, 3), 'force_mask': batch['coord_mask'],
            'stress': torch.randn(4, 6), 'stress_mask': stress_mask,
            'drop_feature_mask': torch.zeros(4, dtype=torch.bool),
        }
        losses = model.pretraining_losses(batch, targets, {})
        changed = dict(targets)
        changed['energy'] = targets['energy'].clone()
        changed['energy'][~energy_mask] = 1e5
        changed['stress'] = targets['stress'].clone()
        changed['stress'][~stress_mask] = -1e5
        changed['force'] = targets['force'].clone()
        changed['force'][:, 0] = 1e5
        changed_losses = model.pretraining_losses(batch, changed, {})
        for key in ('energy', 'stress', 'force'):
            self.assertTrue(torch.allclose(losses[key], changed_losses[key], atol=1e-6, rtol=1e-6),
                            key)
        losses['total'].backward()

        valid = {'align_idx': torch.tensor([[0, 1], [2, 3]]),
                 'drop_feature_mask': torch.zeros(4, dtype=torch.bool)}
        self.assertTrue(torch.isfinite(model.pretraining_losses(batch, valid, {})['align']))
        for illegal in (torch.tensor([[0, 0]]), torch.tensor([[0, 1], [0, 2]]),
                        torch.tensor([[0, 4]])):
            with self.assertRaises(ValueError):
                model.pretraining_losses(
                    batch, {'align_idx': illegal,
                            'drop_feature_mask': torch.zeros(4, dtype=torch.bool)}, {})

    def test_mole_matches_per_sample_reference_and_exposes_gate_stats(self):
        mole = MoleFFN(d_model=8, d_ffn=12, num_experts=3, cond_dim=4, dropout=0.0).eval()
        x = torch.randn(4, 5, 8)
        cond = torch.randn(4, 4)
        cond[2] = cond[0]
        cond[3] = cond[1]
        with torch.no_grad():
            actual, stats = mole(x, cond, return_gate=True)
            w = F.softmax(mole.gate(cond), dim=-1)
            w1 = torch.einsum('be,efd->bfd', w, mole.w1)
            b1 = torch.einsum('be,ef->bf', w, mole.b1)
            hidden = torch.einsum('bld,bfd->blf', x, w1) + b1.unsqueeze(1)
            hidden = mole.act(hidden)
            w2 = torch.einsum('be,edf->bdf', w, mole.w2)
            b2 = torch.einsum('be,ed->bd', w, mole.b2)
            expected = torch.einsum('blf,bdf->bld', hidden, w2) + b2.unsqueeze(1)
            permuted = mole(x[[2, 0, 3, 1]], cond[[2, 0, 3, 1]])
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5),
                        (actual - expected).abs().max().item())
        self.assertTrue(torch.allclose(actual[[2, 0, 3, 1]], permuted, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(stats['weights'], w, atol=1e-6, rtol=1e-6))
        self.assertTrue((stats['entropy'] >= 0).all())

    def test_mole_grouped_backward_matches_per_sample_reference(self):
        mole = MoleFFN(d_model=8, d_ffn=12, num_experts=3, cond_dim=4, dropout=0.0)
        x = torch.randn(4, 5, 8)
        cond = torch.randn(4, 4, requires_grad=True)
        with torch.no_grad():
            cond.data[2] = cond.data[0]
            cond.data[3] = cond.data[1]
        actual = mole(x, cond)
        actual.sum().backward()
        expected_cond_grad = cond.grad.detach().clone()
        expected_parameter_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in mole.named_parameters()
        }
        mole.zero_grad()
        cond.grad = None
        w = F.softmax(mole.gate(cond), dim=-1)
        w1 = torch.einsum('be,efd->bfd', w, mole.w1)
        b1 = torch.einsum('be,ef->bf', w, mole.b1)
        hidden = torch.einsum('bld,bfd->blf', x, w1) + b1.unsqueeze(1)
        hidden = mole.act(hidden)
        w2 = torch.einsum('be,edf->bdf', w, mole.w2)
        b2 = torch.einsum('be,ed->bd', w, mole.b2)
        expected = torch.einsum('blf,bdf->bld', hidden, w2) + b2.unsqueeze(1)
        expected.sum().backward()
        torch.testing.assert_close(cond.grad, expected_cond_grad, rtol=1e-5, atol=1e-6)
        for name, parameter in mole.named_parameters():
            torch.testing.assert_close(parameter.grad, expected_parameter_grads[name],
                                       rtol=1e-5, atol=1e-6)

    def test_sequence_without_pair_pool_weights_and_no_noise_drop(self):
        model = MuseEncoder(small_cfg()).eval()
        batch = make_batch()
        for key in ('bond_mat', 'spd_mat', 'coords', 'coord_mask'):
            batch.pop(key)
        batch['token_kind'] = torch.full_like(batch['token_ids'], 3)
        batch['pool_weights'] = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        with torch.no_grad():
            output = model.encode(batch)
            self.assertEqual(output['embedding'].shape, (2, model.cfg.embed_dim))
            h = torch.randn(1, 3, model.cfg.d_model)
            pad = torch.ones(1, 3, dtype=torch.bool)
            weights = torch.tensor([[0.0, 1.0, 0.0]])
            pooled = model.readout_graph(h, pad, weights)
            expected = model.readout(torch.cat([h[:, 0], h[:, 1]], dim=-1))
            torch.testing.assert_close(pooled, expected)
            targets = {'descriptor': torch.randn(2, 208)}
            first = model.pretraining_losses(batch, targets, {})
            second = model.pretraining_losses(
                batch, {**targets, 'drop_feature_mask': torch.zeros(2, dtype=torch.bool)}, {})
        self.assertTrue(torch.allclose(first['total'], second['total'], atol=0, rtol=0))

    def test_crystal_composition_readout_preserves_other_domains_and_batch_independence(self):
        cfg = small_cfg(crystal_composition_features=True)
        legacy = MuseEncoder(small_cfg()).eval()
        model = MuseEncoder(cfg).eval()
        model.load_state_dict(legacy.state_dict(), strict=False)
        batch = make_batch()
        batch['domain'] = torch.tensor([2, 0])
        batch['global_features'] = torch.zeros(2, cfg.num_elements - 1)
        with torch.no_grad():
            actual = model.encode(batch)
            expected = legacy.encode(batch)
            torch.testing.assert_close(actual['embedding'], expected['embedding'])
            without_features = dict(batch)
            without_features.pop('global_features')
            torch.testing.assert_close(
                model.encode(without_features)['embedding'], expected['embedding'])
            batch['global_features'][0] = torch.randn(cfg.num_elements - 1)
            full = model.encode(batch)['embedding']
            single = model.encode(slice_record(batch, 0))['embedding']
            torch.testing.assert_close(full[0:1], single)
        losses = model.pretraining_losses(
            batch,
            {'energy': torch.zeros(2), 'energy_mask': torch.ones(2, dtype=torch.bool),
             'drop_feature_mask': torch.zeros(2, dtype=torch.bool)},
            {'energy': 10.0},
        )
        losses['total'].backward()
        self.assertGreater(model.composition_readout.weight.grad.abs().sum().item(), 0.0)

    def test_conservative_force_uses_energy_gradient_and_second_order(self):
        cfg = small_cfg(conservative_force=True)
        model = MuseEncoder(cfg)
        batch = make_batch(length=5, lengths=[5, 4])
        targets = {
            'force': torch.randn(2, 5, 3),
            'force_mask': batch['coord_mask'],
            'drop_feature_mask': torch.zeros(2, dtype=torch.bool),
        }
        losses = model.pretraining_losses(batch, targets, {'force': 30.0})
        self.assertIn('force', losses)
        self.assertTrue(torch.isfinite(losses['force']))
        losses['total'].backward()
        self.assertGreater(model.heads.energy[-1].weight.grad.abs().sum().item(), 0.0)
        direct_grad = model.heads.force.weight.grad
        self.assertTrue(direct_grad is None or direct_grad.abs().sum().item() == 0.0)

    def test_encode_eval_token_projection_and_batch_independence(self):
        cfg = small_cfg()
        model = MuseEncoder(cfg).eval()
        batch = make_batch(batch_size=3, length=24, lengths=[24, 17, 9])
        with torch.no_grad():
            full = model.encode(batch)
            self.assertEqual(full['tokens'].shape, (3, 24, cfg.embed_dim))
            for i in range(3):
                single = model.encode(slice_record(batch, i))
                self.assertTrue(torch.allclose(full['embedding'][i:i + 1], single['embedding'],
                                               atol=2e-5, rtol=2e-5))
                n = single['tokens'].shape[1]
                self.assertTrue(torch.allclose(full['tokens'][i:i + 1, :n], single['tokens'],
                                               atol=2e-5, rtol=2e-5))

    def test_sdpa_matches_manual_attention_forward_and_backward(self):
        from muse_encoder.layers import PairBiasAttention

        torch.manual_seed(7)
        kwargs = dict(d_model=32, n_heads=4, dropout=0.0, use_rope=True, max_len=64)
        manual = PairBiasAttention(**kwargs, use_sdpa=False).eval()
        sdpa = PairBiasAttention(**kwargs, use_sdpa=True).eval()
        sdpa.load_state_dict(manual.state_dict())
        x = torch.randn(2, 17, 32, requires_grad=True)
        pad_mask = torch.ones(2, 17, dtype=torch.bool)
        pad_mask[1, 11:] = False
        pair_bias = torch.randn(2, 4, 17, 17) * 0.3
        out_manual = manual(x, pad_mask, pair_bias)
        out_sdpa = sdpa(x, pad_mask, pair_bias)
        self.assertTrue(torch.allclose(out_manual, out_sdpa, atol=1e-5, rtol=1e-5))
        grad = torch.randn_like(out_manual)
        x_manual = x.detach().clone().requires_grad_(True)
        x_sdpa = x.detach().clone().requires_grad_(True)
        manual(x_manual, pad_mask, pair_bias).backward(grad)
        sdpa(x_sdpa, pad_mask, pair_bias).backward(grad)
        self.assertTrue(torch.allclose(x_manual.grad, x_sdpa.grad, atol=1e-4, rtol=1e-4))

    def test_rope_extends_without_new_parameters(self):
        from muse_encoder.layers import RotaryEmbedding

        extended = RotaryEmbedding(8, max_len=4)
        reference = RotaryEmbedding(8, max_len=9)
        x = torch.randn(1, 2, 9, 8)
        actual = extended(x)
        expected = reference(x)
        self.assertEqual(actual.shape, x.shape)
        self.assertEqual(extended.cos.shape[0], 9)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(extended.state_dict(), {})

    def test_gradient_checkpointing_preserves_numerics(self):
        torch.manual_seed(11)
        cfg = small_cfg()
        model_ref = MuseEncoder(cfg)
        model_ref.train()
        torch.manual_seed(11)
        cfg_ckpt = small_cfg(gradient_checkpointing=True)
        model_ckpt = MuseEncoder(cfg_ckpt)
        model_ckpt.train()
        model_ckpt.load_state_dict(model_ref.state_dict())
        torch.manual_seed(23)
        batch = make_batch(batch_size=2, length=24)
        targets = {
            'mlm_labels': torch.randint(1, 20, (2, 24)),
            'mlm_head': torch.ones(2, dtype=torch.long),
        }
        losses_ref = model_ref.pretraining_losses(batch, targets, {'mlm': 1.0})
        torch.manual_seed(23)
        batch2 = make_batch(batch_size=2, length=24)
        losses_ckpt = model_ckpt.pretraining_losses(batch2, targets, {'mlm': 1.0})
        self.assertTrue(torch.allclose(losses_ref['total'], losses_ckpt['total'],
                                       atol=1e-6, rtol=1e-6))
        losses_ref['total'].backward()
        losses_ckpt['total'].backward()
        for (name, p_ref), (_, p_ckpt) in zip(model_ref.named_parameters(),
                                              model_ckpt.named_parameters()):
            g_ref = p_ref.grad
            g_ckpt = p_ckpt.grad
            if g_ref is None and g_ckpt is None:
                continue
            self.assertTrue(torch.allclose(g_ref, g_ckpt, atol=1e-5, rtol=1e-5), name)


if __name__ == '__main__':
    unittest.main()
