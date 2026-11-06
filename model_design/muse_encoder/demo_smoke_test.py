"""端到端烟测：四域前向 → 六组预训练损失 → 冻结编码 → 合规校验。

用小配置（4 层）跑通全链路；正式训练把 MuseConfig 换成报告 §6.3 的完整尺寸。

运行：python -m muse_encoder.demo_smoke_test   （在 muse_encoder 的父目录下）
"""
from __future__ import annotations

import numpy as np
import torch

from muse_encoder import (MuseConfig, MuseEncoder, RecordFeaturizer, collate,
                          encode_subset, check_probe_params)


def small_cfg() -> MuseConfig:
    return MuseConfig(d_model=128, n_layers=4, n_heads=4, d_ffn=256, num_experts=4,
                      mole_every=2, embed_dim=128, contrastive_dim=64,
                      descriptor_dim=16, max_seq_len=128, dropout=0.0)


def make_records(fz: RecordFeaturizer):
    """四域各造一条合成记录（无 RDKit，走降级特征路径）。"""
    recs = []
    # 小分子：乙醇 SMILES
    recs.append(fz.featurize_molecule('CCO'))
    # 聚合物：PSMILES（聚乙烯重复单元）
    recs.append(fz.featurize_polymer(psmiles='[*]CC[*]'))
    # 晶体：2x2x2 NaCl 式小晶胞
    lattice = np.eye(3) * 5.64
    frac = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
                     [0.5, 0.5, 0.5], [0, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0]])
    elems = ['Na', 'Cl', 'Cl', 'Cl', 'Cl', 'Cl', 'Cl', 'Cl']
    recs.append(fz.featurize_crystal(lattice, frac, elems))
    # 蛋白序列
    recs.append(fz.featurize_sequence('MKALIVLGTV'))
    return recs


def to_torch(batch):
    return {k: torch.from_numpy(v) for k, v in batch.items()}


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = small_cfg()
    fz = RecordFeaturizer(cfg)
    model = MuseEncoder(cfg)

    # ---------- 1. 前向 + 读出形状 ----------
    recs = make_records(fz)
    batch = to_torch(collate(recs, cfg))
    out = model.encode(batch)
    assert out['embedding'].shape == (4, cfg.embed_dim), out['embedding'].shape
    assert out['tokens'].shape[0] == 4
    print('[1] encode 形状 OK:  embedding', tuple(out['embedding'].shape),
          ' tokens', tuple(out['tokens'].shape))

    # ---------- 2. 批次无关性（同一记录换 batch 组成，表征须逐位一致） ----------
    model.eval()
    b_single = to_torch(collate([recs[0]], cfg))
    with torch.no_grad():
        e_single = model.encode(b_single)['embedding']
        e_in_batch = model.encode(batch)['embedding'][0:1]
    diff = (e_single - e_in_batch).abs().max().item()
    assert diff < 1e-5, f'批次无关性被破坏: {diff}'
    print(f'[2] 批次无关性 OK:  max|Δ| = {diff:.2e}')

    # ---------- 3. 六组预训练损失 ----------
    model.train()
    B, L = batch['token_ids'].shape
    mlm_labels = torch.full((B, L), -100, dtype=torch.long)
    mlm_labels[:, 1:6] = batch['token_ids'][:, 1:6]   # 伪标签：重建前几个 token
    mlm_head = torch.tensor([3, 3, 3, 2])             # 原子域用 element 头，蛋白用 residue 头
    targets = {
        'mlm_labels': mlm_labels, 'mlm_head': mlm_head,
        'noise': torch.randn(B, L, 3), 'noise_mask': batch['coord_mask'],
        'energy': torch.randn(B),
        'force': torch.randn(B, L, 3), 'force_mask': batch['coord_mask'],
        'stress': torch.randn(B, 6),
        'descriptor': torch.randn(B, cfg.descriptor_dim),
        'align_idx': torch.tensor([[0, 1], [2, 3]]),
        'aux_props': torch.randn(B, 4),
    }
    weights = {'mlm': 1.0, 'denoise': 1.0, 'energy': 1.0, 'force': 30.0,
               'stress': 1.0, 'descriptor': 1.5, 'align': 1.0, 'aux': 1.0}
    losses = model.pretraining_losses(batch, targets, weights)
    for k in ('mlm', 'denoise', 'energy', 'force', 'stress', 'descriptor',
              'align', 'aux', 'total'):
        assert k in losses and torch.isfinite(losses[k]), k
    losses['total'].backward()
    gnorm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert np.isfinite(gnorm) and gnorm > 0
    print('[3] 六组预训练损失 + 反向传播 OK:  '
          + '  '.join(f'{k}={losses[k].item():.3f}' for k in
                      ('mlm', 'denoise', 'energy', 'force', 'descriptor', 'total')))

    # ---------- 4. 推理接口（embeddings.npy / tokens.npy） ----------
    model.eval()
    # 4a. 普通子集（分子）：只出 embeddings
    mol_records = [
        {'id': 'mol_0', 'sample_type': 'molecule', 'inputs': {'smiles': 'CCO'}},
        {'id': 'mol_1', 'sample_type': 'molecule', 'inputs': {'smiles': 'c1ccccc1'}},
    ]
    ids, embs, _, _ = encode_subset(model, fz, cfg, mol_records, batch_size=2,
                                    token_level=False)
    assert embs.shape == (2, cfg.embed_dim) and embs.dtype == np.float32
    assert np.isfinite(embs).all()
    # 4b. 逐残基子集（蛋白）：embeddings + tokens + offsets
    prot_records = [
        {'id': 'prot_0', 'sample_type': 'protein', 'inputs': {'sequence': 'MKALIVLG'}},
        {'id': 'prot_1', 'sample_type': 'protein', 'inputs': {'sequence': 'GGCCPPTT'}},
    ]
    ids, embs2, tokens, offsets = encode_subset(model, fz, cfg, prot_records,
                                                batch_size=2, token_level=True)
    L_tot = sum(len(r['inputs']['sequence']) for r in prot_records)
    assert tokens.shape == (L_tot, cfg.embed_dim), (tokens.shape, L_tot)
    assert offsets.tolist() == [0, 8, 16], offsets
    print(f'[4] 推理接口 OK:  embeddings {embs.shape}/{embs2.shape}, '
          f'tokens {tokens.shape}, offsets {offsets.tolist()}')

    # ---------- 5. MLP 探针合规（≤2 层、≤100K，D=1024 正式配置） ----------
    n_reg = check_probe_params(1024, 48, 1)
    n_multi = check_probe_params(1024, 48, 21)          # CATH 多分类级别
    print(f'[5] MLP 合规 OK:  回归头 {n_reg} 参数,  21 类分类头 {n_multi} 参数 (上限 100K)')

    # ---------- 6. 参数量（完整配置，meta device 计数不占内存） ----------
    full = MuseConfig()
    with torch.device('meta'):
        m_full = MuseEncoder(full)
    n_full = sum(p.numel() for p in m_full.parameters())
    del m_full
    print(f'[6] 完整配置参数量: {n_full/1e6:.1f}M  '
          f'(主干 24L/1024d + 8 专家 MoLE, 规划量级 ~300-400M)')

    # ---------- 7. 掩码策略（序列 MLM / 原子 / 肽结构语义，PepLand） ----------
    from muse_encoder import apply_masking
    g = torch.Generator().manual_seed(1)
    seq_ids = batch['token_ids'][:2].clone()           # 原子级记录当序列演示 MLM 接口
    n_elig = batch['pad_mask'][:2, 1:].sum().item()
    hits = 0
    for _ in range(200):                                # 记录很短，多次采样看期望掩码率
        _, l_seq = apply_masking('sequence', seq_ids, batch['pad_mask'][:2],
                                 mask_id=2, vocab_size=128, generator=g)
        hits += int((l_seq != -100).sum())
    ratio = hits / (200 * n_elig)
    assert 0.08 < ratio < 0.25, ratio
    # 肽：侧链标注可用 → sidechain/fragment/bulk 混合；缺失标注 → span 降级
    pept_ids = batch['token_ids'][0:1].clone()
    grp = torch.tensor([[ -1, 0, 0, 1, 1, 2, 2, -1, -1, -1, -1]])   # 3 个残基
    scf = torch.tensor([[False, False, True, False, True, False, True,
                         False, False, False, False]])
    frg = torch.tensor([[ -1, 0, 0, 1, 1, 1, 2, -1, -1, -1, -1]])
    m_p, l_p = apply_masking('peptide', pept_ids, batch['pad_mask'][0:1],
                             group_ids=grp, sidechain_flags=scf, fragment_ids=frg,
                             generator=g)
    assert (l_p != -100).any() and l_p[0, 0].item() == -100   # CLS 永不掩
    m_p2, l_p2 = apply_masking('peptide', pept_ids, batch['pad_mask'][0:1],
                               generator=g)                    # 无标注 → span 降级
    assert (l_p2 != -100).any()
    n_masked_atoms = int((l_p != -100).sum())
    print(f'[7] 掩码策略 OK:  序列MLM率 {ratio:.2f},  肽掩码 {n_masked_atoms} 原子 '
          f'(含降级路径),  CLS 不掩 ✓')

    # ---------- 8. GRU 池化读出（PepLand 证据的可选读出） ----------
    cfg_gru = small_cfg(); cfg_gru.pooling = 'gru'
    m_gru = MuseEncoder(cfg_gru).eval()
    out_gru = m_gru.encode(batch)
    assert out_gru['embedding'].shape == (4, cfg_gru.embed_dim)
    with torch.no_grad():
        d_gru = (m_gru.encode(to_torch(collate([recs[0]], cfg_gru)))['embedding']
                 - out_gru['embedding'][0:1]).abs().max().item()
    assert d_gru < 1e-4, d_gru
    print(f'[8] GRU 池化读出 OK:  embedding {tuple(out_gru["embedding"].shape)}, '
          f'批次无关 max|Δ|={d_gru:.1e}')

    print('\n全部烟测通过 ✓')


if __name__ == '__main__':
    main()
