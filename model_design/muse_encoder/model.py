"""MuseEncoder：四域前端 + 共享 Transformer 主干 + 统一 D 读出 + 预训练头。

对应报告 §5.2 / §6：
- 前端把各域输入映射为同一维度的 token 序列 + 对偏置矩阵 + 域/模态嵌入；
- 主干为 pre-norm Transformer（几何感知注意力 + 按域软路由的 MoLE）；
- 图与逐 token 读出分别投影到统一 embed_dim，推理不含批次统计量；
- 预训练覆盖 MLM、噪声/坐标/距离、能量·力·应力、双读出 MTR、对比与 aux。
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .config import MuseConfig
from .layers import Block


def _broadcast_mask(mask, reference: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(reference, dtype=torch.bool)
    mask = mask.to(device=reference.device, dtype=torch.bool)
    if mask.dim() == reference.dim() + 1 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if mask.shape == reference.shape[:-1]:
        mask = mask.unsqueeze(-1)
    try:
        return torch.broadcast_to(mask, reference.shape)
    except RuntimeError as exc:
        raise ValueError(f'掩码形状 {tuple(mask.shape)} 无法匹配标签 {tuple(reference.shape)}') from exc


def _effective_atom_mask(mask: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(dtype=torch.bool) & pad_mask.to(dtype=torch.bool)
    valid = valid.clone()
    valid[:, 0] = False
    return valid


def _kabsch_l1(pred: torch.Tensor, target: torch.Tensor,
               valid: torch.Tensor):
    """逐记录只用有效原子做带反射约束的 Kabsch 对齐，再计算坐标 L1。"""
    total = torch.zeros((), device=pred.device, dtype=torch.float32)
    count = 0
    for b in range(pred.shape[0]):
        m = valid[b]
        if not m.any():
            continue
        p = pred[b, m].float()
        q = target[b, m].float()
        p_center = p.mean(dim=0)
        q_center = q.mean(dim=0)
        p0 = p - p_center
        q0 = q - q_center
        if p.shape[0] == 1:
            aligned = p0 + q_center
        else:
            u, _, vh = torch.linalg.svd(p0.transpose(0, 1) @ q0)
            rotation = u @ vh
            if torch.det(rotation) < 0:
                u = u.clone()
                u[:, -1] = -u[:, -1]
                rotation = u @ vh
            aligned = p0 @ rotation + q_center
        total = total + (aligned - q).abs().sum()
        count += q.numel()
    if count == 0:
        return None
    return total / count


def _pair_distance_l1(pred: torch.Tensor, target: torch.Tensor,
                      valid: torch.Tensor):
    pair_mask = valid[:, :, None] & valid[:, None, :]
    pair_mask = torch.triu(pair_mask, diagonal=1)
    if not pair_mask.any():
        return None
    zero_pred = torch.zeros_like(pred)
    zero_target = torch.zeros_like(target)
    safe_pred = torch.where(valid.unsqueeze(-1), pred, zero_pred)
    safe_target = torch.where(valid.unsqueeze(-1), target, zero_target)
    pred_dist = torch.cdist(safe_pred.float(), safe_pred.float())
    target_dist = torch.cdist(safe_target.float(), safe_target.float())
    return (pred_dist - target_dist).abs()[pair_mask].mean()


# ---------------------------------------------------------------- 对偏置编码器
class PairBiasEncoder(nn.Module):
    """bond-type、SPD 桶、距离高斯基与可选 PBC 平移 → [B,H,L,L] 加性偏置。"""

    def __init__(self, cfg: MuseConfig):
        super().__init__()
        self.cfg = cfg
        self.bond_emb = nn.Embedding(cfg.num_bond_types, cfg.n_heads)
        self.spd_emb = nn.Embedding(cfg.num_spd_buckets, cfg.n_heads)
        self.dist_centers = nn.Parameter(torch.linspace(0.0, 10.0, cfg.num_dist_basis),
                                         requires_grad=False)
        self.dist_gamma = 10.0
        self.dist_proj = nn.Linear(cfg.num_dist_basis, cfg.n_heads)
        self.pbc_shift_proj = nn.Linear(3, cfg.n_heads, bias=False)
        self.multiplicity_emb = nn.Embedding(cfg.crystal_max_neighbors + 1, cfg.n_heads)

    def forward(self, bond_mat, spd_mat, coords, coord_mask, mask_pair=None,
                pair_disp=None, pbc_shift=None, drop_feature_mask=None,
                pair_basis=None, pair_multiplicity=None, pair_basis_mask=None,
                pair_multiplicity_mask=None):
        """mask_pair 为旧参数名；drop_feature_mask 可显式覆盖它。"""
        B, L, _ = bond_mat.shape
        bias = self.bond_emb(bond_mat.clamp(0, self.cfg.num_bond_types - 1)) \
             + self.spd_emb(spd_mat.clamp(0, self.cfg.num_spd_buckets - 1))
        if pair_basis is None or pair_basis_mask is not None:
            if pair_disp is None:
                distance = torch.cdist(coords, coords)
            else:
                if pair_disp.shape != (B, L, L, 3):
                    raise ValueError(f'pair_disp 应为 {(B, L, L, 3)}，实际为 {tuple(pair_disp.shape)}')
                distance = torch.linalg.vector_norm(pair_disp, dim=-1)
            dynamic_basis = torch.exp(-self.dist_gamma * (distance.unsqueeze(-1) - self.dist_centers) ** 2)
            if pair_basis is None:
                pair_basis = dynamic_basis
            else:
                if pair_basis_mask.shape != (B,):
                    raise ValueError(f'pair_basis_mask 应为 {(B,)}')
                pair_basis = torch.where(pair_basis_mask.bool()[:, None, None, None],
                                         pair_basis, dynamic_basis)
        if pair_basis.shape != (B, L, L, self.cfg.num_dist_basis):
            raise ValueError(f'pair_basis 应为 {(B, L, L, self.cfg.num_dist_basis)}，实际为 {tuple(pair_basis.shape)}')
        pair_basis = pair_basis * coord_mask[:, :, None, None] * coord_mask[:, None, :, None]
        bias = bias + self.dist_proj(pair_basis)
        if pair_multiplicity is not None:
            multiplicity_bias = self.multiplicity_emb(
                pair_multiplicity.long().clamp(0, self.cfg.crystal_max_neighbors))
            if pair_multiplicity_mask is not None:
                if pair_multiplicity_mask.shape != (B,):
                    raise ValueError(f'pair_multiplicity_mask 应为 {(B,)}')
                multiplicity_bias = multiplicity_bias * pair_multiplicity_mask.bool()[:, None, None, None]
            bias = bias + multiplicity_bias
        bias = bias.permute(0, 3, 1, 2).contiguous()
        if pbc_shift is not None:
            if pbc_shift.shape != (B, L, L, 3):
                raise ValueError(f'pbc_shift 应为 {(B, L, L, 3)}，实际为 {tuple(pbc_shift.shape)}')
            bias = bias + self.pbc_shift_proj(pbc_shift).permute(0, 3, 1, 2)
        drop = drop_feature_mask if drop_feature_mask is not None else mask_pair
        if drop is not None:
            drop = drop.to(device=bias.device, dtype=torch.bool)
            if drop.shape != (B,):
                raise ValueError(f'drop_feature_mask 应为 {(B,)}，实际为 {tuple(drop.shape)}')
            bias = bias * (~drop)[:, None, None, None]
        return bias


# ---------------------------------------------------------------- 预训练头
class PretrainHeads(nn.Module):
    def __init__(self, cfg: MuseConfig):
        super().__init__()
        D = cfg.d_model
        E = cfg.embed_dim
        hidden = max(1, D // 2)
        self.mlm_smiles = nn.Linear(D, cfg.smiles_vocab_size)
        self.mlm_psmiles = nn.Linear(D, cfg.psmiles_vocab_size)
        self.mlm_residue = nn.Linear(D, cfg.residue_vocab_size)
        self.mlm_element = nn.Linear(D, cfg.num_elements)
        self.denoise = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, 3))
        self.coordinate = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, 3))
        self.energy = nn.Sequential(nn.Linear(E, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.force = nn.Linear(D, 3)
        self.stress = nn.Sequential(nn.Linear(E, hidden), nn.SiLU(), nn.Linear(hidden, 6))
        self.descriptor_mid = nn.Linear(E, cfg.descriptor_dim)
        self.descriptor_out = nn.Linear(E, cfg.descriptor_dim)
        self.aux_legacy = nn.Linear(E, cfg.descriptor_dim)
        self.aux_specs = {}
        self.aux_heads = nn.ModuleDict()
        for name, raw_spec in cfg.aux_specs.items():
            if not name or '.' in name:
                raise ValueError(f'aux 任务名不能为空或包含点号: {name!r}')
            kind = raw_spec['kind']
            task_type = raw_spec['task_type']
            dim = int(raw_spec['dim'])
            if kind not in ('graph', 'atom', 'bond'):
                raise ValueError(f'aux[{name}] kind 必须为 graph/atom/bond')
            if task_type not in ('regression', 'vector_regression', 'atom_regression',
                                 'bond_regression', 'binary_classification',
                                 'multiclass_classification', 'sequence_classification'):
                raise ValueError(f'aux[{name}] 不支持 task_type={task_type}')
            if dim <= 0:
                raise ValueError(f'aux[{name}] dim 必须为正数')
            if task_type == 'multiclass_classification' and (kind != 'graph' or dim < 2):
                raise ValueError(f'aux[{name}] 多分类需要 graph kind 且 dim>=2')
            if task_type == 'sequence_classification' and (kind != 'atom' or dim < 2):
                raise ValueError(f'aux[{name}] 逐 token 分类需要 atom kind 且 dim>=2')
            self.aux_specs[name] = {'kind': kind, 'dim': dim, 'task_type': task_type}
            self.aux_heads[name] = nn.Linear(E if kind == 'graph' else D, dim)
        self.proj = nn.Sequential(nn.Linear(E, E), nn.SiLU(),
                                  nn.Linear(E, cfg.contrastive_dim))

    @property
    def aux(self):
        return self.aux_heads


# ---------------------------------------------------------------- 主模型
class MuseEncoder(nn.Module):
    def __init__(self, cfg: MuseConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model

        self.element_emb = nn.Embedding(cfg.num_elements, D)
        self.smiles_emb = nn.Embedding(cfg.smiles_vocab_size, D)
        self.psmiles_emb = nn.Embedding(cfg.psmiles_vocab_size, D)
        self.residue_emb = nn.Embedding(cfg.residue_vocab_size, D)
        self.atom_feature_embs = nn.ModuleList([
            nn.Embedding(size, D) for size in cfg.atom_feature_vocab_sizes
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.domain_emb = nn.Embedding(cfg.num_domains, D)
        self.modality_emb = nn.Embedding(cfg.num_modalities, D)
        self.segment_emb = nn.Embedding(cfg.num_segments, D)
        self.spacegroup_emb = nn.Embedding(cfg.num_spacegroups, D)
        self.scalar_cond = nn.Linear(cfg.scalar_cond_dim, D)
        self.sigma_cond = nn.Sequential(nn.Linear(1, D), nn.SiLU(), nn.Linear(D, D))

        self.pair_encoder = PairBiasEncoder(cfg)
        self.blocks = nn.ModuleList([Block(cfg, i, cond_dim=D) for i in range(cfg.n_layers)])
        self.final_ln = nn.LayerNorm(D, eps=cfg.layer_norm_eps)

        self.readout = nn.Linear(2 * D, cfg.embed_dim)
        self.composition_readout = (
            nn.Linear(cfg.num_elements - 1, cfg.embed_dim, bias=False)
            if cfg.crystal_composition_features else None
        )
        self.token_readout = nn.Linear(D, cfg.embed_dim)
        if cfg.pooling == 'gru':
            self.gru = nn.GRU(D, D, batch_first=True)
        else:
            self.gru = None

        self.heads = PretrainHeads(cfg)
        self.dropout = nn.Dropout(cfg.dropout)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # ---------------------------------------------------------------- 前端嵌入
    def embed_tokens(self, batch: Dict[str, torch.Tensor],
                     drop_feature_mask=None) -> torch.Tensor:
        """支持旧记录级前端与 token_kind 混合前端；位置 0 使用共享 [CLS]。"""
        tok = batch['token_ids'].long()
        domain = batch['domain'].long()
        B, L = tok.shape
        D = self.cfg.d_model
        if 'token_kind' in batch:
            kind = batch['token_kind'].long()
            if kind.shape != tok.shape:
                raise ValueError('token_kind 必须与 token_ids 同为 [B,L]')
        else:
            is_atom = batch['use_atom_features'].bool()
            record_kind = torch.full_like(domain, 3)
            record_kind = torch.where(domain == 0, torch.ones_like(record_kind), record_kind)
            record_kind = torch.where(domain == 1, torch.full_like(record_kind, 2), record_kind)
            record_kind = torch.where(is_atom, torch.zeros_like(record_kind), record_kind)
            kind = record_kind[:, None].expand(B, L)
        if ((kind < 0) | (kind > 3)).any():
            raise ValueError('token_kind 只能取 0=element/1=smiles/2=psmiles/3=residue')

        x = torch.zeros(B, L, D, device=tok.device, dtype=self.element_emb.weight.dtype)
        tables = ((0, self.element_emb, self.cfg.num_elements),
                  (1, self.smiles_emb, self.cfg.smiles_vocab_size),
                  (2, self.psmiles_emb, self.cfg.psmiles_vocab_size),
                  (3, self.residue_emb, self.cfg.residue_vocab_size))
        for kind_id, embedding, vocab_size in tables:
            selected = kind == kind_id
            if selected.any():
                x[selected] = embedding(tok[selected].clamp(0, vocab_size - 1))
        x[:, 0] = self.cls_token.squeeze(0)

        drop = drop_feature_mask
        if drop is None:
            drop = batch.get('drop_feature_mask')
        if drop is not None:
            drop = drop.to(device=tok.device, dtype=torch.bool)
            if drop.shape != (B,):
                raise ValueError(f'drop_feature_mask 应为 {(B,)}，实际为 {tuple(drop.shape)}')

        if 'atom_features' in batch:
            features = batch['atom_features'].long()
            expected = (B, L, len(self.atom_feature_embs))
            if features.shape != expected:
                raise ValueError(f'atom_features 应为 {expected}，实际为 {tuple(features.shape)}')
            feature_x = torch.zeros_like(x)
            for i, embedding in enumerate(self.atom_feature_embs):
                ids = features[..., i].clamp(0, embedding.num_embeddings - 1)
                feature_x = feature_x + embedding(ids)
            atom_valid = batch.get('atom_feature_mask')
            if atom_valid is None:
                atom_valid = (kind == 0).clone()
            else:
                if atom_valid.shape != (B, L):
                    raise ValueError(f'atom_feature_mask 应为 {(B, L)}')
                atom_valid = atom_valid.bool().clone()
            atom_valid[:, 0] = False
            if 'pad_mask' in batch:
                atom_valid &= batch['pad_mask'].bool()
            if drop is not None:
                atom_valid &= (~drop)[:, None]
            x = x + feature_x * atom_valid.unsqueeze(-1)

        x = x + self.domain_emb(domain)[:, None, :] \
              + self.modality_emb(batch['modality'].long())[:, None, :]
        if 'segment_ids' in batch:
            if batch['segment_ids'].shape != tok.shape:
                raise ValueError('segment_ids 必须与 token_ids 同为 [B,L]')
            x = x + self.segment_emb(batch['segment_ids'].long().clamp(0, self.cfg.num_segments - 1))
        if 'spacegroup_id' in batch:
            x = x + self.spacegroup_emb(
                batch['spacegroup_id'].long().clamp(0, self.cfg.num_spacegroups - 1))[:, None, :]
        if 'scalar_cond' in batch:
            scalar = batch['scalar_cond'].to(dtype=x.dtype)
            if scalar.dim() == 1 and self.cfg.scalar_cond_dim == 1:
                scalar = scalar.unsqueeze(-1)
            x = x + self.scalar_cond(scalar)[:, None, :]
        if 'denoise_sigma' in batch:
            sigma = batch['denoise_sigma'].to(dtype=x.dtype)
            if sigma.dim() == 1:
                sigma = sigma.unsqueeze(-1)
            if sigma.shape != (B, 1):
                raise ValueError(f'denoise_sigma 应为 [B] 或 {(B, 1)}')
            active = batch.get('denoise_active')
            if active is None:
                active = torch.ones(B, dtype=torch.bool, device=tok.device)
            if active.shape != (B,):
                raise ValueError(f'denoise_active 应为 {(B,)}')
            x = x + self.sigma_cond(sigma)[:, None, :] * active.bool()[:, None, None]
        return self.dropout(x)

    # ---------------------------------------------------------------- 主干前向
    def forward_backbone(self, batch: Dict[str, torch.Tensor],
                         mask_pair_prob: float = 0.0,
                         return_mid: bool = False,
                         drop_feature_mask=None):
        B = batch['token_ids'].shape[0]
        drop = drop_feature_mask
        if drop is None:
            drop = batch.get('drop_feature_mask')
        if drop is None:
            drop = torch.rand(B, device=batch['token_ids'].device) < mask_pair_prob
        else:
            drop = drop.to(device=batch['token_ids'].device, dtype=torch.bool)
            if drop.shape != (B,):
                raise ValueError(f'drop_feature_mask 应为 {(B,)}，实际为 {tuple(drop.shape)}')
        x = self.embed_tokens(batch, drop_feature_mask=drop)
        pair_bias = None
        if 'bond_mat' in batch and 'spd_mat' in batch:
            coords = batch.get('coords')
            if coords is None:
                coords = torch.zeros((*batch['token_ids'].shape, 3),
                                     device=x.device, dtype=x.dtype)
            coord_mask = batch.get('coord_mask')
            if coord_mask is None:
                coord_mask = torch.zeros_like(batch['pad_mask'], dtype=torch.bool)
            pair_basis = batch.get('pair_basis')
            pair_basis_mask = batch.get('pair_basis_mask')
            pair_disp = batch.get('pair_disp')
            if self.cfg.conservative_force and coords.requires_grad:
                pair_basis = None
                pair_basis_mask = None
                pair_disp = coords[:, None, :, :] - coords[:, :, None, :]
                if 'pbc_shift' in batch and 'lattice' in batch:
                    shift_cart = torch.einsum(
                        'bxyi,bjk->bxyk',
                        batch['pbc_shift'].to(dtype=coords.dtype),
                        batch['lattice'].to(dtype=coords.dtype),
                    )
                    pair_disp = pair_disp + shift_cart
            pair_bias = self.pair_encoder(
                batch['bond_mat'], batch['spd_mat'], coords, coord_mask,
                mask_pair=drop, pair_disp=pair_disp,
                pbc_shift=batch.get('pbc_shift'), pair_basis=pair_basis,
                pair_multiplicity=batch.get('pair_multiplicity'),
                pair_basis_mask=pair_basis_mask,
                pair_multiplicity_mask=batch.get('pair_multiplicity_mask'))
        cond = self.domain_emb(batch['domain'].long())
        mid = None
        mid_idx = max(0, self.cfg.n_layers // 2 - 1)
        for i, blk in enumerate(self.blocks):
            if self.cfg.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, batch['pad_mask'], pair_bias, cond, use_reentrant=False)
            else:
                x = blk(x, batch['pad_mask'], pair_bias, cond)
            if return_mid and i == mid_idx:
                mid = x
        return self.final_ln(x), mid

    # ---------------------------------------------------------------- 读出
    def readout_graph(self, h: torch.Tensor, pad_mask: torch.Tensor,
                      pool_weights=None, global_features=None) -> torch.Tensor:
        """[CLS] ⊕ mean-pool（或 GRU 终态）→ 统一表征，只统计有效 token。"""
        cls = h[:, 0]
        m = pad_mask.clone()
        m[:, 0] = False
        if self.gru is not None:
            lengths = m.sum(1).clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                h[:, 1:], lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hT = self.gru(packed)
            pooled = hT.squeeze(0)
            output = self.readout(torch.cat([cls, pooled], dim=-1))
        else:
            weights = m.to(dtype=h.dtype) if pool_weights is None \
                else pool_weights.to(dtype=h.dtype) * m.to(dtype=h.dtype)
            denom = weights.sum(1, keepdim=True).clamp(min=1e-12)
            mean = (h * weights.unsqueeze(-1)).sum(1) / denom
            output = self.readout(torch.cat([cls, mean], dim=-1))
        if self.composition_readout is not None:
            expected = (h.shape[0], self.cfg.num_elements - 1)
            if global_features is None:
                global_features = output.new_zeros(expected)
            if global_features.shape != expected:
                raise ValueError(f'global_features 应为 {expected}，实际为 {tuple(global_features.shape)}')
            output = output + self.composition_readout(global_features.to(dtype=output.dtype))
        return output

    def encode(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """推理入口：embedding/tokens 分别经独立投影，均为 [..., embed_dim]。"""
        h, _ = self.forward_backbone(batch, mask_pair_prob=0.0)
        return {'embedding': self.readout_graph(
                    h, batch['pad_mask'], batch.get('pool_weights'), batch.get('global_features')),
                'tokens': self.token_readout(h)}

    # ---------------------------------------------------------------- 预训练损失
    def pretraining_losses(self, batch: Dict[str, torch.Tensor],
                           targets: Dict[str, torch.Tensor],
                           weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """按可用标签计算加权损失；targets/batch 可显式给出 drop_feature_mask。

        新坐标标签为 targets['coords']（也接受 clean_coords/coordinate），noise 仍是
        加在 batch 已加噪坐标上的真实噪声。aux_records 是 {name,value,mask} 的列表。
        """
        weights = weights or {}
        if self.cfg.conservative_force and 'force' in targets:
            batch = dict(batch)
            if 'coords' not in batch or 'coord_mask' not in batch:
                raise ValueError('保守力阶段需要 coords/coord_mask')
            batch['coords'] = batch['coords'].detach().requires_grad_(True)
        drop = targets.get('drop_feature_mask', batch.get('drop_feature_mask'))
        if drop is None and 'noise' not in targets:
            drop = torch.zeros(batch['token_ids'].shape[0], dtype=torch.bool,
                               device=batch['token_ids'].device)
        h, mid = self.forward_backbone(
            batch, mask_pair_prob=self.cfg.pair_mask_prob, return_mid=True,
            drop_feature_mask=drop)
        g = self.readout_graph(
            h, batch['pad_mask'], batch.get('pool_weights'), batch.get('global_features'))
        out: Dict[str, torch.Tensor] = {}

        if 'mlm_labels' in targets:
            labels = targets['mlm_labels'].long()
            if 'mlm_head' in targets:
                head_ids = targets['mlm_head'].long()
            elif 'token_kind' in batch:
                head_map = torch.tensor([3, 0, 1, 2], device=h.device)
                head_ids = head_map[batch['token_kind'].long()]
            else:
                head_ids = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)
            loss, count = h.new_zeros(()), 0
            for hid, head in ((0, self.heads.mlm_smiles), (1, self.heads.mlm_psmiles),
                              (2, self.heads.mlm_residue), (3, self.heads.mlm_element)):
                selected = head_ids == hid
                selected_labels = labels[selected]
                if selected_labels.ne(-100).any():
                    logits = head(h[selected])
                    loss = loss + F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), selected_labels.reshape(-1),
                        ignore_index=-100)
                    count += 1
            if count:
                out['mlm'] = loss / count * weights.get('mlm', 1.0)

        if 'noise' in targets:
            pred = self.heads.denoise(h)
            m = _effective_atom_mask(
                targets.get('noise_mask', batch['coord_mask']), batch['pad_mask'])
            if m.any():
                diff = (pred - targets['noise']) ** 2
                value = (diff * m.unsqueeze(-1)).sum() / (m.sum() * 3.0)
                out['denoise'] = value * weights.get('denoise', 1.0)

        coord_target = None
        for key in ('coords', 'clean_coords', 'coordinate'):
            if key in targets:
                coord_target = targets[key]
                break
        if coord_target is not None:
            pred = self.heads.coordinate(h)
            m = _effective_atom_mask(
                targets.get('coord_mask', batch['coord_mask']), batch['pad_mask'])
            with torch.autocast(device_type=h.device.type, enabled=False):
                coordinate = _kabsch_l1(pred, coord_target, m)
                distance = _pair_distance_l1(pred, coord_target, m)
            if coordinate is not None:
                out['coordinate'] = coordinate * weights.get('coordinate', 1.0)
            if distance is not None:
                out['distance'] = distance * weights.get('distance', 1.0)

        if 'energy' in targets:
            pred = self.heads.energy(g).squeeze(-1)
            y = targets['energy']
            m = _broadcast_mask(targets.get('energy_mask'), y)
            if m.any():
                value = (F.huber_loss(pred, y, delta=1.0, reduction='none') * m).sum() / m.sum()
                out['energy'] = value * weights.get('energy', 1.0)
        if 'force' in targets:
            m = _effective_atom_mask(
                targets.get('force_mask', batch['coord_mask']), batch['pad_mask'])
            if m.any():
                if self.cfg.conservative_force:
                    potential = self.heads.energy(g).sum()
                    pred = -torch.autograd.grad(
                        potential, batch['coords'], create_graph=True,
                        retain_graph=True,
                    )[0]
                else:
                    pred = self.heads.force(h)
                diff = (pred - targets['force']) ** 2
                value = (diff * m.unsqueeze(-1)).sum() / (m.sum() * 3.0)
                out['force'] = value * weights.get('force', 30.0)
        if 'stress' in targets:
            pred = self.heads.stress(g)
            y = targets['stress']
            m = _broadcast_mask(targets.get('stress_mask'), y)
            if m.any():
                value = (F.huber_loss(pred, y, delta=1.0, reduction='none') * m).sum() / m.sum()
                out['stress'] = value * weights.get('stress', 1.0)

        if 'descriptor' in targets:
            y = targets['descriptor']
            m = _broadcast_mask(targets.get('descriptor_mask'), y)

            def descriptor_loss(pred):
                return (F.mse_loss(pred, y, reduction='none') * m).sum() / m.sum()

            if m.any():
                loss = descriptor_loss(self.heads.descriptor_out(g))
                if mid is not None:
                    g_mid = self.readout_graph(
                        mid, batch['pad_mask'], batch.get('pool_weights'), batch.get('global_features'))
                    loss = loss + descriptor_loss(self.heads.descriptor_mid(g_mid))
                out['descriptor'] = loss * weights.get('descriptor', 1.0)

        if 'align_idx' in targets:
            idx = targets['align_idx'].long()
            if idx.numel():
                if idx.dim() != 2 or idx.shape[1] != 2:
                    raise ValueError('align_idx 必须为 [N,2]')
                if int(idx.min()) < 0 or int(idx.max()) >= h.shape[0]:
                    raise ValueError('align_idx 含越界样本索引')
                if (idx[:, 0] == idx[:, 1]).any():
                    raise ValueError('InfoNCE 正样本对不允许自配对')
                if torch.unique(idx).numel() != idx.numel():
                    raise ValueError('InfoNCE 不允许样本索引在多个正样本对中重复')
                idx = idx.to(device=h.device)
                z = F.normalize(self.heads.proj(g), dim=-1)
                zi, zj = z[idx[:, 0]], z[idx[:, 1]]
                logits = zi @ zj.t() / 0.07
                labels = torch.arange(len(idx), device=h.device)
                out['align'] = 0.5 * (F.cross_entropy(logits, labels)
                                      + F.cross_entropy(logits.t(), labels)) \
                    * weights.get('align', 1.0)

        aux_losses = []
        records = targets.get('aux_records')
        if isinstance(records, dict):
            records = [records]
        for record in records or []:
            name = record['name']
            if name not in self.heads.aux_specs:
                raise KeyError(f'aux_records 引用了未配置的任务: {name}')
            spec = self.heads.aux_specs[name]
            head = self.heads.aux_heads[name]
            if spec['kind'] == 'graph':
                pred = head(g)
            elif spec['kind'] == 'atom':
                pred = head(h)
            else:
                pair_h = h[:, :, None, :] + h[:, None, :, :]
                pred = head(pair_h)
            raw_value = record['value']
            mask = record.get('mask')
            if mask is None:
                if spec['kind'] == 'atom':
                    mask = batch['pad_mask']
                elif spec['kind'] == 'bond':
                    mask = batch['pad_mask'][:, :, None] & batch['pad_mask'][:, None, :]
            if spec['task_type'] in ('multiclass_classification', 'sequence_classification'):
                if raw_value.shape == pred.shape:
                    value = raw_value.to(dtype=pred.dtype)
                    full_mask = _broadcast_mask(mask, value)
                    valid = full_mask.any(dim=-1)
                    if not torch.equal(full_mask, valid.unsqueeze(-1).expand_as(full_mask)):
                        raise ValueError(f'aux[{name}] 分类 mask 必须在类别维一致')
                    if not valid.any():
                        continue
                    value = torch.where(valid.unsqueeze(-1), value, torch.zeros_like(value))
                    sums = value.sum(dim=-1)
                    if (not torch.isfinite(value).all() or (value < 0).any()
                            or (sums[valid] <= 0).any()):
                        raise ValueError(f'aux[{name}] 分类分布必须有限、非负且有效位置和为正')
                    value = value / sums.clamp_min(1e-12).unsqueeze(-1)
                    element_loss = -(value * F.log_softmax(pred, dim=-1)).sum(dim=-1)
                else:
                    target = raw_value
                    if target.shape == (*pred.shape[:-1], 1):
                        target = target.squeeze(-1)
                    if target.shape != pred.shape[:-1]:
                        raise ValueError(f'aux[{name}] 类别索引形状 {tuple(target.shape)} 与预测 '
                                         f'{tuple(pred.shape)} 不一致')
                    valid = _broadcast_mask(mask, target)
                    if not valid.any():
                        continue
                    target = target.to(dtype=pred.dtype)
                    target = torch.where(valid, target, torch.zeros_like(target))
                    invalid = ((~torch.isfinite(target)) | (target < 0)
                               | (target >= pred.shape[-1]) | (target != target.round()))
                    if invalid[valid].any():
                        raise ValueError(f'aux[{name}] 类别索引越界或非整数')
                    element_loss = F.cross_entropy(
                        pred.reshape(-1, pred.shape[-1]), target.long().reshape(-1),
                        reduction='none').reshape(valid.shape)
                aux_losses.append(
                    (element_loss * valid.to(dtype=pred.dtype)).sum() / valid.sum())
                continue
            value = raw_value.to(dtype=pred.dtype)
            if spec['dim'] == 1 and value.shape == pred.shape[:-1]:
                pred = pred.squeeze(-1)
            if value.shape != pred.shape:
                raise ValueError(f'aux[{name}] 标签形状 {tuple(value.shape)} 与预测 '
                                 f'{tuple(pred.shape)} 不一致')
            mask = _broadcast_mask(mask, value)
            if not mask.any():
                continue
            if spec['task_type'] == 'binary_classification':
                element_loss = F.binary_cross_entropy_with_logits(pred, value, reduction='none')
            else:
                element_loss = F.huber_loss(pred, value, delta=1.0, reduction='none')
            aux_losses.append((element_loss * mask).sum() / mask.sum())
        if aux_losses:
            out['aux'] = torch.stack(aux_losses).mean() * weights.get('aux', 1.0)

        if 'aux_props' in targets:
            y = targets['aux_props']
            if y.shape[1] > self.cfg.descriptor_dim:
                raise ValueError('legacy aux_props 宽度不能超过 descriptor_dim')
            pred = self.heads.aux_legacy(g)[:, :y.shape[1]]
            m = _broadcast_mask(targets.get('aux_mask'), y)
            if m.any():
                legacy = ((pred - y) ** 2 * m).sum() / m.sum()
                legacy = legacy * weights.get('aux', 1.0)
                out['aux'] = out.get('aux', g.new_zeros(())) + legacy

        total = g.new_zeros(())
        for value in out.values():
            total = total + value
        out['total'] = total
        return out


# ---------------------------------------------------------------- MLP 探针合规校验
def check_probe_params(embed_dim: int, hidden: int, out_dim: int,
                       max_params: int = 100_000) -> int:
    """Linear(D→hidden)→act→Linear(hidden→out) 的参数量，须 ≤ 100K。"""
    n = embed_dim * hidden + hidden + hidden * out_dim + out_dim
    assert n <= max_params, f'MLP 参数 {n} 超过上限 {max_params}'
    return n
