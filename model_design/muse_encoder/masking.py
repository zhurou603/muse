"""预训练掩码与坐标破坏（训练期专用）。

序列使用 15% / 80-10-10 MLM；原子和肽使用独立的原子 mask id，dummy 0
仍是可预测的普通原子类型。肽默认按 Bulk/SideChain/Fragment=30/40/30
逐样本采样，strict=True 时要求所选策略的真实结构标注可用。
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

IGNORE = -100
DEFAULT_ATOM_MASK_ID = 119
DEFAULT_SEQUENCE_SPECIAL_IDS = (0, 1, 2, 3, 4, 5)  # PAD/UNK/MASK/CLS/SEP/EOS
PEPTIDE_MASK_MIX = {'bulk': 0.30, 'sidechain': 0.40, 'fragment': 0.30}


def _check_batch(token_ids: torch.Tensor, pad_mask: torch.Tensor):
    if token_ids.ndim != 2 or token_ids.shape[1] == 0:
        raise ValueError('token_ids 必须是非空的 [B,L] 张量')
    if pad_mask.shape != token_ids.shape:
        raise ValueError('pad_mask 必须与 token_ids 同形状')


def random_mlm_mask(token_ids: torch.Tensor, pad_mask: torch.Tensor,
                    mask_id: int, vocab_size: int, mask_prob: float = 0.15,
                    generator: Optional[torch.Generator] = None,
                    special_token_ids: Optional[Sequence[int]] = None):
    """标准 MLM；默认特殊 id 与 tokenizers.SPECIALS 一致。

    special_token_ids 可适配其他词表。随机替换仅从词表内非特殊 id 抽取，
    返回 labels 时未选位置始终为 IGNORE。
    """
    _check_batch(token_ids, pad_mask)
    if not 0.0 <= mask_prob <= 1.0:
        raise ValueError('mask_prob 必须在 [0,1]')
    if not 0 <= mask_id < vocab_size:
        raise ValueError('mask_id 必须位于词表内')
    specials = set(DEFAULT_SEQUENCE_SPECIAL_IDS if special_token_ids is None
                   else special_token_ids)
    specials.add(mask_id)
    specials = {i for i in specials if 0 <= i < vocab_size}
    dev = token_ids.device
    special_tensor = torch.tensor(sorted(specials), dtype=torch.long, device=dev)
    normal_ids = torch.arange(vocab_size, dtype=torch.long, device=dev)
    normal_ids = normal_ids[~torch.isin(normal_ids, special_tensor)]
    if normal_ids.numel() == 0:
        raise ValueError('词表中没有可用于随机替换的普通 token')

    eligible = pad_mask.bool() & ~torch.isin(token_ids, special_tensor)
    eligible[:, 0] = False
    sel = (torch.rand(token_ids.shape, generator=generator, device=dev)
           < mask_prob) & eligible
    labels = torch.full_like(token_ids, IGNORE)
    labels[sel] = token_ids[sel]
    masked = token_ids.clone()

    replace = torch.rand(token_ids.shape, generator=generator, device=dev)
    masked[sel & (replace < 0.8)] = mask_id
    random_replace = sel & (replace >= 0.8) & (replace < 0.9)
    n_random = int(random_replace.sum())
    if n_random:
        normal_index = torch.randint(normal_ids.numel(), (n_random,),
                                     generator=generator, device=dev)
        masked[random_replace] = normal_ids[normal_index]
    return masked, labels


def random_atom_mask(element_ids: torch.Tensor, pad_mask: torch.Tensor,
                     mask_prob: float = 0.15,
                     generator: Optional[torch.Generator] = None,
                     atom_mask_id: int = DEFAULT_ATOM_MASK_ID):
    """随机掩 15% 原子类型；dummy id 0 与 atom_mask_id 不共用。"""
    _check_batch(element_ids, pad_mask)
    if not 0.0 <= mask_prob <= 1.0:
        raise ValueError('mask_prob 必须在 [0,1]')
    if atom_mask_id < 0:
        raise ValueError('atom_mask_id 必须非负')
    eligible = pad_mask.bool()
    eligible[:, 0] = False
    sel = (torch.rand(element_ids.shape, generator=generator,
                     device=element_ids.device) < mask_prob) & eligible
    labels = torch.full_like(element_ids, IGNORE)
    labels[sel] = element_ids[sel]
    masked = element_ids.clone()
    masked[sel] = atom_mask_id
    return masked, labels


def sample_coordinate_noise(coords: torch.Tensor, coord_mask: torch.Tensor,
                            sigma_min: float = 0.04, sigma_max: float = 1.0,
                            generator: Optional[torch.Generator] = None,
                            return_noise_mask: bool = False):
    """逐样本 LogUniform sigma 的高斯坐标去噪。

    输入可为 [B,L,3] 或 [L,3]。默认返回 noisy,true_noise,sigma；
    return_noise_mask=True 时额外返回实际参与去噪的布尔 mask。CLS、PAD
    和非有限坐标不参与；有限无效坐标保持原值，非有限无效坐标置零以免扩散 NaN。
    """
    if coords.ndim not in (2, 3) or coords.shape[-1] != 3:
        raise ValueError('coords 必须是 [B,L,3] 或 [L,3]')
    if not coords.is_floating_point():
        raise ValueError('coords 必须是浮点张量')
    if not 0 < sigma_min <= sigma_max:
        raise ValueError('需要 0 < sigma_min <= sigma_max')
    unbatched = coords.ndim == 2
    coords_b = coords.unsqueeze(0) if unbatched else coords
    mask_b = coord_mask.unsqueeze(0) if unbatched else coord_mask
    if mask_b.shape != coords_b.shape[:-1]:
        raise ValueError('coord_mask 必须与 coords 的前缀形状一致')

    finite = torch.isfinite(coords_b).all(dim=-1)
    positions = torch.arange(coords_b.shape[1], device=coords.device) > 0
    noise_mask = mask_b.bool() & finite & positions.unsqueeze(0)
    epsilon = torch.randn(coords_b.shape, generator=generator,
                          device=coords.device, dtype=coords.dtype)
    true_noise = epsilon * noise_mask.unsqueeze(-1).to(coords.dtype)
    uniform = torch.rand(coords_b.shape[0], generator=generator,
                         device=coords.device, dtype=coords.dtype)
    log_min = torch.log(torch.as_tensor(sigma_min, dtype=coords.dtype,
                                       device=coords.device))
    log_range = torch.log(torch.as_tensor(sigma_max / sigma_min,
                                         dtype=coords.dtype, device=coords.device))
    sigma = torch.exp(log_min + uniform * log_range)
    base = torch.where(finite.unsqueeze(-1), coords_b, torch.zeros_like(coords_b))
    noisy = base + sigma.view(-1, 1, 1) * true_noise

    if unbatched:
        noisy, true_noise, sigma = noisy[0], true_noise[0], sigma[0]
        noise_mask = noise_mask[0]
    result = (noisy, true_noise, sigma)
    return result + (noise_mask,) if return_noise_mask else result


def _unit_mask(strategy: str, b: int, eligible: torch.Tensor,
               length: int, mask_prob: float,
               group_ids: Optional[torch.Tensor],
               sidechain_flags: Optional[torch.Tensor],
               fragment_ids: Optional[torch.Tensor],
               generator: Optional[torch.Generator]) -> torch.Tensor:
    dev = eligible.device
    sel = torch.zeros(length, dtype=torch.bool, device=dev)
    positions = torch.nonzero(eligible).squeeze(-1)
    if positions.numel() == 0 or mask_prob == 0:
        return sel

    ids = group_ids if strategy == 'bulk' else fragment_ids
    if strategy in ('bulk', 'fragment'):
        if ids is None:
            return sel
        valid_ids = ids[b][positions]
        valid_ids = valid_ids[valid_ids >= 0]
        units = valid_ids.unique()
        if units.numel() == 0:
            return sel
        n_pick = max(1, int(units.numel() * mask_prob))
        order = torch.randperm(units.numel(), generator=generator, device=dev)
        picked = units[order[:n_pick]]
        return torch.isin(ids[b], picked) & eligible

    if strategy == 'sidechain' and sidechain_flags is not None:
        sidechain = positions[sidechain_flags[b][positions].bool()]
        if sidechain.numel() == 0:
            return sel
        n_pick = max(1, int(sidechain.numel() * mask_prob))
        order = torch.randperm(sidechain.numel(), generator=generator, device=dev)
        sel[sidechain[order[:n_pick]]] = True
    return sel


def _span_mask(eligible: torch.Tensor, mask_prob: float,
               generator: Optional[torch.Generator]) -> torch.Tensor:
    dev = eligible.device
    sel = torch.zeros(eligible.shape[0], dtype=torch.bool, device=dev)
    positions = torch.nonzero(eligible).squeeze(-1)
    if positions.numel() == 0 or mask_prob == 0:
        return sel
    n_spans = max(1, int(positions.numel() * mask_prob))
    starts = positions[torch.randperm(positions.numel(), generator=generator,
                                      device=dev)[:n_spans]]
    for start in starts:
        span = torch.arange(int(start), min(int(start) + 4, eligible.shape[0]),
                            device=dev)
        sel[span] = eligible[span]
    return sel


def peptide_aware_mask(element_ids: torch.Tensor, pad_mask: torch.Tensor,
                       group_ids: Optional[torch.Tensor] = None,
                       sidechain_flags: Optional[torch.Tensor] = None,
                       fragment_ids: Optional[torch.Tensor] = None,
                       mask_prob: float = 0.15,
                       generator: Optional[torch.Generator] = None,
                       atom_mask_id: int = DEFAULT_ATOM_MASK_ID,
                       strict: bool = False, strategy: Optional[str] = None,
                       return_strategies: bool = False):
    """肽结构语义掩码；strategy 可显式指定 bulk/sidechain/fragment。

    strict=True 时，所选策略缺标注或不能产生任何 mask 会抛出 ValueError，
    不会改走其他策略或 span。非 strict 保留旧烟测的语义策略/span 降级。
    return_strategies=True 时第三个返回值记录每条样本实际执行的策略。
    """
    _check_batch(element_ids, pad_mask)
    if not 0.0 <= mask_prob <= 1.0:
        raise ValueError('mask_prob 必须在 [0,1]')
    if atom_mask_id < 0:
        raise ValueError('atom_mask_id 必须非负')
    for name, annotation in (('group_ids', group_ids),
                             ('sidechain_flags', sidechain_flags),
                             ('fragment_ids', fragment_ids)):
        if annotation is not None and annotation.shape != element_ids.shape:
            raise ValueError(f'{name} 必须与 element_ids 同形状')

    strategies = list(PEPTIDE_MASK_MIX)
    if strategy is not None and strategy not in strategies:
        raise ValueError(f'未知肽掩码策略: {strategy}')
    batch_size, length = element_ids.shape
    dev = element_ids.device
    if strategy is None:
        probs = torch.tensor([PEPTIDE_MASK_MIX[s] for s in strategies], device=dev)
        choice = torch.multinomial(probs.expand(batch_size, -1), 1,
                                   generator=generator).squeeze(-1)
        requested = [strategies[int(i)] for i in choice]
    else:
        requested = [strategy] * batch_size

    labels = torch.full_like(element_ids, IGNORE)
    masked = element_ids.clone()
    executed = []
    positions = torch.arange(length, device=dev) > 0
    for b, requested_strategy in enumerate(requested):
        eligible = pad_mask[b].bool() & positions
        attempts = [requested_strategy]
        if not strict:
            attempts += [s for s in strategies if s != requested_strategy]
        sel = torch.zeros(length, dtype=torch.bool, device=dev)
        actual = requested_strategy
        for candidate in attempts:
            sel = _unit_mask(candidate, b, eligible, length, mask_prob,
                             group_ids, sidechain_flags, fragment_ids, generator)
            if sel.any():
                actual = candidate
                break
        if not sel.any() and not strict:
            sel = _span_mask(eligible, mask_prob, generator)
            actual = 'span' if sel.any() else 'none'
        if strict and not sel.any():
            raise ValueError(f'样本 {b} 的 {requested_strategy} 策略缺少有效标注，'
                             '无法产生 mask；strict 模式禁止 fallback')
        sel[0] = False
        labels[b][sel] = element_ids[b][sel]
        masked[b][sel] = atom_mask_id
        executed.append(actual)
    result = (masked, labels)
    return result + (executed,) if return_strategies else result


def apply_masking(domain: str, token_ids: torch.Tensor, pad_mask: torch.Tensor,
                  mask_id: int = 0, vocab_size: int = 0,
                  cfg_mask_prob: float = 0.15,
                  generator: Optional[torch.Generator] = None,
                  atom_mask_id: int = DEFAULT_ATOM_MASK_ID,
                  special_token_ids: Optional[Sequence[int]] = None,
                  **peptide_kw):
    """按 domain ∈ {sequence, atom, peptide} 路由，保留原位置参数接口。"""
    if domain == 'sequence':
        return random_mlm_mask(token_ids, pad_mask, mask_id, vocab_size,
                               cfg_mask_prob, generator, special_token_ids)
    if domain == 'peptide':
        return peptide_aware_mask(token_ids, pad_mask, mask_prob=cfg_mask_prob,
                                  generator=generator, atom_mask_id=atom_mask_id,
                                  **peptide_kw)
    return random_atom_mask(token_ids, pad_mask, cfg_mask_prob, generator,
                            atom_mask_id)
