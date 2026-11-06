"""共享主干组件：RoPE、对偏置自注意力、MoLE 专家 FFN、Transformer 块。

设计对应报告 §6.3：
- 几何感知注意力 = 标准 MHSA + RoPE + 加性对特征偏置（Uni-Mol2 双轨的轻量版）；
- MoLE（Mixture of Linear Experts, UMA）：FFN 线性层换为 E 个专家的凸组合，
  门控由域嵌入软路由 —— 同一权重文件、同一网络，无硬分支；
- pre-norm + 无 BatchNorm —— 保证表征的批次无关性。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- RoPE
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv, persistent=False)
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv)                       # [max_len, dim/2]
        self.register_buffer('cos', freqs.cos(), persistent=False)
        self.register_buffer('sin', freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, H, L, Dh]，对最后维施加旋转。"""
        L = x.shape[2]
        if L > self.cos.shape[0]:
            t = torch.arange(L, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            self.cos = freqs.cos()
            self.sin = freqs.sin()
        cos = self.cos[:L].unsqueeze(0).unsqueeze(0)      # [1,1,L,Dh/2]
        sin = self.sin[:L].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        x_even = x1 * cos - x2 * sin
        x_odd = x1 * sin + x2 * cos
        return torch.stack([x_even, x_odd], dim=-1).flatten(-2)


# ---------------------------------------------------------------- 对偏置注意力
class PairBiasAttention(nn.Module):
    """MHSA + 可选 RoPE + 加性 pair bias [B, H, L, L]。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float, use_rope: bool,
                 max_len: int, use_sdpa: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.dh, max_len) if use_rope else None
        self.scale = 1.0 / math.sqrt(self.dh)
        self.use_sdpa = use_sdpa

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor,
                pair_bias: torch.Tensor | None) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                  # [B,H,L,Dh]
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        if self.use_sdpa:
            # SDPA/mem-efficient：不物化 [B,H,L,L] 注意力概率，正式规模下省几十 GB。
            neg = torch.finfo(q.dtype).min
            pad_bias = (~pad_mask[:, None, None, :]).to(q.dtype) * neg  # [B,1,1,L] 广播
            attn_mask = pair_bias + pad_bias if pair_bias is not None else pad_bias
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.drop.p if self.training else 0.0)
        else:
            att = (q @ k.transpose(-2, -1)) * self.scale      # [B,H,L,L]
            if pair_bias is not None:
                att = att + pair_bias
            neg = torch.finfo(att.dtype).min
            att = att.masked_fill(~pad_mask[:, None, None, :], neg)
            p = F.softmax(att, dim=-1)
            p = self.drop(p)
            y = p @ v
        y = y.transpose(1, 2).reshape(B, L, D)
        return self.out(y)


# ---------------------------------------------------------------- MoLE FFN
class _MoleLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, expert_weight, expert_bias, groups):
        ctx.save_for_backward(x, weights, expert_weight, expert_bias, groups)
        output = None
        for group in groups.unique():
            indices = torch.nonzero(groups == group, as_tuple=False).flatten()
            group_weight = weights[indices].mean(dim=0)
            weight = torch.einsum('e,efd->fd', group_weight, expert_weight)
            bias = torch.einsum('e,ef->f', group_weight, expert_bias)
            value = F.linear(x.index_select(0, indices), weight, bias)
            if output is None:
                output = value.new_zeros(*x.shape[:-1], expert_weight.shape[1])
            output = output.index_copy(0, indices, value)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weights, expert_weight, expert_bias, groups = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = grad_weights = None
        grad_output = grad_output.float()
        x_float = x.float()
        weights_float = weights.float()
        if ctx.needs_input_grad[0]:
            grad_x = None
            for group in groups.unique():
                indices = torch.nonzero(groups == group, as_tuple=False).flatten()
                group_weight = weights_float[indices].mean(dim=0)
                weight = torch.einsum('e,efd->fd', group_weight, expert_weight)
                value = F.linear(grad_output.index_select(0, indices), weight.t())
                if grad_x is None:
                    grad_x = value.new_zeros(x.shape)
                grad_x = grad_x.index_copy(0, indices, value)
            grad_x = grad_x.to(x.dtype)
        if ctx.needs_input_grad[1]:
            grad_weights = weights.new_zeros(weights.shape)
            for expert in range(expert_weight.shape[0]):
                value = F.linear(x_float, expert_weight[expert], expert_bias[expert])
                grad_weights[:, expert] = (grad_output * value).sum(dim=(1, 2))
        if ctx.needs_input_grad[2]:
            grad_weight = torch.einsum('be,blf,bld->efd', weights_float, grad_output, x_float)
        if ctx.needs_input_grad[3]:
            grad_bias = torch.einsum('be,blf->ef', weights_float, grad_output)
        return grad_x, grad_weights, grad_weight, grad_bias, None


class MoleFFN(nn.Module):
    """UMA 式 MoLE：前向按域合并专家，反向保留逐样本门控与专家梯度。"""

    def __init__(self, d_model: int, d_ffn: int, num_experts: int, cond_dim: int,
                 dropout: float):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, d_ffn, d_model))
        self.b1 = nn.Parameter(torch.zeros(num_experts, d_ffn))
        self.w2 = nn.Parameter(torch.empty(num_experts, d_model, d_ffn))
        self.b2 = nn.Parameter(torch.zeros(num_experts, d_model))
        nn.init.normal_(self.w1, std=d_model ** -0.5)
        nn.init.normal_(self.w2, std=d_ffn ** -0.5)
        self.gate = nn.Linear(cond_dim, num_experts)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def gate_stats(self, cond: torch.Tensor):
        logits = self.gate(cond)
        weights = F.softmax(logits, dim=-1)
        entropy = -(weights * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        return {'weights': weights, 'entropy': entropy}

    def forward(self, x: torch.Tensor, cond: torch.Tensor, return_gate: bool = False):
        stats = self.gate_stats(cond)
        weights = stats['weights']
        _, groups = torch.unique(cond.detach(), dim=0, return_inverse=True)
        hidden = _MoleLinearFunction.apply(x, weights, self.w1, self.b1, groups)
        hidden = self.drop(self.act(hidden))
        out = _MoleLinearFunction.apply(hidden, weights, self.w2, self.b2, groups)
        if return_gate:
            return out, stats
        return out


class StandardFFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model))

    def forward(self, x, cond=None):
        return self.net(x)


# ---------------------------------------------------------------- Transformer 块
class Block(nn.Module):
    def __init__(self, cfg, layer_idx: int, cond_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.attn = PairBiasAttention(cfg.d_model, cfg.n_heads, cfg.dropout,
                                      cfg.use_rope, cfg.max_seq_len, cfg.use_sdpa)
        self.ln2 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        if cfg.num_experts > 1 and (layer_idx % cfg.mole_every == cfg.mole_every - 1):
            self.ffn = MoleFFN(cfg.d_model, cfg.d_ffn, cfg.num_experts, cond_dim, cfg.dropout)
            self.is_mole = True
        else:
            self.ffn = StandardFFN(cfg.d_model, cfg.d_ffn, cfg.dropout)
            self.is_mole = False
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, pad_mask, pair_bias, cond):
        x = x + self.drop(self.attn(self.ln1(x), pad_mask, pair_bias))
        ffn_in = self.ln2(x)
        x = x + self.drop(self.ffn(ffn_in, cond) if self.is_mole else self.ffn(ffn_in))
        return x
