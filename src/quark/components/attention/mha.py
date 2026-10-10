"""Multi-head attention through PyTorch SDPA. GQA/MQA come from `n_kv_heads` < `n_heads`.

Implements the qk_norm toggle and applies a position encoding's attention bias (ALiBi).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from quark.components.base import Attention, BuildContext, LayerCache, PositionEncoding, mark_fused, mark_residual
from quark.components.registry import register


@register("attention", "mha")
class MHA(Attention):
    summary = "Multi-head attention via SDPA, with a KV cache; grouped-query when n_kv_heads < n_heads."
    paper = "arXiv:1706.03762"
    maturity = "established"
    supports = frozenset({"qk_norm"})

    def __init__(self, ctx: BuildContext):
        super().__init__()
        cfg = ctx.cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads or cfg.n_heads
        self.head_dim = cfg.head_dim
        q_dim, kv_dim = self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim
        self.qkv = mark_fused(nn.Linear(cfg.d_model, q_dim + 2 * kv_dim, bias=False), [q_dim, kv_dim, kv_dim])
        self.out = mark_residual(nn.Linear(q_dim, cfg.d_model, bias=False))
        if ctx.enabled("qk_norm"):  # per-head norm of q and k, before the position encoding
            self.q_norm = ctx.make("norm", self.head_dim)
            self.k_norm = ctx.make("norm", self.head_dim)

    def forward(self, x: torch.Tensor, pos: PositionEncoding, cache: LayerCache | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).split(
            [self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim],
            dim=-1,
        )
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if hasattr(self, "q_norm"):
            q, k = self.q_norm(q), self.k_norm(k)

        past = cache.length if cache is not None else 0
        q, k = pos.rotate(q, k, past)
        if cache is not None:
            if past:
                k = torch.cat((cache.k, k), dim=2)
                v = torch.cat((cache.v, v), dim=2)
            cache.k, cache.v = k, v

        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        bias = pos.attn_bias(t, past + t, past, x.device, q.dtype)
        if bias is not None:
            causal = torch.ones(t, past + t, dtype=torch.bool, device=x.device).tril(diagonal=past)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.masked_fill(~causal, float("-inf")))
        elif past == 0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif t == 1:
            y = F.scaled_dot_product_attention(q, k, v)  # one new token attends to everything cached
        else:
            # Several new tokens after a cached prefix: query i may see keys up to past + i.
            mask = torch.ones(t, past + t, dtype=torch.bool, device=x.device).tril(diagonal=past)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(b, t, -1))
