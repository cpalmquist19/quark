"""Baseline decoder-only GPT: RoPE, RMSNorm (pre-norm), SwiGLU, SDPA attention, tied LM head.

Grouped-query attention comes for free via `n_kv_heads`. A simple per-layer KV cache supports
generation. Later phases swap these parts out through the component registry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from quark.config import ModelConfig


@dataclass
class LayerCache:
    k: torch.Tensor | None = None  # (B, n_kv_heads, T_past, head_dim)
    v: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]


class KVCache:
    def __init__(self, n_layers: int):
        self.layers = [LayerCache() for _ in range(n_layers)]

    @property
    def length(self) -> int:
        return self.layers[0].length


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class Rotary(nn.Module):
    """Precomputed RoPE tables (half-split layout, as in Llama)."""

    def __init__(self, head_dim: int, max_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        angles = torch.outer(torch.arange(max_len, dtype=torch.float32), inv_freq)  # (T, head_dim/2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, start: int) -> torch.Tensor:
        # x: (B, H, T, head_dim)
        t = x.shape[2]
        cos = self.cos[start : start + t].to(x.dtype)
        sin = self.sin[start : start + t].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads or cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, (self.n_heads + 2 * self.n_kv_heads) * self.head_dim, bias=False)
        self.out = nn.Linear(self.n_heads * self.head_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, rope: Rotary, cache: LayerCache | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).split(
            [self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim],
            dim=-1,
        )
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        past = cache.length if cache is not None else 0
        q, k = rope(q, past), rope(k, past)
        if cache is not None:
            if past:
                k = torch.cat((cache.k, k), dim=2)
                v = torch.cat((cache.v, v), dim=2)
            cache.k, cache.v = k, v

        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        if past == 0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif t == 1:
            y = F.scaled_dot_product_attention(q, k, v)  # one new token attends to everything cached
        else:
            # Several new tokens after a cached prefix: query i may see keys up to past + i.
            mask = torch.ones(t, past + t, dtype=torch.bool, device=x.device).tril(diagonal=past)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(b, t, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.hidden = swiglu_hidden(cfg)
        self.gate_up = nn.Linear(cfg.d_model, 2 * self.hidden, bias=False)
        self.down = nn.Linear(self.hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


def swiglu_hidden(cfg: ModelConfig) -> int:
    return 64 * math.ceil(cfg.ffn_mult * cfg.d_model / 64)  # multiple of 64 for efficient matmuls


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, rope: Rotary, cache: LayerCache | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope, cache)
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = Rotary(cfg.head_dim, cfg.seq_len)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.apply(self._init_weights)
        # GPT-2 style: shrink the layers that write into the residual stream by depth.
        for block in self.blocks:
            for proj in (block.attn.out, block.mlp.down):
                nn.init.normal_(proj.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        """Returns the mean cross-entropy loss if `targets` is given, else logits (B, T, vocab)."""
        start = cache.length if cache is not None else 0
        if start + idx.shape[1] > self.cfg.seq_len:
            raise ValueError(f"sequence length {start + idx.shape[1]} exceeds seq_len {self.cfg.seq_len}")
        x = self.tok_emb(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, self.rope, cache.layers[i] if cache is not None else None)
        logits = F.linear(self.norm(x), self.tok_emb.weight)  # tied LM head
        if targets is None:
            return logits
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-1)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.tok_emb.weight.numel() if non_embedding else n

    def flops_per_token(self) -> float:
        """Training FLOPs per token (forward + backward), PaLM-style estimate."""
        c = self.cfg
        return 6 * self.num_params(non_embedding=True) + 12 * c.n_layers * c.d_model * c.seq_len
