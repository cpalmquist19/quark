"""Decoder-only GPT assembled from component slots (docs/component_registry.md).

The baseline choices are RoPE, RMSNorm (pre-norm), SwiGLU, SDPA attention and a tied LM head;
`cfg.slots` picks others from the registry. The slot attribute names (`tok_emb`, `blocks`, `norm`)
are part of the checkpoint format, so they must not change.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from quark.components import KVCache, build_context
from quark.config import HardwareProfile, ModelConfig


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig, hardware: HardwareProfile | None = None):
        super().__init__()
        self.cfg = cfg
        ctx = build_context(cfg, hardware)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = ctx.make("position")
        self.blocks = ctx.make("block")
        self.norm = ctx.make("norm", cfg.d_model)
        self.head = ctx.make("head")
        self.apply(self._init_weights)
        # GPT-2 style: shrink the projections that write into the residual stream by (unrolled) depth.
        residual_std = 0.02 / math.sqrt(2 * self.blocks.unrolled_depth)
        for module in self.modules():
            if getattr(module, "_quark_residual", False):
                nn.init.normal_(module.weight, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def new_cache(self) -> KVCache:
        return KVCache(self.blocks.n_cache_layers)

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
        x = self.pos.embed(self.tok_emb(idx), start)
        x = self.blocks(x, self.pos, cache)
        logits = self.head(self.norm(x), self.tok_emb)
        if targets is None:
            return logits
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-1)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.tok_emb.weight.numel() if non_embedding else n

    def flops_per_token(self) -> float:
        """Training FLOPs per token (forward + backward), PaLM-style estimate."""
        c = self.cfg
        return 6 * self.num_params(non_embedding=True) + 12 * self.blocks.unrolled_depth * c.d_model * c.seq_len
