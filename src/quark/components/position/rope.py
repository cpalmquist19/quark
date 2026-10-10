"""Rotary position embedding."""

from __future__ import annotations

import torch

from quark.components.base import BuildContext, PositionEncoding
from quark.components.registry import ComponentParams, register


@register("position", "rope")
class RoPE(PositionEncoding):
    summary = "Rotary position embedding: rotates q/k by position (half-split layout, as in Llama)."
    paper = "arXiv:2104.09864"
    maturity = "established"

    class Params(ComponentParams):
        base: float = 10000.0

    def __init__(self, ctx: BuildContext):
        super().__init__()
        head_dim, base = ctx.cfg.head_dim, ctx.params_for(self).base
        inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        angles = torch.outer(torch.arange(ctx.cfg.seq_len, dtype=torch.float32), inv_freq)  # (T, head_dim/2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def rotate(self, q: torch.Tensor, k: torch.Tensor, start: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._rotate(q, start), self._rotate(k, start)

    def _rotate(self, x: torch.Tensor, start: int) -> torch.Tensor:
        t = x.shape[2]
        cos = self.cos[start : start + t].to(x.dtype)
        sin = self.sin[start : start + t].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
