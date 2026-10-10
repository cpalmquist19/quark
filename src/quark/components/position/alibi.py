"""ALiBi: a fixed, per-head linear penalty on attention scores by query-key distance."""

from __future__ import annotations

import math

import torch

from quark.components.base import BuildContext, PositionEncoding
from quark.components.registry import register


def alibi_slopes(n_heads: int) -> list[float]:
    """Geometric slopes from the paper; non-power-of-2 head counts interleave the next power's."""

    def power_of_2(n: int) -> list[float]:
        start = 2 ** (-8 / n)
        return [start ** (i + 1) for i in range(n)]

    closest = 2 ** math.floor(math.log2(n_heads))
    if closest == n_heads:
        return power_of_2(n_heads)
    return power_of_2(closest) + power_of_2(2 * closest)[0::2][: n_heads - closest]


@register("position", "alibi")
class ALiBi(PositionEncoding):
    summary = "Adds -slope_h * distance to attention scores; no parameters, extrapolates past seq_len."
    paper = "arXiv:2108.12409"
    maturity = "established"
    requires = ("attention:mha",)  # attention types must apply attn_bias

    def __init__(self, ctx: BuildContext):
        super().__init__()
        slopes = torch.tensor(alibi_slopes(ctx.cfg.n_heads), dtype=torch.float32)
        self.register_buffer("slopes", slopes.view(-1, 1, 1), persistent=False)

    def attn_bias(self, tq: int, tk: int, start: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        q_pos = torch.arange(start, start + tq, device=device).view(-1, 1)
        k_pos = torch.arange(tk, device=device).view(1, -1)
        return (self.slopes * (k_pos - q_pos)).to(dtype)  # <= 0 for every visible key
