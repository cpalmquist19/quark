"""SwiGLU feed-forward."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from quark.components.base import MLP, BuildContext, mark_residual
from quark.components.registry import register


@register("mlp", "swiglu")
class SwiGLU(MLP):
    summary = "Gated feed-forward: down(silu(gate(x)) * up(x)); hidden = ffn_mult * d_model."
    paper = "arXiv:2002.05202"
    maturity = "established"

    def __init__(self, ctx: BuildContext):
        super().__init__()
        cfg = ctx.cfg
        self.hidden = 64 * math.ceil(cfg.ffn_mult * cfg.d_model / 64)  # multiple of 64 for efficient matmuls
        self.gate_up = nn.Linear(cfg.d_model, 2 * self.hidden, bias=False)
        self.down = mark_residual(nn.Linear(self.hidden, cfg.d_model, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)
