"""Squared-ReLU feed-forward (Primer)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from quark.components.base import MLP, BuildContext, mark_residual
from quark.components.registry import register


@register("mlp", "relu2")
class ReLU2(MLP):
    summary = "down(relu(up(x))^2); hidden = 1.5 * ffn_mult * d_model, so params match SwiGLU."
    paper = "arXiv:2109.08668"
    maturity = "promising"

    def __init__(self, ctx: BuildContext):
        super().__init__()
        cfg = ctx.cfg
        # Two matrices instead of SwiGLU's three: 1.5x the hidden width keeps the parameter count
        # equal at the same ffn_mult, so ablations against SwiGLU compare like with like.
        self.hidden = 64 * math.ceil(1.5 * cfg.ffn_mult * cfg.d_model / 64)
        self.up = nn.Linear(cfg.d_model, self.hidden, bias=False)
        self.down = mark_residual(nn.Linear(self.hidden, cfg.d_model, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.relu(self.up(x)).square())
