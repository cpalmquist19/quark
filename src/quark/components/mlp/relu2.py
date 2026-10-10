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
    summary = "down(relu(up(x))^2); 1.5x SwiGLU's hidden width, so params match SwiGLU exactly."
    paper = "arXiv:2109.08668"
    maturity = "promising"

    def __init__(self, ctx: BuildContext):
        super().__init__()
        cfg = ctx.cfg
        # Two matrices instead of SwiGLU's three: 1.5x SwiGLU's hidden width (a multiple of 64, so this
        # is a multiple of 32) keeps the parameter count exactly equal, so ablations compare like with like.
        self.hidden = 3 * (64 * math.ceil(cfg.ffn_mult * cfg.d_model / 64)) // 2
        self.up = nn.Linear(cfg.d_model, self.hidden, bias=False)
        self.down = mark_residual(nn.Linear(self.hidden, cfg.d_model, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.relu(self.up(x)).square())
