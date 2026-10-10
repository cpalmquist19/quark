"""RMSNorm with a learned gain."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from quark.components.base import BuildContext, Norm
from quark.components.registry import ComponentParams, register


@register("norm", "rmsnorm")
class RMSNorm(Norm):
    summary = "Root-mean-square norm with a learned gain (no mean subtraction, no bias)."
    paper = "arXiv:1910.07467"
    maturity = "established"

    class Params(ComponentParams):
        eps: float = 1e-6

    def __init__(self, ctx: BuildContext, dim: int):
        super().__init__()
        self.eps = ctx.params_for(self).eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)
