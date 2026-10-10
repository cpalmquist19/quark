"""LM head that reuses the token embedding matrix."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from quark.components.base import BuildContext, Head
from quark.components.registry import register


@register("head", "tied")
class TiedHead(Head):
    summary = "Logits from the token embedding matrix (no extra parameters)."
    paper = "arXiv:1608.05859"
    maturity = "established"
    supports = frozenset({"z_loss"})

    def __init__(self, ctx: BuildContext):
        super().__init__()
        if ctx.enabled("z_loss"):
            self.z_loss_coef = ctx.params["z_loss"].coef

    def forward(self, x: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
        return F.linear(x, embedding.weight)
