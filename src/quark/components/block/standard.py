"""The standard stack: n_layers pre-norm layers, each attention then MLP on the residual stream."""

from __future__ import annotations

import torch
import torch.nn as nn

from quark.components.base import BuildContext, KVCache, LayerCache, PositionEncoding, Stack
from quark.components.registry import register


class PreNormLayer(nn.Module):
    def __init__(self, ctx: BuildContext):
        super().__init__()
        d = ctx.cfg.d_model
        self.attn_norm = ctx.make("norm", d)
        self.attn = ctx.make("attention")
        self.mlp_norm = ctx.make("norm", d)
        self.mlp = ctx.make("mlp")

    def forward(self, x: torch.Tensor, pos: PositionEncoding, cache: LayerCache | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), pos, cache)
        return x + self.mlp(self.mlp_norm(x))


@register("block", "standard")
class StandardStack(Stack):
    summary = "n_layers independent pre-norm layers (attention, then MLP)."
    maturity = "established"

    def __init__(self, ctx: BuildContext):
        super().__init__()
        n = ctx.cfg.n_layers
        for i in range(n):
            self.add_module(str(i), PreNormLayer(ctx))  # children named 0..n-1, as in nn.ModuleList
        self.n_cache_layers = self.unrolled_depth = n

    def forward(self, x: torch.Tensor, pos: PositionEncoding, cache: KVCache | None = None) -> torch.Tensor:
        for i, layer in enumerate(self.children()):
            x = layer(x, pos, cache.layers[i] if cache is not None else None)
        return x
