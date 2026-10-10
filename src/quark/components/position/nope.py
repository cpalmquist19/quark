"""No position encoding: causal masking alone lets the model infer order."""

from __future__ import annotations

from quark.components.base import BuildContext, PositionEncoding
from quark.components.registry import register


@register("position", "nope")
class NoPE(PositionEncoding):
    summary = "No explicit position signal; a causal decoder infers position from the mask."
    paper = "arXiv:2305.19466"
    maturity = "promising"

    def __init__(self, ctx: BuildContext):
        super().__init__()
