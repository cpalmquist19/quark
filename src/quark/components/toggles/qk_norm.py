"""QK-norm: normalize queries and keys per head before the attention dot product."""

from __future__ import annotations

from quark.components.base import Toggle
from quark.components.registry import register


@register("toggle", "qk_norm")
class QKNorm(Toggle):
    summary = "Per-head norm (the model's norm type) on q and k; bounds attention logits, steadies fp16."
    paper = "arXiv:2010.04245"
    maturity = "established"
    implemented_by = "attention"
