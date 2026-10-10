"""z-loss: penalize log(sum(exp(logits)))^2 so logits don't drift large."""

from __future__ import annotations

from pydantic import Field

from quark.components.base import Toggle
from quark.components.registry import ComponentParams, register


@register("toggle", "z_loss")
class ZLoss(Toggle):
    summary = "Adds coef * logsumexp(logits)^2 to the training loss; keeps the softmax normalizer near 0."
    paper = "arXiv:2204.02311"
    maturity = "established"
    implemented_by = "head"

    class Params(ComponentParams):
        coef: float = Field(1e-4, gt=0)  # PaLM's value
