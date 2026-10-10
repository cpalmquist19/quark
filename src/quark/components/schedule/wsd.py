"""Warmup-stable-decay: hold the peak LR, then decay over the final fraction of training."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field

from quark.components.base import Schedule
from quark.components.registry import ComponentParams, register


@register("schedule", "wsd")
class WSD(Schedule):
    summary = "Linear warmup, constant LR, then decay to lr * min_lr_ratio over the last decay_frac."
    paper = "arXiv:2404.06395"
    maturity = "established"

    class Params(ComponentParams):
        decay_frac: float = Field(0.2, gt=0, le=1)  # share of total steps spent decaying
        decay: Literal["linear", "1-sqrt"] = "linear"  # 1-sqrt: arXiv:2405.18392

    def lr_at(self, step: int) -> float:
        t, p = self.train, self.params
        if step < t.warmup_steps:
            return t.lr * (step + 1) / t.warmup_steps
        decay_steps = max(1, round(p.decay_frac * self.total_steps))
        decay_start = self.total_steps - decay_steps
        if step < decay_start:
            return t.lr
        progress = min(1.0, (step - decay_start) / decay_steps)
        drop = progress if p.decay == "linear" else math.sqrt(progress)
        min_lr = t.lr * t.min_lr_ratio
        return min_lr + (t.lr - min_lr) * (1 - drop)
