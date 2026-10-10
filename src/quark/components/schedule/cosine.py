"""Linear warmup, then cosine decay to lr * min_lr_ratio."""

from __future__ import annotations

import math

from quark.components.base import Schedule
from quark.components.registry import register


@register("schedule", "cosine")
class Cosine(Schedule):
    summary = "Linear warmup over warmup_steps, then cosine decay to lr * min_lr_ratio."
    paper = "arXiv:1608.03983"
    maturity = "established"

    def lr_at(self, step: int) -> float:
        t = self.train
        if step < t.warmup_steps:
            return t.lr * (step + 1) / t.warmup_steps
        progress = min(1.0, (step - t.warmup_steps) / max(1, self.total_steps - t.warmup_steps))
        min_lr = t.lr * t.min_lr_ratio
        return min_lr + 0.5 * (t.lr - min_lr) * (1 + math.cos(math.pi * progress))
