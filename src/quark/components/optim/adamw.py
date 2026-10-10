"""AdamW with decoupled weight decay on matrices and embeddings only."""

from __future__ import annotations

import torch
import torch.nn as nn

from quark.components.base import OptimizerComponent
from quark.components.registry import ComponentParams, register
from quark.config import TrainConfig


@register("optimizer", "adamw")
class AdamW(OptimizerComponent):
    summary = "Adam with decoupled weight decay; norm gains and other vectors aren't decayed."
    paper = "arXiv:1711.05101"
    maturity = "established"

    @classmethod
    def build(
        cls, model: nn.Module, train: TrainConfig, params: ComponentParams, device: torch.device
    ) -> torch.optim.Optimizer:
        decay = [p for p in model.parameters() if p.dim() >= 2]  # matrices and embeddings
        no_decay = [p for p in model.parameters() if p.dim() < 2]  # norm gains
        groups = [
            {"params": decay, "weight_decay": train.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=train.lr, betas=train.betas, fused=device.type == "cuda")
