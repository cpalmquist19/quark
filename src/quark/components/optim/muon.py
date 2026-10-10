"""Muon for hidden matrices, AdamW for embeddings and vectors, as one optimizer.

Muon replaces each matrix's momentum with its nearest semi-orthogonal matrix (Newton-Schulz), so
every direction gets a similar step size. Updates are rescaled by 0.2 * sqrt(max(rows, cols)) to
match AdamW's update RMS (Moonlight), so the shared `lr` and `weight_decay` carry over from AdamW.
Fused weights (q/k/v, gate/up; see `mark_fused`) are orthogonalized part by part.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from pydantic import Field

from quark.components.base import OptimizerComponent
from quark.components.registry import ComponentParams, Prior, Rule, register
from quark.config import TrainConfig


def newton_schulz(g: torch.Tensor, steps: int, dtype: torch.dtype) -> torch.Tensor:
    """Approximate U V^T of g's SVD with the quintic iteration from Keller Jordan's Muon."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.to(dtype)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return x.T if tall else x


class MuonWithAdamW(torch.optim.Optimizer):
    """Param groups with use_muon=True get Muon; the rest get AdamW. One state dict for both."""

    def __init__(self, groups: list[dict], lr: float, betas: tuple[float, float], momentum: float,
                 nesterov: bool, ns_steps: int, ns_dtype: torch.dtype, eps: float = 1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=0.0, use_muon=False, parts=None,
                        momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(groups, defaults)
        self.ns_dtype = ns_dtype  # runtime choice (bf16 where native, else fp32); not saved

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            (self._muon if group["use_muon"] else self._adamw)(group)
        return loss

    def _muon(self, group: dict) -> None:
        lr, wd, mom = group["lr"], group["weight_decay"], group["momentum"]
        for p, parts in zip(group["params"], group["parts"]):
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state["momentum_buffer"] = torch.zeros_like(p)
            buf = state["momentum_buffer"]
            buf.lerp_(p.grad, 1 - mom)
            g = p.grad.lerp(buf, mom) if group["nesterov"] else buf
            update = torch.cat([
                newton_schulz(chunk, group["ns_steps"], self.ns_dtype) * (0.2 * math.sqrt(max(chunk.shape)))
                for chunk in g.split(parts or [g.shape[0]], dim=0)
            ])
            p.mul_(1 - lr * wd)
            p.add_(update.to(p.dtype), alpha=-lr)

    def _adamw(self, group: dict) -> None:
        lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            m, v = state["exp_avg"], state["exp_avg_sq"]
            m.lerp_(p.grad, 1 - beta1)
            v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
            denom = (v / (1 - beta2 ** state["step"])).sqrt_().add_(eps)
            p.mul_(1 - lr * wd)
            p.addcdiv_(m, denom, value=-lr / (1 - beta1 ** state["step"]))


@register("optimizer", "muon")
class Muon(OptimizerComponent):
    summary = "Orthogonalized momentum (Newton-Schulz) for hidden matrices; AdamW for embeddings/vectors."
    paper = "arXiv:2502.16982"
    maturity = "established"
    prior = Prior(1.3, "medium", (51, 253), "constant token-efficiency gain over AdamW (ml_papers3 A1; plan §7.4)")
    rules = (
        Rule("info", lambda ctx: ctx.hardware is not None and not ctx.hardware.bf16,
             "Newton-Schulz runs in fp32 on this GPU (fp16 can overflow); slower, same result"),
    )

    class Params(ComponentParams):
        momentum: float = Field(0.95, ge=0, lt=1)
        nesterov: bool = True
        ns_steps: int = Field(5, gt=0)

    @classmethod
    def build(
        cls, model: nn.Module, train: TrainConfig, params: ComponentParams, device: torch.device
    ) -> torch.optim.Optimizer:
        embeddings = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
        fused = {id(m.weight): m._quark_fused for m in model.modules() if hasattr(m, "_quark_fused")}
        matrices = [p for p in model.parameters() if p.dim() >= 2 and id(p) not in embeddings]
        tables = [p for p in model.parameters() if id(p) in embeddings]  # the tied head lives here
        vectors = [p for p in model.parameters() if p.dim() < 2]
        groups = [
            {"params": matrices, "use_muon": True, "weight_decay": train.weight_decay,
             "parts": [fused.get(id(p)) for p in matrices]},
            {"params": tables, "weight_decay": train.weight_decay},
            {"params": vectors, "weight_decay": 0.0},
        ]
        native_bf16 = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
        return MuonWithAdamW(
            [g for g in groups if g["params"]],
            lr=train.lr,
            betas=train.betas,
            momentum=params.momentum,
            nesterov=params.nesterov,
            ns_steps=params.ns_steps,
            ns_dtype=torch.bfloat16 if native_bf16 else torch.float32,
        )
