"""Slot interfaces, the KV cache and the build context (docs/component_registry.md §2–3).

Model-slot components are `nn.Module`s built with a `BuildContext`; sub-components are built
through it (`ctx.make("norm", dim)`), so no component hard-codes another slot's choice.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from quark.components.registry import TOGGLE, Component, ComponentParams, define_slot

if TYPE_CHECKING:
    from quark.config import HardwareProfile, ModelConfig, TrainConfig


# --- KV cache -----------------------------------------------------------------------------------


@dataclass
class LayerCache:
    k: torch.Tensor | None = None  # (B, n_kv_heads, T_past, head_dim)
    v: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]


class KVCache:
    """One LayerCache per attention application (a looped stack needs one per unrolled layer)."""

    def __init__(self, n_layers: int):
        self.layers = [LayerCache() for _ in range(n_layers)]

    @property
    def length(self) -> int:
        return self.layers[0].length


# --- build context ------------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildContext:
    cfg: ModelConfig
    choices: Mapping[str, type[Component]]  # model slot -> chosen component class
    params: Mapping[str, ComponentParams]  # slot name or toggle name -> validated params
    toggles: frozenset[str]  # enabled toggle names
    hardware: HardwareProfile | None = None

    def make(self, slot: str, *args, **kwargs) -> nn.Module:
        return self.choices[slot](self, *args, **kwargs)

    def params_for(self, component: type[Component] | Component) -> ComponentParams:
        return self.params[component.name if component.slot == TOGGLE else component.slot]

    def enabled(self, toggle: str) -> bool:
        return toggle in self.toggles


def mark_residual(linear: nn.Linear) -> nn.Linear:
    """Flag a projection that writes into the residual stream; its init is scaled by depth."""
    linear._quark_residual = True
    return linear


def mark_fused(linear: nn.Linear, sizes: list[int]) -> nn.Linear:
    """Flag a weight that stacks several matrices along its output dim (e.g. q, k, v).

    Matrix-aware optimizers such as Muon treat each part as its own matrix.
    """
    if sum(sizes) != linear.out_features:
        raise ValueError(f"fused sizes {sizes} don't add up to out_features {linear.out_features}")
    linear._quark_fused = list(sizes)
    return linear


# --- model slots --------------------------------------------------------------------------------


class PositionEncoding(Component, nn.Module):
    """Built once per model and passed to every attention call. Override the hooks you need."""

    def embed(self, x: torch.Tensor, start: int) -> torch.Tensor:
        """Token embeddings (B, T, d_model) -> embeddings with position information added."""
        return x

    def rotate(self, q: torch.Tensor, k: torch.Tensor, start: int) -> tuple[torch.Tensor, torch.Tensor]:
        """q, k: (B, heads, T, head_dim); `start` is the absolute position of the first token."""
        return q, k

    def attn_bias(self, tq: int, tk: int, start: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        """Additive attention-score bias (n_heads, tq, tk) for queries at start..start+tq-1 and
        keys at 0..tk-1, or None. Attention components that apply it must say so in `requires`."""
        return None


class Attention(Component, nn.Module):
    """forward(x, pos, cache) -> (B, T, d_model). Must support the KV cache."""

    def forward(self, x: torch.Tensor, pos: PositionEncoding, cache: LayerCache | None = None) -> torch.Tensor:
        raise NotImplementedError


class Norm(Component, nn.Module):
    """Constructed as `ctx.make("norm", dim)`."""


class MLP(Component, nn.Module):
    """forward(x) -> x, (B, T, d_model)."""


class Stack(Component, nn.Module):
    """The whole layer stack: layout, residual wiring, cache size and unrolled depth."""

    n_cache_layers: int  # LayerCache entries needed by one forward pass
    unrolled_depth: int  # layers applied per token (> unique layers for looped stacks)

    def forward(self, x: torch.Tensor, pos: PositionEncoding, cache: KVCache | None = None) -> torch.Tensor:
        raise NotImplementedError


class Head(Component, nn.Module):
    """forward(hidden, embedding) -> logits (B, T, vocab); loss(logits, targets) -> scalar."""

    z_loss_coef = 0.0  # set by heads that implement the z_loss toggle

    def forward(self, x: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
        raise NotImplementedError

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean cross-entropy (targets of -1 are ignored). Auxiliary terms such as z-loss are added
        in training mode only, so validation loss stays plain cross-entropy and comparable."""
        logits, targets = logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)
        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        if self.z_loss_coef and self.training:
            keep = targets != -1
            loss = loss + self.z_loss_coef * torch.logsumexp(logits[keep], dim=-1).square().mean()
        return loss


# --- train slots --------------------------------------------------------------------------------


class OptimizerComponent(Component):
    @classmethod
    def build(
        cls, model: nn.Module, train: TrainConfig, params: ComponentParams, device: torch.device
    ) -> torch.optim.Optimizer:
        raise NotImplementedError


class Schedule(Component):
    """Learning rate at each optimizer step, before the trainer's divergence back-off."""

    def __init__(self, train: TrainConfig, params: ComponentParams, total_steps: int):
        self.train, self.params, self.total_steps = train, params, total_steps

    def lr_at(self, step: int) -> float:
        raise NotImplementedError


# --- toggles ------------------------------------------------------------------------------------


class Toggle(Component):
    """An independent add-on. The component in `implemented_by` must list it in `supports`."""

    implemented_by: ClassVar[str] = ""


define_slot("position", "model", PositionEncoding, "Where position information enters the model.")
define_slot("attention", "model", Attention, "Token mixing.")
define_slot("norm", "model", Norm, "Normalization layer type.")
define_slot("mlp", "model", MLP, "Per-token feed-forward.")
define_slot("block", "model", Stack, "The layer stack: layout, residuals, looping.")
define_slot("head", "model", Head, "Hidden states to logits.")
define_slot("optimizer", "train", OptimizerComponent, "Parameter update rule.")
define_slot("schedule", "train", Schedule, "Learning-rate schedule.")
define_slot(TOGGLE, None, Toggle, "Independent add-ons implemented by a slot's component.")
