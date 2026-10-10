"""Resolving a config against the registry and validating it (docs/component_registry.md §5).

`resolve` turns slot names, params and toggles into component classes; `check` adds the
requires/conflicts, hardware and callable rules. Errors make a config invalid; warnings carry a
planner penalty; info reports a build-time auto-fix. Every issue says why.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from pydantic import ValidationError

from quark.components.base import BuildContext, OptimizerComponent, Schedule
from quark.components.registry import (
    _SLOTS,
    TOGGLE,
    Component,
    ComponentParams,
    Ref,
    Severity,
    Stage,
    _suggest,
    get,
    slots,
)
from quark.config import ModelConfig, QuarkConfig

if TYPE_CHECKING:
    from quark.config import HardwareProfile

ALL_STAGES: tuple[Stage, ...] = ("model", "train")


@dataclass(frozen=True)
class Issue:
    severity: Severity
    source: str  # "attention:mha", "toggle:qk_norm" or "config"
    message: str
    penalty: float = 0.0

    def __str__(self) -> str:
        return f"{self.severity}: {self.source}: {self.message}"


class ComponentError(ValueError):
    def __init__(self, issues: list[Issue]):
        self.issues = issues
        super().__init__("invalid component config:\n" + "\n".join(f"  - {i.source}: {i.message}" for i in issues))


@dataclass(frozen=True)
class Selection:
    choices: dict[str, type[Component]]  # slot -> chosen component (checked stages only)
    params: dict[str, ComponentParams]  # slot name or toggle name -> validated params
    toggles: dict[str, type[Component]]  # enabled toggle name -> toggle class

    @property
    def names(self) -> dict[str, str]:
        return {slot: cls.name for slot, cls in self.choices.items()}


@dataclass
class CheckContext:
    """What a callable rule can look at."""

    cfg: QuarkConfig
    hardware: HardwareProfile | None
    purpose: Mapping[str, Any] | None
    selection: Selection

    def has(self, ref: str) -> bool:
        """True if the config selects this ref, e.g. ctx.has("optimizer:muon")."""
        return Ref.parse(ref).satisfied(self.selection.names, self.selection.toggles)

    @cached_property
    def params_m(self) -> float:
        """Exact total parameter count in millions (built on the meta device)."""
        from quark.model import count_params

        return count_params(self.cfg.model) / 1e6


def resolve(cfg: QuarkConfig, stages: Collection[Stage] = ALL_STAGES) -> tuple[Selection | None, list[Issue]]:
    """Look up every chosen component and validate its params. Selection is None if anything failed."""
    issues: list[Issue] = []
    expected = {s.name for stage in stages for s in slots(stage)}
    chosen: dict[str, str] = {}
    raw_params: dict[str, Any] = {}
    if "model" in stages:
        chosen.update(cfg.model.slots)
        raw_params.update(cfg.model.params)
    if "train" in stages:
        chosen.update(optimizer=cfg.train.optimizer, schedule=cfg.train.schedule)
        raw_params.update(cfg.train.params)

    for slot in sorted(expected - chosen.keys()):
        issues.append(Issue("error", "config", f"no choice for slot {slot!r}"))
    choices: dict[str, type[Component]] = {}
    for slot, name in chosen.items():
        if slot not in expected:
            issues.append(Issue("error", "config", f"unknown slot {slot!r}{_suggest(slot, expected)}"))
            continue
        try:
            choices[slot] = get(slot, name)
        except KeyError as err:
            issues.append(Issue("error", f"{slot}:{name}", err.args[0]))
    for key in raw_params.keys() - expected:
        issues.append(Issue("error", "config", f"params given for unknown slot {key!r}{_suggest(key, expected)}"))

    params: dict[str, ComponentParams] = {}
    for slot, cls in choices.items():
        if (p := _validate_params(cls, raw_params.get(slot), issues)) is not None:
            params[slot] = p

    toggles: dict[str, type[Component]] = {}
    if "model" in stages:  # toggles are stored on the model config
        for name, value in cfg.model.toggles.items():
            if value is False or value is None:
                continue
            try:
                cls = get(TOGGLE, name)
            except KeyError as err:
                issues.append(Issue("error", f"toggle:{name}", err.args[0]))
                continue
            if _SLOTS[cls.implemented_by].stage not in stages:
                continue
            if value is not True and not isinstance(value, dict):
                issues.append(Issue("error", cls.key(), f"value must be true, false or a dict of params, not {value!r}"))
                continue
            if (p := _validate_params(cls, None if value is True else value, issues)) is not None:
                params[name] = p
            toggles[name] = cls
            impl = choices.get(cls.implemented_by)
            if impl is not None and name not in impl.supports:
                issues.append(Issue("error", cls.key(), f"{impl.key()} doesn't implement this toggle"))

    if any(i.severity == "error" for i in issues):
        return None, issues
    return Selection(choices, params, toggles), issues


def check(
    cfg: QuarkConfig,
    hardware: HardwareProfile | None = None,
    purpose: Mapping[str, Any] | None = None,
    stages: Collection[Stage] = ALL_STAGES,
) -> list[Issue]:
    """All issues for `cfg`. Hardware rules are skipped when `hardware` is None."""
    selection, issues = resolve(cfg, stages)
    if selection is None:
        return issues
    ctx = CheckContext(cfg, hardware, purpose, selection)
    names, toggled = selection.names, selection.toggles.keys()
    checkable = set(names) | {TOGGLE}  # refs into unchecked stages can't be judged here
    for cls in (*selection.choices.values(), *selection.toggles.values()):
        source = cls.key()
        for text in cls.requires:
            ref = Ref.parse(text)
            if ref.slot in checkable and not ref.satisfied(names, toggled):
                issues.append(Issue("error", source, f"requires {ref}"))
        for text in cls.conflicts:
            ref = Ref.parse(text)
            if ref.slot in checkable and ref.satisfied(names, toggled):
                issues.append(Issue("error", source, f"conflicts with {ref}"))
        if hardware is not None:
            issues.extend(_hardware_issues(cls, hardware))
        for rule in cls.rules:
            if rule.when(ctx):
                issues.append(Issue(rule.severity, source, rule.reason, rule.penalty))
    return issues


def raise_for_errors(issues: list[Issue]) -> None:
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        raise ComponentError(errors)


def build_context(cfg: ModelConfig, hardware: HardwareProfile | None = None) -> BuildContext:
    """Resolve the model slots for building. Raises ComponentError on unknown names or bad params."""
    selection, issues = resolve(QuarkConfig(model=cfg), stages=("model",))
    if selection is None:
        raise_for_errors(issues)
    return BuildContext(cfg, selection.choices, selection.params, frozenset(selection.toggles), hardware)


def make_optimizer(cfg: QuarkConfig, model: nn.Module, device: torch.device) -> torch.optim.Optimizer:
    cls, params = _train_choice(cfg, "optimizer")
    assert issubclass(cls, OptimizerComponent)
    return cls.build(model, cfg.train, params, device)


def make_schedule(cfg: QuarkConfig, total_steps: int) -> Schedule:
    cls, params = _train_choice(cfg, "schedule")
    assert issubclass(cls, Schedule)
    return cls(cfg.train, params, total_steps)


def _train_choice(cfg: QuarkConfig, slot: str) -> tuple[type[Component], ComponentParams]:
    selection, issues = resolve(cfg, stages=("train",))
    if selection is None:
        raise_for_errors(issues)
    return selection.choices[slot], selection.params[slot]


def _validate_params(cls: type[Component], raw: Any, issues: list[Issue]) -> ComponentParams | None:
    try:
        return cls.Params.model_validate(raw or {})
    except ValidationError as err:
        for e in err.errors():
            field = ".".join(str(part) for part in e["loc"]) or "params"
            issues.append(Issue("error", cls.key(), f"param {field}: {e['msg']}"))
        return None


def _hardware_issues(cls: type[Component], hw: HardwareProfile) -> list[Issue]:
    req, source, problems = cls.hw, cls.key(), []
    if req.min_compute is not None and float(hw.compute_capability) < req.min_compute:
        problems.append(f"needs compute capability >= {req.min_compute:g}; {hw.name} has {hw.compute_capability}")
    if req.bf16 and not hw.bf16:
        problems.append(f"needs native bf16; {hw.name} doesn't have it")
    if req.triton and not hw.triton:
        problems.append(f"needs Triton; not available on {hw.name}")
    if req.flash_attn_pkg and not hw.flash_attn_pkg:
        problems.append(f"needs the flash-attn package; not available on {hw.name}")
    return [Issue("error", source, p) for p in problems]
