"""Slots, the Component base class and `@register` (docs/component_registry.md §2–3).

A slot is a mutually exclusive choice (one position encoding, one optimizer, ...). Toggles are
independent add-ons registered in the pseudo-slot "toggle"; each names the slot whose component
implements it. Slot interfaces are defined in `quark.components.base`, which calls `define_slot`.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, TypeVar

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from quark.components.check import CheckContext

Maturity = Literal["experimental", "promising", "established"]
Severity = Literal["error", "warning", "info"]
Stage = Literal["model", "train"]
TOGGLE = "toggle"


class ComponentParams(BaseModel):
    """Base for a component's settings. Defaults should be the paper's values."""

    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True)
class HwReq:
    """Hard hardware requirements, checked against the hardware profile."""

    min_compute: float | None = None  # e.g. 8.0 for Ampere-only kernels
    bf16: bool = False
    triton: bool = False
    flash_attn_pkg: bool = False


@dataclass(frozen=True)
class Prior:
    """A paper's claimed effect, for the planner's quality estimate (plan §7.4)."""

    multiplier: float  # effective-compute multiplier; 1.0 = no effect
    confidence: Literal["low", "medium", "high"]
    validated_params_m: tuple[float, float]  # model sizes (M params) the claim was measured at
    source: str = ""


@dataclass(frozen=True)
class Rule:
    """A soft or informational rule. `when(ctx)` returns True when the reason applies."""

    severity: Severity
    when: Callable[[CheckContext], bool]
    reason: str
    penalty: float = 0.0  # planner ranking penalty for warnings


@dataclass(frozen=True)
class Ref:
    """A reference in `requires` / `conflicts`: "slot:name", "slot:a|b" or "toggle:name"."""

    slot: str
    names: frozenset[str]

    @classmethod
    def parse(cls, text: str) -> Ref:
        slot, sep, names = text.partition(":")
        if not sep or not slot or not names:
            raise ValueError(f"bad component ref {text!r}; expected 'slot:name' or 'slot:a|b'")
        if slot not in _SLOTS:
            raise ValueError(f"component ref {text!r} names unknown slot {slot!r}")
        return cls(slot, frozenset(names.split("|")))

    def satisfied(self, chosen: Mapping[str, str], toggles: Collection[str]) -> bool:
        if self.slot == TOGGLE:
            return bool(self.names & set(toggles))
        return chosen.get(self.slot) in self.names

    def __str__(self) -> str:
        return f"{self.slot}:{'|'.join(sorted(self.names))}"


class Component:
    """Metadata shared by every technique. Subclass a slot interface from `base`, not this."""

    slot: ClassVar[str] = ""  # set by @register
    name: ClassVar[str] = ""  # set by @register
    summary: ClassVar[str] = ""
    paper: ClassVar[str | None] = None
    maturity: ClassVar[Maturity] = "experimental"
    requires: ClassVar[tuple[str, ...]] = ()
    conflicts: ClassVar[tuple[str, ...]] = ()
    hw: ClassVar[HwReq] = HwReq()
    rules: ClassVar[tuple[Rule, ...]] = ()
    supports: ClassVar[frozenset[str]] = frozenset()  # toggles this component implements
    prior: ClassVar[Prior | None] = None
    Params: ClassVar[type[ComponentParams]] = ComponentParams

    @classmethod
    def key(cls) -> str:
        return f"{cls.slot}:{cls.name}"


@dataclass(frozen=True)
class SlotSpec:
    name: str
    stage: Stage | None  # None for toggles: their stage is the stage of `implemented_by`
    base: type
    summary: str


_SLOTS: dict[str, SlotSpec] = {}
_REGISTRY: dict[str, dict[str, type[Component]]] = {}

C = TypeVar("C", bound=type[Component])


def define_slot(name: str, stage: Stage | None, base: type, summary: str) -> None:
    _SLOTS[name] = SlotSpec(name, stage, base, summary)
    _REGISTRY.setdefault(name, {})


def register(slot: str, name: str) -> Callable[[C], C]:
    """Class decorator: add a component to the registry under `slot:name`."""

    def decorate(cls: C) -> C:
        spec = _SLOTS.get(slot)
        if spec is None:
            raise ValueError(f"unknown slot {slot!r} (known: {', '.join(_SLOTS)})")
        if not issubclass(cls, spec.base):
            raise TypeError(f"{cls.__name__} must subclass {spec.base.__name__} to register in slot {slot!r}")
        existing = _REGISTRY[slot].get(name)
        if existing is not None and existing.__qualname__ != cls.__qualname__:  # same class again = module reload
            raise ValueError(f"{slot}:{name} is already registered by {existing.__qualname__}")
        if slot == TOGGLE:
            if name in _SLOTS:
                raise ValueError(f"toggle name {name!r} clashes with a slot name")
            target = getattr(cls, "implemented_by", "")
            if target not in _SLOTS or target == TOGGLE:
                raise ValueError(f"toggle {name!r} must set implemented_by to a slot (got {target!r})")
        for ref in (*cls.requires, *cls.conflicts):
            Ref.parse(ref)
        cls.slot, cls.name = slot, name
        _REGISTRY[slot][name] = cls
        return cls

    return decorate


def slots(stage: Stage | None = None) -> list[SlotSpec]:
    """Slot specs in definition order. `stage` filters model/train slots; toggles are excluded then."""
    return [s for s in _SLOTS.values() if stage is None or s.stage == stage]


def names(slot: str) -> list[str]:
    return sorted(_REGISTRY.get(slot, {}))


def get(slot: str, name: str) -> type[Component]:
    """Look up a component, with a "did you mean" hint for typos."""
    if slot not in _SLOTS:
        raise KeyError(f"unknown slot {slot!r}{_suggest(slot, _SLOTS)}")
    try:
        return _REGISTRY[slot][name]
    except KeyError:
        known = names(slot)
        raise KeyError(
            f"unknown {slot} {name!r}{_suggest(name, known)} (known: {', '.join(known) or 'none yet'})"
        ) from None


def components(slot: str | None = None) -> list[type[Component]]:
    keys = [slot] if slot is not None else list(_SLOTS)
    return [_REGISTRY[k][n] for k in keys for n in names(k)]


def _suggest(word: str, options: Collection[str]) -> str:
    close = difflib.get_close_matches(word, list(options), n=1)
    return f"; did you mean {close[0]!r}?" if close else ""
