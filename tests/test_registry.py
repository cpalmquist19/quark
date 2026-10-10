"""The registry and rule engine, exercised with throwaway components."""

import json

import pytest

from quark.components import (
    ComponentParams,
    HwReq,
    Rule,
    Schedule,
    Toggle,
    check,
    get,
    register,
    registry,
)
from quark.components.schedule.cosine import Cosine
from quark.config import ModelConfig, QuarkConfig, TrainConfig
from quark.hardware import load_profile

from .conftest import tiny_config


@pytest.fixture(autouse=True)
def scratch_registry():
    """Components registered in a test disappear afterwards."""
    snapshot = {slot: dict(entries) for slot, entries in registry._REGISTRY.items()}
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(snapshot)


def with_schedule(name: str, **model_updates) -> QuarkConfig:
    model = ModelConfig().model_copy(update=model_updates)
    return QuarkConfig(model=model, train=TrainConfig(schedule=name))


def messages(issues, severity="error"):
    return [f"{i.source}: {i.message}" for i in issues if i.severity == severity]


# --- registration -------------------------------------------------------------------------------


def test_register_validates_slot_interface_and_refs():
    with pytest.raises(ValueError, match="unknown slot 'positon'"):
        register("positon", "x")(Cosine)
    with pytest.raises(TypeError, match="must subclass PositionEncoding"):
        register("position", "x")(type("NotPosition", (Schedule,), {"summary": "x"}))
    with pytest.raises(ValueError, match="already registered"):
        register("schedule", "cosine")(type("Other", (Schedule,), {}))
    with pytest.raises(ValueError, match="names unknown slot"):
        register("schedule", "bad_ref")(type("BadRef", (Schedule,), {"requires": ("positon:rope",)}))
    with pytest.raises(ValueError, match="must set implemented_by"):
        register("toggle", "orphan")(type("Orphan", (Toggle,), {}))


def test_lookup_suggests_close_names():
    with pytest.raises(KeyError, match="did you mean 'cosine'"):
        get("schedule", "cosin")


# --- resolve: names, params, toggles ------------------------------------------------------------


def test_unknown_names_and_param_slots_are_errors():
    cfg = QuarkConfig(
        model=ModelConfig(slots={**ModelConfig().slots, "mlp": "swigloo"}, params={"attn": {"x": 1}}),
        train=TrainConfig(optimizer="adamw"),
    )
    errors = messages(check(cfg))
    assert any("unknown mlp 'swigloo'; did you mean 'swiglu'" in e for e in errors)
    assert any("params given for unknown slot 'attn'; did you mean 'attention'" in e for e in errors)


def test_params_are_validated_with_defaults():
    @register("schedule", "with_params")
    class WithParams(Cosine):
        summary = "test"

        class Params(ComponentParams):
            decay_frac: float = 0.2

    ok = QuarkConfig(train=TrainConfig(schedule="with_params", params={"schedule": {"decay_frac": 0.5}}))
    assert messages(check(ok)) == []
    bad = QuarkConfig(train=TrainConfig(schedule="with_params", params={"schedule": {"decay_fraction": 0.5}}))
    assert any("param decay_fraction: Extra inputs are not permitted" in e for e in messages(check(bad)))


def test_toggles_must_be_implemented_by_the_chosen_component():
    @register("toggle", "flag")
    class Flag(Toggle):
        summary = "test"
        implemented_by = "schedule"

        class Params(ComponentParams):
            strength: float = 1.0

    @register("schedule", "flag_aware")
    class FlagAware(Cosine):
        summary = "test"
        supports = frozenset({"flag"})

    assert messages(check(with_schedule("cosine", toggles={"flag": True}))) == [
        "toggle:flag: schedule:cosine doesn't implement this toggle"
    ]
    assert messages(check(with_schedule("flag_aware", toggles={"flag": {"strength": 2.0}}))) == []
    assert messages(check(with_schedule("cosine", toggles={"flag": False}))) == []  # off = ignored
    assert any("unknown toggle 'flg'" in e for e in messages(check(with_schedule("cosine", toggles={"flg": True}))))
    assert any("param strength" in e for e in messages(check(with_schedule("flag_aware", toggles={"flag": {"strength": "x"}}))))


# --- rules --------------------------------------------------------------------------------------


def test_requires_and_conflicts():
    @register("schedule", "needs_alibi")
    class NeedsAlibi(Cosine):
        summary = "test"
        requires = ("position:alibi|nope",)

    @register("schedule", "hates_rope")
    class HatesRope(Cosine):
        summary = "test"
        conflicts = ("position:rope",)

    assert messages(check(with_schedule("needs_alibi"))) == ["schedule:needs_alibi: requires position:alibi|nope"]
    assert messages(check(with_schedule("hates_rope"))) == ["schedule:hates_rope: conflicts with position:rope"]
    # A model-only check can't judge refs from a train component, and doesn't include it.
    assert messages(check(with_schedule("needs_alibi"), stages=("model",))) == []


def test_hardware_requirements():
    @register("schedule", "bf16_only")
    class Bf16Only(Cosine):
        summary = "test"
        hw = HwReq(bf16=True, min_compute=8.0)

    cfg = with_schedule("bf16_only")
    errors = messages(check(cfg, hardware=load_profile("colab-t4")))
    assert errors == [
        "schedule:bf16_only: needs compute capability >= 8; colab-t4 has 7.5",
        "schedule:bf16_only: needs native bf16; colab-t4 doesn't have it",
    ]
    assert messages(check(cfg, hardware=load_profile("rtx3070ti-laptop"))) == []
    assert messages(check(cfg)) == []  # unknown hardware: hardware rules are skipped


def test_callable_rules_see_exact_param_count():
    @register("schedule", "small_models_only")
    class SmallOnly(Cosine):
        summary = "test"
        rules = (
            Rule("warning", lambda ctx: ctx.params_m > 10, "no gain measured above 10M params", penalty=0.2),
            Rule("info", lambda ctx: ctx.has("position:rope"), "uses fp32 tables with RoPE"),
        )

    small = check(with_schedule("small_models_only", d_model=64, n_heads=4, n_layers=2))
    assert messages(small, "warning") == []
    big = check(with_schedule("small_models_only"))  # default ModelConfig: ~13M params
    assert messages(big, "warning") == ["schedule:small_models_only: no gain measured above 10M params"]
    assert [i.penalty for i in big if i.severity == "warning"] == [0.2]
    assert messages(big, "info") == ["schedule:small_models_only: uses fp32 tables with RoPE"]


def test_trainer_refuses_invalid_configs_and_records_notes(data_dir, tmp_path):
    from quark.components import ComponentError
    from quark.train import Trainer

    @register("schedule", "noted")
    class Noted(Cosine):
        summary = "test"
        rules = (Rule("warning", lambda ctx: True, "experimental schedule"),)

    bad = tiny_config().model_copy(update={"train": tiny_config().train.model_copy(update={"optimizer": "adamm"})})
    with pytest.raises(ComponentError, match="unknown optimizer 'adamm'"):
        Trainer(bad, data_dir, tmp_path / "bad", device="cpu", log=lambda _: None)

    cfg = tiny_config(schedule="noted")
    logged = []
    Trainer(cfg, data_dir, tmp_path / "run", device="cpu", log=logged.append)
    assert "warning: schedule:noted: experimental schedule" in logged
    events = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert {"event": "component_warning", "message": "schedule:noted: experimental schedule"}.items() <= events[0].items()
