"""Contract tests every registered component gets for free (docs/component_registry.md §7).

Each model component is swapped into a tiny baseline config, and each model toggle is switched on
in it; combinations the registry rejects are skipped with the reason. A new component only needs
tests for what is specific to it.
"""

import math

import pytest
import torch

from quark.components import TOGGLE, check, components, get, make_optimizer, make_schedule, names, slots
from quark.config import ModelConfig, QuarkConfig, TrainConfig
from quark.model import build_model, count_params

VOCAB = 97
BASE = ModelConfig(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=4, seq_len=24)
MODEL_SLOTS = {spec.name for spec in slots("model")}
MODEL_VARIANTS = [
    pytest.param(spec.name, name, id=f"{spec.name}:{name}") for spec in slots("model") for name in names(spec.name)
] + [
    pytest.param(TOGGLE, name, id=f"toggle:{name}")
    for name in names(TOGGLE)
    if get(TOGGLE, name).implemented_by in MODEL_SLOTS
]


def config_with(slot: str, name: str) -> ModelConfig:
    if slot == TOGGLE:
        cfg = BASE.model_copy(update={"toggles": {name: True}})
    else:
        cfg = BASE.model_copy(update={"slots": {**BASE.slots, slot: name}})
    errors = [i for i in check(QuarkConfig(model=cfg), stages=("model",)) if i.severity == "error"]
    if errors:
        pytest.skip(f"not valid with the baseline: {'; '.join(i.message for i in errors)}")
    return cfg


def make(cfg: ModelConfig, seed: int = 0):
    torch.manual_seed(seed)
    return build_model(cfg).eval()


# --- model slots --------------------------------------------------------------------------------


@pytest.mark.parametrize("slot, name", MODEL_VARIANTS)
def test_shapes_and_initial_loss(slot, name):
    model = make(config_with(slot, name))
    idx = torch.randint(0, VOCAB, (3, 24))
    assert model(idx).shape == (3, 24, VOCAB)
    loss = model(idx, targets=torch.randint(0, VOCAB, (3, 24)))
    assert torch.isfinite(loss)
    assert abs(loss.item() - math.log(VOCAB)) < 0.5  # a fresh model predicts roughly uniformly


@pytest.mark.parametrize("slot, name", MODEL_VARIANTS)
def test_causal(slot, name):
    model = make(config_with(slot, name))
    a = torch.randint(0, VOCAB, (1, 24))
    b = a.clone()
    b[0, 12:] = torch.randint(0, VOCAB, (12,))
    with torch.no_grad():
        assert torch.allclose(model(a)[0, :12], model(b)[0, :12], atol=1e-5)


@pytest.mark.parametrize("slot, name", MODEL_VARIANTS)
def test_kv_cache_matches_full_forward(slot, name):
    model = make(config_with(slot, name))
    idx = torch.randint(0, VOCAB, (2, 20))
    with torch.no_grad():
        full = model(idx)
        cache = model.new_cache()
        prefill = model(idx[:, :12], cache=cache)
        steps = [model(idx[:, i : i + 1], cache=cache) for i in range(12, 16)]
        chunk = model(idx[:, 16:20], cache=cache)
    assert torch.allclose(prefill, full[:, :12], atol=1e-5)
    for offset, step in enumerate(steps):
        assert torch.allclose(step[:, 0], full[:, 12 + offset], atol=1e-5)
    assert torch.allclose(chunk, full[:, 16:20], atol=1e-5)


@pytest.mark.parametrize("slot, name", MODEL_VARIANTS)
def test_overfits_one_batch(slot, name):
    model = make(config_with(slot, name)).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    batch = torch.randint(0, VOCAB, (4, 25))
    x, y = batch[:, :-1], batch[:, 1:]
    for _ in range(300):
        loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


@pytest.mark.parametrize("slot, name", MODEL_VARIANTS)
def test_meta_param_count_matches_real(slot, name):
    cfg = config_with(slot, name)
    assert count_params(cfg) == make(cfg).num_params()


# --- train slots --------------------------------------------------------------------------------


@pytest.mark.parametrize("name", names("optimizer"))
def test_optimizer_reduces_loss(name):
    cfg = QuarkConfig(model=BASE, train=TrainConfig(optimizer=name, lr=3e-3))
    model = make(BASE).train()
    opt = make_optimizer(cfg, model, torch.device("cpu"))
    batch = torch.randint(0, VOCAB, (4, 25))
    first = None
    for _ in range(30):
        loss = model(batch[:, :-1], batch[:, 1:])
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first - 0.5


@pytest.mark.parametrize("name", names("schedule"))
def test_schedule_warms_up_and_stays_in_range(name):
    train = TrainConfig(schedule=name, lr=1e-3, warmup_steps=10)
    schedule = make_schedule(QuarkConfig(train=train), total_steps=100)
    lrs = [schedule.lr_at(step) for step in range(100)]
    assert all(math.isfinite(lr) and 0 < lr <= train.lr * (1 + 1e-9) for lr in lrs)
    assert lrs[:10] == sorted(lrs[:10])  # non-decreasing through warmup
    assert lrs[-1] <= lrs[10]


# --- metadata -----------------------------------------------------------------------------------


@pytest.mark.parametrize("component", components(), ids=lambda c: c.key())
def test_metadata_is_complete(component):
    assert component.summary, "every component needs a one-line summary"
    assert component.maturity in ("experimental", "promising", "established")
    component.Params()  # every param has a default
    for toggle in component.supports:
        assert toggle in names(TOGGLE), f"supports unknown toggle {toggle!r}"
