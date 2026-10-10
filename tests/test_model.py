import pytest
import torch

from quark.components import ComponentError
from quark.config import ModelConfig
from quark.config.presets import get_preset
from quark.model import build_model, count_params

CFG = ModelConfig(vocab_size=97, d_model=32, n_layers=2, n_heads=4, seq_len=24)


def make(cfg=CFG, seed=0):
    torch.manual_seed(seed)
    return build_model(cfg).eval()


def test_shapes_and_loss():
    model = make()
    idx = torch.randint(0, 97, (3, 24))
    assert model(idx).shape == (3, 24, 97)
    loss = model(idx, targets=torch.randint(0, 97, (3, 24)))
    assert loss.ndim == 0 and torch.isfinite(loss)
    # A fresh model predicts roughly uniformly: loss ~ ln(vocab).
    assert abs(loss.item() - torch.log(torch.tensor(97.0)).item()) < 0.5


def test_causal():
    model = make()
    a = torch.randint(0, 97, (1, 24))
    b = a.clone()
    b[0, 12:] = torch.randint(0, 97, (12,))  # change only the future
    with torch.no_grad():
        assert torch.allclose(model(a)[0, :12], model(b)[0, :12], atol=1e-5)


def test_tied_head_and_param_count():
    model = make()
    n_emb = 97 * 32
    assert model.num_params() - model.num_params(non_embedding=True) == n_emb
    # No separate output matrix exists: the head reuses the embedding.
    assert not any(p.shape == (97, 32) for name, p in model.named_parameters() if "tok_emb" not in name)


@pytest.mark.parametrize("n_kv_heads", [None, 2, 1])
def test_kv_cache_matches_full_forward(n_kv_heads):
    model = make(CFG.model_copy(update={"n_kv_heads": n_kv_heads}))
    idx = torch.randint(0, 97, (2, 20))
    with torch.no_grad():
        full = model(idx)
        cache = model.new_cache()
        prefill = model(idx[:, :12], cache=cache)
        steps = [model(idx[:, i : i + 1], cache=cache) for i in range(12, 16)]
        chunk = model(idx[:, 16:20], cache=cache)  # several tokens after a cached prefix
    assert torch.allclose(prefill, full[:, :12], atol=1e-5)
    for offset, step in enumerate(steps):
        assert torch.allclose(step[:, 0], full[:, 12 + offset], atol=1e-5)
    assert torch.allclose(chunk, full[:, 16:20], atol=1e-5)


def test_rejects_too_long_input():
    with pytest.raises(ValueError, match="exceeds seq_len"):
        make()(torch.zeros(1, 25, dtype=torch.long))


def test_unknown_components_fail_clearly():
    with pytest.raises(ComponentError, match=r"unknown position 'ropee'; did you mean 'rope'\?"):
        build_model(CFG.model_copy(update={"slots": {**CFG.slots, "position": "ropee"}}))
    with pytest.raises(ComponentError, match="param base: Input should be a valid number"):
        build_model(CFG.model_copy(update={"params": {"position": {"base": "big"}}}))


def test_component_params_reach_the_model():
    model = build_model(CFG.model_copy(update={"params": {"position": {"base": 500.0}}}))
    default = make()
    assert not torch.allclose(model.pos.cos, default.pos.cos)


# --- compatibility with checkpoints and configs written before the component registry ---------


def test_checkpoint_keys_are_stable():
    keys = list(make().state_dict())
    assert keys == [
        "tok_emb.weight",
        *(
            f"blocks.{i}.{name}"
            for i in range(2)
            for name in (
                "attn_norm.weight",
                "attn.qkv.weight",
                "attn.out.weight",
                "mlp_norm.weight",
                "mlp.gate_up.weight",
                "mlp.down.weight",
            )
        ),
        "norm.weight",
    ]


def test_init_matches_pre_registry_model():
    # Fingerprint of build_model(CFG) with seed 0 from the Phase 1 GPT, so the same seed keeps
    # giving the same init and earlier baselines stay comparable.
    weights = make().state_dict().values()
    fingerprint = sum((i + 1) * t.double().abs().sum().item() for i, t in enumerate(weights))
    assert fingerprint == pytest.approx(5026.476320238139, rel=1e-9)


def test_config_hashes_are_stable():
    # Hashes from before the registry: a changed hash would stop in-progress runs from resuming.
    assert get_preset("smoke").config_hash() == "1fe908b783fd"
    assert get_preset("tinystories-small").config_hash() == "37cee61dd2e4"


def test_meta_param_count_is_exact():
    assert count_params(CFG) == make().num_params()
    assert count_params(CFG, non_embedding=True) == make().num_params(non_embedding=True)
    assert count_params(get_preset("tinystories-small").model) == 17_308_032


def test_overfits_one_batch():
    torch.manual_seed(0)
    model = build_model(CFG)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    batch = torch.randint(0, 97, (4, 25))
    x, y = batch[:, :-1], batch[:, 1:]
    for _ in range(300):
        loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
