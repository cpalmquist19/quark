import pytest
import torch

from quark.config import ModelConfig
from quark.model import KVCache, build_model

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
        cache = KVCache(CFG.n_layers)
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


def test_unsupported_slots_fail_clearly():
    with pytest.raises(NotImplementedError, match="Phase 1 supports only"):
        build_model(CFG.model_copy(update={"slots": {**CFG.slots, "position": "alibi"}}))


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
