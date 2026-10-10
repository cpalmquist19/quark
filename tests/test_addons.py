"""Behaviour specific to individual components; the generic contract is in test_components.py."""

import math

import pytest
import torch

from quark.components import make_optimizer, make_schedule
from quark.components.optim.muon import MuonWithAdamW, newton_schulz
from quark.components.position.alibi import alibi_slopes
from quark.config import ModelConfig, QuarkConfig, TrainConfig
from quark.model import build_model, count_params

from .conftest import tiny_config

VOCAB = 97
BASE = ModelConfig(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=4, seq_len=24)


def variant(**updates) -> ModelConfig:
    return BASE.model_copy(update=updates)


# --- position -----------------------------------------------------------------------------------


def test_alibi_slopes_follow_the_paper():
    assert alibi_slopes(8) == [2.0 ** -(i + 1) for i in range(8)]
    # 6 heads: the 4-head slopes, then every other 8-head slope.
    assert alibi_slopes(6) == pytest.approx([2**-2, 2**-4, 2**-6, 2**-8, 2**-1, 2**-3])


def test_alibi_bias_penalizes_distance():
    model = build_model(variant(slots={**BASE.slots, "position": "alibi"}))
    bias = model.pos.attn_bias(3, 5, 2, torch.device("cpu"), torch.float32)  # queries at 2..4
    assert bias.shape == (4, 3, 5)
    assert torch.all(bias[:, 0, 2] == 0) and torch.all(bias[:, 0, :2] < 0)  # self = 0, past < 0
    assert bias[0, 2, 0] == pytest.approx(-4 * alibi_slopes(4)[0])


@pytest.mark.parametrize("position", ["alibi", "nope"])
def test_position_alternatives_work_with_gqa_cache(position):
    torch.manual_seed(0)
    model = build_model(variant(slots={**BASE.slots, "position": position}, n_kv_heads=2)).eval()
    idx = torch.randint(0, VOCAB, (2, 16))
    with torch.no_grad():
        full = model(idx)
        cache = model.new_cache()
        model(idx[:, :10], cache=cache)
        chunk = model(idx[:, 10:16], cache=cache)
    assert torch.allclose(chunk, full[:, 10:16], atol=1e-5)


# --- toggles ------------------------------------------------------------------------------------


def test_qk_norm_adds_one_gain_per_head_dim():
    with_norm = count_params(variant(toggles={"qk_norm": True}))
    assert with_norm - count_params(BASE) == BASE.n_layers * 2 * BASE.head_dim


def test_z_loss_is_training_only():
    torch.manual_seed(0)
    plain = build_model(BASE)
    torch.manual_seed(0)
    z = build_model(variant(toggles={"z_loss": {"coef": 1e-2}}))
    assert z.head.z_loss_coef == 1e-2
    x, y = torch.randint(0, VOCAB, (2, 24)), torch.randint(0, VOCAB, (2, 24))
    plain.eval(), z.eval()
    assert torch.equal(z(x, y), plain(x, y))  # validation loss stays plain cross-entropy
    z.train()
    with torch.no_grad():
        lse = torch.logsumexp(z(x).float(), dim=-1)
        expected = plain(x, y) + 1e-2 * lse.square().mean()
    assert z(x, y).item() == pytest.approx(expected.item(), rel=1e-5)


# --- mlp ----------------------------------------------------------------------------------------


def test_relu2_matches_swiglu_params_at_the_default_width():
    cfg = ModelConfig()  # d_model 384, ffn_mult 8/3
    assert count_params(cfg.model_copy(update={"slots": {**cfg.slots, "mlp": "relu2"}})) == count_params(cfg)


# --- schedules ----------------------------------------------------------------------------------


@pytest.mark.parametrize("decay", ["linear", "1-sqrt"])
def test_wsd_holds_then_decays_to_min(decay):
    train = TrainConfig(schedule="wsd", lr=1e-3, warmup_steps=10, min_lr_ratio=0.1,
                        params={"schedule": {"decay_frac": 0.2, "decay": decay}})
    s = make_schedule(QuarkConfig(train=train), total_steps=100)
    assert all(s.lr_at(step) == 1e-3 for step in range(10, 80))
    assert s.lr_at(99) == pytest.approx(1e-4 + 0.9e-3 * (1 - (19 / 20 if decay == "linear" else math.sqrt(19 / 20))))
    assert s.lr_at(100) == pytest.approx(1e-4)


def test_wsd_1_sqrt_decays_faster_early():
    def lr(decay):
        train = TrainConfig(schedule="wsd", params={"schedule": {"decay": decay}})
        return make_schedule(QuarkConfig(train=train), total_steps=1000).lr_at(850)

    assert lr("1-sqrt") < lr("linear")


# --- muon ---------------------------------------------------------------------------------------


def test_newton_schulz_orthogonalizes():
    torch.manual_seed(0)
    for shape in [(64, 32), (32, 64)]:
        sv = torch.linalg.svdvals(newton_schulz(torch.randn(shape), 5, torch.float32))
        assert sv.min() > 0.5 and sv.max() < 1.3  # the quintic targets ~[0.7, 1.2], not exactly 1


def test_muon_routes_params_and_splits_fused_matrices():
    model = build_model(variant(n_kv_heads=2))
    opt = make_optimizer(QuarkConfig(model=BASE, train=TrainConfig(optimizer="muon")), model, torch.device("cpu"))
    assert isinstance(opt, MuonWithAdamW)
    muon, tables, vectors = opt.param_groups
    assert muon["use_muon"] and not tables["use_muon"] and not vectors["use_muon"]
    assert tables["params"] == [model.tok_emb.weight]  # the tied head/embedding stays on AdamW
    assert vectors["weight_decay"] == 0.0 and all(p.dim() == 1 for p in vectors["params"])
    attn = model.blocks.get_submodule("0").attn
    parts = dict(zip(map(id, muon["params"]), muon["parts"]))
    assert parts[id(attn.qkv.weight)] == [32, 16, 16]  # q, k, v with 2 kv heads
    assert parts[id(attn.out.weight)] is None


def test_muon_state_round_trips():
    def run(resume_after: int | None):
        torch.manual_seed(0)
        model = build_model(BASE)
        cfg = QuarkConfig(model=BASE, train=TrainConfig(optimizer="muon", lr=3e-3))
        opt = make_optimizer(cfg, model, torch.device("cpu"))
        batch = torch.randint(0, VOCAB, (4, 25), generator=torch.Generator().manual_seed(1))
        for step in range(6):
            if step == resume_after:
                state = opt.state_dict()
                opt = make_optimizer(cfg, model, torch.device("cpu"))
                opt.load_state_dict(state)
            loss = model(batch[:, :-1], batch[:, 1:])
            opt.zero_grad()
            loss.backward()
            opt.step()
        return model.state_dict()

    straight, resumed = run(None), run(3)
    assert all(torch.equal(straight[k], resumed[k]) for k in straight)


# --- end to end ---------------------------------------------------------------------------------


def test_trainer_runs_and_resumes_with_the_new_components(data_dir, tmp_path):
    from quark.train import Trainer

    cfg = tiny_config(optimizer="muon", schedule="wsd")
    cfg = cfg.model_copy(update={"model": cfg.model.model_copy(update={
        "slots": {**cfg.model.slots, "position": "alibi", "mlp": "relu2"},
        "toggles": {"qk_norm": True, "z_loss": True},
    })})
    first = Trainer(cfg, data_dir, tmp_path / "run", device="cpu", max_steps=20, log=lambda _: None).fit()
    assert first.stop_reason == "max_steps"
    resumed = Trainer(cfg, data_dir, tmp_path / "run", device="cpu", resume_from=tmp_path / "run" / "latest.pt",
                      log=lambda _: None)
    assert resumed.step == 20
    result = resumed.fit()
    assert result.stop_reason == "completed" and result.best_val < math.log(512)
