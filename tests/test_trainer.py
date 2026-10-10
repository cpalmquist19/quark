import json
import math

import pytest
import torch

from quark.infer import generate_text, load_for_inference
from quark.train import ConfigMismatchError, Trainer
from quark.train.checkpoint import load_checkpoint
from quark.train.monitors import EarlyStopper, GapMonitor, SpikeDetector
from tests.conftest import tiny_config


def run(cfg, data_dir, run_dir, **kwargs):
    logs = []
    trainer = Trainer(cfg, data_dir, run_dir, device="cpu", log=logs.append, **kwargs)
    return trainer, trainer.fit(), logs


def test_fit_produces_run_artifacts(data_dir, tmp_path):
    trainer, result, logs = run(tiny_config(), data_dir, tmp_path / "run")
    assert result.stop_reason == "completed" and result.steps == 40
    for name in ("latest.pt", "best.pt", "metrics.jsonl", "training.png", "config.yaml", "run.json", "tokenizer.json"):
        assert (tmp_path / "run" / name).exists(), name
    assert len(list((tmp_path / "run").glob("step_*.pt"))) == 2  # keep_last

    records = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    evals = [r for r in records if r["type"] == "eval"]
    assert len(evals) == 4
    assert evals[-1]["val_loss"] < math.log(300)  # learned something beyond uniform
    train = [r for r in records if r["type"] == "train"]
    assert {"loss", "loss_ema", "lr", "grad_norm", "tokens_per_sec", "eta_s", "epoch"} <= set(train[-1])
    # Warmup then cosine decay to min_lr_ratio * lr.
    assert train[0]["lr"] < train[2]["lr"]
    assert trainer.lr_at(39) == pytest.approx(3e-4, rel=0.05)


def test_resume_is_exact(data_dir, tmp_path):
    cfg = tiny_config()
    _, straight, _ = run(cfg, data_dir, tmp_path / "a")
    _, half, _ = run(cfg, data_dir, tmp_path / "b", max_steps=20)
    assert half.stop_reason == "max_steps" and half.steps == 20
    _, resumed, logs = run(cfg, data_dir, tmp_path / "b", resume_from=tmp_path / "b" / "latest.pt")
    assert any("resumed" in line for line in logs)
    assert resumed.steps == 40

    a = load_checkpoint(tmp_path / "a" / "latest.pt")["model"]
    b = load_checkpoint(tmp_path / "b" / "latest.pt")["model"]
    for key in a:
        assert torch.allclose(a[key], b[key], atol=1e-6), key


def test_resume_refuses_changed_config(data_dir, tmp_path):
    run(tiny_config(), data_dir, tmp_path / "r", max_steps=10)
    changed = tiny_config(lr=1e-2)
    with pytest.raises(ConfigMismatchError):
        Trainer(changed, data_dir, tmp_path / "r", device="cpu", resume_from=tmp_path / "r" / "latest.pt", log=lambda _: None)
    # Runtime-only settings may change freely.
    Trainer(tiny_config(eval_every_steps=7), data_dir, tmp_path / "r", device="cpu",
            resume_from=tmp_path / "r" / "latest.pt", log=lambda _: None)


def test_generation_from_checkpoint(data_dir, tmp_path):
    run(tiny_config(), data_dir, tmp_path / "g")
    model, tok, cfg = load_for_inference(tmp_path / "g" / "best.pt", device="cpu")
    text = generate_text(model, tok, "The cat", max_new_tokens=40, temperature=0)
    assert text.startswith("The cat") and len(text) > len("The cat")
    assert text == generate_text(model, tok, "The cat", max_new_tokens=40, temperature=0)  # greedy is deterministic
    # Longer than seq_len (32): the KV cache must slide the window without errors.
    long = generate_text(model, tok, "The", max_new_tokens=100, temperature=1.0, seed=1, stop_at_eot=False)
    assert len(tok.encode(long).ids) > 32


def test_early_stopper():
    stop = EarlyStopper(patience=2, min_delta=0.01)
    assert stop.update(2.0) and not stop.should_stop(epoch=0)
    assert not stop.update(1.995)  # improvement smaller than min_delta
    assert not stop.update(2.1)
    assert stop.should_stop(epoch=0)
    stop2 = EarlyStopper(patience=4, min_delta=0.0)
    stop2.update(1.0)
    stop2.update(1.1)
    stop2.update(1.2)
    assert not stop2.should_stop(epoch=0) and stop2.should_stop(epoch=1.2)  # patience halves after epoch 1


def test_gap_monitor():
    gap = GapMonitor(window=3)
    assert gap.update(2.0, 2.1) is None
    assert gap.update(1.9, 2.1) is None
    assert "overfitting" in gap.update(1.8, 2.2)
    healthy = GapMonitor(window=3)
    for t, v in [(2.0, 2.1), (1.9, 2.0), (1.8, 1.9)]:
        assert healthy.update(t, v) is None
    # Gap grows only because train falls faster, while val still improves: not overfitting.
    lagging = GapMonitor(window=3)
    for t, v in [(3.0, 2.9), (2.6, 2.6), (2.3, 2.4)]:
        assert lagging.update(t, v) is None


def test_spike_detector():
    spikes = SpikeDetector(sigma=6, window=20)
    for i in range(20):
        assert spikes.check(3.0 + 0.01 * (i % 3)) is None
    assert "spike" in spikes.check(9.0)
    assert "non-finite" in spikes.check(float("nan"))


def test_divergence_rolls_back(data_dir, tmp_path, monkeypatch):
    cfg = tiny_config(max_tokens=30 * 4 * 32)
    trainer = Trainer(cfg, data_dir, tmp_path / "d", device="cpu", log=lambda _: None)
    real_step = trainer._train_step
    calls = {"n": 0}

    def flaky_step():
        calls["n"] += 1
        loss, gn, lr, skipped = real_step()
        return (float("nan") if calls["n"] == 15 else loss), gn, lr, skipped

    monkeypatch.setattr(trainer, "_train_step", flaky_step)
    result = trainer.fit()
    assert trainer.rollbacks == 1 and trainer.lr_mult == 0.5
    assert result.stop_reason == "completed"
    events = [json.loads(line) for line in (tmp_path / "d" / "metrics.jsonl").read_text().splitlines()]
    assert any(e["type"] == "event" and e["event"] == "divergence" for e in events)
