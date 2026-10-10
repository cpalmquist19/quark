"""Hand-picked starting configs until the planner (Phase 3) can produce them.

LR and batch for `tinystories-small` follow the Step Law refit for models under 60M params
(ml_papers3.txt D1): lr* = 0.0985 * N^-0.508 * D^0.238 gives ~2.4e-3, and the optimal batch is
~28k tokens, for N ~ 14M non-embedding params and D ~ 300M tokens. We use 2e-3 and 32k tokens.

`ablation-s` / `ablation-m` are the rungs of the component-testing ladder (plan §18.2): the
chatbot target's seq_len (2048), the same deep-and-thin aspect ratio (d_model / n_layers = 32) and
~6 tokens per non-embedding param, with the D1 LR at each size. Both use 16k-token batches: D1's
optimum at these budgets is 4-19k tokens, and 16k keeps the T4 busy at one value for both rungs.
They are instruments for comparing components, not models to keep.
"""

from __future__ import annotations

from collections.abc import Callable

from quark.config.schema import ModelConfig, QuarkConfig, TrainConfig


def _smoke() -> QuarkConfig:
    """~1M params; a few minutes end to end. For checking the pipeline, not for quality."""
    return QuarkConfig(
        name="smoke",
        model=ModelConfig(d_model=128, n_layers=2, n_heads=4, seq_len=128),
        train=TrainConfig(
            lr=1e-3,
            warmup_steps=20,
            max_tokens=2_000_000,
            micro_batch=16,
            eval_every_steps=50,
            eval_batches=5,
            checkpoint_every_steps=100,
        ),
    )


def _tinystories_small() -> QuarkConfig:
    """~17M params (14M non-embedding). Coherent TinyStories after ~100-200M tokens."""
    return QuarkConfig(
        name="tinystories-small",
        model=ModelConfig(d_model=384, n_layers=8, n_heads=6, seq_len=512),
        train=TrainConfig(
            lr=2e-3,
            warmup_steps=200,
            max_tokens=200_000_000,
            micro_batch=32,
            grad_accum=2,  # 32 x 2 x 512 = 32,768 tokens per optimizer step
            eval_every_steps=250,
            checkpoint_every_steps=500,
        ),
    )


def _ablation(name: str, d_model: int, n_layers: int, n_heads: int, lr: float, max_tokens: int,
              eval_every: int) -> QuarkConfig:
    return QuarkConfig(
        name=name,
        model=ModelConfig(d_model=d_model, n_layers=n_layers, n_heads=n_heads, seq_len=2048),
        train=TrainConfig(
            lr=lr,
            warmup_steps=200,
            max_tokens=max_tokens,
            micro_batch=8,  # 8 x 2048 = 16,384 tokens per optimizer step
            eval_every_steps=eval_every,
            eval_batches=20,
            checkpoint_every_steps=1000,
        ),
    )


def _ablation_s() -> QuarkConfig:
    """~8.5M params (6.4M non-embedding), 40M tokens: ~2.4k steps. The cheap rung; every component."""
    return _ablation("ablation-s", 256, 8, 4, lr=2.2e-3, max_tokens=40_000_000, eval_every=250)


def _ablation_m() -> QuarkConfig:
    """~38M params (34M non-embedding), 200M tokens: ~12k steps. Finalists from ablation-s only."""
    return _ablation("ablation-m", 448, 14, 7, lr=1.4e-3, max_tokens=200_000_000, eval_every=1000)


PRESETS: dict[str, Callable[[], QuarkConfig]] = {
    "smoke": _smoke,
    "tinystories-small": _tinystories_small,
    "ablation-s": _ablation_s,
    "ablation-m": _ablation_m,
}


def get_preset(name: str) -> QuarkConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r} (known: {', '.join(PRESETS)})")
    return PRESETS[name]()
