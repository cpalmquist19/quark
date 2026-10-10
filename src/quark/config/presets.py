"""Hand-picked starting configs until the planner (Phase 3) can produce them.

LR and batch for `tinystories-small` follow the Step Law refit for models under 60M params
(ml_papers3.txt D1): lr* = 0.0985 * N^-0.508 * D^0.238 gives ~2.4e-3, and the optimal batch is
~28k tokens, for N ~ 14M non-embedding params and D ~ 300M tokens. We use 2e-3 and 32k tokens.
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


PRESETS: dict[str, Callable[[], QuarkConfig]] = {
    "smoke": _smoke,
    "tinystories-small": _tinystories_small,
}


def get_preset(name: str) -> QuarkConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r} (known: {', '.join(PRESETS)})")
    return PRESETS[name]()
