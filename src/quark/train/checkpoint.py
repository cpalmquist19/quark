"""Atomic, portable checkpoints (plan §11.3).

Weights are stored in fp32 and precision is chosen at load time, so a checkpoint moves freely
between the bf16 laptop and the fp16 Colab T4.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(state: dict, path: str | Path) -> Path:
    """Write to a temp file, then rename, so a crash or disconnect mid-save can't corrupt `path`."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path) -> dict:
    # weights_only=False: our own files also hold RNG states and plain Python/numpy objects.
    return torch.load(path, map_location="cpu", weights_only=False)


def rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])


def prune_step_checkpoints(run_dir: str | Path, keep: int) -> None:
    """Keep the newest `keep` step_<n>.pt files; latest.pt and best.pt are never touched."""
    steps = sorted(Path(run_dir).glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for old in steps[: max(0, len(steps) - keep)]:
        old.unlink(missing_ok=True)
