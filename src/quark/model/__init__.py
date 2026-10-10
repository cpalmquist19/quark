"""GPT assembled from component slots."""

import torch

from quark.components import KVCache, check, raise_for_errors
from quark.config import HardwareProfile, ModelConfig, QuarkConfig
from quark.model.gpt import GPT


def build_model(cfg: ModelConfig, hardware: HardwareProfile | None = None) -> GPT:
    """Build the model for `cfg`. Raises ComponentError, with reasons, if the slot choices are invalid."""
    raise_for_errors(check(QuarkConfig(model=cfg), hardware, stages=("model",)))
    return GPT(cfg, hardware)


def count_params(cfg: ModelConfig, non_embedding: bool = False) -> int:
    """Exact parameter count, without allocating memory (the model is built on the meta device)."""
    with torch.device("meta"):
        return GPT(cfg).num_params(non_embedding)


__all__ = ["GPT", "KVCache", "build_model", "count_params"]
