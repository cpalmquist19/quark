"""Baseline GPT assembled from component slots."""

from quark.config import DEFAULT_SLOTS, ModelConfig
from quark.model.gpt import GPT, KVCache


def build_model(cfg: ModelConfig) -> GPT:
    """Build the model for `cfg`. Phase 1 supports only the baseline slot choices."""
    unsupported = {slot: name for slot, name in cfg.slots.items() if DEFAULT_SLOTS.get(slot) != name}
    if unsupported:
        raise NotImplementedError(
            f"slot choices {unsupported} aren't available yet; Phase 1 supports only {DEFAULT_SLOTS}"
        )
    if cfg.toggles:
        raise NotImplementedError(f"toggles {sorted(cfg.toggles)} aren't available yet (component registry, Phase 2)")
    return GPT(cfg)


__all__ = ["GPT", "KVCache", "build_model"]
