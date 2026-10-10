"""Text generation from a checkpoint, with a KV cache (plan §12)."""

from __future__ import annotations

from pathlib import Path

import torch

from quark.config import QuarkConfig
from quark.data.tokenizer import eot_id, load_tokenizer
from quark.model import GPT, KVCache, build_model
from quark.train.checkpoint import load_checkpoint


def load_for_inference(checkpoint: str | Path, device: str | torch.device | None = None):
    """Returns (model, tokenizer, config). The tokenizer is read from the checkpoint's run folder."""
    checkpoint = Path(checkpoint)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = load_checkpoint(checkpoint)
    cfg = QuarkConfig.model_validate(state["config"])
    model = build_model(cfg.model)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    tokenizer = load_tokenizer(checkpoint.parent / "tokenizer.json")
    return model, tokenizer, cfg


def _autocast(device: torch.device):
    if device.type != "cuda":
        return torch.autocast("cpu", enabled=False)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    return torch.autocast("cuda", dtype=dtype)


@torch.no_grad()
def generate_ids(
    model: GPT,
    prompt_ids: list[int],
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.9,
    stop_id: int | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Sample continuation ids. temperature 0 = greedy. Stops at `stop_id` if given."""
    device = next(model.parameters()).device
    seq_len = model.cfg.seq_len
    ids = list(prompt_ids) or ([stop_id] if stop_id is not None else [0])
    new: list[int] = []
    cache = KVCache(model.cfg.n_layers)
    pending = ids[-seq_len:]  # tokens not yet in the cache

    with _autocast(device):
        while len(new) < max_new_tokens:
            if cache.length + len(pending) > seq_len:
                # Context is full: re-prefill the last half window (amortizes the cost of sliding).
                cache = KVCache(model.cfg.n_layers)
                pending = (ids + new)[-(seq_len // 2) :]
            x = torch.tensor([pending], device=device)
            logits = model(x, cache=cache)[0, -1].float()
            next_id = _sample(logits, temperature, top_k, top_p, generator)
            new.append(next_id)
            pending = [next_id]
            if stop_id is not None and next_id == stop_id:
                break
    return new


def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float, generator) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if 0 < top_p < 1:
        sorted_probs, order = probs.sort(descending=True)
        keep = sorted_probs.cumsum(0) - sorted_probs < top_p  # always keeps the top token
        sorted_probs = sorted_probs * keep
        probs = torch.zeros_like(probs).scatter(0, order, sorted_probs)
    return int(torch.multinomial(probs / probs.sum(), 1, generator=generator))


def generate_text(
    model: GPT,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.9,
    seed: int | None = None,
    stop_at_eot: bool = True,
) -> str:
    stop = eot_id(tokenizer) if stop_at_eot else None
    generator = None
    if seed is not None:
        generator = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    ids = generate_ids(model, tokenizer.encode(prompt).ids, max_new_tokens, temperature, top_k, top_p, stop, generator)
    if stop is not None and ids and ids[-1] == stop:
        ids = ids[:-1]
    return prompt + tokenizer.decode(ids)
