"""Tokenizers: train a byte-level BPE on the data, or load a pretrained one (plan §7.4 rule)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from quark.config import TokenizerConfig
from quark.data.sources import EOT

# Short names for pretrained tokenizers on the Hugging Face Hub.
PRETRAINED = {
    "smollm2": "HuggingFaceTB/SmolLM2-135M",
    "gpt2": "openai-community/gpt2",
}


def train_bpe(texts: Iterable[str], vocab_size: int) -> Tokenizer:
    """Byte-level BPE (GPT-2 style): any text round-trips, no unknown tokens."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def load_pretrained(name: str) -> Tokenizer:
    return Tokenizer.from_pretrained(PRETRAINED.get(name, name))


def build_tokenizer(cfg: TokenizerConfig, sample_texts: Iterable[str]) -> Tokenizer:
    if cfg.type == "train":
        return train_bpe(sample_texts, cfg.size)
    return load_pretrained(cfg.name)


def eot_id(tok: Tokenizer) -> int:
    for token in (EOT, "</s>", "<|im_end|>", "<eos>"):
        token_id = tok.token_to_id(token)
        if token_id is not None:
            return token_id
    raise ValueError("tokenizer has no end-of-text token")


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))
