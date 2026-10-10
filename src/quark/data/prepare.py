"""Tokenize a text source once into memory-mapped token files (nanoGPT style).

Output directory:
  tokenizer.json   the tokenizer used (copied into every run so checkpoints are self-contained)
  train.bin        uint16 (or uint32 for vocabularies > 65,535) token ids, documents separated by EOT
  val.bin          same format; documents are assigned by a hash of their text, so the split is
                   deterministic and stable even if the source order changes
  meta.json        written last: its presence means the directory is complete
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from itertools import islice
from pathlib import Path

import numpy as np

import quark
from quark.config import DataConfig
from quark.data.sources import iter_documents
from quark.data.tokenizer import build_tokenizer, eot_id

BATCH_DOCS = 1000


def is_val(text: str, val_fraction: float) -> bool:
    bucket = int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big") % 1_000_000
    return bucket < val_fraction * 1_000_000


def prepare_dataset(
    data: DataConfig,
    out_dir: str | Path,
    max_docs: int | None = None,
    tokenizer_sample_docs: int = 200_000,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Tokenize `data.source` into `out_dir`. Returns the meta dict. `progress(docs, tokens)`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").unlink(missing_ok=True)
    started = time.time()

    def docs(limit: int | None = max_docs):
        return iter_documents(data.source, data.split, data.text_field, limit)

    sample = tokenizer_sample_docs if max_docs is None else min(tokenizer_sample_docs, max_docs)
    tok = build_tokenizer(data.tokenizer, docs(sample))
    tok.save(str(out / "tokenizer.json"))
    vocab_size = tok.get_vocab_size()
    eot = eot_id(tok)
    dtype = np.uint16 if vocab_size <= 65_535 else np.uint32

    counts = {"train": [0, 0], "val": [0, 0]}  # [docs, tokens]
    files = {split: open(out / f"{split}.bin.tmp", "wb") for split in counts}
    try:
        stream = docs()
        while batch := list(islice(stream, BATCH_DOCS)):
            for text, enc in zip(batch, tok.encode_batch(batch)):
                split = "val" if is_val(text, data.val_fraction) else "train"
                ids = np.asarray(enc.ids + [eot], dtype=dtype)
                files[split].write(ids.tobytes())
                counts[split][0] += 1
                counts[split][1] += len(ids)
            if progress is not None:
                progress(counts["train"][0] + counts["val"][0], counts["train"][1] + counts["val"][1])
    finally:
        for fh in files.values():
            fh.close()
    for split in counts:
        os.replace(out / f"{split}.bin.tmp", out / f"{split}.bin")

    if counts["val"][1] == 0 or counts["train"][1] == 0:
        raise ValueError(f"empty split after tokenizing {data.source}: {counts}; check val_fraction / source")

    meta = {
        "source": data.source,
        "split": data.split,
        "text_field": data.text_field,
        "tokenizer": data.tokenizer.model_dump(),
        "vocab_size": vocab_size,
        "eot_id": eot,
        "dtype": np.dtype(dtype).name,
        "val_fraction": data.val_fraction,
        "train_docs": counts["train"][0],
        "train_tokens": counts["train"][1],
        "val_docs": counts["val"][0],
        "val_tokens": counts["val"][1],
        "max_docs": max_docs,
        "seconds": round(time.time() - started, 1),
        "quark_version": quark.__version__,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def default_data_dir(data: DataConfig, root: str | Path = "data") -> Path:
    """e.g. data/tinystories-bpe8192 for roneneldan/TinyStories with the default tokenizer."""
    stem = Path(data.source.rstrip("/\\")).name.lower().replace(" ", "-") or "data"
    return Path(root) / f"{stem}-{data.tokenizer.tag()}"
