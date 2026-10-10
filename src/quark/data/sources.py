"""Iterate raw text documents from a Hugging Face dataset or local files.

Local sources: a file or a folder (searched recursively) of
  - .txt      one document per file, or several separated by <|endoftext|>
  - .jsonl    one JSON object per line, text in `text_field`
  - .parquet  text in column `text_field`
Anything else is treated as a Hugging Face dataset name.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

EOT = "<|endoftext|>"
LOCAL_SUFFIXES = (".txt", ".jsonl", ".parquet")


def iter_documents(
    source: str,
    split: str = "train",
    text_field: str = "text",
    max_docs: int | None = None,
) -> Iterator[str]:
    docs = _iter_local(Path(source), text_field) if Path(source).exists() else _iter_hf(source, split, text_field)
    for count, doc in enumerate(docs):
        if max_docs is not None and count >= max_docs:
            return
        yield doc


def _iter_local(path: Path, text_field: str) -> Iterator[str]:
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.suffix in LOCAL_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"no {'/'.join(LOCAL_SUFFIXES)} files found in {path}")
    for file in files:
        if file.suffix == ".txt":
            text = file.read_text(encoding="utf-8", errors="replace")
            for doc in text.split(EOT) if EOT in text else [text]:
                if doc.strip():
                    yield doc.strip()
        elif file.suffix == ".jsonl":
            with file.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line)[text_field]
        elif file.suffix == ".parquet":
            import pyarrow.parquet as pq

            for batch in pq.ParquetFile(file).iter_batches(columns=[text_field], batch_size=1000):
                yield from batch.column(0).to_pylist()
        else:
            raise ValueError(f"unsupported file type: {file}")


def _iter_hf(name: str, split: str, text_field: str) -> Iterator[str]:
    from datasets import load_dataset

    dataset = load_dataset(name, split=split)
    for batch in dataset.iter(batch_size=1000):
        yield from batch[text_field]
