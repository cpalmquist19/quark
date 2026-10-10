"""Data mixtures: several filtered sources, one shared tokenizer, weights chosen at training time.

A recipe (mixes/<name>.yaml) names the sources. Each is a stream (Hugging Face dataset or local
files) with an optional filter and a token cap. Building a mix has two CPU-only steps:

1. collect: stream each source, keep the documents its filter accepts, and cache them as jsonl
   shards. Sources that read the same stream are collected in one pass. Collection is resumable:
   the stream position is saved with every flush, so a long scan survives a disconnect.
2. prepare: train one tokenizer on a sample of every source, then tokenize each source into its
   own train/val files.

data/mix-<name>-<tokenizer>/
  raw/<source>/part-<scanned docs>.jsonl, raw/<stream>.state.json     collected text
  <source>/train.bin, <source>/val.bin                                 tokens per source
  tokenizer.json, meta.json                                            meta.json is written last

Which sources a model trains on, and how often, is `data.weights` in the training config, so domain
weights can be compare arms without re-preparing anything.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Iterator
from itertools import chain
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

import quark
from quark.config import TokenizerConfig
from quark.data.domains import DocFilter
from quark.data.prepare import TOKENIZER_SAMPLE_CHARS, _char_capped, token_dtype, tokenize_to
from quark.data.sources import EOT, LOCAL_SUFFIXES
from quark.data.tokenizer import build_tokenizer, eot_id

FLUSH_DOCS = 20_000  # scanned documents between shard writes and state saves
_NAME = re.compile(r"^[a-z0-9_]+$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MixFilter(_Strict):
    domain: str | None = None  # a key of quark.data.domains.DOMAINS
    min_hits: int = 3
    min_distinct: int = 2
    min_per_1k_words: float = 2.0
    columns: dict[str, list[Any]] = Field(default_factory=dict)  # e.g. {"audience": ["young_children"]}
    exclude_sites: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)

    def build(self) -> DocFilter:
        return DocFilter(
            self.domain, self.min_hits, self.min_distinct, self.min_per_1k_words, self.columns or None,
            tuple(self.exclude_sites), tuple(self.exclude_patterns),
        )


class MixSource(_Strict):
    source: str  # HF dataset name or local file/folder
    subset: str | None = None
    split: str = "train"
    text_field: str = "text"
    tokens: int = Field(gt=0)  # collect until about this many tokens are kept
    filter: MixFilter | None = None
    note: str = ""  # why this source is in the mix

    @property
    def stream_key(self) -> str:
        return f"{self.source}|{self.subset}|{self.split}"


class MixRecipe(_Strict):
    name: str
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    val_fraction: float = Field(0.005, gt=0, lt=0.5)
    sources: dict[str, MixSource]

    @model_validator(mode="after")
    def _check(self) -> MixRecipe:
        for name in (self.name.replace("-", "_"), *self.sources):
            if not _NAME.match(name):
                raise ValueError(f"{name!r}: use lowercase letters, digits and '_' (and '-' in the mix name)")
        if not self.sources:
            raise ValueError("a mix needs at least one source")
        for name, src in self.sources.items():
            if src.filter is not None:
                try:
                    src.filter.build()  # unknown domains and bad regexes fail here, not hours into a scan
                except re.error as err:
                    raise ValueError(f"source {name!r}: bad exclude pattern: {err}") from err
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> MixRecipe:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def mix_dir(recipe: MixRecipe, root: str | Path = "data") -> Path:
    return Path(root) / f"mix-{recipe.name}-{recipe.tokenizer.tag()}"


# --- collect ------------------------------------------------------------------------------------


def collect(
    recipe: MixRecipe,
    root: str | Path = "data",
    only: list[str] | None = None,
    log: Callable[[str], None] = print,
    flush_docs: int = FLUSH_DOCS,
    max_docs: int | None = None,
) -> dict:
    """Collect every source (or `only` these), resuming where earlier runs stopped. Returns progress."""
    raw = mix_dir(recipe, root) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    names = [n for n in recipe.sources if only is None or n in only]
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(recipe.sources[name].stream_key, []).append(name)
    for key, members in groups.items():
        _collect_stream(recipe, raw, key, members, log, flush_docs, max_docs)
    return collect_status(recipe, root)


def collect_status(recipe: MixRecipe, root: str | Path = "data") -> dict:
    """{source: {"tokens": kept so far, "target": cap, "done": bool}} from the saved states."""
    raw = mix_dir(recipe, root) / "raw"
    status = {}
    for name, src in recipe.sources.items():
        state = _load_state(raw / _state_name(src.stream_key))
        kept = state["sources"].get(name, {"tokens": 0}) if state else {"tokens": 0}
        status[name] = {
            "tokens": kept["tokens"],
            "target": src.tokens,
            "done": bool(state and (kept["tokens"] >= src.tokens or state.get("exhausted"))),
        }
    return status


def _collect_stream(recipe, raw: Path, key: str, members: list[str], log, flush_docs: int, max_docs: int | None):
    first = recipe.sources[members[0]]
    state_path = raw / _state_name(key)
    state = _load_state(state_path) or {"stream": key, "scanned_docs": 0, "scanned_tokens": 0, "stream_state": None,
                                        "exhausted": False, "sources": {}}
    for name in members:
        state["sources"].setdefault(name, {"docs": 0, "tokens": 0})
        (raw / name).mkdir(exist_ok=True)
        for part in (raw / name).glob("part-*.jsonl"):  # written after the last saved state: redo them
            if int(part.stem.split("-")[1]) > state["scanned_docs"]:
                part.unlink()
    open_names = [n for n in members if state["sources"][n]["tokens"] < recipe.sources[n].tokens]
    if state["exhausted"] or not open_names:
        log(f"{key}: already collected")
        return

    filters = {n: (recipe.sources[n].filter.build() if recipe.sources[n].filter else None) for n in members}
    rows, stream = _open_stream(first, state)
    buffers: dict[str, list[str]] = {n: [] for n in members}
    started, scanned_at_start = time.time(), state["scanned_docs"]
    log(f"{key}: collecting {', '.join(open_names)}" + (f" (resuming after {state['scanned_docs']:,} docs)" if scanned_at_start else ""))

    def flush():
        for name, texts in buffers.items():
            if texts:
                part = raw / name / f"part-{state['scanned_docs']:012d}.jsonl"
                tmp = part.with_suffix(".tmp")
                tmp.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts), encoding="utf-8")
                tmp.replace(part)
                texts.clear()
        if stream is not None:
            state["stream_state"] = stream.state_dict()
        _save_state(state_path, state)
        rate = (state["scanned_docs"] - scanned_at_start) / max(1e-9, time.time() - started)
        kept = ", ".join(
            f"{n} {state['sources'][n]['tokens'] / 1e6:,.1f}M/{recipe.sources[n].tokens / 1e6:,.0f}M" for n in members
        )
        log(f"  scanned {state['scanned_docs']:,} docs ({state['scanned_tokens'] / 1e6:,.0f}M tokens, "
            f"{rate:,.0f} docs/s) | kept {kept}")

    stopped = False  # True when we stop before the stream ends (all caps reached, or max_docs)
    for row in rows:
        text = row[first.text_field]
        tokens = row.get("token_count") or len(text) // 4  # estimate; exact counts come from tokenizing
        state["scanned_docs"] += 1
        state["scanned_tokens"] += tokens
        for name in open_names:
            kept = state["sources"][name]
            if kept["tokens"] >= recipe.sources[name].tokens:
                continue
            if filters[name] is None or filters[name].accepts(row, first.text_field):
                buffers[name].append(text)
                kept["docs"] += 1
                kept["tokens"] += tokens
        if state["scanned_docs"] % flush_docs == 0:
            flush()
            open_names = [n for n in open_names if state["sources"][n]["tokens"] < recipe.sources[n].tokens]
            if not open_names or (max_docs is not None and state["scanned_docs"] - scanned_at_start >= max_docs):
                stopped = True
                break
    state["exhausted"] = not stopped
    flush()
    if state["exhausted"]:
        log(f"{key}: stream exhausted after {state['scanned_docs']:,} docs")


def _open_stream(src: MixSource, state: dict) -> tuple[Iterator[dict], Any]:
    """Rows from the source, positioned after the documents already scanned."""
    path = Path(src.source)
    if path.exists():
        rows = _local_rows(path, src.text_field)
        for _ in range(state["scanned_docs"]):  # local files: skip what was scanned
            next(rows, None)
        return rows, None
    from datasets import load_dataset

    stream = load_dataset(src.source, src.subset, split=src.split, streaming=True)
    if state["stream_state"] is not None:
        stream.load_state_dict(state["stream_state"])
    return iter(stream), stream


def _local_rows(path: Path, text_field: str) -> Iterator[dict]:
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.suffix in LOCAL_SUFFIXES)
    for file in files:
        if file.suffix == ".jsonl":
            with file.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line)
        elif file.suffix == ".parquet":
            import pyarrow.parquet as pq

            for batch in pq.ParquetFile(file).iter_batches(batch_size=1000):
                yield from batch.to_pylist()
        else:
            text = file.read_text(encoding="utf-8", errors="replace")
            for doc in text.split(EOT) if EOT in text else [text]:
                if doc.strip():
                    yield {text_field: doc.strip()}


def _state_name(key: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", key.lower()).strip("-")[:60]
    return f"{slug}-{hashlib.blake2b(key.encode(), digest_size=4).hexdigest()}.state.json"


def _load_state(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    tmp.replace(path)


def iter_collected(recipe: MixRecipe, name: str, root: str | Path = "data") -> Iterator[str]:
    for part in sorted((mix_dir(recipe, root) / "raw" / name).glob("part-*.jsonl")):
        with part.open(encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)["text"]


# --- prepare ------------------------------------------------------------------------------------


def prepare_mix(
    recipe: MixRecipe,
    root: str | Path = "data",
    progress: Callable[[str, int, int], None] | None = None,
    allow_partial: bool = False,
) -> dict:
    """Train the shared tokenizer and tokenize every collected source. Returns the meta dict."""
    out = mix_dir(recipe, root)
    status = collect_status(recipe, root)
    unfinished = [n for n, s in status.items() if not s["done"]]
    if unfinished and not allow_partial:
        raise RuntimeError(f"sources not fully collected yet: {', '.join(unfinished)} (run collect first)")
    (out / "meta.json").unlink(missing_ok=True)
    started = time.time()

    # Tokenizer: every source contributes to the sample in proportion to its size, at least 5%.
    caps = {n: s.tokens for n, s in recipe.sources.items()}
    shares = {n: max(0.05, c / sum(caps.values())) for n, c in caps.items()}
    norm = sum(shares.values())
    sample = chain.from_iterable(
        _char_capped(iter_collected(recipe, n, root), int(TOKENIZER_SAMPLE_CHARS * s / norm)) for n, s in shares.items()
    )
    tok = build_tokenizer(recipe.tokenizer, sample)
    tok.save(str(out / "tokenizer.json"))

    sources = {}
    for name, src in recipe.sources.items():
        (out / name).mkdir(exist_ok=True)
        report = (lambda d, t, n=name: progress(n, d, t)) if progress else None
        counts = tokenize_to(out / name, iter_collected(recipe, name, root), tok, recipe.val_fraction, None, report, name)
        sources[name] = {
            "source": src.source, "subset": src.subset, "filter": src.filter.model_dump() if src.filter else None,
            "train_docs": counts["train"][0], "train_tokens": counts["train"][1],
            "val_docs": counts["val"][0], "val_tokens": counts["val"][1],
        }
    meta = {
        "kind": "mix",
        "name": recipe.name,
        "tokenizer": recipe.tokenizer.model_dump(),
        "vocab_size": tok.get_vocab_size(),
        "eot_id": eot_id(tok),
        "dtype": token_dtype(tok.get_vocab_size()).__name__,
        "val_fraction": recipe.val_fraction,
        "sources": sources,
        "seconds": round(time.time() - started, 1),
        "quark_version": quark.__version__,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
