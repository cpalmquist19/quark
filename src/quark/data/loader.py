"""Read prepared token files and serve training batches with a resumable position.

Training data is cut into non-overlapping windows of seq_len + 1 tokens, shuffled with a fresh
seeded permutation each epoch. The loader state is just (epoch, position), so a resumed run
continues on unseen windows instead of replaying old ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from quark.data.tokenizer import load_tokenizer


class TokenData:
    def __init__(self, data_dir: str | Path):
        self.dir = Path(data_dir)
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{self.dir} is not a prepared data directory (no meta.json); run `quark data prepare`")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dtype = np.dtype(self.meta["dtype"])
        self.train = np.memmap(self.dir / "train.bin", dtype=dtype, mode="r")
        self.val = np.memmap(self.dir / "val.bin", dtype=dtype, mode="r")

    @property
    def vocab_size(self) -> int:
        return self.meta["vocab_size"]

    def tokenizer(self):
        return load_tokenizer(self.dir / "tokenizer.json")


class BatchLoader:
    def __init__(
        self,
        tokens: np.ndarray,
        seq_len: int,
        batch_size: int,
        seed: int = 0,
        shuffle: bool = True,
        device: str | torch.device = "cpu",
    ):
        self.tokens = tokens
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.device = torch.device(device)
        self.n_windows = (len(tokens) - 1) // seq_len
        if self.n_windows < batch_size:
            raise ValueError(
                f"only {self.n_windows} windows of {seq_len} tokens available, fewer than batch size {batch_size}"
            )
        self.epoch = 0
        self.pos = 0
        self._order = self._make_order()

    def _make_order(self) -> torch.Tensor:
        if not self.shuffle:
            return torch.arange(self.n_windows)
        gen = torch.Generator().manual_seed(self.seed * 100_003 + self.epoch)
        return torch.randperm(self.n_windows, generator=gen)

    @property
    def epoch_progress(self) -> float:
        """Epochs completed, fractional (1.5 = halfway through the second pass)."""
        return self.epoch + self.pos / self.n_windows

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pos + self.batch_size > self.n_windows:  # drop the ragged tail, start a new epoch
            self.epoch += 1
            self.pos = 0
            self._order = self._make_order()
        starts = self._order[self.pos : self.pos + self.batch_size].tolist()
        self.pos += self.batch_size
        return self._gather(starts)

    def _gather(self, window_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.seq_len
        rows = np.stack([self.tokens[i * t : i * t + t + 1].astype(np.int64) for i in window_ids])
        batch = torch.from_numpy(rows)
        if self.device.type == "cuda":
            batch = batch.pin_memory().to(self.device, non_blocking=True)
        return batch[:, :-1], batch[:, 1:]

    def fixed_batches(self, n_batches: int):
        """The same first `n_batches` (in order) every call, for comparable evaluation losses."""
        n = min(n_batches, self.n_windows // self.batch_size)
        for b in range(n):
            yield self._gather(list(range(b * self.batch_size, (b + 1) * self.batch_size)))

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "pos": self.pos, "seed": self.seed, "n_windows": self.n_windows}

    def load_state_dict(self, state: dict) -> None:
        if state["n_windows"] != self.n_windows:
            raise ValueError(
                f"data changed since the checkpoint ({state['n_windows']} windows then, {self.n_windows} now)"
            )
        self.seed, self.epoch, self.pos = state["seed"], state["epoch"], state["pos"]
        self._order = self._make_order()


# --- mixtures -----------------------------------------------------------------------------------


class SourceTokens:
    def __init__(self, directory: Path, dtype: np.dtype):
        self.train = np.memmap(directory / "train.bin", dtype=dtype, mode="r")
        self.val = np.memmap(directory / "val.bin", dtype=dtype, mode="r")


class MixData:
    """A prepared mix (quark.data.mix): one tokenizer, train/val token files per source."""

    def __init__(self, data_dir: str | Path):
        self.dir = Path(data_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        dtype = np.dtype(self.meta["dtype"])
        self.sources = {name: SourceTokens(self.dir / name, dtype) for name in self.meta["sources"]}

    @property
    def vocab_size(self) -> int:
        return self.meta["vocab_size"]

    @property
    def train_tokens(self) -> int:
        return sum(len(s.train) for s in self.sources.values())

    def tokenizer(self):
        return load_tokenizer(self.dir / "tokenizer.json")

    def resolve_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """Normalized training weights; {} means in proportion to each source's train tokens."""
        unknown = set(weights) - set(self.sources)
        if unknown:
            raise ValueError(f"data.weights names unknown sources {sorted(unknown)} (mix has {sorted(self.sources)})")
        raw = weights or {name: float(len(s.train)) for name, s in self.sources.items()}
        total = sum(raw.values())
        return {name: raw.get(name, 0.0) / total for name in self.sources if raw.get(name, 0.0) > 0}


def load_data(data_dir: str | Path) -> TokenData | MixData:
    meta_path = Path(data_dir) / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")).get("kind") == "mix":
        return MixData(data_dir)
    return TokenData(data_dir)


class _WindowStream:
    """One source's windows of seq_len + 1 tokens in a fresh seeded order each epoch."""

    def __init__(self, tokens: np.ndarray, seq_len: int, seed: int):
        self.tokens, self.seq_len, self.seed = tokens, seq_len, seed
        self.n_windows = (len(tokens) - 1) // seq_len
        if self.n_windows < 1:
            raise ValueError(f"a source has fewer than {seq_len + 1} tokens")
        self.epoch = self.pos = 0
        self._order = self._make_order()

    def _make_order(self) -> np.ndarray:
        return np.random.default_rng((self.seed, self.epoch)).permutation(self.n_windows)

    def next_window(self) -> np.ndarray:
        if self.pos == self.n_windows:
            self.epoch, self.pos = self.epoch + 1, 0
            self._order = self._make_order()
        i = int(self._order[self.pos])
        self.pos += 1
        return self.tokens[i * self.seq_len : i * self.seq_len + self.seq_len + 1].astype(np.int64)

    @property
    def epoch_progress(self) -> float:
        return self.epoch + self.pos / self.n_windows

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "pos": self.pos, "n_windows": self.n_windows}

    def load_state_dict(self, state: dict) -> None:
        if state["n_windows"] != self.n_windows:
            raise ValueError(f"data changed since the checkpoint ({state['n_windows']} windows then, {self.n_windows} now)")
        self.epoch, self.pos = state["epoch"], state["pos"]
        self._order = self._make_order()


class MixtureLoader:
    """Each sequence comes from a source drawn by weight; each source walks its own shuffled windows.

    Draws are seeded by (seed, batch number), so a resumed run continues the same sequence of sources.
    """

    def __init__(self, data: MixData, weights: dict[str, float], seq_len: int, batch_size: int, seed: int = 0,
                 device: str | torch.device = "cpu"):
        self.weights = weights  # normalized, from MixData.resolve_weights
        self.names = list(weights)
        self.batch_size, self.seed, self.device = batch_size, seed, torch.device(device)
        self.streams = {n: _WindowStream(data.sources[n].train, seq_len, seed * 1009 + i) for i, n in enumerate(self.names)}
        self._probs = np.array([weights[n] for n in self.names])
        self.draws = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        picks = np.random.default_rng((self.seed, 7, self.draws)).choice(len(self.names), self.batch_size, p=self._probs)
        self.draws += 1
        rows = np.stack([self.streams[self.names[i]].next_window() for i in picks])
        batch = torch.from_numpy(rows)
        if self.device.type == "cuda":
            batch = batch.pin_memory().to(self.device, non_blocking=True)
        return batch[:, :-1], batch[:, 1:]

    @property
    def epoch_progress(self) -> float:
        """Epochs of the most-repeated source: what matters for overfitting."""
        return max(s.epoch_progress for s in self.streams.values())

    @property
    def source_epochs(self) -> dict[str, float]:
        return {n: s.epoch_progress for n, s in self.streams.items()}

    def state_dict(self) -> dict:
        return {"kind": "mix", "draws": self.draws, "seed": self.seed, "weights": self.weights,
                "streams": {n: s.state_dict() for n, s in self.streams.items()}}

    def load_state_dict(self, state: dict) -> None:
        if state.get("kind") != "mix" or set(state["streams"]) != set(self.streams):
            raise ValueError("the checkpoint's data loader doesn't match this mix (different sources or weights)")
        self.draws, self.seed = state["draws"], state["seed"]
        for n, s in state["streams"].items():
            self.streams[n].load_state_dict(s)
