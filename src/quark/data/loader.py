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
