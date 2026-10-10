"""Training health monitors: early stopping, generalization gap, loss-spike detection (plan §11.4)."""

from __future__ import annotations

import math
from collections import deque


class EarlyStopper:
    """Stop when val loss hasn't improved by `min_delta` for `patience` evals.

    Patience is halved once training is past the first epoch, because repeated data is when
    overfitting typically starts.
    """

    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.bad_evals = 0

    def update(self, val_loss: float) -> bool:
        """Record an eval. Returns True if it's a new best."""
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.bad_evals = 0
            return True
        self.best = min(self.best, val_loss)
        self.bad_evals += 1
        return False

    def should_stop(self, epoch: float) -> bool:
        patience = max(1, self.patience // 2) if epoch >= 1 else self.patience
        return self.bad_evals >= patience

    def state_dict(self) -> dict:
        return {"best": self.best, "bad_evals": self.bad_evals}

    def load_state_dict(self, state: dict) -> None:
        self.best, self.bad_evals = state["best"], state["bad_evals"]


class GapMonitor:
    """Detects the classic overfitting signature: train loss falling, val loss no longer improving,
    and val - train growing. (A growing gap alone isn't enough: early on the train-loss EMA lags
    behind, so the gap grows even while val loss is still dropping nicely.)
    """

    def __init__(self, window: int):
        self.window = window
        self.history: deque[tuple[float, float]] = deque(maxlen=window)  # (train, val)

    def update(self, train_loss: float, val_loss: float) -> str | None:
        self.history.append((train_loss, val_loss))
        if len(self.history) < self.window:
            return None
        gaps = [v - t for t, v in self.history]
        trains = [t for t, _ in self.history]
        vals = [v for _, v in self.history]
        growing = all(b > a for a, b in zip(gaps, gaps[1:]))
        val_stalled = vals[-1] >= min(vals[:-1])  # the latest eval isn't a new low within the window
        if growing and val_stalled and trains[-1] < trains[0]:
            return f"val-train gap grew for {self.window} evals ({gaps[0]:.3f} -> {gaps[-1]:.3f}): likely overfitting"
        return None

    def state_dict(self) -> dict:
        return {"history": list(self.history)}

    def load_state_dict(self, state: dict) -> None:
        self.history = deque((tuple(x) for x in state["history"]), maxlen=self.window)


class SpikeDetector:
    """Flags a step loss that is non-finite or far above the recent mean."""

    def __init__(self, sigma: float, window: int):
        self.sigma = sigma
        self.losses: deque[float] = deque(maxlen=window)

    def check(self, loss: float) -> str | None:
        """Returns a reason if `loss` is a spike; otherwise records it and returns None."""
        if not math.isfinite(loss):
            return f"non-finite loss ({loss})"
        if len(self.losses) >= self.losses.maxlen // 2:
            mean = sum(self.losses) / len(self.losses)
            std = math.sqrt(sum((x - mean) ** 2 for x in self.losses) / len(self.losses))
            # Floor the std so a very flat loss curve doesn't make normal noise look like a spike.
            threshold = mean + self.sigma * max(std, 0.02 * mean)
            if loss > threshold:
                return f"loss spike ({loss:.3f} > {threshold:.3f})"
        self.losses.append(loss)
        return None

    def reset(self) -> None:
        self.losses.clear()
