"""Training dashboard: the four "loss going down" panels from plan §11.2, built from metrics.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

# The object API (Figure, not pyplot) needs no GUI backend: it saves PNGs from the CLI and still
# displays inline in Jupyter/Colab.
from matplotlib.figure import Figure


def read_metrics(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def make_figure(records: list[dict], title: str = "") -> Figure:
    train = [r for r in records if r["type"] == "train"]
    evals = [r for r in records if r["type"] == "eval"]
    fig = Figure(figsize=(12, 7))
    axes = fig.subplots(2, 2)
    (ax_loss, ax_lr), (ax_gn, ax_tps) = axes

    def col(rows, key):
        return [r["tokens"] / 1e6 for r in rows if r.get(key) is not None], [r[key] for r in rows if r.get(key) is not None]

    x, y = col(train, "loss")
    ax_loss.plot(x, y, color="tab:blue", alpha=0.25, linewidth=0.8, label="train (raw)")
    x, y = col(train, "loss_ema")
    ax_loss.plot(x, y, color="tab:blue", linewidth=2, label="train (EMA)")
    x, y = col(evals, "val_loss")
    ax_loss.plot(x, y, "o-", color="tab:orange", markersize=4, label="val" + (" (mix)" if evals and any(k.startswith("val_loss/") for k in evals[-1]) else ""))
    sources = sorted({k.removeprefix("val_loss/") for r in evals for k in r if k.startswith("val_loss/")})
    for i, name in enumerate(sources):
        x, y = col(evals, f"val_loss/{name}")
        ax_loss.plot(x, y, "--", color=f"C{(i + 2) % 10}", linewidth=1, alpha=0.8, label=f"val: {name}")
    ax_loss.set(title="Loss", xlabel="tokens (M)", ylabel="cross-entropy")
    ax_loss.legend()
    if train:
        lows = [r["loss_ema"] for r in train[len(train) // 10 :]] or [train[-1]["loss_ema"]]
        ax_loss.set_ylim(min(lows + y) * 0.97 if y else min(lows) * 0.97, max(lows) * 1.15)

    x, y = col(train, "lr")
    ax_lr.plot(x, y, color="tab:green")
    ax_lr.set(title="Learning rate", xlabel="tokens (M)")

    x, y = col(train, "grad_norm")
    ax_gn.plot(x, y, color="tab:red", linewidth=0.8)
    ax_gn.set(title="Gradient norm (pre-clip)", xlabel="tokens (M)", yscale="log")

    x, y = col(train, "tokens_per_sec")
    ax_tps.plot(x, [v / 1e3 for v in y], color="tab:purple", label="tokens/sec (k)")
    ax_tps.set(title="Throughput and memory", xlabel="tokens (M)", ylabel="k tokens/sec")
    x, y = col(train, "mem_peak_gb")
    if y:
        ax_mem = ax_tps.twinx()
        ax_mem.plot(x, y, color="tab:gray", linestyle="--", label="peak GPU mem (GB)")
        ax_mem.set_ylabel("GB")

    for ax in axes.flat:
        ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def save_plot(run_dir: str | Path, title: str = "") -> Path | None:
    run_dir = Path(run_dir)
    records = read_metrics(run_dir / "metrics.jsonl")
    if not records:
        return None
    fig = make_figure(records, title)
    out = run_dir / "training.png"
    fig.savefig(out, dpi=110)
    return out
