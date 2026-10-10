"""Paired-seed comparisons of components (plan §15 Phase 2, §18.2).

A spec names a base config, arms (each a partial override of the base; `{}` is the base itself), a
reference arm and seeds. Every arm trains once per seed on the same token budget. Paired seeds share
the data order and, when shapes match, the initial weights, so per-seed differences cancel most of
the run-to-run noise.

Runs go seed by seed (all arms for seed 1, then seed 2, ...), so an interrupted compare has balanced
results. Re-running skips finished runs and resumes unfinished ones from latest.pt, which is what
makes this usable on Colab.

experiments/<name>/
  spec.yaml                       the spec as last run
  runs/<arm>/seed<n>/             ordinary Trainer run folders, plus result.json when finished
  results.json, report.md, curves.png
"""

from __future__ import annotations

import copy
import json
import math
import re
import shutil
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field, model_validator

from quark.components import ComponentError, Issue, check
from quark.config import HardwareProfile, QuarkConfig
from quark.config.presets import PRESETS, get_preset
from quark.train.plots import read_metrics

EXPERIMENTS_DIR = Path("experiments")
SESSION_MARGIN_MIN = 20  # stop this long before a session's hard limit (same margin as the trainer)
MIN_RUN_START_MIN = 5  # don't start or resume a run with less session time left than this
_ARM_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class CompareSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    base: str  # preset name (e.g. "ablation-s") or path to a config.yaml
    arms: dict[str, dict[str, Any]]  # arm -> override of the base config, e.g. {"train": {"optimizer": "muon"}}
    reference: str | None = None  # the arm others are compared with; default: the first arm
    seeds: list[int] = Field(default_factory=lambda: [1, 2], min_length=1)
    max_tokens: int | None = Field(None, gt=0)  # override the base budget for every arm
    data: dict[str, Any] = Field(default_factory=dict)  # DataConfig override for every arm: what was trained on
    keep_checkpoints: bool = False  # delete .pt files when a run finishes; results live in metrics.jsonl

    @model_validator(mode="after")
    def _check(self) -> CompareSpec:
        if not self.arms:
            raise ValueError("a compare needs at least one arm")
        for name in (self.name, *self.arms):
            if not _ARM_NAME.match(name):
                raise ValueError(f"{name!r}: names may only use letters, digits, '_', '-' and '.'")
        if self.reference is not None and self.reference not in self.arms:
            raise ValueError(f"reference {self.reference!r} is not one of the arms")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct")
        return self

    @property
    def ref(self) -> str:
        return self.reference or next(iter(self.arms))

    @classmethod
    def from_yaml(cls, path: str | Path) -> CompareSpec:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False), encoding="utf-8")


def base_config(spec: CompareSpec) -> QuarkConfig:
    return get_preset(spec.base) if spec.base in PRESETS else QuarkConfig.from_yaml(Path(spec.base))


def arm_config(spec: CompareSpec, arm: str, seed: int) -> QuarkConfig:
    """The base config with the arm's overrides, the seed and the compare's fixed-budget settings."""
    data = _deep_merge(base_config(spec).model_dump(mode="json"), {"data": spec.data})
    data = _deep_merge(data, spec.arms[arm])
    data["name"] = f"{spec.name}-{arm}"
    train = data["train"]
    train["seed"] = seed
    if spec.max_tokens is not None:
        train["max_tokens"] = spec.max_tokens
    train["early_stopping"]["enabled"] = False  # equal budgets: every arm trains to the end
    train["keep_last"] = 0  # no rolling step copies; latest.pt is enough to resume
    return QuarkConfig.model_validate(data)


def validate(spec: CompareSpec, hardware: HardwareProfile | None = None) -> dict[str, list[Issue]]:
    """Check every arm before any GPU time is spent. Raises ComponentError naming the arms at fault."""
    issues = {arm: check(arm_config(spec, arm, spec.seeds[0]), hardware) for arm in spec.arms}
    errors = [
        Issue("error", f"{arm} / {i.source}", i.message) for arm, found in issues.items() for i in found
        if i.severity == "error"
    ]
    if errors:
        raise ComponentError(errors)
    return issues


def run_compare(
    spec: CompareSpec,
    data_dir: str | Path,
    out_dir: str | Path | None = None,
    hardware: HardwareProfile | None = None,
    device: str | None = None,
    live_plot: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    """Run (or continue) every arm x seed, then write the report. Returns the summary."""
    from quark.train import Trainer

    out = Path(out_dir) if out_dir is not None else EXPERIMENTS_DIR / spec.name
    out.mkdir(parents=True, exist_ok=True)
    validate(spec, hardware)
    spec.to_yaml(out / "spec.yaml")

    session = hardware.session if hardware is not None else None
    deadline = (
        time.time() + (session.max_hours * 60 - SESSION_MARGIN_MIN) * 60 if session and session.max_hours else None
    )
    total = len(spec.arms) * len(spec.seeds)
    done = 0
    for seed in spec.seeds:
        for arm in spec.arms:
            done += 1
            cfg = arm_config(spec, arm, seed)
            run_dir = out / "runs" / arm / f"seed{seed}"
            result_path = run_dir / "result.json"
            if result_path.exists():
                _check_unchanged(result_path, cfg, arm)
                continue
            if deadline is not None:
                left_min = (deadline - time.time()) / 60
                if left_min < MIN_RUN_START_MIN:
                    log(f"session time nearly used up; stopping. Run the compare again to continue ({done - 1}/{total} done).")
                    return write_report(spec, out)
                cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update={"time_limit_min": left_min})})

            latest = run_dir / "latest.pt"
            log(f"\n=== [{done}/{total}] arm {arm!r}, seed {seed}{' (resuming)' if latest.exists() else ''} ===")
            trainer = Trainer(
                cfg, data_dir, run_dir, hardware=hardware, resume_from=latest if latest.exists() else None,
                device=device, live_plot=live_plot, log=log,
            )
            result = trainer.fit()
            if result.stop_reason not in ("completed", "diverged"):
                log(f"run stopped ({result.stop_reason}); run the compare again to resume it.")
                return write_report(spec, out)
            _write_result(result_path, trainer, result, arm, seed)
            if not spec.keep_checkpoints:
                for ckpt in run_dir.glob("*.pt"):
                    ckpt.unlink()
            del trainer
            write_report(spec, out)  # partial results stay visible as the compare goes
    return write_report(spec, out)


# --- results ------------------------------------------------------------------------------------


def _write_result(path: Path, trainer, result, arm: str, seed: int) -> None:
    train = [r for r in read_metrics(trainer.run_dir / "metrics.jsonl") if r["type"] == "train"]
    steady = train[2:] or train  # the first log windows include warm-up and compilation
    record = {
        "arm": arm,
        "seed": seed,
        "config_hash": trainer.cfg.config_hash(),
        "stop_reason": result.stop_reason,
        "steps": result.steps,
        "tokens": result.tokens,
        "final_val": result.last_val,
        "best_val": result.best_val,
        "params": trainer.raw_model.num_params(),
        "params_non_embedding": trainer.raw_model.num_params(non_embedding=True),
        "tokens_per_sec": statistics.median(r["tokens_per_sec"] for r in steady) if steady else None,
        "peak_mem_gb": max((r["mem_peak_gb"] for r in train if "mem_peak_gb" in r), default=None),
        "train_time_s": trainer.train_time_s,
        "precision": trainer.precision,
        "gpu": trainer.hardware.gpu_name if trainer.hardware else None,
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def _check_unchanged(result_path: Path, cfg: QuarkConfig, arm: str) -> None:
    recorded = json.loads(result_path.read_text(encoding="utf-8"))["config_hash"]
    if recorded != cfg.config_hash():
        raise ValueError(
            f"arm {arm!r} changed since {result_path.parent} was trained. Rename the arm, or delete "
            f"{result_path.parent.parent} to rerun it."
        )


def _val_curve(run_dir: Path) -> tuple[list[float], list[float], list[float]]:
    """(tokens, val_loss, time_s) from metrics.jsonl; for repeated steps (resume, rollback) the last wins."""
    by_step: dict[int, dict] = {}
    for r in read_metrics(run_dir / "metrics.jsonl"):
        if r["type"] == "eval":
            by_step[r["step"]] = r
    rows = [by_step[s] for s in sorted(by_step)]
    return [r["tokens"] for r in rows], [r["val_loss"] for r in rows], [r.get("time_s", math.nan) for r in rows]


def summarize(spec: CompareSpec, out: Path) -> dict:
    results: dict[str, dict[int, dict]] = {arm: {} for arm in spec.arms}
    for arm in spec.arms:
        for seed in spec.seeds:
            path = out / "runs" / arm / f"seed{seed}" / "result.json"
            if path.exists():
                results[arm][seed] = json.loads(path.read_text(encoding="utf-8"))

    ref = spec.ref
    ref_runs = results[ref]
    slope = _late_slope([_val_curve(out / "runs" / ref / f"seed{s}") for s in ref_runs])
    ref_speed = _mean([r["tokens_per_sec"] for r in ref_runs.values()])
    ref_vals = [r["final_val"] for r in ref_runs.values() if r["final_val"] is not None]

    arms = {}
    for arm, runs in results.items():
        finals = {s: r["final_val"] for s, r in runs.items() if r["final_val"] is not None}
        paired = [finals[s] - ref_runs[s]["final_val"] for s in finals if s in ref_runs and arm != ref]
        speed = _mean([r["tokens_per_sec"] for r in runs.values()])
        speed_ratio = speed / ref_speed if speed and ref_speed else None
        mean_d = _mean(paired)
        se = statistics.stdev(paired) / math.sqrt(len(paired)) if len(paired) >= 2 else None
        arms[arm] = {
            "seeds_done": sorted(runs),
            "final_val": finals,
            "mean_final_val": _mean(list(finals.values())),
            "delta": mean_d,
            "delta_se": se,
            "seeds_better": sum(d < 0 for d in paired),
            "paired": len(paired),
            "delta_equal_time": (
                mean_d + slope * math.log(speed_ratio) if mean_d is not None and slope is not None and speed_ratio else None
            ),
            "speed_ratio": speed_ratio,
            "tokens_per_sec": speed,
            "params": next((r["params"] for r in runs.values()), None),
            "peak_mem_gb": max((r["peak_mem_gb"] for r in runs.values() if r["peak_mem_gb"] is not None), default=None),
            "diverged": any(r["stop_reason"] == "diverged" for r in runs.values()),
            "verdict": _verdict(arm == ref, paired, se, any(r["stop_reason"] == "diverged" for r in runs.values())),
        }
    return {
        "name": spec.name,
        "base": spec.base,
        "reference": ref,
        "seeds": spec.seeds,
        "runs_done": sum(len(r) for r in results.values()),
        "runs_total": len(spec.arms) * len(spec.seeds),
        "tokens_per_run": next((r["tokens"] for runs in results.values() for r in runs.values()), None),
        "reference_seed_spread": statistics.stdev(ref_vals) if len(ref_vals) >= 2 else None,
        "late_slope_per_log_tokens": slope,
        "arms": arms,
    }


def _verdict(is_ref: bool, paired: list[float], se: float | None, diverged: bool) -> str:
    if is_ref:
        return "reference"
    if diverged:
        return "diverged"
    if not paired:
        return "pending"
    if se is None:
        return "1 seed: no verdict"
    mean = sum(paired) / len(paired)
    agree = all(d < 0 for d in paired) or all(d > 0 for d in paired)
    if agree and abs(mean) > 2 * se:
        return "better" if mean < 0 else "worse"
    return "within noise"


def _late_slope(curves: list[tuple[list[float], list[float], list[float]]]) -> float | None:
    """d(val loss)/d(ln tokens) over the last two thirds of the reference runs, averaged over seeds."""
    slopes = []
    for tokens, vals, _ in curves:
        pts = [(math.log(t), v) for t, v in zip(tokens, vals) if t >= tokens[-1] / 3] if tokens else []
        if len(pts) >= 3:
            x, y = np.array(pts).T
            slopes.append(float(np.polyfit(x, y, 1)[0]))
    return _mean(slopes)


def _mean(values: list[float | None]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


# --- report -------------------------------------------------------------------------------------


def write_report(spec: CompareSpec, out: Path) -> dict:
    summary = summarize(spec, out)
    (out / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "report.md").write_text(render_report(summary), encoding="utf-8")
    if summary["runs_done"]:
        make_compare_figure(spec, out).savefig(out / "curves.png", dpi=110)
    return summary


def render_report(s: dict) -> str:
    def num(v, fmt="{:+.4f}"):
        return "–" if v is None else fmt.format(v)

    budget = f"{_si(s['tokens_per_run'])} tokens per run" if s["tokens_per_run"] else "budget pending"
    lines = [
        f"# Compare: {s['name']}",
        "",
        f"Base `{s['base']}`, reference arm `{s['reference']}`, seeds {s['seeds']}, {budget}. "
        f"Runs finished: {s['runs_done']} / {s['runs_total']}.",
        "",
        "| arm | params | final val (mean) | Δ vs ref (equal tokens) | seeds better | Δ ≈ equal time | speed | peak mem | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, a in s["arms"].items():
        delta = "–" if a["delta"] is None else num(a["delta"]) + (f" ± {a['delta_se']:.4f}" if a["delta_se"] else "")
        speed = "–" if a["tokens_per_sec"] is None else f"{_si(a['tokens_per_sec'])} tok/s"
        if a["speed_ratio"] and arm != s["reference"]:
            speed += f" ({a['speed_ratio']:.2f}×)"
        params = "–" if a["params"] is None else _si(a["params"])
        better = f"{a['seeds_better']}/{a['paired']}" if a["paired"] else "–"
        cells = [
            arm, params, num(a["mean_final_val"], "{:.4f}"), delta, better, num(a["delta_equal_time"]), speed,
            num(a["peak_mem_gb"], "{:.2f} GB"), a["verdict"],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Final val loss per seed:", "", "| arm | " + " | ".join(f"seed {x}" for x in s["seeds"]) + " |",
              "|---|" + "---|" * len(s["seeds"])]
    for arm, a in s["arms"].items():
        lines.append(f"| {arm} | " + " | ".join(num(a["final_val"].get(x, a["final_val"].get(str(x))), "{:.4f}") for x in s["seeds"]) + " |")
    spread = s["reference_seed_spread"]
    lines += [
        "",
        "How to read this:",
        "",
        "- Δ < 0 means the arm reached a lower validation loss than the reference (better). Each Δ is the mean of "
        "per-seed differences (arm minus reference, same seed), ± one standard error.",
        f"- Seed noise: the reference's final val loss varies by {num(spread, '{:.4f}')} (std) across seeds.",
        "- **better / worse**: the mean Δ is beyond 2 standard errors and every seed agrees. Otherwise **within "
        "noise**. With 2 seeds this is a weak test; add a seed for close calls.",
        "- **Δ ≈ equal time** adjusts Δ for speed: an arm that is slower would have seen fewer tokens in the same "
        "time. It uses the reference's loss-vs-log(tokens) slope late in training "
        f"({num(s['late_slope_per_log_tokens'], '{:.3f}')} per e-fold of tokens); it's an estimate that ignores the "
        "schedule's shape.",
        "- Speed is measured in each run. A GPU that changes clock state between runs (laptop power or thermal "
        "limits, a shared GPU) skews the speed and equal-time columns; the loss columns aren't affected.",
        "- curves.png shows the val curves and how each arm's gap to the reference evolved: a gap still widening "
        "at the end suggests a longer run would favor that arm more.",
    ]
    return "\n".join(lines) + "\n"


def _si(n: float) -> str:
    """8_520_000 -> '8.52M', 40_000_000 -> '40M', 36_032 -> '36K'."""
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= scale:
            return f"{n / scale:.3g}{suffix}"
    return f"{n:.3g}"


def make_compare_figure(spec: CompareSpec, out: Path) -> Figure:
    fig = Figure(figsize=(12, 4.5))
    ax_val, ax_delta = fig.subplots(1, 2)
    ref = spec.ref
    ref_curves = {s: _val_curve(out / "runs" / ref / f"seed{s}") for s in spec.seeds}
    for i, arm in enumerate(spec.arms):
        color = f"C{i % 10}"
        curves = {s: _val_curve(out / "runs" / arm / f"seed{s}") for s in spec.seeds}
        curves = {s: c for s, c in curves.items() if c[0]}
        for j, (seed, (tokens, vals, _)) in enumerate(curves.items()):
            ax_val.plot([t / 1e6 for t in tokens], vals, color=color, alpha=0.8, linewidth=1.2,
                        label=arm if j == 0 else None)
        if arm == ref:
            continue
        deltas = []
        for seed, (tokens, vals, _) in curves.items():
            r_tokens, r_vals, _ = ref_curves.get(seed, ([], [], []))
            if not r_tokens:
                continue
            grid = [t for t in r_tokens if tokens[0] <= t <= tokens[-1]]
            deltas.append((grid, np.interp(grid, tokens, vals) - np.interp(grid, r_tokens, r_vals)))
        if deltas:
            grid = min((g for g, _ in deltas), key=len)
            mean = np.mean([d[: len(grid)] for _, d in deltas], axis=0)
            ax_delta.plot([t / 1e6 for t in grid], mean, "o-", color=color, markersize=3, label=arm)
    ax_val.set(title="Validation loss (each line = one seed)", xlabel="tokens (M)", ylabel="cross-entropy")
    tail = [v for arm in spec.arms for s in spec.seeds for v in _val_curve(out / "runs" / arm / f"seed{s}")[1][-3:]]
    if tail:
        lo, hi = min(tail), max(tail)
        ax_val.set_ylim(lo - 0.05, hi + 4 * (hi - lo) + 0.3)
    ax_delta.axhline(0, color="black", linewidth=0.8)
    ax_delta.set(title=f"Δ val loss vs {ref} (mean over seeds; < 0 is better)", xlabel="tokens (M)")
    for ax in (ax_val, ax_delta):
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.suptitle(f"compare: {spec.name}")
    fig.tight_layout()
    return fig


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged
