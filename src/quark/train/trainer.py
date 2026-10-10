"""The training loop (plan §11): stats, live plots, checkpoints, early stopping, divergence
rollback, session time guard and exact resume. The CLI and both notebooks use this one class.
"""

from __future__ import annotations

import io
import json
import math
import shutil
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

import quark
from quark.components import check, make_optimizer, make_schedule, raise_for_errors
from quark.config import HardwareProfile, QuarkConfig
from quark.data import BatchLoader, TokenData
from quark.model import build_model
from quark.train.checkpoint import load_checkpoint, prune_step_checkpoints, rng_state, save_checkpoint, set_rng_state
from quark.train.monitors import EarlyStopper, GapMonitor, SpikeDetector
from quark.train.plots import make_figure, read_metrics, save_plot

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
SESSION_MARGIN_MIN = 20  # stop this long before a session's hard limit, to save and sync to Drive


class ConfigMismatchError(RuntimeError):
    pass


@dataclass
class TrainResult:
    run_dir: Path
    steps: int
    tokens: int
    best_val: float
    last_val: float | None
    stop_reason: str


class Trainer:
    def __init__(
        self,
        cfg: QuarkConfig,
        data_dir: str | Path,
        run_dir: str | Path,
        hardware: HardwareProfile | None = None,
        resume_from: str | Path | None = None,
        allow_config_change: bool = False,
        live_plot: bool = False,
        max_steps: int | None = None,
        device: str | torch.device | None = None,
        log: Callable[[str], None] = print,
    ):
        self.cfg = cfg
        self.tcfg = cfg.train
        self.run_dir = Path(run_dir)
        self.data_dir = Path(data_dir)
        self.live_plot = live_plot
        self.max_steps = max_steps
        self.log = log
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.hardware = hardware if hardware is not None else self._detect_hardware()
        self.precision = self._resolve_precision()
        self.issues = check(cfg, self.hardware)  # component rules; errors stop here, before any work
        raise_for_errors(self.issues)
        for issue in self.issues:
            self.log(f"{'warning' if issue.severity == 'warning' else 'note'}: {issue.source}: {issue.message}")

        torch.manual_seed(self.tcfg.seed)
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")  # TF32 where the GPU supports it

        # Data
        self.data = TokenData(self.data_dir)
        if self.data.vocab_size > cfg.model.vocab_size:
            raise ValueError(
                f"tokenizer has {self.data.vocab_size} tokens but model.vocab_size is {cfg.model.vocab_size}"
            )
        seq = cfg.model.seq_len
        self.train_loader = BatchLoader(self.data.train, seq, self.tcfg.micro_batch, self.tcfg.seed, True, self.device)
        self.val_loader = BatchLoader(self.data.val, seq, self.tcfg.micro_batch, 0, False, self.device)

        # Model and optimizer
        self.raw_model = build_model(cfg.model, self.hardware).to(self.device)
        self.model = torch.compile(self.raw_model) if self.tcfg.compile else self.raw_model
        self.optimizer = make_optimizer(cfg, self.raw_model, self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.precision == "fp16" and self.device.type == "cuda")

        # Schedule and bookkeeping
        self.tokens_per_step = self.tcfg.tokens_per_step(seq)
        self.total_steps = max(1, self.tcfg.max_tokens // self.tokens_per_step)
        self.schedule = make_schedule(cfg, self.total_steps)
        self.step = 0
        self.tokens = 0
        self.train_time_s = 0.0
        self.loss_ema: float | None = None
        self.lr_mult = 1.0
        self.rollbacks = 0
        es = self.tcfg.early_stopping
        self.stopper = EarlyStopper(es.patience, es.min_delta)
        self.gap = GapMonitor(es.gap_window)
        self.spikes = SpikeDetector(self.tcfg.divergence.spike_sigma, self.tcfg.divergence.window)
        self.last_val: float | None = None
        self._display = None

        session = self.hardware.session if self.hardware else None
        self.checkpoint_every_min = self.tcfg.checkpoint_every_min or (session.checkpoint_every_min if session else None)
        self.time_limit_min = self.tcfg.time_limit_min or (
            session.max_hours * 60 - SESSION_MARGIN_MIN if session and session.max_hours else None
        )

        if resume_from is not None:
            self._resume(Path(resume_from), allow_config_change)
        self._init_run_dir()

    # --- setup -----------------------------------------------------------------------------------

    def _detect_hardware(self) -> HardwareProfile | None:
        if self.device.type != "cuda":
            return None
        from quark.hardware import detect_profile

        return detect_profile(self.device.index or 0)

    def _resolve_precision(self) -> str:
        if self.device.type != "cuda":
            return "fp32"
        requested = self.tcfg.precision
        native_bf16 = torch.cuda.get_device_capability(self.device)[0] >= 8
        if requested == "auto":
            return self.hardware.precision if self.hardware else ("bf16" if native_bf16 else "fp16")
        if requested == "bf16" and not native_bf16:
            self.log("warning: bf16 isn't native on this GPU; using fp16 + GradScaler instead")
            return "fp16"
        return requested

    def _init_run_dir(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.to_yaml(self.run_dir / "config.yaml")
        tokenizer_copy = self.run_dir / "tokenizer.json"
        if not tokenizer_copy.exists():
            shutil.copyfile(self.data.dir / "tokenizer.json", tokenizer_copy)
        run_info = {
            "data_dir": str(self.data_dir.resolve()),
            "config_hash": self.cfg.config_hash(),
            "hardware": self.hardware.name if self.hardware else "cpu",
            "gpu": self.hardware.gpu_name if self.hardware else None,
            "precision": self.precision,
            "quark_version": quark.__version__,
        }
        (self.run_dir / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
        for issue in self.issues:
            self._event(f"component_{issue.severity}", f"{issue.source}: {issue.message}")

    # --- schedule --------------------------------------------------------------------------------

    def lr_at(self, step: int) -> float:
        return self.schedule.lr_at(step) * self.lr_mult

    def _autocast(self):
        if self.precision == "fp32":
            return nullcontext()
        return torch.autocast(self.device.type, dtype=DTYPES[self.precision])

    # --- training --------------------------------------------------------------------------------

    def fit(self) -> TrainResult:
        t = self.tcfg
        self.log(self._banner())
        self.model.train()
        started = last_ckpt = time.time()
        window = self._new_window()
        last_eval_step = -1
        stop_reason = "completed"
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        try:
            while self.step < self.total_steps:
                if self.max_steps is not None and self.step >= self.max_steps:
                    stop_reason = "max_steps"
                    break

                loss, grad_norm, lr, skipped = self._train_step()
                self.step += 1
                self.tokens += self.tokens_per_step
                window["steps"] += 1
                window["clipped"] += int(t.grad_clip is not None and grad_norm > t.grad_clip)
                window["skipped"] += int(skipped)

                if t.divergence.enabled and (problem := self.spikes.check(loss)):
                    if not self._rollback(problem):
                        stop_reason = "diverged"
                        break
                    continue

                self.loss_ema = loss if self.loss_ema is None else 0.98 * self.loss_ema + 0.02 * loss

                if self.step % t.log_every_steps == 0 or self.step == 1:
                    self._log_train(loss, grad_norm, lr, window)
                    window = self._new_window()

                pause_start = time.time()  # eval and checkpoint time doesn't count toward throughput
                if self.step % t.eval_every_steps == 0:
                    last_eval_step = self.step
                    if reason := self._evaluate_and_check():
                        stop_reason = reason
                        break

                now = time.time()
                if self.step % t.checkpoint_every_steps == 0 or (
                    self.checkpoint_every_min and now - last_ckpt >= self.checkpoint_every_min * 60
                ):
                    self._save_latest(step_copy=self.step % t.checkpoint_every_steps == 0)
                    last_ckpt = now
                window["paused"] += time.time() - pause_start

                if self.time_limit_min and (now - started) / 60 >= self.time_limit_min:
                    stop_reason = "time_limit"
                    break
        except KeyboardInterrupt:
            stop_reason = "interrupted"
        finally:
            self.train_time_s += time.time() - started

        if stop_reason != "diverged" and last_eval_step != self.step:
            self._evaluate_and_check(final=True)
        self._save_latest(step_copy=False)
        self._plot()
        self.log(f"stopped: {stop_reason} at step {self.step:,} ({self.tokens / 1e6:.1f}M tokens). Run: {self.run_dir}")
        return TrainResult(self.run_dir, self.step, self.tokens, self.stopper.best, self.last_val, stop_reason)

    def _new_window(self) -> dict:
        return {"t": time.time(), "tokens": self.tokens, "steps": 0, "clipped": 0, "skipped": 0, "paused": 0.0}

    def _train_step(self) -> tuple[float, float, float, bool]:
        t = self.tcfg
        lr = self.lr_at(self.step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        loss_sum = torch.zeros((), device=self.device)
        for _ in range(t.grad_accum):
            x, y = self.train_loader.next_batch()
            with self._autocast():
                loss = self.model(x, y)
            self.scaler.scale(loss / t.grad_accum).backward()
            loss_sum += loss.detach()

        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), t.grad_clip or float("inf"))
        skipped = False
        if self.scaler.is_enabled():  # fp16: the scaler skips steps with inf/nan grads and lowers the scale
            scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            skipped = self.scaler.get_scale() < scale
        elif torch.isfinite(grad_norm):
            self.optimizer.step()
        else:
            skipped = True
        self.optimizer.zero_grad(set_to_none=True)
        return (loss_sum / t.grad_accum).item(), grad_norm.item(), lr, skipped

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        losses = []
        for x, y in self.val_loader.fixed_batches(self.tcfg.eval_batches):
            with self._autocast():
                losses.append(self.model(x, y).item())
        self.model.train()
        return sum(losses) / len(losses)

    def _evaluate_and_check(self, final: bool = False) -> str | None:
        """Evaluate, log, save best, update plots. Returns a stop reason, if any."""
        es = self.tcfg.early_stopping
        val = self.evaluate()
        self.last_val = val
        improved = self.stopper.update(val)
        if improved:
            save_checkpoint(self._state(), self.run_dir / "best.pt")
        gap_warning = self.gap.update(self.loss_ema, val) if self.loss_ema is not None else None
        epoch = self.train_loader.epoch_progress
        self._write_metrics(
            {
                "type": "eval",
                "step": self.step,
                "tokens": self.tokens,
                "val_loss": val,
                "val_ppl": math.exp(min(val, 20)),
                "train_loss_ema": self.loss_ema,
                "best_val": self.stopper.best,
                "epoch": epoch,
            }
        )
        self.log(
            f"eval  step {self.step:>7,} | val {val:.4f} (ppl {math.exp(min(val, 20)):.2f}) | "
            f"best {self.stopper.best:.4f}{'  * saved best.pt' if improved else ''}"
        )
        if gap_warning:
            self.log(f"warning: {gap_warning}")
            self._event("overfitting_warning", gap_warning)
        self._plot()
        if final:
            return None
        if gap_warning and es.stop_on_gap:
            return "overfitting"
        if es.enabled and self.stopper.should_stop(epoch):
            return "early_stopping"
        return None

    def _rollback(self, problem: str) -> bool:
        """Restore latest.pt after a loss spike. Returns False if training should stop."""
        div = self.tcfg.divergence
        self.rollbacks += 1
        latest = self.run_dir / "latest.pt"
        self._event("divergence", problem)
        if self.rollbacks > div.max_rollbacks:
            self.log(f"error: {problem}; gave up after {div.max_rollbacks} rollbacks")
            return False
        if not latest.exists():
            if "non-finite" in problem:
                self.log(f"error: {problem} before the first checkpoint; stopping")
                return False
            self.log(f"warning: {problem} (no checkpoint yet to roll back to; continuing)")
            return True
        rollbacks, lr_mult = self.rollbacks, self.lr_mult * div.lr_backoff
        # Keep the data loader where it is, so the batches that caused the spike are skipped.
        loader_state = self.train_loader.state_dict()
        self._restore(load_checkpoint(latest))
        self.train_loader.load_state_dict(loader_state)
        self.rollbacks, self.lr_mult = rollbacks, lr_mult
        self.spikes.reset()
        self.log(f"warning: {problem}; rolled back to step {self.step:,}, LR x{self.lr_mult:g}")
        return True

    # --- checkpoints -----------------------------------------------------------------------------

    def _state(self) -> dict:
        return {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.step,
            "tokens": self.tokens,
            "train_time_s": self.train_time_s,
            "loader": self.train_loader.state_dict(),
            "rng": rng_state(),
            "config": self.cfg.model_dump(mode="json"),
            "config_hash": self.cfg.config_hash(),
            "loss_ema": self.loss_ema,
            "lr_mult": self.lr_mult,
            "rollbacks": self.rollbacks,
            "early_stopper": self.stopper.state_dict(),
            "gap": self.gap.state_dict(),
            "precision": self.precision,
            "quark_version": quark.__version__,
        }

    def _restore(self, state: dict) -> None:
        self.raw_model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if state["scaler"] and self.scaler.is_enabled():  # empty when saved from a bf16/fp32 run
            self.scaler.load_state_dict(state["scaler"])
        self.step, self.tokens = state["step"], state["tokens"]
        self.train_time_s = state["train_time_s"]
        self.train_loader.load_state_dict(state["loader"])
        set_rng_state(state["rng"])
        self.loss_ema, self.lr_mult, self.rollbacks = state["loss_ema"], state["lr_mult"], state["rollbacks"]
        self.stopper.load_state_dict(state["early_stopper"])
        self.gap.load_state_dict(state["gap"])

    def _resume(self, path: Path, allow_config_change: bool) -> None:
        state = load_checkpoint(path)
        if state["config_hash"] != self.cfg.config_hash() and not allow_config_change:
            raise ConfigMismatchError(
                f"{path} was trained with a different config (hash {state['config_hash']} vs "
                f"{self.cfg.config_hash()}). Pass allow_config_change to continue anyway."
            )
        self._restore(state)
        self.log(f"resumed from {path} at step {self.step:,} ({self.tokens / 1e6:.1f}M tokens)")

    def _save_latest(self, step_copy: bool) -> None:
        latest = save_checkpoint(self._state(), self.run_dir / "latest.pt")
        if step_copy and self.tcfg.keep_last > 0:
            shutil.copyfile(latest, self.run_dir / f"step_{self.step}.pt")
            prune_step_checkpoints(self.run_dir, self.tcfg.keep_last)

    # --- logging ---------------------------------------------------------------------------------

    def _log_train(self, loss: float, grad_norm: float, lr: float, window: dict) -> None:
        elapsed = max(1e-9, time.time() - window["t"] - window["paused"])
        steps = max(1, window["steps"])
        tps = (self.tokens - window["tokens"]) / elapsed
        eta_s = (self.total_steps - self.step) * elapsed / steps
        record = {
            "type": "train",
            "step": self.step,
            "tokens": self.tokens,
            "loss": loss,
            "loss_ema": self.loss_ema,
            "lr": lr,
            "grad_norm": grad_norm,
            "clip_frac": window["clipped"] / steps,
            "skipped_steps": window["skipped"],
            "tokens_per_sec": tps,
            "step_ms": 1000 * elapsed / steps,
            "tflops": self.raw_model.flops_per_token() * tps / 1e12,
            "epoch": self.train_loader.epoch_progress,
            "eta_s": eta_s,
        }
        peak = self.hardware.measured.matmul_tflops if self.hardware else None
        if peak:
            record["mfu"] = record["tflops"] / peak
        if self.device.type == "cuda":
            record["mem_alloc_gb"] = torch.cuda.memory_allocated(self.device) / 1024**3
            record["mem_peak_gb"] = torch.cuda.max_memory_allocated(self.device) / 1024**3
        if self.scaler.is_enabled():
            record["loss_scale"] = self.scaler.get_scale()
        self._write_metrics(record)

        mem = f" | mem {record['mem_alloc_gb']:.1f}/{record['mem_peak_gb']:.1f} GB" if "mem_peak_gb" in record else ""
        self.log(
            f"step {self.step:>7,}/{self.total_steps:,} | loss {loss:.4f} (ema {self.loss_ema:.4f}) | "
            f"lr {lr:.2e} | gn {grad_norm:.2f} | {tps / 1e3:.1f}k tok/s{mem} | "
            f"ep {record['epoch']:.2f} | ETA {_fmt_duration(eta_s)}"
        )

    def _write_metrics(self, record: dict) -> None:
        with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def _event(self, kind: str, message: str) -> None:
        self._write_metrics({"type": "event", "event": kind, "message": message, "step": self.step, "tokens": self.tokens})

    def _plot(self) -> None:
        title = f"{self.cfg.name}: {self.raw_model.num_params() / 1e6:.1f}M params"
        save_plot(self.run_dir, title)
        if self.live_plot:
            self._show_live(title)

    def _show_live(self, title: str) -> None:
        try:
            from IPython.display import Image, display
        except ImportError:
            return
        buf = io.BytesIO()
        make_figure(read_metrics(self.run_dir / "metrics.jsonl"), title).savefig(buf, format="png", dpi=90)
        image = Image(data=buf.getvalue())
        if self._display is None:
            self._display = display(image, display_id=True)
        else:
            self._display.update(image)

    def _banner(self) -> str:
        m = self.raw_model
        gpu = self.hardware.gpu_name if self.hardware else self.device.type
        return (
            f"training {self.cfg.name}: {m.num_params() / 1e6:.2f}M params "
            f"({m.num_params(non_embedding=True) / 1e6:.2f}M non-embedding) on {gpu}, {self.precision}\n"
            f"  {self.tokens_per_step:,} tokens/step x {self.total_steps:,} steps = "
            f"{self.total_steps * self.tokens_per_step / 1e6:.0f}M tokens "
            f"({self.total_steps * self.tokens_per_step / max(1, len(self.data.train)):.2f} epochs of "
            f"{len(self.data.train) / 1e6:.0f}M train tokens)\n"
            f"  run dir: {self.run_dir}"
        )


def dry_run(cfg: QuarkConfig, data_dir: str | Path, steps: int = 3) -> dict:
    """Run a few real optimizer steps and report measured peak memory and speed (plan §7.3 verify).

    Nothing is saved; the model is thrown away afterwards.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(cfg, data_dir, Path(tmp) / "run", log=lambda _: None)
        cuda = trainer.device.type == "cuda"
        trainer._train_step()  # warm-up step (allocator, kernels)
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(trainer.device)
        start = time.time()
        for _ in range(steps):
            trainer._train_step()
        if cuda:
            torch.cuda.synchronize()
        seconds = (time.time() - start) / steps
        result = {
            "params_m": trainer.raw_model.num_params() / 1e6,
            "precision": trainer.precision,
            "step_s": seconds,
            "tokens_per_sec": trainer.tokens_per_step / seconds,
            "peak_mem_gb": torch.cuda.max_memory_allocated(trainer.device) / 1024**3 if cuda else None,
            "est_hours": trainer.total_steps * seconds / 3600,
        }
        del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    return f"{seconds // 60}m{seconds % 60:02d}s"
