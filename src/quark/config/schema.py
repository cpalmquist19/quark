"""Config schema shared by the planner, generator and trainer.

One `QuarkConfig` describes a model, how to train it, what data to use, and which hardware
profile it targets. It round-trips through YAML, and `config_hash()` identifies the parts that
make a checkpoint incompatible (model, train and data), so resume can detect config changes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- Hardware -----------------------------------------------------------------------------------


class Measured(_Strict):
    """Throughput numbers filled in by `quark gpu bench` (Phase 3). None until measured."""

    matmul_tflops: float | None = None
    ref_model_tokens_per_sec: float | None = None


class Session(_Strict):
    """Limits for time-boxed environments such as Colab. None means unlimited."""

    max_hours: float | None = None
    weekly_hours: float | None = None
    checkpoint_every_min: float | None = None


class HardwareProfile(_Strict):
    name: str
    gpu_name: str
    vram_gb: float
    usable_vram_gb: float
    compute_capability: str  # e.g. "8.6"
    bf16: bool  # native bf16 (Ampere+); T4 emulates it slowly, so it counts as False
    tf32: bool
    flash_attn_pkg: bool  # the separate `flash-attn` package, not PyTorch's built-in SDPA kernel
    sdpa_kernels: list[Literal["flash", "mem_efficient", "math"]]
    triton: bool
    measured: Measured = Field(default_factory=Measured)
    session: Session = Field(default_factory=Session)
    storage: Literal["local", "gdrive"] = "local"

    @property
    def precision(self) -> Literal["bf16", "fp16"]:
        """Mixed-precision dtype to train with: bf16 where native, otherwise fp16 + GradScaler."""
        return "bf16" if self.bf16 else "fp16"

    @model_validator(mode="after")
    def _check_memory(self) -> HardwareProfile:
        if not 0 < self.usable_vram_gb <= self.vram_gb:
            raise ValueError("usable_vram_gb must be > 0 and <= vram_gb")
        return self


# --- Model / train / data -----------------------------------------------------------------------

# Baseline choice per model slot (plan §5.3). Alternatives come from the component registry
# (quark.components); optimizer and schedule are the train-stage slots on TrainConfig.
DEFAULT_SLOTS: dict[str, str] = {
    "position": "rope",
    "attention": "mha",
    "norm": "rmsnorm",
    "mlp": "swiglu",
    "block": "standard",
    "head": "tied",
}


class ModelConfig(_Strict):
    vocab_size: int = Field(8192, gt=0)
    d_model: int = Field(384, gt=0)
    n_layers: int = Field(6, gt=0)
    n_heads: int = Field(6, gt=0)
    n_kv_heads: int | None = None  # None = same as n_heads (plain MHA)
    ffn_mult: float = Field(8 / 3, gt=0)  # SwiGLU hidden = ffn_mult * d_model (rounded in the model)
    seq_len: int = Field(512, gt=0)
    slots: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_SLOTS))
    toggles: dict[str, Any] = Field(default_factory=dict)  # add-ons: {"qk_norm": True} or {"z_loss": {"coef": 1e-4}}
    params: dict[str, dict[str, Any]] = Field(default_factory=dict)  # per-slot settings: {"position": {"base": 5e5}}

    @model_validator(mode="after")
    def _check_shapes(self) -> ModelConfig:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        kv = self.n_kv_heads or self.n_heads
        if self.n_heads % kv:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({kv})")
        missing = set(DEFAULT_SLOTS) - set(self.slots)
        if missing:
            raise ValueError(f"missing slot choices: {sorted(missing)}")
        return self

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class EarlyStoppingConfig(_Strict):
    enabled: bool = True
    patience: int = Field(5, gt=0)  # evals without a min_delta improvement in val loss
    min_delta: float = Field(0.005, ge=0)
    gap_window: int = Field(5, gt=1)  # evals over which a growing val-train gap triggers a warning
    stop_on_gap: bool = False  # warn only by default


class DivergenceConfig(_Strict):
    enabled: bool = True
    spike_sigma: float = Field(6.0, gt=0)  # a loss this many std devs above the recent mean is a spike
    window: int = Field(100, gt=10)  # recent steps used for the mean/std
    max_rollbacks: int = Field(3, ge=0)
    lr_backoff: float = Field(0.5, gt=0, le=1)  # LR multiplier applied after each rollback


class TrainConfig(_Strict):
    optimizer: str = "adamw"  # a registered optimizer component
    lr: float = Field(3e-4, gt=0)
    min_lr_ratio: float = Field(0.1, ge=0, le=1)
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = Field(0.1, ge=0)
    schedule: str = "cosine"  # a registered schedule component
    warmup_steps: int = Field(200, ge=0)
    max_tokens: int = Field(100_000_000, gt=0)
    micro_batch: int = Field(16, gt=0)
    grad_accum: int = Field(1, gt=0)
    grad_clip: float | None = 1.0
    precision: Literal["auto", "bf16", "fp16", "fp32"] = "auto"  # auto = from the hardware profile
    seed: int = 1337
    params: dict[str, dict[str, Any]] = Field(default_factory=dict)  # per-slot settings: {"schedule": {...}}

    # Runtime settings: they don't change what is learned, so they're excluded from config_hash()
    # and can be changed freely between resumes.
    compile: bool = False  # torch.compile; needs Triton (Linux/WSL2 or triton-windows)
    log_every_steps: int = Field(10, gt=0)
    eval_every_steps: int = Field(250, gt=0)
    eval_batches: int = Field(20, gt=0)
    checkpoint_every_steps: int = Field(1000, gt=0)
    checkpoint_every_min: float | None = None  # time-based saves; defaults from the hardware session
    keep_last: int = Field(2, ge=0)  # rolling step checkpoints kept besides latest.pt and best.pt
    time_limit_min: float | None = None  # stop + save before this; defaults from the hardware session
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    divergence: DivergenceConfig = Field(default_factory=DivergenceConfig)

    def tokens_per_step(self, seq_len: int) -> int:
        return self.micro_batch * self.grad_accum * seq_len


RUNTIME_TRAIN_FIELDS = frozenset(
    {
        "compile",
        "log_every_steps",
        "eval_every_steps",
        "eval_batches",
        "checkpoint_every_steps",
        "checkpoint_every_min",
        "keep_last",
        "time_limit_min",
        "early_stopping",
        "divergence",
    }
)


class TokenizerConfig(_Strict):
    # Plan §7.4 tokenizer rule: custom ~8k BPE by default, a pretrained one for larger/broad models.
    type: Literal["train", "pretrained"] = "train"
    size: int | None = 8192  # vocab size when type == "train"
    name: str | None = None  # e.g. "smollm2" when type == "pretrained"

    @model_validator(mode="after")
    def _check(self) -> TokenizerConfig:
        if self.type == "train" and not self.size:
            raise ValueError("tokenizer.size is required when type is 'train'")
        if self.type == "pretrained" and not self.name:
            raise ValueError("tokenizer.name is required when type is 'pretrained'")
        return self

    def tag(self) -> str:
        """Short label for directory names, e.g. 'bpe8192' or 'smollm2'."""
        return f"bpe{self.size}" if self.type == "train" else str(self.name).replace("/", "-")


class DataConfig(_Strict):
    source: str = "roneneldan/TinyStories"  # HF dataset name or a local file/folder
    subset: str | None = None  # HF dataset config, e.g. "sample-10BT" for HuggingFaceFW/fineweb-edu
    split: str = "train"  # HF split to read; train/val are split by Quark (hash-based)
    token_limit: int | None = Field(None, gt=0)  # stop after this many tokens; HF sources are then streamed
    text_field: str = "text"
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    val_fraction: float = Field(0.005, gt=0, lt=0.5)


# Fields added after configs were already in use: config_hash() skips them while unset.
HASH_OMIT_IF_DEFAULT = (("model", "params"), ("train", "params"), ("data", "subset"), ("data", "token_limit"))


class QuarkConfig(_Strict):
    name: str = "untitled"
    hardware: str | None = None  # hardware profile name; None = detect at train time
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    def config_hash(self) -> str:
        """Short, stable hash of everything that affects checkpoint compatibility.

        `name` and `hardware` are excluded on purpose: checkpoints are portable across GPUs. So are
        runtime settings such as logging cadence and early stopping (RUNTIME_TRAIN_FIELDS).
        Fields added after Phase 1 are left out while unset (HASH_OMIT_IF_DEFAULT), so configs written
        before they existed keep their hash and their runs can still resume.
        """
        payload = self.model_dump(include={"model", "train", "data"}, mode="json")
        for field in RUNTIME_TRAIN_FIELDS:
            payload["train"].pop(field, None)
        for section, field in HASH_OMIT_IF_DEFAULT:
            if payload[section][field] in (None, {}):
                del payload[section][field]
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def to_yaml(self, path: str | Path | None = None) -> str:
        text = yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_yaml(cls, source: str | Path) -> QuarkConfig:
        """Load from a YAML file path or a YAML string."""
        text = Path(source).read_text(encoding="utf-8") if _looks_like_path(source) else str(source)
        return cls.model_validate(yaml.safe_load(text) or {})


def _looks_like_path(source: str | Path) -> bool:
    if isinstance(source, Path):
        return True
    return "\n" not in source and source.endswith((".yaml", ".yml"))
