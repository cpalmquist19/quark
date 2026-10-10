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

# Baseline choice per slot (plan §5.3). Later phases add alternatives through the component registry.
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
    toggles: dict[str, Any] = Field(default_factory=dict)  # independent on/off add-ons, e.g. {"qk_norm": True}

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


class TrainConfig(_Strict):
    optimizer: str = "adamw"
    lr: float = Field(3e-4, gt=0)
    weight_decay: float = Field(0.1, ge=0)
    schedule: str = "cosine"
    warmup_steps: int = Field(200, ge=0)
    max_tokens: int = Field(100_000_000, gt=0)
    micro_batch: int = Field(16, gt=0)
    grad_accum: int = Field(1, gt=0)
    grad_clip: float | None = 1.0
    precision: Literal["auto", "bf16", "fp16", "fp32"] = "auto"  # auto = from the hardware profile
    eval_every_steps: int = Field(250, gt=0)
    checkpoint_every_steps: int = Field(1000, gt=0)
    checkpoint_every_min: float | None = None  # time-based saves (Colab)
    seed: int = 1337


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


class DataConfig(_Strict):
    source: str = "roneneldan/TinyStories"  # HF dataset name or a local path
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    val_fraction: float = Field(0.005, gt=0, lt=0.5)


class QuarkConfig(_Strict):
    name: str = "untitled"
    hardware: str | None = None  # hardware profile name; None = detect at train time
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    def config_hash(self) -> str:
        """Short, stable hash of everything that affects checkpoint compatibility.

        `name` and `hardware` are excluded on purpose: checkpoints are portable across GPUs.
        """
        payload = self.model_dump(include={"model", "train", "data"}, mode="json")
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
