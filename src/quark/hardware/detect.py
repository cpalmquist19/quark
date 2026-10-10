"""Build a HardwareProfile from the GPU PyTorch can see.

Kernel support is probed by actually running tiny SDPA calls with each backend forced, because
spec sheets and build flags don't reliably say what works (e.g. flash SDPA on Windows builds).
"""

from __future__ import annotations

import importlib.util
import math
import os
import re
import sys
import warnings

from quark.config import HardwareProfile, Session

# Left free for the CUDA context, allocator fragmentation and the display (calibrated later).
VRAM_RESERVE_GB = 0.8


class NoGpuError(RuntimeError):
    pass


def detect_profile(device: int = 0) -> HardwareProfile:
    import torch

    if not torch.cuda.is_available():
        if torch.version.cuda is None:
            hint = (
                f"PyTorch {torch.__version__} is a CPU-only build. Install the CUDA build "
                "(in this project: `uv sync`)."
            )
        else:
            hint = "PyTorch has CUDA support but no GPU is visible. Check the NVIDIA driver (`nvidia-smi`)."
        raise NoGpuError(f"No CUDA GPU available. {hint}")

    props = torch.cuda.get_device_properties(device)
    major, minor = props.major, props.minor
    vram_gb = props.total_memory / 1024**3
    usable = max(0.1, math.floor((vram_gb - VRAM_RESERVE_GB) * 10) / 10)
    on_colab = _on_colab()

    return HardwareProfile(
        name=f"local-{_slug(props.name)}",
        gpu_name=props.name,
        vram_gb=round(vram_gb, 2),
        usable_vram_gb=usable,
        compute_capability=f"{major}.{minor}",
        bf16=major >= 8,  # torch.cuda.is_bf16_supported() is True on T4 too (emulated), so don't use it
        tf32=major >= 8,
        flash_attn_pkg=importlib.util.find_spec("flash_attn") is not None,
        sdpa_kernels=probe_sdpa_kernels(device),
        triton=importlib.util.find_spec("triton") is not None,
        session=Session(max_hours=12, weekly_hours=30, checkpoint_every_min=15) if on_colab else Session(),
        storage="gdrive" if on_colab else "local",
    )


def probe_sdpa_kernels(device: int = 0) -> list[str]:
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backends = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
    }
    q = torch.randn(1, 2, 64, 64, device=f"cuda:{device}", dtype=torch.float16)
    working = []
    for name, backend in backends.items():
        try:
            # A failed backend emits "kernel not used because..." warnings before raising; that's the
            # expected outcome of a probe, not something to show the user.
            with warnings.catch_warnings(), sdpa_kernel(backend):
                warnings.simplefilter("ignore", UserWarning)
                F.scaled_dot_product_attention(q, q, q, is_causal=True)
            torch.cuda.synchronize(device)
            working.append(name)
        except RuntimeError:
            pass
    return working


def _on_colab() -> bool:
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def _slug(text: str) -> str:
    text = re.sub(r"^nvidia\s+(geforce\s+)?", "", text.strip(), flags=re.I)
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
