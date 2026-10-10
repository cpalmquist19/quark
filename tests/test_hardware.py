import pytest
import torch

from quark.hardware import NoGpuError, detect_profile, list_profiles, load_profile, load_profile_file, save_profile

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_presets_load():
    names = {profile.name for profile, _ in list_profiles()}
    assert {"rtx3070ti-laptop", "colab-t4"} <= names

    t4 = load_profile("colab-t4")
    assert t4.precision == "fp16"
    assert "flash" not in t4.sdpa_kernels
    assert t4.storage == "gdrive" and t4.session.max_hours == 12

    laptop = load_profile("rtx3070ti-laptop")
    assert laptop.precision == "bf16"


def test_unknown_profile_lists_known_names():
    with pytest.raises(KeyError, match="colab-t4"):
        load_profile("no-such-gpu")


def test_detect_without_cuda_explains_why(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(NoGpuError, match="No CUDA GPU"):
        detect_profile()


@requires_cuda
def test_detect_real_gpu(tmp_path):
    profile = detect_profile()
    assert profile.vram_gb > 0
    assert 0 < profile.usable_vram_gb < profile.vram_gb
    assert "math" in profile.sdpa_kernels  # the fallback kernel always works

    path = save_profile(profile, tmp_path)
    assert load_profile_file(path) == profile
