"""Load, list and save hardware profiles.

Built-in presets ship with the package (`quark/hardware/presets/`). Profiles you save with
`quark gpu detect --save` go to `./hardware/` in the current project and take precedence.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from quark.config import HardwareProfile

PRESET_DIR = Path(__file__).parent / "presets"


def user_dir() -> Path:
    return Path.cwd() / "hardware"


def _profile_files() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for directory in (PRESET_DIR, user_dir()):  # user profiles override presets with the same name
        if directory.is_dir():
            for path in sorted(directory.glob("*.yaml")):
                files[path.stem] = path
    return files


def list_profiles() -> list[tuple[HardwareProfile, Path]]:
    return [(load_profile_file(path), path) for path in _profile_files().values()]


def load_profile(name: str) -> HardwareProfile:
    files = _profile_files()
    if name not in files:
        known = ", ".join(sorted(files)) or "none"
        raise KeyError(f"unknown hardware profile {name!r} (known: {known})")
    return load_profile_file(files[name])


def load_profile_file(path: str | Path) -> HardwareProfile:
    return HardwareProfile.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def save_profile(profile: HardwareProfile, directory: str | Path | None = None) -> Path:
    directory = Path(directory) if directory is not None else user_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{profile.name}.yaml"
    path.write_text(yaml.safe_dump(profile.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    return path
