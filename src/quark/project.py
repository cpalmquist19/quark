"""Model folders and runs.

models/<name>/
  config.yaml
  runs/<YYYYmmdd-HHMMSS>/
    config.yaml       the exact config this run trained with
    run.json          data directory, hardware, config hash
    tokenizer.json    copied from the data directory, so checkpoints are self-contained
    latest.pt, best.pt, step_<n>.pt
    metrics.jsonl, training.png
"""

from __future__ import annotations

import time
from pathlib import Path

from quark.config import QuarkConfig

MODELS_DIR = Path("models")
CHECKPOINT_NAMES = ("latest", "best")


def model_dir(name_or_path: str | Path, root: str | Path | None = None) -> Path:
    """A model folder from a name (models/<name>) or a path to the folder itself."""
    path = Path(name_or_path)
    if (path / "config.yaml").exists() or path.is_absolute() or len(path.parts) > 1:
        return path
    return Path(root or MODELS_DIR) / path


def create_model(name: str, cfg: QuarkConfig, root: str | Path | None = None, force: bool = False) -> Path:
    directory = model_dir(name, root)
    config_path = directory / "config.yaml"
    if config_path.exists() and not force:
        raise FileExistsError(f"{config_path} already exists (use force to overwrite)")
    directory.mkdir(parents=True, exist_ok=True)
    cfg = cfg.model_copy(update={"name": directory.name})
    cfg.to_yaml(config_path)
    return directory


def load_model_config(directory: str | Path) -> QuarkConfig:
    path = Path(directory) / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no config.yaml in {directory}; create one with `quark new`")
    return QuarkConfig.from_yaml(path)


def new_run_dir(directory: str | Path) -> Path:
    run = Path(directory) / "runs" / time.strftime("%Y%m%d-%H%M%S")
    while run.exists():  # two runs started within the same second
        run = run.with_name(run.name + "b")
    run.mkdir(parents=True)
    return run


def latest_run(directory: str | Path) -> Path | None:
    runs = sorted((Path(directory) / "runs").glob("*/"))
    return runs[-1] if runs else None


def plan_run(directory: str | Path, resume: str = "auto") -> tuple[Path, Path | None]:
    """Decide where to train: returns (run_dir, checkpoint to resume from or None).

    resume = "auto"   continue the most recent run if it has a latest.pt, else start a new run
             "new"    always start a new run
             "latest" | "best" | <path>   resume that checkpoint (error if missing)
    """
    if resume == "new":
        return new_run_dir(directory), None
    if resume == "auto":
        run = latest_run(directory)
        if run is not None and (run / "latest.pt").exists():
            return run, run / "latest.pt"
        return new_run_dir(directory), None
    checkpoint = resolve_checkpoint(directory, resume)
    return checkpoint.parent, checkpoint


def resolve_checkpoint(directory: str | Path, which: str = "latest") -> Path:
    """`which` is 'latest' or 'best' (from the most recent run that has it) or a checkpoint path."""
    if which not in CHECKPOINT_NAMES:
        path = Path(which)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint {path} not found")
        return path
    for run in sorted((Path(directory) / "runs").glob("*/"), reverse=True):
        path = run / f"{which}.pt"
        if path.exists():
            return path
    raise FileNotFoundError(f"no {which}.pt in any run under {Path(directory) / 'runs'}")
