"""Writes the notebooks from one cell list: quark_local.ipynb, quark_colab.ipynb (train one model),
quark_compare_colab.ipynb (run a component comparison on Colab) and quark_data_colab.ipynb (build a data
mix on a CPU-only Colab runtime).

Phase 1 hand-written templates (plan §10). Targets share every cell except the target-specific ones,
so they can't drift apart. Phase 4 replaces this with `quark notebook`. The compare notebook embeds
experiments/first-batch.yaml as its default spec; the data notebook embeds mixes/chatbot-v1.yaml.

    uv run python notebooks/build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
REPO = "https://github.com/cpalmquist19/quark.git"
DEFAULT_SPEC = (HERE.parent / "experiments" / "first-batch.yaml").read_text(encoding="utf-8").strip()
assert "'''" not in DEFAULT_SPEC  # it's embedded in a ''' string in the compare notebook
DEFAULT_MIX = (HERE.parent / "mixes" / "chatbot-v1.yaml").read_text(encoding="utf-8").strip()
assert "'''" not in DEFAULT_MIX

# (targets, kind, source). targets: "both" (= local + colab), a target name, or a tuple of names;
# kind: "md" | "code".
CELLS: list[tuple[str | tuple[str, ...], str, str]] = [
    ("data", "md", """# Quark: build a data mix on Google Colab (CPU)

Streams each source of a mix recipe, keeps the documents its filter accepts, then trains one tokenizer and tokenizes every source (`docs/domain_data_survey.md`, plan §18). Everything is saved to **Google Drive**, where the training and compare notebooks pick it up.

**No GPU needed:** *Runtime → Change runtime type → CPU*, so this doesn't use GPU quota. Then *Runtime → Run all*.

**Long scans are resumable.** Collection saves its stream position every ~20,000 documents. If the session ends, *Run all* again and it continues where it stopped. Keep the browser tab open while it runs."""),
    ("compare", "md", """# Quark: compare components on Google Colab

Trains every **arm** of a comparison once per **seed** on the same token budget and reports how each arm differs from the reference, with paired seeds to cancel run-to-run noise (plan §18.2). Data, runs and the report live on **Google Drive**.

**Before running:** *Runtime → Change runtime type → T4 GPU*. Edit the **Settings** cell (the spec), then *Runtime → Run all*.

**Interrupted?** Just *Run all* again: finished runs are skipped and an unfinished run resumes from its last checkpoint. The compare also stops on its own ~20 minutes before the 12-hour session limit."""),
    ("local", "md", """# Quark: train a model locally

Trains a Quark model on this machine's GPU with a live dashboard, best/latest checkpoints, early stopping and automatic resume.

**Before running:** select the project's `.venv` as the notebook kernel (VS Code: *Select Kernel → Python Environments → .venv*). If it isn't there, run `uv sync --extra dev` in the repo first.

Edit only the **Settings** cell, then *Run All*. Re-running the notebook resumes the latest run automatically."""),
    ("colab", "md", """# Quark: train a model on Google Colab

Trains a Quark model on Colab's free T4 GPU. Checkpoints, data and plots live on **Google Drive**, so a disconnected session loses at most ~15 minutes: re-run all cells and training resumes where it stopped.

**Before running:** *Runtime → Change runtime type → T4 GPU*. Edit only the **Settings** cell, then *Runtime → Run all*. Drive will ask for permission once per session."""),

    (("local", "colab", "compare"), "md", "## 1. Environment check"),
    (("colab", "compare"), "code", """!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv"""),
    (("colab", "compare", "data"), "md", "## 2. Google Drive (persistent storage)"),
    (("colab", "compare", "data"), "code", """from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
DRIVE_ROOT = Path("/content/drive/MyDrive/quark")  # models, checkpoints and cached data live here
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
print("Drive folder:", DRIVE_ROOT)"""),
    (("colab", "compare", "data"), "md", "## 3. Install Quark"),
    (("colab", "compare", "data"), "code", """QUARK_REF = "main"  # a branch, tag or commit of the quark repo
%pip install -q "git+""" + REPO + """@{QUARK_REF}\""""),

    (("local", "colab", "compare"), "code", """import sys
import torch
import quark
from quark.hardware import detect_profile, load_profile

print(f"python {sys.version.split()[0]} | torch {torch.__version__} (CUDA {torch.version.cuda}) | quark {quark.__version__}")
assert torch.cuda.is_available(), "No GPU visible to PyTorch." + (
    " Runtime -> Change runtime type -> T4 GPU." if "google.colab" in sys.modules
    else " Is the notebook using the project's .venv kernel (CUDA build of torch)?")

hw = detect_profile()
print(f"GPU: {hw.gpu_name} | {hw.vram_gb:g} GB ({hw.usable_vram_gb:g} usable) | precision {hw.precision} | SDPA {hw.sdpa_kernels}")
TARGET = "colab-t4" if "google.colab" in sys.modules else "rtx3070ti-laptop"
if hw.compute_capability != load_profile(TARGET).compute_capability:
    print(f"warning: this GPU differs from the expected target ({TARGET}); settings may need adjusting")"""),

    ("both", "md", """## Settings
The only cell you're expected to edit."""),
    ("local", "code", """from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()  # the quark repo
MODELS_ROOT = ROOT / "models"
DATA_ROOT = ROOT / "data"

MODEL_NAME = "ts-small"                  # folder under models/
PRESET = "tinystories-small"             # used only when the model folder doesn't exist yet
DATA_SOURCE = "roneneldan/TinyStories"   # Hugging Face dataset, or a local file/folder
TOKENIZER = "train:8192"                 # train:<vocab> or pretrained:<name> (plan §7.4 rule)
RESUME = "auto"                          # auto | new | latest | best | path to a .pt file
MAX_STEPS = None                         # e.g. 200 for a quick test; None = full run"""),
    ("colab", "code", """MODELS_ROOT = DRIVE_ROOT / "models"
DATA_ROOT = DRIVE_ROOT / "data"          # tokenized data is cached on Drive, then copied to local disk
LOCAL_DATA_ROOT = Path("/content/data")

MODEL_NAME = "ts-small"                  # folder under quark/models on Drive
PRESET = "tinystories-small"             # used only when the model folder doesn't exist yet
DATA_SOURCE = "roneneldan/TinyStories"   # Hugging Face dataset, or a file/folder on Drive
TOKENIZER = "train:8192"                 # train:<vocab> or pretrained:<name> (plan §7.4 rule)
RESUME = "auto"                          # auto | new | latest | best | path to a .pt file
MAX_STEPS = None                         # e.g. 200 for a quick test; None = full run"""),

    ("both", "md", "## Model config"),
    ("both", "code", """from quark.config import TokenizerConfig
from quark.config.presets import get_preset
from quark.project import create_model, load_model_config, model_dir

MODEL_DIR = model_dir(MODEL_NAME, root=MODELS_ROOT)
if not (MODEL_DIR / "config.yaml").exists():
    create_model(MODEL_NAME, get_preset(PRESET), root=MODELS_ROOT)
    print(f"created {MODEL_DIR / 'config.yaml'} from preset {PRESET!r}")
cfg = load_model_config(MODEL_DIR)

kind, _, value = TOKENIZER.partition(":")
tok_cfg = TokenizerConfig(type="train", size=int(value)) if kind == "train" else TokenizerConfig(type="pretrained", name=value, size=None)
cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update={"source": DATA_SOURCE, "tokenizer": tok_cfg})})
m, t = cfg.model, cfg.train
print(f"{cfg.name}: d_model {m.d_model}, {m.n_layers} layers, {m.n_heads} heads, seq_len {m.seq_len}, vocab {m.vocab_size}")
print(f"train: lr {t.lr}, {t.tokens_per_step(m.seq_len):,} tokens/step, {t.max_tokens / 1e6:.0f}M tokens, precision {t.precision}")"""),

    ("both", "md", "## Data\nTokenized once, then reused. The first run downloads the dataset and trains the tokenizer (several minutes for TinyStories)."),
    ("compare", "md", """## Settings
The experiment spec. Each arm is a partial override of the base config (`{}` = the base itself); the first arm is the reference unless `reference` is set. `data` is what every arm trains on. Change `name` for a new experiment: results go to `MyDrive/quark/experiments/<name>/`."""),
    ("compare", "code", """EXPERIMENTS_ROOT = DRIVE_ROOT / "experiments"
DATA_ROOT = DRIVE_ROOT / "data"          # tokenized data is cached on Drive, then copied to local disk
LOCAL_DATA_ROOT = Path("/content/data")

SPEC_YAML = '''
""" + DEFAULT_SPEC + """
'''"""),
    ("compare", "md", "## Check the arms\nEvery arm is validated before any GPU time is spent; invalid combinations stop here with the reason."),
    ("compare", "code", """import yaml
from quark.compare import CompareSpec, arm_config, validate
from quark.model import count_params

spec = CompareSpec.model_validate(yaml.safe_load(SPEC_YAML))
EXP_DIR = EXPERIMENTS_ROOT / spec.name
issues = validate(spec, hw)
print(f"{spec.name}: {len(spec.arms)} arms x {len(spec.seeds)} seeds = {len(spec.arms) * len(spec.seeds)} runs -> {EXP_DIR}")
for arm in spec.arms:
    arm_cfg = arm_config(spec, arm, spec.seeds[0])
    notes = "; ".join(f"{i.severity}: {i.message}" for i in issues[arm])
    tag = " (reference)" if arm == spec.ref else ""
    print(f"  {arm + tag:24s} {count_params(arm_cfg.model) / 1e6:6.2f}M params  {arm_cfg.train.max_tokens / 1e6:,.0f}M tokens  {notes}")
cfg = arm_config(spec, spec.ref, spec.seeds[0])  # the data settings every arm shares"""),
    ("compare", "md", "## Data\nTokenized once, then reused by every arm and every later experiment on the same data."),
    ("local", "code", """from quark.data import TokenData, default_data_dir, prepare_dataset

DATA_DIR = default_data_dir(cfg.data, root=DATA_ROOT)
if not (DATA_DIR / "meta.json").exists():
    print(f"preparing {DATA_SOURCE} -> {DATA_DIR} ...")
    prepare_dataset(cfg.data, DATA_DIR, progress=lambda d, n: d % 100_000 == 0 and print(f"  {d:,} docs, {n / 1e6:,.0f}M tokens"))"""),
    (("colab", "compare"), "code", """import shutil
from quark.data import default_data_dir, prepare_dataset

DRIVE_DATA_DIR = default_data_dir(cfg.data, root=DATA_ROOT)
DATA_DIR = LOCAL_DATA_ROOT / DRIVE_DATA_DIR.name  # local copy: much faster random reads than Drive
if not (DRIVE_DATA_DIR / "meta.json").exists():
    if cfg.data.mix:
        raise FileNotFoundError(f"mix {cfg.data.mix!r} isn't prepared on Drive yet ({DRIVE_DATA_DIR}); "
                                "build it with notebooks/quark_data_colab.ipynb first")
    print(f"preparing {cfg.data.source} -> {DATA_DIR} (then caching to Drive) ...")
    prepare_dataset(cfg.data, DATA_DIR, progress=lambda d, n: d % 100_000 == 0 and print(f"  {d:,} docs, {n / 1e6:,.0f}M tokens"))
    shutil.copytree(DATA_DIR, DRIVE_DATA_DIR, dirs_exist_ok=True)
elif not (DATA_DIR / "meta.json").exists():
    print(f"copying cached data from Drive to {DATA_DIR} ...")
    shutil.copytree(DRIVE_DATA_DIR, DATA_DIR, dirs_exist_ok=True, ignore=shutil.ignore_patterns("raw"))"""),
    (("local", "colab", "compare"), "code", """from quark.data import MixData, load_data

data = load_data(DATA_DIR)
meta = data.meta
if isinstance(data, MixData):
    weights = data.resolve_weights(cfg.data.weights)
    print(f"mix {meta['name']} | vocab {meta['vocab_size']:,}")
    for name, src in meta["sources"].items():
        w = weights.get(name, 0.0)
        epochs = w * cfg.train.max_tokens / max(1, src["train_tokens"])
        print(f"  {name:18s} train {src['train_tokens'] / 1e6:7,.1f}M | val {src['val_tokens'] / 1e6:5,.2f}M | "
              f"weight {w:4.0%} | {epochs:.2f} epochs planned")
    sample_tokens = next(iter(data.sources.values())).val
else:
    print(f"vocab {meta['vocab_size']:,} | train {meta['train_tokens'] / 1e6:,.1f}M tokens ({meta['train_docs']:,} docs) "
          f"| val {meta['val_tokens'] / 1e6:,.2f}M tokens ({meta['val_docs']:,} docs)")
    epochs = cfg.train.max_tokens / meta["train_tokens"]
    print(f"planned training: {epochs:.2f} epochs" + ("  (multi-epoch: watch for overfitting)" if epochs > 1 else ""))
    sample_tokens = data.val
print("\\nsample (decoded from val):\\n" + data.tokenizer().decode(sample_tokens[:150].tolist()))"""),

    ("both", "md", "## Model and memory check\nOne real training step on the GPU: the actual peak memory, not an estimate."),
    ("both", "code", """from quark.train import dry_run

check = dry_run(cfg, DATA_DIR)
print(f"{check['params_m']:.2f}M params | {check['precision']} | peak GPU memory {check['peak_mem_gb']:.2f} GB "
      f"of {hw.usable_vram_gb:g} usable | {check['tokens_per_sec'] / 1e3:.1f}k tokens/s "
      f"| full run ~{check['est_hours']:.1f} h")
if check["peak_mem_gb"] > hw.usable_vram_gb:
    print("warning: this config won't fit; lower train.micro_batch and raise train.grad_accum in config.yaml")"""),

    ("both", "md", "## Resume detection\n`RESUME = \"auto\"` continues the latest run if it has a checkpoint; otherwise a new run starts."),
    ("both", "code", """from quark.project import plan_run

RUN_DIR, CHECKPOINT = plan_run(MODEL_DIR, RESUME)
print(("resuming " + str(CHECKPOINT)) if CHECKPOINT else ("new run: " + str(RUN_DIR)))"""),

    ("both", "md", "## Train\nLive dashboard below; stats print every few steps. Interrupting the cell (stop button) saves `latest.pt` first."),
    ("colab", "md", "On Colab, `latest.pt` is also saved every 15 minutes, and training stops cleanly ~20 minutes before the 12-hour session limit."),
    ("both", "code", """from quark.train import Trainer

trainer = Trainer(cfg, DATA_DIR, RUN_DIR, hardware=hw, resume_from=CHECKPOINT, live_plot=True, max_steps=MAX_STEPS)
result = trainer.fit()"""),

    ("both", "md", "## Results"),
    ("both", "code", """from IPython.display import Image, display

print(f"stopped: {result.stop_reason} | steps {result.steps:,} | tokens {result.tokens / 1e6:,.1f}M | "
      f"best val loss {result.best_val:.4f} | last val loss {result.last_val:.4f}")
print(f"run folder: {result.run_dir}")
display(Image(filename=str(result.run_dir / "training.png")))"""),

    ("both", "md", "## Samples from the best checkpoint"),
    ("both", "code", """from quark.infer import generate_text, load_for_inference

model, tokenizer, _ = load_for_inference(result.run_dir / "best.pt")
for i, prompt in enumerate(["Once upon a time", "The little dog", "Lily wanted to"]):
    print(f"--- {prompt!r} ---")
    print(generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_p=0.9, seed=i), "\\n")"""),

    ("both", "md", "## Export\nExporting to Hugging Face / safetensors format comes in Phase 6 (`quark export`). Checkpoints in the run folder already work with `quark generate` and `load_for_inference`."),

    ("compare", "md", "## Time estimate\nOne real training step of the reference arm. Slower arms (e.g. Muon) take longer."),
    ("compare", "code", """from quark.train import dry_run

done = len(list(EXP_DIR.glob("runs/*/*/result.json")))
total = len(spec.arms) * len(spec.seeds)
check = dry_run(cfg, DATA_DIR)
print(f"reference arm: {check['tokens_per_sec'] / 1e3:.1f}k tokens/s, peak GPU memory {check['peak_mem_gb']:.2f} GB "
      f"of {hw.usable_vram_gb:g} usable | ~{check['est_hours'] * 60:.0f} min per run")
print(f"{total - done} of {total} runs left: ~{check['est_hours'] * (total - done):.1f} h")"""),
    ("compare", "md", "## Run\nRuns go seed by seed (every arm for the first seed, then the next), so an interrupted compare still has balanced results. Each run shows its own dashboard; the report updates after every run."),
    ("compare", "code", """from quark.compare import run_compare

summary = run_compare(spec, DATA_DIR, EXP_DIR, hardware=hw, live_plot=True)"""),
    ("compare", "md", "## Report\nΔ < 0 means the arm beat the reference. The report explains the verdicts; `report.md`, `results.json` and `curves.png` are in the experiment folder on Drive."),
    ("compare", "code", """from IPython.display import Image, Markdown, display

display(Markdown((EXP_DIR / "report.md").read_text(encoding="utf-8")))
if (EXP_DIR / "curves.png").exists():
    display(Image(filename=str(EXP_DIR / "curves.png")))"""),

    ("data", "md", """## Settings
The mix recipe: sources, filters and token targets. Training weights are chosen later, in the training config or compare spec. Changing a source's filter after collecting it means deleting its folder under `raw/` first."""),
    ("data", "code", """DATA_ROOT = DRIVE_ROOT / "data"

# A raw string, because the filters' regexes contain backslashes.
RECIPE_YAML = r'''
""" + DEFAULT_MIX + """
'''"""),
    ("data", "code", """import yaml
from quark.data.mix import MixRecipe, collect_status, mix_dir

recipe = MixRecipe.model_validate(yaml.safe_load(RECIPE_YAML))
MIX_DIR = mix_dir(recipe, DATA_ROOT)
print(f"mix {recipe.name} -> {MIX_DIR}")
for name, st in collect_status(recipe, DATA_ROOT).items():
    print(f"  {name:18s} {st['tokens'] / 1e6:8,.1f}M / {st['target'] / 1e6:,.0f}M tokens  {'done' if st['done'] else ''}")"""),
    ("data", "md", "## Collect\nOne pass per stream: all FineWeb-Edu sources are filled from the same scan. Progress prints at every save."),
    ("data", "code", """from quark.data.mix import collect

status = collect(recipe, DATA_ROOT)"""),
    ("data", "md", "## Prepare\nTrains the shared tokenizer, then tokenizes each source into its own train/val files."),
    ("data", "code", """from quark.data.mix import prepare_mix

if (MIX_DIR / "meta.json").exists():
    print("already prepared:", MIX_DIR)
else:
    meta = prepare_mix(recipe, DATA_ROOT, progress=lambda name, d, t: d % 50_000 == 0 and print(f"  {name}: {d:,} docs"))
    for name, src in meta["sources"].items():
        print(f"{name:18s} train {src['train_tokens'] / 1e6:7,.1f}M tokens ({src['train_docs']:,} docs) | val {src['val_tokens'] / 1e6:.2f}M")"""),
    ("data", "md", "Done. To train on this mix, set `data.mix` to the mix name (and the same `data.tokenizer`) in a model config or compare spec, plus `data.weights` to choose how often each source is used."),
]


def _targets(spec: str | tuple[str, ...]) -> tuple[str, ...]:
    if spec == "both":
        return ("local", "colab")
    return (spec,) if isinstance(spec, str) else spec


def build(target: str) -> dict:
    cells = []
    for cell_targets, kind, source in CELLS:
        if target not in _targets(cell_targets):
            continue
        lines = source.splitlines(keepends=True)
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines})
    metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    if target in ("colab", "compare"):
        metadata["accelerator"] = "GPU"
        metadata["colab"] = {"provenance": [], "gpuType": "T4"}
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


NOTEBOOKS = {
    "local": "quark_local.ipynb",
    "colab": "quark_colab.ipynb",
    "compare": "quark_compare_colab.ipynb",
    "data": "quark_data_colab.ipynb",
}

if __name__ == "__main__":
    for target, filename in NOTEBOOKS.items():
        path = HERE / filename
        path.write_text(json.dumps(build(target), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("wrote", path)
