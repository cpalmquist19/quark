# Quark

A composable small-transformer workbench. Start from a baseline GPT, add techniques from recent papers, let a
planner pick the best valid configuration for your GPU and goal, then train it locally or on Google Colab.

- Plan: [docs/project_plan_opus_5.5_medium.md](docs/project_plan_opus_5.5_medium.md)
- Research notes: [docs/ml_papers3.txt](docs/ml_papers3.txt)

## Setup (Windows, local GPU)

Requires [uv](https://docs.astral.sh/uv/). This creates `.venv/` with the CUDA build of PyTorch:

```powershell
uv sync --extra dev
uv run quark gpu detect
```

`gpu detect` should report your GPU with `cuda: yes`. If it says `cuda: no`, PyTorch is the CPU build. Re-run
`uv sync`.

## Train a TinyStories model (command line)

```powershell
uv run quark new ts-small --preset tinystories-small    # writes models/ts-small/config.yaml
uv run quark data prepare                               # downloads TinyStories, trains an 8k BPE, tokenizes (~5 min)
uv run quark train ts-small --data data/tinystories-bpe8192
uv run quark generate ts-small --prompt "Once upon a time" --samples 3
```

- Stop training any time with Ctrl+C. It saves `latest.pt` first. Continue with
  `uv run quark train ts-small --resume latest`.
- Each run lives in `models/<name>/runs/<timestamp>/`: `best.pt`, `latest.pt`, `metrics.jsonl`, `training.png`
  (loss / learning rate / grad norm / throughput), and the config and tokenizer it used.
- Settings live in `models/<name>/config.yaml`. Resuming with a changed config is refused unless you pass
  `--allow-config-change`. Logging, eval, checkpoint and early-stopping settings can change freely.

## Train in a notebook

- **Local:** open [notebooks/quark_local.ipynb](notebooks/quark_local.ipynb), select the `.venv` kernel, edit the
  Settings cell, Run All.
- **Google Colab (free T4):** upload [notebooks/quark_colab.ipynb](notebooks/quark_colab.ipynb) to Colab (or open
  it from GitHub), set the runtime to T4 GPU, Run All. It installs Quark from GitHub and keeps data and
  checkpoints on Google Drive. After a disconnect, Run All again and it resumes.

Both notebooks show a live training dashboard. They're generated from one template by
`notebooks/build_notebooks.py`, so edit that and re-run it rather than editing the `.ipynb` files.

## Commands

```
quark version | gpu detect [--save] | gpu list
quark new <name> [--preset smoke|tinystories-small]
quark data prepare [--source ...] [--tokenizer train:8192|pretrained:smollm2] [--max-docs N]
quark train <model> --data <dir> | --resume latest|best|<path> [--max-steps N]
quark generate <model> [--ckpt best|latest|<path>] [--prompt ...] [--samples N] [--temperature ...]
```

`uv run pytest` runs the tests.

## Status

- Phase 0 (scaffold): done.
- Phase 1 (baseline GPT, data, trainer, generation, notebooks): done. See the plan for the exit-criteria run.
- Next: Phase 2, the component registry and first add-ons (Muon, WSD, QK-norm, GQA, DaRoPE, ...).
