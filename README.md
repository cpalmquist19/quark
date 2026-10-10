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

## Commands (so far)

```powershell
uv run quark version            # package, Python and PyTorch versions
uv run quark gpu detect         # build a hardware profile from the current GPU
uv run quark gpu detect --save  # ...and write it to hardware/local-<name>.yaml
uv run quark gpu list           # list known hardware profiles
uv run pytest                   # run the tests
```

## Status

Phase 0 (scaffold) is done. Next up is Phase 1: baseline GPT, data preparation, trainer, notebooks.
