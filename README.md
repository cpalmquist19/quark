# Quark

A composable small-transformer workbench. Start from a baseline GPT, add techniques from recent papers, let a
planner pick the best valid configuration for your GPU and goal, then train it on Google Colab (or locally).

- Plan and roadmap: [docs/project_plan_opus_5.5_medium.md](docs/project_plan_opus_5.5_medium.md)
- Component registry design: [docs/component_registry.md](docs/component_registry.md)
- Research notes: [docs/ml_papers3.txt](docs/ml_papers3.txt)

Training currently runs on **Colab (free T4)**; the local machine is used for development, tests and short
smoke runs.

**Target model:** a ~125–350M-param chatbot (2048-token context) trained on general educational English with
neurodivergence, education and pre-K/kindergarten content emphasized; final training on a paid GPU. Components
are chosen by small, cheap experiments designed to transfer to that size. See
[plan §18](docs/project_plan_opus_5.5_medium.md#18-target-model-a-domain-focused-chatbot-decided-2026-10-10).

## Setup (Windows, for development)

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

## Train on Google Colab

1. Upload [notebooks/quark_colab.ipynb](notebooks/quark_colab.ipynb) to Colab (or open it from GitHub).
2. *Runtime → Change runtime type → T4 GPU*.
3. Edit the **Settings** cell: `MODEL_NAME` (a folder under `MyDrive/quark/models/`), `PRESET` (used only when
   that folder doesn't exist yet), `RESUME`, `MAX_STEPS`.
4. *Runtime → Run all*. Approve Google Drive access.

Things to know:

- **Colab installs Quark from GitHub** (`QUARK_REF = "main"`). Local changes reach Colab only after they're
  pushed. Set `QUARK_REF` to a commit or tag to pin a version.
- **Everything persistent lives on Drive** under `MyDrive/quark/`: tokenized data (`data/`, prepared once and
  reused) and models (`models/<name>/config.yaml` and `runs/<timestamp>/`). `latest.pt` is saved every 15
  minutes.
- **Resuming:** after a disconnect, *Run all* again. With `RESUME = "auto"` it continues the newest run of
  `MODEL_NAME`. If that run already finished there is nothing left to train, so use a new `MODEL_NAME` (or
  `RESUME = "new"`) for a fresh run.
- **Always run from the top.** If Colab restarts the session (for example after the install), variables from
  earlier cells are gone; an error like `name 'hw' is not defined` means an earlier cell didn't run.
- **Changing the model's components on Colab:** edit `MyDrive/quark/models/<name>/config.yaml` (Colab's file
  browser can open it) before training starts. See [Choose components](#choose-components).
- **Getting a model back to this machine:** download the run folder from Drive and keep the layout:
  `models/<name>/config.yaml` and `models/<name>/runs/<timestamp>/` with `best.pt` and `tokenizer.json`
  (`latest.pt` only if you'll continue training). Then `uv run quark generate <name>`.

The local notebook, [notebooks/quark_local.ipynb](notebooks/quark_local.ipynb), works the same way on this
machine's GPU (select the `.venv` kernel). Both notebooks are generated from one template by
`notebooks/build_notebooks.py`, so edit that and re-run it rather than editing the `.ipynb` files.

## Choose components

A model is built from **slots** (one choice each) and **toggles** (independent add-ons). List what's available:

```powershell
uv run quark components list            # everything, grouped by slot
uv run quark components show optimizer:muon
```

Pick them in the model's `config.yaml`:

```yaml
model:
  slots: {position: rope, attention: mha, norm: rmsnorm, mlp: swiglu, block: standard, head: tied}
  toggles: {qk_norm: true, z_loss: {coef: 1.0e-4}}
  params: {position: {base: 10000}}       # per-slot settings; omitted = paper defaults
train:
  optimizer: muon                         # adamw | muon
  schedule: wsd                           # cosine | wsd
  params: {schedule: {decay_frac: 0.2}}
```

Invalid combinations fail before training starts, with the reason (for example
`unknown position 'ropee'; did you mean 'rope'?` or `requires attention:mha`). Grouped-query attention is
`model.n_kv_heads`, not a component.

## Compare components

`quark compare` decides whether a component helps. A spec (see
[experiments/first-batch.yaml](experiments/first-batch.yaml)) names a base config, **arms** (each a partial
override, `{}` = the base) and **seeds**. Every arm trains once per seed on the same token budget; paired seeds
share data order and initial weights, so per-seed differences cancel most run-to-run noise.

- **On Colab:** open [notebooks/quark_compare_colab.ipynb](notebooks/quark_compare_colab.ipynb), edit the spec in
  its Settings cell, *Run all*. Results go to `MyDrive/quark/experiments/<name>/`. If the session ends, *Run all*
  again: finished runs are skipped and an unfinished one resumes. It stops by itself ~20 minutes before
  Colab's 12-hour limit.
- **Locally:** `uv run quark compare check experiments/first-batch.yaml`, then
  `uv run quark compare run experiments/first-batch.yaml --data <prepared data folder>`.
- **The report** (`report.md`, `curves.png`, `results.json`) gives, per arm: the paired Δ in final val loss vs
  the reference (± standard error), how many seeds agree, an estimated Δ at equal wall-clock time, speed, peak
  memory and a verdict (*better* / *worse* / *within noise*). The curves show whether a gap was still widening.
- **Ladder presets:** `ablation-s` (~8.5M params, 40M tokens, seq_len 2048) for every component; `ablation-m`
  (~38M, 200M tokens) for finalists. Why a ladder, and which results transfer to the target model: plan §18.2.
- Checkpoints of finished runs are deleted to save Drive space (`keep_checkpoints: true` keeps them); the
  metrics stay.

## Commands

```
quark version | gpu detect [--save] | gpu list
quark components list [<slot>] | components show <slot:name>
quark compare check <spec.yaml> | compare run <spec.yaml> --data <dir> | compare report <experiment dir>
quark new <name> [--preset smoke|tinystories-small|ablation-s|ablation-m]
quark data prepare [--source ...] [--tokenizer train:8192|pretrained:smollm2] [--max-docs N]
quark train <model> --data <dir> | --resume latest|best|<path> [--max-steps N]
quark generate <model> [--ckpt best|latest|<path>] [--prompt ...] [--samples N] [--temperature ...]
```

`uv run pytest` runs the tests.

## Status

- Phase 0 (scaffold): done.
- Phase 1 (baseline GPT, data, trainer, generation, notebooks): done, exit criteria met on Colab (val loss 1.48, resume from Drive verified).
- Phase 2: component registry done ([design](docs/component_registry.md)); the baseline is now built from registered components.
- First add-ons done: Muon, WSD, QK-norm, z-loss, ALiBi, NoPE, ReLU² (`quark components list`).
- `quark compare` (paired-seed ablations), the `ablation-s` / `ablation-m` ladder presets and the Colab compare
  notebook: done.
- Next: data mixtures with per-domain validation, then the first experiments on the target mix.
