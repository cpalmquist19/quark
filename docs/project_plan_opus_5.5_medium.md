# Quark — Project Plan

A composable transformer workbench: start with a baseline GPT, bolt on techniques from papers (RoPE variants, YaRN, MTP, looped layers, attention variants, Muon, WSD, …), let a planner find the **valid** and **best-ranked** configurations for a given GPU and purpose, then generate the model code plus ready-to-run notebooks (local and Google Colab) that train with live charts, best-checkpointing, early stopping and resume.

---

## 1. Goals and non-goals

**Goals**
1. A baseline decoder-only transformer that is small, readable, and correct.
2. A **component registry**: every technique is a plug-in with metadata (what it needs, what it conflicts with, what hardware it needs, at what scale it helps, what it costs).
3. A **hardware profile** system: describe a GPU (or auto-detect it) → memory, compute capability, bf16/fp16 support, FlashAttention support, measured throughput.
4. A **planner** that:
   - enumerates candidate configurations,
   - removes invalid ones (hard constraints),
   - penalizes inadvisable ones (soft constraints, with an explanation),
   - estimates memory, parameter count, FLOPs, training time, tokens/sec, and expected quality,
   - ranks the survivors using **user-configurable weights**.
5. A **generator** that takes a chosen configuration and writes a self-contained model folder into the project.
6. **Notebook generation** for local and Colab targets, with steps in the correct order.
7. **Training** with standard stats, live loss curves, best/latest checkpoints, overfitting/divergence detection, and full resume.
8. **Inference** (text generation) from any saved checkpoint.

**North star:** the best-capability model we can train on our GPUs, with **far fewer parameters than is typical for its quality class**. The 2026 literature gives two levers for this, and the plan is built around them. The first is *parameter reuse at equal compute* (looped/Hyperloop blocks, fixed-point "attractor" blocks). The second is *nearly-free per-layer upgrades validated at 50–350M scale*. Paper-by-paper evidence and verdicts: [ml_papers3.txt](ml_papers3.txt). Entry IDs such as `B2` or `C1` below refer to that file.

**Product target (2026-10-10):** a ~125–350M-param chatbot with 2048-token context, trained on FineWeb-Edu with neurodivergence, education and pre-K/kindergarten content upweighted. See §18; it decides how components are tested.

**Non-goals (for now)**
- Evals. Instead, a clean export interface (Section 12) so your separate eval project can consume checkpoints.
- Multi-GPU / distributed training. Design so it can be added later, but don't build it now.
- Fine-tuning recipes (SFT/RLHF). ~~Possibly a later phase.~~ **Changed 2026-10-10:** chat SFT is now Phase 2c, for the chatbot target (§18). RLHF remains a non-goal.

---

## 2. Hardware reality check (your two targets)

This drives many of the constraint rules, so it's worth stating up front.

Since 2026-10-10 training runs on Colab only; the laptop is for development and short smoke tests.

| | RTX 3070 Ti **Laptop** (local) | Colab Free (almost always a **T4**) |
|---|---|---|
| VRAM | 8 GB | ~15 GB usable |
| Architecture | Ampere, sm_86 | Turing, sm_75 |
| bf16 | ✅ native | ❌ (emulated and slow) → **must use fp16 + GradScaler** |
| TF32 | ✅ | ❌ |
| FlashAttention-2 (`flash-attn` pkg) | ✅ (on Linux/WSL; painful on native Windows) | ❌ requires sm_80+ |
| PyTorch SDPA kernels | mem-efficient (+ math). **Probed 2026-10-09: official Windows torch wheels are not built with flash SDPA**; flash needs Linux/WSL2 | mem-efficient only |
| `torch.compile` / Triton / FlexAttention | Needs `triton-windows` on native Windows, or WSL2 | ✅ (Linux) |
| Session limits | none | ~12 h max, idle disconnects, ~30 h/week for you |
| Persistent storage | local disk | must checkpoint to **Google Drive** |

Consequences built into the rules:
- Anything that assumes bf16 (Muon's Newton–Schulz in bf16, bf16 optimizer states) needs an fp16/fp32 fallback on T4.
- fp16 training is more prone to overflow → on fp16 targets the planner should favor stability addons (QK-norm, z-loss, logit soft-capping) and force a GradScaler.
- Colab runs **must** save time-based checkpoints to Drive (e.g. every 15–20 min), not just "best" checkpoints, because the session can die at any moment.
- Python: you have 3.13 locally; PyTorch supports it. Colab manages its own Python. Keep code compatible with 3.10+.
- On native Windows, the recommendation will be: SDPA (no `flash-attn` package), `torch.compile` optional via `triton-windows`. WSL2 is the easy path if you want the full Linux toolchain.

**Throughput should be measured, not looked up.** Spec-sheet TFLOPs are far from real numbers (laptop clocks, thermals, kernel availability). A `quark gpu bench` command runs a short matmul + small-model forward/backward benchmark and stores the result in the hardware profile. The time estimates in the planner use the measured number.

---

## 3. High-level architecture

```
                   ┌───────────────────────┐
  hardware profile │ quark gpu detect/bench│
                   └──────────┬────────────┘
                              │
  purpose + budget ──►  ┌─────▼─────┐   ┌──────────────────────┐
  ranking weights       │  PLANNER  │◄──│ COMPONENT REGISTRY    │
                        │ enumerate │   │ (metadata + rules +   │
                        │ validate  │   │  cost models + code)  │
                        │ estimate  │   └──────────────────────┘
                        │ rank      │
                        └─────┬─────┘
                              │ ranked list (only valid configs)
                        ┌─────▼─────┐
                        │ GENERATOR │──► models/<name>/  (config, frozen code, README)
                        └─────┬─────┘
                              │
                ┌─────────────┼──────────────┐
                ▼             ▼              ▼
        notebook_local  notebook_colab   CLI train / generate
                │             │              │
                └──────► TRAINER (shared library) ◄──── checkpoints, logs, plots
```

Everything is driven by one **config object** (dataclass/pydantic, serialized to YAML). The planner produces configs, the generator freezes them, and the trainer consumes them. There is one code path for training, whether called from CLI or a notebook.

---

## 4. Proposed repository layout

```
quark/
├── PLAN.md
├── pyproject.toml
├── README.md
├── src/quark/
│   ├── config/            # Config schema (ModelConfig, TrainConfig, DataConfig, HardwareProfile)
│   ├── components/        # The registry. One file per technique (as built: docs/component_registry.md).
│   │   ├── registry.py    # @register, Component metadata, slots; base.py = slot interfaces; check.py = validation
│   │   ├── toggles/       # qk_norm.py, z_loss.py, ... (metadata; implemented by a slot's component)
│   │   ├── position/      # rope.py, nope.py, alibi.py, yarn.py, ...
│   │   ├── attention/     # mha.py, gqa.py, mla.py, sliding_window.py, diff_attn.py, ...
│   │   ├── norm/          # layernorm.py, rmsnorm.py, qk_norm.py, layerrope.py
│   │   ├── mlp/           # gelu.py, swiglu.py, relu2.py, moe.py
│   │   ├── block/         # standard.py, looped.py, hyperloop.py, attractor.py, parallel.py
│   │   ├── head/          # lm_head.py, mtp.py (multi-token prediction)
│   │   ├── optim/         # adamw.py, muon.py (+SAMuon-lite, AF-Muon, hyperball), lion.py, adamw8bit.py, schedule_free.py
│   │   ├── schedule/      # cosine.py, wsd.py, linear.py, constant_cooldown.py
│   │   └── tricks/        # z_loss.py, softcap.py, weight_tying.py, grad_checkpoint.py, orca.py, ema.py, ...
│   ├── model/             # Baseline GPT assembled from component slots
│   ├── hardware/          # gpu presets (yaml), detect.py, bench.py
│   ├── planner/
│   │   ├── space.py       # search space definition
│   │   ├── constraints.py # hard + soft rule evaluation
│   │   ├── estimators.py  # memory, flops, time, quality
│   │   ├── rank.py        # pareto + weighted scoring
│   │   └── report.py      # pretty tables / explanations
│   ├── data/              # sources, tokenizers, pre-tokenize to .bin memmaps, loaders
│   ├── train/             # trainer loop, metrics, live plots, checkpointing, early stopping, kd.py (distillation)
│   ├── infer/             # generation, sampling, KV cache
│   ├── generate/          # model folder writer, notebook writer (nbformat + jinja templates)
│   └── cli.py             # `quark ...` entry point
├── hardware/              # GPU profiles: rtx3070ti-laptop.yaml, colab-t4.yaml, ...
├── purposes/              # purpose profiles: tinystories.yaml, code.yaml, long-context.yaml, ...
├── models/                # ★ GENERATED: one folder per built model
│   └── <model-name>/
│       ├── config.yaml
│       ├── model.py       # frozen, self-contained copy of the code for this config
│       ├── plan_report.md # why this config was chosen, estimates, warnings
│       ├── notebook_local.ipynb
│       ├── notebook_colab.ipynb
│       └── runs/<run-id>/ # checkpoints, metrics.jsonl, plots
├── data/                  # raw + tokenized datasets (gitignored)
└── tests/
```

---

## 5. The component registry (the heart of the project)

Implemented design: [component_registry.md](component_registry.md). This section is the original proposal.

Each technique is a class registered under a **slot**. A slot is a mutually exclusive choice (you pick exactly one position encoding, one attention type, one optimizer, …). Some things are **toggles** (independent on/off: QK-norm, z-loss, MTP, weight tying, gradient checkpointing).

Slots keep the search space manageable: the number of combinations is the product of the slot sizes, not 2^(all features).

### 5.1 Component metadata

```python
@register(slot="position", name="yarn")
class YaRN(Component):
    paper = "arXiv:2309.00071"
    summary = "Extends RoPE context by frequency-dependent interpolation + attention temperature."
    maturity = "established"           # experimental | promising | established
    requires = ["position.base:rope"]  # hard: needs a RoPE base
    conflicts = ["position:alibi", "position:nope"]
    hw = HwReq(min_sm=None, needs_bf16=False, needs_triton=False)
    phase = "extension"                # pretrain | extension | both
    helps_when = [Rule("target_ctx > train_ctx", "only useful when extending context")]
    hurts_when = []
    params = {"scale": float, "original_ctx": int}

    def param_count(self, cfg): ...
    def extra_memory(self, cfg, batch): ...
    def extra_flops(self, cfg, tokens): ...
    def quality_prior(self, cfg):        # see 7.4
        ...
    def build(self, cfg) -> nn.Module: ...
```

### 5.2 Rule types

| Type | Effect | Example |
|---|---|---|
| **requires** | invalid without it | YaRN requires RoPE |
| **conflicts** | invalid together | ALiBi + RoPE; logit soft-capping + flash SDPA kernel (needs FlexAttention or eager) |
| **hardware** | invalid on this GPU | `flash-attn` pkg on T4; bf16-only path on T4 |
| **scale** (soft) | penalty + warning | token-level MTP at < ~3B params (hurts small models; latent MTP is the exception, see C6); MoE on 8 GB; value gate below ~50M (no measurable gain, E2) |
| **purpose** (soft) | penalty + warning | YaRN when target context ≤ training context; MLA when inference KV cache isn't a priority; for `long-context`, stacking GQA + sliding window + short train ctx (compounding extension damage, E4); n-gram/Engram memory when data is repeated for >1 epoch ("forking", D4) |
| **interaction** (soft) | penalty + warning | fp16 without QK-norm/z-loss at higher LR; Muon applied to embeddings/LM head without AF-Muon/MuonIO (A4); Hyperball together with scaled WD (they're alternatives, A5/A6); truncated BPTT in looped blocks (lowers the loop's value, B1); optimizer priors measured at much larger batch than ours (A9) |
| **auto-fix** | config is silently corrected | Muon ⇒ automatically splits params into Muon group + AdamW group; T4 ⇒ fp16 + GradScaler; T4 ⇒ Muon Newton–Schulz and fixed-point solvers run in fp32; sink logit ⇒ float attn_mask, so the SDPA mem-efficient kernel is used; looped block ⇒ residual init scaled by *unrolled* depth (G1) |

Every rejection or penalty carries a **human-readable reason**, so the planner can show *why* a config was excluded (`--explain`), which addresses the "not every method should be used everywhere" concern directly.

### 5.3 Initial catalog (proposed, by phase)

| Slot / toggle | Phase 1 (baseline) | Phase 2 | Phase 2b (param-efficiency, see roadmap) | Later |
|---|---|---|---|---|
| position | learned abs, **RoPE** | NoPE, ALiBi, YaRN (ext), partial RoPE, **DaRoPE** ([report](darope_integration_report.md)) | | DAPE, DroPE, NTK-aware |
| attention | **MHA (SDPA)** | GQA, MQA, sliding window, QK-norm | value residual → SATFormer gate (C2); sink logit + value gate (E2) | MLA, differential attention, Gated-DeltaNet / head-wise hybrids (F2, needs WSL2/Triton), FlexAttention variants |
| norm | LayerNorm, **RMSNorm**, pre-norm | post-norm, sandwich norm | **LayerRoPE** depth-shared gains (C1) | NAG norm-agnostic residual (C3) |
| mlp | GELU, **SwiGLU** | ReLU² | FFN width schedule: uniform / front-loaded (F1) | MoE (small expert counts) |
| block | **standard** | | **Hyperloop** `[begin \| mid × r \| end]` + inter-loop hyper-connections (B2, B4, C4, G1); **Attractor** fixed-point block (E1) | LPT shared-KV recursions (B3), dense→looped conversion (B6) |
| head | **tied LM head** | | | latent MTP / LoopMTP (C6, planning purposes only) |
| optimizer | **AdamW** | Muon (+AdamW for non-matrix params), 8-bit AdamW | SAMuon-lite bulk gain (A2), AF-Muon for tied table (A4) | Lion, SOAP, ADANA, schedule-free AdamW |
| norm control | **decoupled WD** | WD ∝ √(overtraining) (A1) | scaled WD = WD·lr/lr_max (A5) **or** Hyperball (A6) | |
| schedule | **cosine w/ warmup** | WSD, linear decay | WSqD horizon-free (A7) | constant + cooldown on demand |
| tricks | grad clipping, grad checkpointing | z-loss, logit soft-capping, µP-style init | ORCA early orthogonal reg (A3), weight EMA / PACE (A10) | GQA-µP transfer (F7) |
| data / objective | **plain next-token** | | stack-task warm-up 1% (E3); weak-teacher KD (F3); masked-input regularization for multi-epoch (A8) | chain distillation for many-epoch runs (F4) |

Bold = baseline default. **Rejected after review** (don't build): Kronecker / fixed input token codes (no param saving with a tied head at our scale, C5/G4), Engram n-gram memory (adds size, overfits under replay, G2), K=V projection sharing (G3), KAN FFNs, X-MoD, token-level MTP for pure LM loss.

Every Phase 2b component has a recent paper validated at ≤350M, and most are very new and not yet replicated. They enter the registry with maturity `promising`, and the ablation results in `runs.db` decide whether they get promoted.

Each new component ships with: a unit test (shapes, causality), a tiny overfit test (can memorize 1 batch), and an ablation recipe (baseline vs baseline+component, same token budget).

---

## 6. Hardware and purpose profiles

### 6.1 Hardware profile (`hardware/*.yaml`)

```yaml
name: rtx3070ti-laptop
vram_gb: 8
usable_vram_gb: 7.2          # leave headroom for CUDA context + fragmentation
compute_capability: 8.6
bf16: true
tf32: true
flash_attn_pkg: false        # native Windows; true under WSL2
sdpa_kernels: [flash, mem_efficient, math]
triton: optional
measured:
  bf16_matmul_tflops: null   # filled by `quark gpu bench`
  ref_model_tokens_per_sec: null
session:
  max_hours: null
  weekly_hours: null
storage: local
```

```yaml
name: colab-t4
vram_gb: 15
usable_vram_gb: 14
compute_capability: 7.5
bf16: false
tf32: false
flash_attn_pkg: false
sdpa_kernels: [mem_efficient, math]
triton: true
session: {max_hours: 12, weekly_hours: 30, checkpoint_every_min: 15}
storage: gdrive
```

`quark gpu detect` builds a profile from `torch.cuda.get_device_properties`. `quark gpu bench` fills in measured numbers.

### 6.2 Purpose profile (`purposes/*.yaml`)

A purpose sets defaults and default ranking weights:

```yaml
name: tinystories
description: Small story generator; fast iteration, short context
dataset_hint: roneneldan/TinyStories
target_ctx: 512
vocab: {type: train, size: 8192}        # default per §7.4 tokenizer rule; or {type: pretrained, name: smollm2}
budget: {hours: 6}
weights: {quality: 0.5, train_time: 0.3, memory_headroom: 0.1, inference_speed: 0.1}
```

Other starter purposes: `code-completion`, `long-context`, `research-ablation` (equal-cost comparisons, favors fast runs), `max-quality-in-budget`.

---

## 7. The planner

### 7.1 Inputs

`quark plan --gpu colab-t4 --purpose tinystories --hours 10 --dataset-tokens 470M --weights quality=0.6,time=0.2,memory=0.2 --top 10`

### 7.2 Enumeration

1. **Slot choices**: Cartesian product of the slot options (after hardware filtering, a few thousand combos at most).
2. **Dimensions**: for each combo, a grid over `d_model`, `n_layers`, `n_heads`/`n_kv_heads`, `seq_len`, micro-batch, grad-accum, and loop count (if looped). Grid values come from sensible shape rules (head_dim 64/128, d_model a multiple of 64, aspect ratio d_model/n_layers within a healthy range). At 50–150M, bias the grid toward **deep-and-thin** shapes (d_model/n_layers ≈ 25, e.g. 23 × 576; D2). Looped blocks add a layout dimension `(n_begin, n_mid, r, n_end)`, and Hyperloop's tested layouts (`[2|4×3|2]`, `[3|4×3|3]`, `[4|10×2|4]`) are the starting grid. FFN width schedule is a further dimension (F1).
3. Everything is analytical (no GPU needed), so evaluating even 100k+ candidates takes seconds.

### 7.3 Validation (hard + soft)

- Run all `requires` / `conflicts` / `hardware` rules → drop invalid configs.
- **Memory fits?** → drop if the estimated peak exceeds `usable_vram_gb`. Try automatic rescue first: enable grad checkpointing, smaller micro-batch with more accumulation, chunked cross-entropy, 8-bit optimizer.
- Apply soft-rule penalties and collect warnings.
- Optional **empirical verification** for the top-K: `quark plan --verify` builds each finalist, runs 3 real forward/backward steps on the GPU, and records `torch.cuda.max_memory_allocated()` and real tokens/sec. This catches estimator errors and is what makes "only valid configurations" trustworthy.

### 7.4 Estimators

**Parameters**: exact, computed from each component's `param_count`.

**Memory** (training peak) ≈
- weights (fp32 master: 4 B/param)
- + gradients (4 B/param)
- + optimizer state (AdamW: 8 B/param; Muon: 4 B/param for matrices; 8-bit AdamW: ~2 B/param)
- + activations (per layer, roughly `seq · batch · d_model · k` bytes; `k` depends on the components; the attention-score term disappears with flash/mem-efficient SDPA; drops to about √layers with checkpointing)
- + **logits**: `batch · seq · vocab · 4` bytes. This is a common surprise: batch 16 × seq 1024 × vocab 50k ≈ 3.3 GB in fp32. Chunked/fused cross-entropy is an auto-fix candidate on 8 GB.
- + MTP extra heads/logits, MoE expert buffers, etc. (component-provided)
- + CUDA context and fragmentation overhead (~0.5–0.8 GB, calibrated)

**FLOPs**: `6 · N_non_embedding · tokens + attention term` (+ component extras; looped blocks multiply compute but not params).

**Time**: `FLOPs / (measured_throughput · utilization_factor)`, where the utilization factor is calibrated from benchmarks and from your own completed runs.

**Quality (expected loss)** — the hardest, least certain estimate, so it's handled carefully:
- Baseline: a Chinchilla-style scaling law `L(N, D) = E + A/N^α + B/D^β` (defaults from the literature, refittable on your own runs).
- Given the time budget, the planner computes the tokens D you can afford for each N, which yields a **compute-optimal size for this GPU and these hours**.
- Each component contributes a **prior** expressed as an effective-compute multiplier (e.g. "Muon ≈ 1.3–2× compute efficiency", reported in papers) along with a **confidence** and a **scale range** where it was validated.
- Priors outside their validated scale range get shrunk toward 1.0 (no effect).
- **Looped/shared-weight models** use the iso-depth law (B1) instead of a flat multiplier: `L = E + A·(N_once + r^φ·N_rec)^(−α) + B·D^(−β)`, with **φ ≈ 0.46** for plain looping, **≈ 0.65 with hyper-connections** (Hyperloop), and ≈ 0.38 if truncated BPTT is used. Training FLOPs still scale with the *unrolled* depth. This is how the planner prices "fewer params at the same compute" against "plain model, more params" honestly.
- **Muon prior**: a roughly constant token-efficiency multiplier across overtraining horizons at 51–253M (A1). Discount priors from papers tested at much larger batch sizes (A9).
- **Repeated data**: when the budget exceeds ~1 epoch, use the SoftQ data-constrained law (A8) instead of Chinchilla's additive form.

**Hyperparameter defaults** (written into the generated config; the user can override):
- AdamW, N < ~60M: recalibrated Step Law (D1): `lr* = 0.0985 · N^−0.508 · D^0.238`, `batch_tokens* = 3.6e-4 · D^0.931`. The original Step Law overestimates the LR by ~4× at this size. For N ≥ 60M, use the original Step Law.
- Weight decay grows with overtraining: `wd = wd_base · √(D / (20·N))` (A1).
- LayerRoPE widens the stable LR basin 3–10× (C1), so when it's enabled the planner can recommend fewer LR sweeps.
- **Tokenizer rule** (decided 2026-10-09): default to a **custom BPE trained on the dataset (~8k vocab)**. Offer the **SmolLM2 tokenizer (49,152)** only when its embedding table would be ≤ ~20% of total params (roughly ≥150M models), when the data is broad web text, code or multilingual, or when weak-teacher KD (F3) is enabled, since KD requires a shared tokenizer. Reason: at 30–80M a 49k vocab spends ~⅓ of all params on the embedding table and inflates logits memory on 8 GB, which works against the north star. The cost is ~15–25% more tokens per text on general English (less on simple text like TinyStories).
- **Learning loop**: every completed run logs `(config, tokens, final val loss)` to `runs.db`. The planner refits the scaling law and component multipliers from your own results over time. Your ablations end up correcting the paper claims for your hardware and data.
- The report always shows quality as a range plus a confidence label, never a false-precision single number.

Rough intuition, to be replaced by real estimates: at ~20 tokens/param, a ~100M-param model needs ~2B tokens ≈ 1.2e18 FLOPs, which is on the order of a day of compute on either of your GPUs. That means the planner will often recommend 20–80M-param models for single-session budgets, and the most useful thing it does is choose **N vs. D** sensibly. For reference, a full single-GPU 135M pipeline at 13B tokens reached ~87% of SmolLM-135M's zero-shot score (D3).

### 7.5 Ranking

1. Compute metrics per valid config: `est_loss`, `params`, `peak_mem`, `mem_headroom`, `train_hours`, `tokens_per_sec`, `inference_tok_per_sec`, `kv_cache_per_token`, `max_ctx`, `risk` (from maturity + soft-rule penalties).
2. Compute the **Pareto frontier** (configs not dominated on every metric).
3. Score with user weights: normalize each metric to [0, 1] across candidates, then compute a weighted sum. Hard filters are also available (`--max-params 50M`, `--max-hours 8`, `--min-ctx 2048`).
4. Output a table plus an explanation per row:

```
#  name                         params  peak mem  est. time  est. val loss    risk  score
1  rope-gqa-swiglu-muon-wsd      42M    6.1 GB     7.4 h    2.31 ±0.08      low   0.87
2  rope-mha-swiglu-adamw-cos     38M    6.4 GB     7.9 h    2.38 ±0.06      low   0.81
...
excluded: 1,912 configs (run with --explain to see reasons)
```

---

## 8. The generator: building a model into the project

`quark build <plan-row-id | config.yaml> --name tiny-story-v1`

Writes `models/tiny-story-v1/`:
- `config.yaml`: the full resolved config (model + train + data + hardware target), including a config hash.
- `model.py`: a **frozen, self-contained** copy of just the components this config uses, rendered into one file. Why freeze it? The library will keep evolving. A model you built in March should still load in June. Also, a single file is easy to embed in a Colab notebook.
- `plan_report.md`: estimates, warnings, why the config ranked where it did, paper links.
- `notebook_local.ipynb` and `notebook_colab.ipynb` (Section 10).
- `runs/`: created at train time.

`quark models list` and `quark models show <name>` let you browse what you've built.

---

## 9. Data pipeline

- **Sources**: Hugging Face dataset name, local `.txt`/`.jsonl`/`.parquet` files, or a folder of text.
- **Tokenizer**: pretrained (GPT-2 via `tiktoken`, or any HF tokenizer) **or** train a small BPE on your data (better for tiny models: a smaller vocab means less of the parameter budget goes to embeddings and the logits tensor stays small). The planner accounts for vocab size in params and memory.
- **Pre-tokenize once** into `uint16`/`uint32` memmap `.bin` shards plus `meta.json` (nanoGPT style): fast, low RAM, resumable.
- **Splits**: train/val (deterministic, hash-based), with a reported **token count**, which the planner uses to warn about multi-epoch training (a key overfitting signal).
- **Colab**: tokenized shards are cached on Google Drive so re-tokenizing isn't repeated each session.
- **Loader state** (shard, offset, RNG) is part of every checkpoint so a resume continues on new data instead of replaying old data.

---

## 10. Notebook generation

Built with `nbformat` + Jinja templates from **one template with target-specific blocks**, so local and Colab never drift apart. Both notebooks call the same `quark` trainer.

**How the code gets into Colab**: the Phase 1 notebooks `pip install git+<repo>@<QUARK_REF>` (default `main`). Code reaches Colab only once it's pushed, and a disconnected run reinstalls on resume, so pushed changes must keep `config_hash()` and checkpoint keys stable (pinned by tests). The planned self-contained mode (frozen `model.py` + pinned trainer embedded via `%%writefile`) arrives with the generator in Phase 4.

### Cell order

| # | Local | Colab |
|---|---|---|
| 1 | Header: model name, config summary, plan report excerpt | same |
| 2 | Environment check (torch, CUDA, GPU name, VRAM); **warns if GPU ≠ target** | same + `!nvidia-smi` |
| 3 | — | **Mount Google Drive**; set `RUN_DIR` on Drive |
| 4 | — | `pip install` pinned dependencies |
| 5 | Write/import model code + config | same (`%%writefile`) |
| 6 | Settings cell (the only cell you're expected to edit): dataset, run name, resume path, budget | same |
| 7 | Data: download → tokenize (or load cached) → show token counts and sample decoded text | same, cached to Drive |
| 8 | Build model, print param count + estimated vs **actual** memory after one dry-run step | same |
| 9 | **Resume detection**: if a checkpoint exists in `RUN_DIR`, load it automatically | same (critical on Colab) |
| 10 | Train with live dashboard (Section 11) | same + **time-based checkpointing** and a **session-time guard** (gracefully save and stop before the ~12 h limit) |
| 11 | Final plots + summary table | same |
| 12 | Sample generations from the best checkpoint | same |
| 13 | Export (Section 12) | same, to Drive |

The precision cell differs automatically: bf16 autocast locally, fp16 + GradScaler on T4.

---

## 11. Training: stats, plots, checkpoints, early stopping, resume

### 11.1 Stats logged

Every N steps:
- train loss (raw + EMA-smoothed), **val loss**, val perplexity
- learning rate (per param group, e.g. Muon vs AdamW)
- gradient norm (pre-clip), and how often clipping fires
- tokens/sec, step time, **MFU** (model FLOPs utilization)
- tokens seen, epoch fraction
- GPU memory allocated/peak
- ETA to the end of the budget
- loss-scale value (fp16 only), plus NaN/Inf counters

### 11.2 Live plots (the "loss going down" charts)

- Notebook: a live-updating figure (matplotlib + `IPython.display` refresh) with 4 panels:
  1. train loss (light raw + bold EMA) with val loss points overlaid, x-axis in **tokens** (log-x toggle)
  2. learning rate schedule (shows warmup, WSD stable/decay phases)
  3. grad norm
  4. throughput (tokens/sec) + memory
- Always written to `metrics.jsonl` + a PNG snapshot each eval, so plots survive disconnects and can be regenerated.
- Optional: TensorBoard or Weights & Biases (toggle). W&B is very convenient on Colab because the charts live outside the session.
- `quark compare run1 run2 ...` overlays loss curves from multiple runs, which is the standard ablation chart you see on X.

### 11.3 Checkpointing

- **best.pt**: written when val loss improves.
- **latest.pt**: written every N steps **and** every M minutes (time-based, for Colab).
- Keep the last K checkpoints; delete older ones.
- **Atomic writes** (write to `.tmp`, then rename) so a disconnect mid-save can't corrupt a checkpoint.
- Contents: model (fp32 weights), optimizer state, scheduler state, GradScaler state, step, tokens seen, data-loader position, RNG states (python/numpy/torch/cuda), config + config hash, metrics history pointer, library version.
- Checkpoints are **portable between local and Colab** (stored in fp32, precision is chosen at load time). That lets you start training on Colab and continue on your laptop, or the other way around.

### 11.4 Early stopping and overfitting detection

- **Patience** on val loss with `min_delta` (e.g. stop if there's no improvement of ≥0.005 over 5 evals).
- **Generalization-gap monitor**: stop or warn if `val_loss - train_loss` keeps growing over a window while train loss keeps dropping (the classic overfitting signature).
- **Multi-epoch awareness**: the planner already warns when the budget exceeds ~1–4 epochs of data. The trainer tightens patience once past epoch 1.
- **Divergence guard**: on a loss spike (> X σ above the EMA) or NaN, roll back to the last good checkpoint, skip the offending batch, optionally lower the LR, and give up after K retries.
- **WSD interaction**: with a WSD schedule, a plateau during the stable phase should trigger **early cooldown** (start the decay phase now) rather than a hard stop, because most of the gain comes during decay. This is one of the real advantages of WSD for budget-limited training.

### 11.5 Resume

- `quark train models/<name> --resume [latest|best|path]`, or automatic resume in notebooks.
- Verifies the config hash. If the config changed, it refuses unless you pass `--allow-config-change` (useful for deliberate changes like extending the schedule or switching GPU).
- Restores the schedule position exactly. With WSD you can also **extend** a finished run by resuming from a pre-decay checkpoint with a longer stable phase.

---

## 12. Inference and export

- `quark generate models/<name> --ckpt best --prompt "Once upon a time" --temperature 0.8 --top-p 0.9 --max-new 200`
- KV cache; YaRN/other context-extension scaling applied at inference time if configured.
- **Export for your eval project**: `quark export models/<name> --format hf|safetensors`, producing a standard folder (`config.json`, `model.safetensors`, tokenizer files, and a small `load_model()` / `generate()` / `loglikelihood(text)` interface). Agreeing on that interface now means evals can plug in later without touching the trainer.

---

## 13. CLI summary

```
quark gpu detect | list | bench
quark components list [<slot>] | show <slot:name>
quark plan --gpu <profile> --purpose <profile> [--hours H] [--weights ...] [--top K] [--verify] [--explain]
quark build <plan-row|config.yaml> --name <model>
quark data prepare --source <hf-name|path> --tokenizer <name|train:8k> --out data/<name>
quark notebook models/<name> --target local|colab|both
quark train models/<name> [--data data/<name>] [--resume latest]
quark generate models/<name> --prompt "..."
quark compare <spec.yaml>        # baseline + arms x paired seeds (Phase 2)
quark export models/<name> --format hf
quark models list | show <name>
```

---

## 14. Tech stack

- **PyTorch ≥ 2.5** (SDPA, FlexAttention, `torch.compile`), Python ≥ 3.10
- `pydantic` (config schema + validation), `pyyaml`
- `typer` + `rich` (CLI + pretty tables)
- `nbformat` + `jinja2` (notebook generation)
- `tiktoken` / `tokenizers` / `datasets` (data)
- `matplotlib` (live plots), optional `wandb` / `tensorboard`
- `safetensors` (export)
- optional: `bitsandbytes` (8-bit optimizers), `flash-attn` (Linux/WSL on Ampere only), `triton-windows`
- `pytest` for tests; `uv` for environment management

Optimizer implementations (Muon, etc.) are vendored as small, readable files with paper/repo attribution, rather than added as heavy dependencies.

---

## 15. Roadmap (build value end-to-end early)

**Current focus (2026-10-10):** Phase 2 experiments. `first-batch` (8 component arms on FineWeb-Edu at
`ablation-s`) is running on Colab; next, build the `chatbot-v1` mix, review its neurodivergence curation, read
`first-batch` and confirm winners at `ablation-m`, then run `experiments/mix-weights.yaml`. The ordered list
with owners is in the README's "Next steps".

**Phase 0: Scaffold — ✅ done 2026-10-09**
- `pyproject`, package layout, config schema, CLI skeleton, `gpu detect`, tests setup, git init. Project venv via `uv sync` (Python 3.12, torch 2.14.1+cu130); `quark version`, `quark gpu detect [--save]`, `quark gpu list`; built-in presets in `src/quark/hardware/presets/`, user-saved profiles in `./hardware/`; 14 tests passing including a real-GPU detection test.

**Phase 1: Baseline end-to-end (the most important milestone) — ✅ done 2026-10-10** (17.3M model on Colab T4: val loss 1.48 after 200M tokens, coherent stories; kill-and-resume from Drive verified)
- Baseline GPT (RoPE, RMSNorm, SwiGLU, MHA via SDPA, tied head), AdamW + cosine.
- Data prepare (HF + local text → memmap), TinyStories as the first dataset.
- Trainer with all of Section 11 (stats, live plots, best/latest checkpoints, early stopping, resume).
- Generate script.
- **Hand-written** template notebooks for local + Colab (not generated yet).
- ✅ Exit criteria: train a ~10–20M model on TinyStories on Colab, kill the Colab session mid-run, resume from Drive, and produce coherent stories. (**Changed 2026-10-10:** training is cloud-only for now, so the local 3070 Ti run is no longer required. Short local smoke tests are still fine.)

**Phase 2: Registry + first addons**
- ✅ Component base class, slots, `@register`, metadata, rules; baseline refactored into components (2026-10-10). Design and decisions: [component_registry.md](component_registry.md). GQA/MQA are the `n_kv_heads` shape dimension rather than separate components.
- ✅ QK-norm, ALiBi/NoPE, ReLU², WSD, Muon, z-loss (2026-10-10; see [component_registry.md §9](component_registry.md)).
- Still to add: DaRoPE, grad checkpointing, 8-bit AdamW.
- ✅ Per-component contract tests (shapes, causality, KV cache, overfit, param count) for every registered component.
- ✅ (2026-10-10) `quark compare`: baseline + arms × **paired seeds** (≥2–3), equal token budget, report with paired deltas, speed and an equal-time estimate. At this scale seed noise is the same size as many of the effects (E2 measured gains of 0.005–0.02 nats). Ablation ladder presets `ablation-s` (8.5M, 40M tokens) and `ablation-m` (38M, 200M tokens) at seq_len 2048 (§18.2). Runs seed by seed, skips finished runs, resumes unfinished ones and respects the Colab session limit; `notebooks/quark_compare_colab.ipynb` runs it on Colab. Spec format: `experiments/first-batch.yaml`.
- ✅ (2026-10-10) **Data mixtures** (added for §18): recipes (`mixes/*.yaml`) of streamed sources with filters (domain keyword density, dataset columns, excluded sites/patterns); resumable CPU collection that fills every source sharing a stream in one pass; a shared tokenizer; training weights in `data.weights` (compare arms, no re-preparation); per-source validation and epoch warnings; per-source deltas in compare reports; `notebooks/quark_data_colab.ipynb` (CPU runtime). First recipe: `mixes/chatbot-v1.yaml`.
- Experiments (in progress): `experiments/first-batch.yaml` (components on FineWeb-Edu), then
  `experiments/mix-weights.yaml` (domain weights on the chatbot-v1 mix); winners confirmed at `ablation-m`.

**Phase 2b: Parameter-efficiency stack (the project's north star)**
Each step is ablated against the previous best at the same token budget on the target data mix (§18), using the s/m ladder, cheapest and most certain first:
1. LayerRoPE (C1): ~40 lines, ~0 FLOPs, fewer params, −0.07 loss at 134M in the paper.
2. Value residual → SATFormer gate (C2).
3. Sink logit (+ projection value gate at ≥100M) (E2).
4. SAMuon-lite bulk gain (A2); scaled WD (A5) *or* Hyperball (A6), never both.
5. AF-Muon for the tied embedding table (A4); ORCA early orthogonal regularization (A3).
6. Front-loaded FFN width schedule (F1).
7. **Architecture bake-off at equal training FLOPs**: plain vs **Hyperloop** (B2, with orthogonal input injection B4, symmetry-broken stream init C4, unrolled-depth init G1) vs **Attractor** fixed-point block (E1). Report params, val loss, tokens/sec and peak memory. The expected outcome is matching or beating plain with ~50–60% of the params. Be ready for a null result too: CART (B7) found no win at small width and ~1B tokens.
8. Data-side extras: stack-task warm-up (E3), weak-teacher KD (F3), masked-input regularization when multi-epoch (A8).
- ✅ Exit criteria: a documented "Quark recommended stack" with measured multipliers in `runs.db`, plus planner priors updated from those runs.

**Phase 2c: Chat model (added 2026-10-10, §18)**
- Chat format and template, special tokens, loss on assistant turns only.
- SFT data: general instruction/chat sets plus domain conversations (education, neurodivergence-informed, early-childhood level), curated for accuracy and safety.
- `quark chat` for interactive testing; a small held-out chat set with per-domain loss.
- Final pretraining run of the recommended stack at the target size on a paid GPU, then SFT.

**Phase 3: Planner**
- Hardware + purpose profiles, `gpu bench`.
- Constraint engine with explanations.
- Param/memory/FLOPs/time estimators; calibrate memory against real measurements (target: within ~10%).
- Scaling-law quality estimate + component priors, including the φ-based looped estimate (B1), SoftQ for repeated data (A8), and Step-Law-small / WD rules (D1, A1).
- Pareto + weighted ranking, `--explain`, `--verify`.

**Phase 4: Generator + notebook generation**
- `quark build` (frozen `model.py` rendering), `plan_report.md`.
- Templated notebook generation for both targets.
- `models list/show`.

**Phase 5: Advanced components**
- YaRN context extension, sliding window, MLA, differential attention, FlexAttention-based variants, schedule-free AdamW, SOAP/Lion/ADANA.
- Looped extras: LPT shared-KV recursions (B3), LoopMTP (C6), dense→looped conversion (B6), LoopCD decoding at inference (B-skips).
- NAG residual (C3), WSqD schedule (A7), weight EMA/PACE (A10), chain distillation (F4).
- Gated-DeltaNet / head-wise hybrids (F2) once WSL2 + Triton kernels are confirmed working (and on T4).
- MoE and token-level MTP stay low priority (scale soft-rules).

**Phase 6: Learning loop + polish**
- `runs.db`; refit scaling law + component multipliers from your runs.
- Export for the eval project; W&B integration; docs.

---

## 16. Risks and how the plan handles them

| Risk | Mitigation |
|---|---|
| Quality predictions are guesses | Show ranges + confidence; learn from your own runs; `research-ablation` purpose makes measurement cheap |
| Memory estimates wrong → OOM | `--verify` dry-runs finalists on the real GPU; calibration factor learned from measurements |
| Combinatorial explosion | Slots (exclusive choices) + analytical estimation + hardware pre-filter |
| Paper techniques that don't transfer to small scale | `scale` soft rules, maturity labels, priors shrunk outside validated ranges |
| Colab disconnects | Time-based atomic checkpoints on Drive, auto-resume cell, session-time guard |
| fp16 instability on T4 | Auto GradScaler, stability addons favored, divergence rollback |
| Windows toolchain friction (Triton, flash-attn) | SDPA-first design; everything works without compile/flash-attn; WSL2 documented as optional upgrade |
| Library changes break old models | Frozen `model.py` + config hash + library version in every checkpoint |
| Scope creep (dozens of papers) | Every component needs tests + an ablation before it's "established"; experimental ones are flagged |
| Fresh, unreplicated papers (several Phase 2b items are days to weeks old) | Enter as `promising`; priors start near 1.0 and move only on our own paired-seed results; a negative ablation demotes the component |
| Gains don't stack (many target the same "curse of depth" / spectrum issue) | Add one at a time against the current best (Phase 2b order); known overlaps are flagged as soft rules (LayerRoPE vs NAG, scaled WD vs Hyperball, SAMuon vs BulkBoost) |

---

## 17. Open questions for you

1. ~~**"DaRope"**~~ **Resolved:** DaRoPE = "RoPE is Dead, Long Live RoPE" (arXiv 2609.34556). See [darope_integration_report.md](darope_integration_report.md).
2. ~~**First purpose / dataset**~~ **Superseded 2026-10-10 by §18** (domain-focused chatbot, ~125–350M, 2048 context, FineWeb-Edu + upweighted domains). **Decided 2026-10-09:** the target is a **small general-English text model**. Phase 1 uses **TinyStories** to prove the pipeline end to end (fast, coherent output at 10–30M). Phase 2b ablations and the main models use **FineWeb-Edu** (a sample sized to the planner's token budget), matching most cited papers so results are comparable. Add a `fineweb-edu` purpose profile next to `tinystories`.
3. ~~**Tokenizer preference**~~ **Resolved:** custom ~8k BPE by default, SmolLM2 tokenizer for larger / broad-data / KD models. The planner picks per model; see "Tokenizer rule" in §7.4.
4. ~~**Experiment tracking**~~ **Decided (default):** local `metrics.jsonl` + PNG plots, mirrored to Drive on Colab. W&B is an optional toggle added in Phase 6. No account or API key is needed to start.
5. ~~**Local environment**~~ **Decided (default):** native Windows. Every recommended component is pure PyTorch + SDPA, and WSL2 currently has no Linux distro installed (only `docker-desktop`). Revisit only for Gated-DeltaNet hybrids (F2). **Setup blocker found 2026-10-09:** the installed PyTorch is `2.13.0+cpu`, which can't see the GPU (RTX 3070 Ti Laptop, 8 GB, driver 595.79). Phase 0 must create a project venv with a CUDA build of PyTorch and verify `torch.cuda.is_available()`.
6. ~~**Repo hosting**~~ **Decided (default):** `git init` locally in Phase 0, with an optional *private* GitHub repo (`gh` is installed). Notebooks stay **self-contained** (code embedded via `%%writefile`), so Colab never needs GitHub credentials. **As built (2026-10-10):** the repo is on GitHub (`cpalmquist19/quark`) and the Phase 1 notebooks install from it; the `%%writefile` mode comes with Phase 4 (see §10).
7. ~~**Eval interface**~~ **Finding:** the eval project (`C:\Code\ai-agent\evals`) scores *agentic tool use* of chat models through LM Studio / OpenAI-compatible APIs. A from-scratch 10–150M base model can't do tool calling, and LM Studio's llama.cpp can't load our custom architectures (Hyperloop, LayerRoPE, …), so that harness isn't a fit for Quark models. **Decision:** Quark gets its own lightweight evals: validation loss/perplexity, plus a few small-model benchmarks (e.g. BLiMP, HellaSwag, ARC-Easy, LAMBADA) via a `loglikelihood(text)` API. An OpenAI-compatible `quark serve` endpoint stays a "later" item, if a Quark model is ever instruction-tuned.

---

## 18. Target model: a domain-focused chatbot (decided 2026-10-10)

The models Phase 2b selects components for are aimed at this target. TinyStories (Phase 1) proved the pipeline; it is no longer the yardstick.

| | Decision |
|---|---|
| Use | A **chatbot**: pretraining, then chat fine-tuning (SFT, Phase 2c) |
| Size | **~125–350M params**, final training on a **paid GPU** (Colab Pro / rented L4 or A100). Ablations still run on the free T4 |
| Context | **2048 tokens** (multi-turn chat); YaRN (Phase 5) can extend it later |
| Data | **FineWeb-Edu as the general-English base**, with three domains upweighted beyond their natural share: **neurodivergence** (autism, ADHD, dyslexia, executive function, sensory processing, special education), **education** (teaching, learning, pedagogy) and **early childhood** |
| Domain decisions (2026-10-10) | **Early childhood = knowledge *about* early childhood** (development, teaching young children, parenting) from filtered web text; no paid synthetic generation. **Neurodivergence = neurodiversity-affirming**, and **children will talk to the model**: the neurodivergence slice is curated (deficit/cure framing left out), and child safety shapes the data and SFT (Phase 2c) |
| Tokenizer | Open. The §7.4 rule points to SmolLM2's 49k tokenizer at this size; a custom ~32k BPE trained on the mix may fit child-level and domain vocabulary better. Decide with a test at the ~40M rung |

### 18.1 Data sources (first survey, 2026-10-10)

Measured supply and quality: [domain_data_survey.md](domain_data_survey.md). In FineWeb-Edu, documents *about* each domain are ~0.74% (neurodivergence), ~4.5% (education) and ~0.86% (early childhood) of tokens, mostly adult-facing. Cosmopedia's young-children text is child-voiced retellings of older topics, not pre-K content.

| Domain | Source | Status |
|---|---|---|
| General English | `HuggingFaceFW/fineweb-edu` | Planned base |
| Young-children knowledge and language | `HuggingFaceTB/cosmopedia` (Apache-2.0, ~25B tokens of synthetic textbooks, stories, WikiHow-style text; `audience` and `format` columns allow filtering to the youngest audiences); `roneneldan/TinyStories` | Confirmed audience values seen so far: `grade_school_students`, `college_students`. Check the full value list before relying on a "young children" filter |
| Education | FineWeb-Edu filtered by topic: the WebOrganizer topic classifier (24 topics, incl. "Education & Jobs" and "Health") and/or keyword filters | Standard domain-upweighting approach |
| Neurodivergence | **No ready-made knowledge corpus found** (only small labeled social-media or diagnostic sets). Keyword/topic-filtered web text first; carefully reviewed synthetic text possibly later | Needs curation |

Cautions:
- **Repetition.** Upweighting a small domain means repeating it. Repeated text loses value after a few epochs (A8), so domain weights trade emphasis against repetition.
- **Content quality.** Web text on neurodivergence includes outdated, stigmatizing or wrong material. That slice needs curation. If the chatbot will talk with children or neurodivergent users, the SFT stage needs care with accuracy and safety, and the model must not present itself as a source of diagnosis or medical advice.

### 18.2 How components are tested so results transfer to the target

1. **Target data.** Ablations train on a scaled-down sample of the target mix, not TinyStories.
2. **Target context.** Ablations use seq_len 2048. Position-encoding differences mostly show at longer context.
3. **Per-domain validation.** Val loss is reported separately for general / education / neurodivergence / early-childhood, so domain emphasis is measured and trade-offs show up.
4. **Scale ladder.** Every component is tested at ~10M (`ablation-s`) and finalists at ~40M (`ablation-m`) on the T4. Trust a gain that holds or grows from s to m; distrust one that shrinks. The final stack gets one confirmation run at the target size on the paid GPU.
5. **Same regime at every rung.** Each rung keeps the target's seq_len, tokenizer family and deep-and-thin shape, with the LR from the D1 size rule. Optimizer arms (e.g. Muon) get a small LR sweep on the s rung.
6. **Per hour as well as per token.** Reports show the loss difference at equal tokens and an estimate at equal wall-clock, because the real budget is GPU hours.
7. **Know what small runs can't show.** Stability add-ons (QK-norm, z-loss) are judged by a high-LR stress arm or adopted on evidence. Schedules (WSD vs cosine) are judged partly on practicality: a multi-session target run benefits from WSD's extendable stable phase.
8. **The data mix is an experiment too.** Domain weights are tuned at the s rung. Small-proxy mixture rankings largely transfer to larger models (RegMix, arXiv 2407.01492), which makes this one of the cheapest, highest-value experiments.
