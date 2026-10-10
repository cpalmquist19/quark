# Component registry: design

Status: implemented 2026-10-10 (Phase 2, step 1: registry + baseline refactor). Plan references: §5 (registry), §7.3 (validation), §7.4 (estimators), §8 (frozen `model.py`), roadmap Phases 2–5.

The registry is the catalog of techniques. Everything else asks it questions:

| Consumer | Question it asks | When |
|---|---|---|
| Model builder | "build the module for `attention: mha`" | Phase 2 (now) |
| Trainer | "build the optimizer / schedule", "is this config valid on this GPU?" | Phase 2 (now) |
| `quark compare` (ablations) | "what are the alternatives in slot X?" | Phase 2 |
| Planner | "which combinations are valid, what do they cost, what's the prior?" | Phase 3 |
| Generator | "which source files make up this config?" (frozen `model.py`) | Phase 4 |
| Learning loop | "update this component's prior from measured runs" | Phase 6 |

## 1. What the project needs from it

Requirements, taken from the roadmap and the Phase 2b list:

1. **Exclusive choices (slots)** and **independent add-ons (toggles)**, so the search space is a product of slot sizes rather than 2^features (§5).
2. **Interfaces that survive the hard cases**, not only the easy swaps:
   - position encodings act in different places: RoPE rotates q/k, learned-absolute adds to embeddings, ALiBi biases attention scores, NoPE does nothing;
   - the `block` slot must own the **whole layer stack**, because Hyperloop (`[begin | mid × r | end]` + hyper-connections) and Attractor (fixed-point solve) aren't "one layer repeated n times";
   - looped stacks reuse weights, so the KV cache length and residual init depth come from the stack (unrolled depth, G1), not from `n_layers`;
   - some techniques live inside another component (QK-norm inside attention, z-loss in the loss, LayerRoPE gains shared across depth).
3. **Rules with human-readable reasons**: requires, conflicts, hardware (hard); scale, purpose, interaction (soft, with a penalty); auto-fix notices (§5.2).
4. **Per-component parameters** with defaults and validation (`rope.base`, `wsd.decay_frac`, `muon.ns_steps`, Hyperloop layout…).
5. **Metadata for the planner**: paper, maturity, the quality prior with its validated scale range (§7.4).
6. **Exact cost numbers** for the planner: parameter count now; memory/FLOPs hooks in Phase 3.
7. **Uniform tests**: every component gets shape, causality, KV-cache and overfit-one-batch tests for free.
8. **Backward compatibility**: existing configs keep their `config_hash()`, and existing checkpoints keep their state-dict keys. This matters now, because Colab runs resume from GitHub `main`.

## 2. Slots and toggles

| Slot | Stage | Interface (base class) | Baseline | Owns |
|---|---|---|---|---|
| `position` | model | `PositionEncoding` | `rope` | `embed(x, start)`, `rotate(q, k, start)` hooks |
| `attention` | model | `Attention` | `mha` | q/k/v projections, SDPA call, KV cache use |
| `norm` | model | `Norm` | `rmsnorm` | a norm layer factory (`ctx.make("norm", dim)`) |
| `mlp` | model | `MLP` | `swiglu` | the feed-forward |
| `block` | model | `Stack` | `standard` | **the whole layer stack**: layer layout, residual wiring, cache size, unrolled depth |
| `head` | model | `Head` | `tied` | hidden → logits |
| `optimizer` | train | `OptimizerComponent` | `adamw` | param grouping + `torch.optim.Optimizer` |
| `schedule` | train | `Schedule` | `cosine` | LR at each step |

**Toggles** are registered in the pseudo-slot `toggle` and declare `implemented_by = "<slot>"`. A toggle doesn't carry code paths of its own. The component in that slot implements it and lists it in `supports`. If a toggle is enabled but the selected component doesn't support it (`qk_norm` with an attention type that ignores it), that's a hard error. This makes "the config says it's on but nothing happens" impossible.

Decisions this table encodes:

- **GQA / MQA are not separate components.** They're `n_kv_heads`, which `mha` already handles. The planner varies them as a shape dimension (§7.2), the same way it varies `n_heads`.
- **Norm placement (pre / post / sandwich)** is a parameter of the stack (`block.params.norm_placement`), not a norm type. `norm` only answers "which norm layer".
- **Grad checkpointing** is a toggle implemented by `block`, because the stack decides what to recompute.
- **z-loss / logit soft-capping** are toggles implemented by `head`, because the head produces the logits.

## 3. A component

One file per technique, and one class that is both the metadata and the implementation:

```python
@register("position", "rope")
class RoPE(PositionEncoding):
    summary = "Rotary position embedding (half-split layout, as in Llama)."
    paper = "arXiv:2104.09864"
    maturity = "established"          # experimental | promising | established
    requires = ()                     # refs: "slot:name", "slot:a|b", "toggle:name"
    conflicts = ()
    hw = HwReq()                      # min compute capability, needs bf16 / triton / flash-attn pkg
    rules = ()                        # soft / info rules: Rule(severity, when, reason, penalty)
    supports = frozenset()            # toggles this component implements
    prior = None                      # Prior(multiplier, confidence, validated_params_m, source)

    class Params(ComponentParams):    # pydantic, extra="forbid"; defaults are the paper's
        base: float = 10000.0

    def __init__(self, ctx: BuildContext): ...
```

- **Model-slot components are `nn.Module`s.** Their constructor takes a `BuildContext`: the model config, the resolved choice for every slot, validated params, enabled toggles and the hardware profile. Sub-components are built through it (`ctx.make("norm", dim)`), so the stack never hard-codes RMSNorm or SwiGLU.
- **Train-slot components** build things: `AdamW.build(model, train_cfg, params, device)` returns an optimizer, and `Cosine(train_cfg, params, total_steps).lr_at(step)` returns the LR.
- **Self-contained files.** A component file imports only base interfaces and torch. This is what lets the Phase 4 generator concatenate the files a config uses into a frozen `model.py`.
- **Registration checks the interface.** A `position` component must subclass `PositionEncoding`, names are unique per slot, and refs must name real slots.

## 4. Config

The existing fields stay exactly as they are:

```yaml
model:
  slots: {position: rope, attention: mha, norm: rmsnorm, mlp: swiglu, block: standard, head: tied}
  toggles: {}               # {qk_norm: true} or {z_loss: {coef: 1.0e-4}}
  params: {}                # NEW, per slot: {position: {base: 500000}}
train:
  optimizer: adamw          # now any registered name (was Literal["adamw"])
  schedule: cosine          # now any registered name (was Literal["cosine"])
  params: {}                # NEW, per slot: {optimizer: {...}, schedule: {...}}
```

`lr`, `warmup_steps`, `weight_decay`, `betas` and `min_lr_ratio` stay as top-level train fields. They're shared by almost every optimizer and schedule, and moving them would change every existing config. Component-specific settings go in `params`.

**Hash compatibility:** `config_hash()` leaves out the new `params` fields while they're empty. Every existing config and checkpoint keeps its hash, so in-progress runs still resume. The test suite pins the pre-registry hashes.

## 5. Validation (`check`)

`check(cfg, hardware=None, purpose=None, stages=("model", "train")) -> list[Issue]`.

Each `Issue` has a `severity` (`error`, `warning` or `info`), a `source` (`"attention:mha"`), a `message` and a `penalty` (soft rules only). The order of evaluation:

1. **Names**: unknown slots, components or toggles become errors, with a "did you mean" suggestion.
2. **Params**: validated against each component's `Params` model. A missing value takes the default; an unknown or invalid key is an error.
3. **Toggles**: each enabled toggle must be listed in `supports` by the component in its `implemented_by` slot.
4. **Declarative rules**: `requires` / `conflicts` refs, and the `hw` requirements checked against the hardware profile (skipped when the hardware is unknown).
5. **Callable rules**: `Rule(severity, when, reason, penalty)`, where `when(ctx)` sees the config, hardware, purpose and a lazily computed exact parameter count. This covers the plan's scale, purpose and interaction rules.

Rules stop after step 1 or 2 if those produced errors, because later rules can't evaluate an unresolved config.

**Auto-fixes don't rewrite the config.** Every auto-fix listed in plan §5.2 is really a build-time behaviour: Muon splits params, NS runs in fp32 on T4, the sink logit forces a float mask, looped init uses unrolled depth. The component adapts at build time using `ctx.hardware`, and an `info` rule reports what it did. Rewriting the config would make `config_hash()` depend on the GPU, which breaks moving checkpoints between machines.

Callers:

- `build_model(cfg)` runs the model-stage check and raises on errors.
- `Trainer` runs the full check with the real hardware: errors raise before anything is built, and warnings and info are logged and written to `metrics.jsonl` as events.
- `quark components list|show` prints the catalog.
- The Phase 3 planner calls `check` on every candidate and turns penalties into ranking and `--explain` output.

## 6. Cost hooks

- **Parameter count: exact and free.** The model is built on the `meta` device (no memory, no compute), so no component has to maintain a hand-written formula. Rules use this, and so will the planner.
- **FLOPs per token:** `6·N_non_emb + 12·unrolled_depth·d_model·seq_len`. The stack reports `unrolled_depth`, so the attention term already counts looped layers. The parameter term doesn't yet, because shared weights are counted once. The first looped stack must override this with its unrolled cost.
- **Memory / extra FLOPs** (`extra_memory`, `extra_flops`) arrive with the planner (Phase 3). The base class gets the hooks then, defaulting to zero, so existing components need no changes.

## 7. Tests every component gets

`tests/test_components.py` is parametrized over the registry. For every registered model component, it swaps that choice into the baseline config (skipping combinations `check` rejects) and asserts:

- output shapes, and loss ≈ ln(vocab) at init;
- causality (changing future tokens doesn't change earlier outputs);
- KV-cache decode equals the full forward pass;
- it can overfit one batch;
- the meta-device parameter count equals the real one.

Optimizers must reduce the loss on a tiny problem. Schedules must warm up, stay within `[0, lr]` and be finite. Rule refs must parse, and every `supports` entry must be a registered toggle.

A new component therefore needs only its own specific tests. The generic contract comes from this harness.

## 8. Scope of this step

**In:**
- registry, base interfaces, `BuildContext`, `check` with `Issue`/`Rule`/`HwReq`/`Prior`;
- baseline split into components (`rope`, `mha`, `rmsnorm`, `swiglu`, `standard`, `tied`, `adamw`, `cosine`);
- `params` fields with hash compatibility;
- Trainer, inference and CLI wired through the registry;
- `quark components list|show`;
- the contract test harness.

**Out (next steps, to discuss):** the actual add-ons (GQA as a planner dimension, QK-norm, ALiBi/NoPE, DaRoPE, ReLU², WSD, Muon, z-loss, grad checkpointing, 8-bit AdamW), `quark compare`, and the attention-bias hook for ALiBi. That hook gets added to `PositionEncoding` together with its first user, so it's tested from day one.

## 9. First add-ons (2026-10-10)

| Component | Notes |
|---|---|
| `toggle:qk_norm` | Implemented by `attention:mha`. Uses the model's norm type over `head_dim`, before the position encoding. |
| `toggle:z_loss` | Implemented by `head:tied`. `coef` defaults to 1e-4 (PaLM). Added in training mode only, so validation loss stays plain cross-entropy and ablations stay comparable. |
| `position:nope` | No position signal. |
| `position:alibi` | Uses the new `PositionEncoding.attn_bias` hook; `requires attention:mha` (the attention type must apply the bias). The float mask means SDPA can't use `is_causal`. |
| `mlp:relu2` | Hidden width is 1.5 × `ffn_mult` × `d_model`, so the parameter count matches SwiGLU at the same `ffn_mult`. |
| `schedule:wsd` | `decay_frac` (0.2), `decay` (`linear` or `1-sqrt`). |
| `optimizer:muon` | Muon for hidden matrices, AdamW for the embedding table and vectors, all in one optimizer and one state dict. Updates are scaled by 0.2·√max(rows, cols), so `lr` and `weight_decay` carry over from AdamW (Moonlight). Newton–Schulz runs in bf16 where native, otherwise in fp32 (T4). Fused weights marked with `mark_fused` (q/k/v, gate/up) are orthogonalized part by part. |

New hooks: `PositionEncoding.attn_bias`, `Head.loss` and `mark_fused`.

Measured on the 3070 Ti at `tinystories-small` size (fp16, 32k tokens/step): Muon takes 335 ms/step against AdamW's 294 ms (+14%), with the same peak memory. Expect more on the T4, where Newton–Schulz runs in fp32. Batching same-shape matrices through Newton–Schulz would cut this if it matters.

**Invariants kept:**
- state-dict keys are unchanged (`blocks.0.attn.qkv.weight` …), so existing checkpoints load;
- the same seed gives bit-identical init to the pre-registry model, so earlier baselines stay comparable;
- config hashes are unchanged.
