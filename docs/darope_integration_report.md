# DaRoPE (Data-aware RoPE): research and integration report

**Paper:** "RoPE is Dead, Long Live RoPE: Towards Scalable Data-aware Positional Encodings": Lévy, Videau, Yehya, King, d'Ascoli, Moreau. [arXiv:2609.34556](https://arxiv.org/abs/2609.34556) (v1 28 Sep 2026, v2 3 Oct 2026).
**Verdict:** **Adopt** as a `position` slot component (`position:darope`), with maturity **`promising`**. It's cheap, it's SDPA-compatible, it works on both target GPUs, and it's simple to implement (~50 lines). Expect little to no in-window gain at Quark's scale. The real benefits are **length extrapolation without YaRN**, **less recency bias**, and better **in-context retrieval**.

---

## 1. Why this paper out of the list

| Candidate (from `ml_papers.txt`) | Fit for Quark | Why not chosen / why chosen |
|---|---|---|
| **DaRoPE** | ✅ Excellent | Named in the original project idea ("DaRope") and still Open Question #1 in the plan. It drops into the existing `position` slot, needs no kernels, and the paper tests down to **124M params** with matched baselines. The authors include Meta FAIR-style researchers working on the Meta Lingua codebase. |
| Looped / recurrent depth | Good | Already planned for Phase 5 (`block/looped.py`). There's no single paper to read, only an X thread plus several follow-ups. |
| Hybrid NoPE (Dawes et al.) | Good | Interesting, but it's an analysis/design pattern rather than a module, and it overlaps with DaRoPE's motivation. |
| Foil / Looped MoE, MoSE, Sigma-MoE-Tiny | Weak | MoE on 8 GB / T4 at 20–80M params is the plan's own "soft-penalty" case. Sigma-MoE-Tiny is ~20B total params. |
| Self-Pruning Transformer | Weak | No arXiv version or code. KV-cache compression matters little at our context lengths. |
| Cross-head sparse attention, FFN sparsity | Weak | Inference-time or kernel-dependent (custom CUDA). Bad fit for Windows/T4 and tiny models. |
| ZipMoE, MoE scaling laws | N/A | Serving and theory, not trainable modules. |

This also resolves **Open Question #1** in [project_plan_opus_5.5_medium.md](project_plan_opus_5.5_medium.md#L525): "DaRope" is this paper. It is *not* DAPE or DroPE.

---

## 2. What the method does

### 2.1 The problem it targets
RoPE rotates each 2-D pair ("band") *j* of a query/key by `m·θ_j`, with `θ_j = base^(-2j/d)`. Low-frequency (**slow**) bands have a wavelength `2π/θ_j` **longer than the training context L**. During training they only ever see a fraction of a full rotation. Past L they produce angles the model has never seen, which is the root cause of RoPE's extrapolation failure (and the thing YaRN/NTK patch at inference). Slow bands also carry most of RoPE's built-in distance decay, i.e. its recency bias.

### 2.2 The fix
Split bands by the wavelength criterion:

```
band j is slow  ⇔  θ_j < 2π / L          K = #fast bands
```

- **Fast bands (j < K):** unchanged RoPE, rotated by the token index `m`.
- **Slow bands (j ≥ K):** rotated by a learned, **bounded, per-head content coordinate** in place of `m`:

```
β_m      = logit((m + 0.5) / L)                        # fixed positional prior (clipped)
c_h(x_m) = L · sigmoid( w_hᵀ x_m + α_h · β_m )          # ∈ [0, L], per head h
angle_j  = m · θ_j            for j < K
angle_j  = c_h(x_m) · θ_j     for j ≥ K
```

Because the coordinate is applied identically to q and k, the slow-band score still depends only on a *difference*, `(c_h(x_n) − c_h(x_m))·θ_j`, but that difference is now a learned, context-dependent "distance". The sigmoid keeps `c_h ∈ [0, L]`, so **slow-band angles stay inside their trained range at any sequence length**. That gives extrapolation with no inference-time rescaling.

Special cases (useful as tests and as sibling components):
- `w_h = 0, α_h = 1` → `c_h = m + 0.5` → **exactly RoPE** (inside the window).
- Drop the slow-band rotation entirely → **HoPE** ("high-frequency RoPE"), a zero-parameter sibling.

### 2.3 Cost
- **Params:** `n_heads · (d_model + 1)` per layer. For a typical Quark model (d=384, 6 heads, 6 layers) that's ~14k params, about 0.05%.
- **Compute:** one tiny `d_model → n_heads` projection plus a sigmoid per token. The paper measured **~3% slower prefill** than RoPE (59.8 vs 61.7 ktok/s, 1B model, A100).
- **Kernels:** rotation happens before attention, so it stays **compatible with SDPA / FlashAttention**. No FlexAttention, Triton, or custom CUDA needed.

---

## 3. Evidence (and how much to trust it)

| Setting | Result |
|---|---|
| Dense LM at 124M / 350M / 1B, in-window perplexity | **Tie.** Every method with a positional signal is within a few tenths of a ppl point. Only NoPE regresses. |
| 1B extrapolation (positions 8–16k, trained at 4k) | DaRoPE ≈ HoPE ≈ 19 ppl; PoPE 42.9; **RoPE-500k > 450**. YaRN is in-window 19.1. |
| Formulation ablation (124M, 8–16k NLL) | DaRoPE 3.957 vs. no-β 4.066, fixed-α 4.449, per-band α 4.815. The design choices matter. |
| Synthetic retrieval | KV recall @256 bindings: **30.5%** vs. ≤7.25% for the others. Return-from-digression @2k: 88.1% vs. RoPE 77.6%. Shuffling the learned coordinates erases these gains, so they're real. |
| Non-text (music, genomics, EEG) | Small but consistent wins over HoPE. |
| 50B MoE (2.6B active, 617B tokens) | ≈ HoPE on short benchmarks (54.7 vs 54.3). Better on LongBench (27.7 vs 26.1) and RepoBench. |

**Caveats to carry into the registry metadata:**
- **Single seed** per method; the long-context controlled evals (RULER, BABILong) are a **tie**.
- **No official code release** that I could find. We implement from the paper's pseudocode.
- **Two weeks old**, not peer-reviewed yet.
- Internal inconsistencies: param overhead is given as both "<0.1%" and "≤0.3%". The init `α = 0.5` doesn't match the RoPE-equivalence point `α = 1`. That's fine, since it's just an init, but worth noting.
- **Under-specified:** whether `x_m` is pre- or post-norm, whether q/k share `w_h`, and how GQA is handled. Our choices are in §4.2.

**Bottom line for Quark's scale (10–100M params, 512–2048 context):** don't expect better TinyStories loss. Do expect better behaviour **beyond the training length** and on **retrieval / non-text** data. That makes DaRoPE a natural *alternative to YaRN* in the `long-context` purpose.

---

## 4. Integration into Quark

### 4.1 Band split at our sizes
`K` depends on `head_dim`, `base` and the training context `L` (computed, matches the paper's own numbers):

| head_dim | base | L=256 | L=512 | L=1024 | L=2048 | L=4096 |
|---|---|---|---|---|---|---|
| 64 | 10k | 19/32 slow | 16/32 | 14/32 | 11/32 | 9/32 |
| 64 | 500k | 22/32 | 21/32 | 19/32 | 17/32 | 16/32 |
| 128 | 10k | 38/64 | 33/64 | 28/64 | 23/64 | 18/64 |

At small L, half or more of the bands are slow, so DaRoPE changes the model *more* at our scale than at the paper's 4k setting. That's another reason to treat it as `promising`, not `established`.

### 4.2 Design decisions (where the paper is silent)
1. **Input to the coordinate:** the **normalized block input** (the same tensor fed to the QKV projection). This is the natural pre-norm choice and keeps the scale stable.
2. **q/k share `c_h`:** required for the relative-offset property, so we don't treat it as optional.
3. **GQA/MQA:** one coordinate per **KV head**, broadcast to the query heads in its group. Each query head must use the same coordinate as the key it attends to, so this is the only consistent choice.
4. **Separate `nn.Linear`, not fused into QKV.** This matters for the **Muon** interaction: Muon would orthogonalize a fused QKV+coord matrix and wreck the 1-row-per-head coordinate projection. `coord.weight` and `alpha` go to the **AdamW group, no weight decay**.
5. **fp32 angles and coordinate projection.** `L·sigmoid(·)` amplifies fp16 error straight into the slow-band angle. This matters on the **T4 (fp16)** target.
6. **KV cache:** the coordinate depends only on token *m*'s own hidden state, so it's causal. Cache **post-rotation** keys exactly as with RoPE; nothing else changes in `infer/`.

### 4.3 Reference implementation (verified)
Drop-in for `src/quark/components/position/darope.py`. Tested on CPU (torch 2.13), see §4.5.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_freqs(head_dim, base):
    return base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)  # (hd/2,)


def apply_rotation(x, angle):
    # x: (B, H, T, hd), angle: (B|1, H|1, T, hd/2). Interleaved-pair convention.
    x1, x2 = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = angle.cos(), angle.sin()
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return out.type_as(x)


class DaRoPE(nn.Module):
    """Data-aware RoPE (arXiv:2609.34556). One instance per attention layer.

    Fast bands (wavelength <= train_ctx) rotate by token index m, like RoPE.
    Slow bands rotate by a bounded per-head content coordinate
        c_h(x_m) = L * sigmoid(w_h . x_m + alpha_h * logit((m + 0.5) / L))
    which is shared by q and k of head h (needed for the relative-offset property).
    """

    def __init__(self, d_model, n_kv_heads, head_dim, train_ctx, base=10_000.0,
                 w_std=0.1, alpha_init=0.5, eps=1e-6):
        super().__init__()
        theta = rope_freqs(head_dim, base)
        self.register_buffer("theta", theta, persistent=False)
        self.K = int((theta >= 2 * math.pi / train_ctx).sum())
        self.L, self.eps = float(train_ctx), eps
        self.coord = nn.Linear(d_model, n_kv_heads, bias=False)  # w_h, one per KV head
        nn.init.normal_(self.coord.weight, std=w_std)
        self.alpha = nn.Parameter(torch.full((n_kv_heads,), alpha_init))

    def angles(self, x, pos):
        # x: (B, T, d_model) normalized block input; pos: (T,) absolute positions
        L = self.L
        m_bar = (pos.float() + 0.5).clamp(self.eps * L, (1 - self.eps) * L)
        beta = torch.logit(m_bar / L)                                   # (T,)
        # fp32 on purpose: L * sigmoid(.) amplifies fp16 error into the slow-band angles
        z = F.linear(x.float(), self.coord.weight.float()) + self.alpha * beta[:, None]  # (B, T, Hkv)
        c = L * torch.sigmoid(z)                                        # (B, T, Hkv) in [0, L]
        fast = pos.float()[:, None] * self.theta[: self.K]              # (T, K)
        slow = c[..., None] * self.theta[self.K:]                       # (B, T, Hkv, S)
        fast = fast[None, :, None, :].expand(*slow.shape[:-1], self.K)
        return torch.cat((fast, slow), -1).transpose(1, 2)              # (B, Hkv, T, hd/2)

    def forward(self, q, k, x, pos):
        # q: (B, Hq, T, hd), k: (B, Hkv, T, hd). Angles computed in fp32 (matters on fp16/T4).
        a = self.angles(x, pos)
        g = q.shape[1] // k.shape[1]
        return apply_rotation(q, a.repeat_interleave(g, dim=1)), apply_rotation(k, a)
```

**One API implication for the baseline:** plain RoPE only needs `(q, k, pos)`, but DaRoPE also needs the hidden state `x`. When the Phase 2 registry defines the `position` slot interface, make it `apply(q, k, x, pos)` so RoPE, HoPE, DaRoPE (and later DAPE-like methods) all fit without changing the attention module. Doing this now is free; retrofitting it later touches every attention variant.

### 4.4 Registry entry

```python
@register(slot="position", name="darope")
class DaRoPEComponent(Component):
    paper = "arXiv:2609.34556"
    summary = "RoPE on fast bands; bounded, learned content coordinate on slow bands."
    maturity = "promising"
    requires = []                                   # self-contained (owns its RoPE freqs)
    conflicts = ["position:alibi", "position:nope", "position:yarn"]  # YaRN rescales the slow bands DaRoPE replaces
    hw = HwReq(min_sm=None, needs_bf16=False, needs_triton=False)     # works on 3070 Ti and T4
    phase = "pretrain"                              # must be trained in; not a post-hoc extension
    params = {"base": float, "w_std": float, "alpha_init": float}
    helps_when = [
        Rule("target_ctx > train_ctx", "extrapolates without YaRN"),
        Rule("purpose in {long-context, code-completion}", "retrieval / reduced recency bias"),
        Rule("data_modality != 'text'", "largest wins on music/genomics/EEG in the paper"),
    ]
    hurts_when = [
        Rule("n_slow_bands(cfg) == 0", "no slow bands: identical to RoPE, pure overhead"),
    ]

    def param_count(self, cfg):  return cfg.n_layers * cfg.n_kv_heads * (cfg.d_model + 1)
    def extra_flops(self, cfg, tokens):  return 2 * tokens * cfg.n_layers * cfg.d_model * cfg.n_kv_heads
    def extra_memory(self, cfg, batch):  return 0  # (B,T,Hkv) fp32 per layer, negligible
    def quality_prior(self, cfg):
        # In-window: ~no effect. Report a gain only when the purpose evaluates beyond train_ctx.
        return Prior(compute_mult=1.0, conf="medium", scale_range=(124e6, 50e9))
```

**Planner rules to add:**
- *Interaction (auto-fix):* optimizer = Muon → route `*.coord.weight` and `*.alpha` to the AdamW group with `weight_decay=0`.
- *Interaction (soft):* fp16 target → force fp32 angle computation (already in the reference code; flag it if someone swaps the implementation).
- *Scale (soft):* model < 124M → shrink the prior toward 1.0 and add a warning: "outside validated scale".
- *Purpose:* in `long-context`, rank `darope` vs. `rope+yarn` on an *eval-at-k×train_ctx* metric, not in-window loss.

Add **`position:hope`** alongside it: the same split logic with slow bands left unrotated and zero params. In the paper it matched DaRoPE on dense LM extrapolation, so it's the right cheap baseline for every DaRoPE ablation.

### 4.5 Tests (already passing on the sketch)

| Test | Result |
|---|---|
| Shapes with GQA (4 q-heads / 2 kv-heads); gradients reach `w_h`, `α_h` | ✅ |
| Causality: perturbing tokens ≥ 40 leaves outputs < 40 unchanged (with SDPA `is_causal`) | ✅ max diff 0.0 |
| `w=0, α=1` reproduces plain-RoPE attention scores | ✅ max diff 7.6e-6 |
| Slow-band angles bounded at **8× train length** | ✅ max = bound |
| KV-cache: token-by-token key rotation == full-sequence rotation | ✅ max diff 1.2e-6 |
| fp16 inputs → finite fp16 outputs | ✅ |

These map directly onto the plan's "every component ships with a unit test" requirement. Add the standard tiny-overfit test once the trainer exists.

### 4.6 Ablation recipe
Purpose `research-ablation`, TinyStories, ~20M params, `train_ctx = 512`, equal token budget:

1. `rope` (baseline), `hope`, `darope`, `rope + yarn(×4)`
2. Report **val loss at 512** (expect a tie) and **val loss at 1024 / 2048** (where DaRoPE/HoPE should win clearly and plain RoPE should blow up).
3. Optionally, a needle/KV-recall probe at 512 for the retrieval claim.
4. Feed results into `runs.db` so the planner's `quality_prior` for `darope` is learned from *our* data, not the paper's.

This takes a few hours on the 3070 Ti and is the cheapest way to confirm the paper at our scale before marking it `established`.

---

## 5. Where it lands in the roadmap

- **Phase 1:** no change, but **define the position API as `apply(q, k, x, pos)`** (§4.3).
- **Phase 2:** add `hope` + `darope` next to NoPE/ALiBi (move it up from "Later"). It has no dependencies and is cheaper to build than YaRN.
- **Phase 5:** check interactions with **looped blocks**. The coordinate is recomputed from each iteration's hidden state, which should work naturally, but it's untested. Also check the interaction with **sliding window** (slow bands matter less when the window ≪ L).

## 6. Open items
1. Watch for an official code release (Meta Lingua based). If one appears, diff it against §4.3, especially pre/post-norm input and GQA handling.
2. Is separate `w_h` for q and k worth testing? It breaks the strict relative form, so it's low priority.
3. Does a smaller `base` (10k) or larger (500k) favour DaRoPE at L=512? Run it as a planner sweep dimension once the planner exists.

**Sources:** [arXiv abstract](https://arxiv.org/abs/2609.34556) · [HTML v2 (method, tables, appendix)](https://arxiv.org/html/2609.34556v2)
