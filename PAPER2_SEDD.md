# Paper 2 — Score-based vs marginal-based composition in discrete diffusion

> **Status**: planning + infrastructure stage. No experiment run yet. This
> document is the working draft for a follow-up to Paper 1 (PoE-MDLM).

## 0. One-liner

We tested marginal-based PoE composition on MDLM and identified a
robust N=3 plateau (Paper 1). We now test the orthogonal hypothesis:
**is the plateau caused by the per-position factorization implicit in the
MDLM-PoE composition, or is it intrinsic to the composition itself?**

The discriminator is **score-based composition on SEDD** (Lou, Meng,
Ermon 2024) — same composition formula in log-space, but applied to
*scores* (transition log-ratios) which carry sequence-level coherence
through the τ-leaping sampler. If the plateau lifts under SEDD, the
bottleneck was factorization. If it stays, it's intrinsic to PoE — and
our μ-fix (Paper 1) gains generality.

## 1. Theoretical pivot

### 1.1 The factorization issue in MDLM-PoE

MDLM at step $t$ predicts a per-position categorical distribution
$p_\theta(x_0^{(i)} \mid x_t)$. The PoE composition we use in Paper 1
sums logits per-position:

$$\ell_\text{PoE}(x_t)_i = \ell_b(x_t)_i + \sum_k \lambda_k\,(\ell_k - \ell_b)_i$$

This treats positions as conditionally independent given $x_t$. Cross-position
correlations induced by the experts are lost. Paper 1's
MH-token-swap experiment (§12.2) **already confirmed that MDLM-PoE samples
sit near the modes of the factorized distribution** — i.e., the issue is
not insufficient sampling but a property of the distribution itself.

### 1.2 What SEDD changes

SEDD (Lou et al. 2024, ICML Best Paper) trains a **concrete score**:

$$s_\theta(x, t)_y = \frac{p_t(y)}{p_t(x)} \quad \text{for } y \neq x$$

This is a *transition ratio*, not a marginal distribution. Crucially, the
PoE identity in log-space holds **exactly at the sequence level**:

$$\frac{p_\text{PoE}(y)}{p_\text{PoE}(x)} = \frac{p_b(y)}{p_b(x)} \cdot \prod_k \left(\frac{p_k(y)/p_k(x)}{p_b(y)/p_b(x)}\right)^{\lambda_k}$$

$$\Rightarrow\quad \log s_\text{PoE} = \log s_b + \sum_k \lambda_k(\log s_k - \log s_b)$$

Same algebraic form as the MDLM logit composition, but the underlying
quantities are log-ratios with sequence-level semantics. Combined with
SEDD's τ-leaping sampler (which respects cross-position correlations
encoded in the transition matrix), this gives an exact composition.

### 1.3 The discriminator experiment

| Backbone | Composition | Sampler | What it measures |
|---|---|---|---|
| MDLM | PoE-logits | categorical, per-position | Paper 1: factorized PoE |
| SEDD | PoE-scores | τ-leaping | Paper 2: exact-sequence PoE |

If SEDD lifts the N=3 plateau without μ-fix, the MDLM bottleneck was
factorization. If SEDD plateaus too, μ-fix (Paper 1) addresses an
intrinsic issue and gains weight.

## 2. Hypotheses (to be tested)

**H1.** PoE-2 super-additivity holds on SEDD-small (110M, OWT-comparable to
MDLM-OWT). Mean ratio $> 1$ on the 10 pairs from Paper 1.

**H2.** SEDD lifts the N=3 plateau on triplets that failed under MDLM-PoE.
Specifically, formal × positive × concrete (Paper 1 ratio = 0.55 with
canonical μ; 0.71 with μ-fix at 0). Target on SEDD: $\geq 1.0$ without μ-fix.

**H3.** μ-fix on SEDD shows a **flatter** sweep than on MDLM. The bell
shape of Paper 1 §13.2 was sharp because the canonical μ caused
factorization-induced collapse. Under exact composition, varying μ should
have a milder effect.

**H4.** The cross-backbone κ↔deficit collapse (Paper 1 §10) is reproduced
on SEDD-small vs SEDD-medium, OR partially repaired — this would
strengthen the "capacity vs composition" thread.

## 3. Architecture (parallel track in this repo)

```
external/sedd/                       # vendored Lou et al. (read-only)
  ├── model/transformer.py           # DDiT, score model
  ├── sampling.py                    # PC sampler with score_fn hook
  ├── graph_lib.py                   # absorbing graph + score_entropy()
  ├── losses.py                      # DSE loss
  └── load_model.py                  # SEDD.from_pretrained(...)

src/sedd_composition/                # OUR code, parallel to src/composition/
  ├── load.py                        # load_sedd_from_hub() + sys.path setup
  ├── poe_score.py                   # PoEScoreCompositionModel (analog of poe_sampler.PoECompositionModel)
  ├── sampler.py                     # PoEScoreSampler (wraps Lou's pc_sampler)
  ├── train_lora.py                  # LoRA training with score-entropy loss
  └── README.md

scripts/sedd_*.py                    # Phase-by-phase experiments, parallel namespace
  sedd_00_load_smoke.py              # Sanity: load SEDD-small, sample, decode
  sedd_01_train_lora.py              # Train one LoRA expert (smoke test)
  sedd_02_train_all_experts.sh       # Phase-3 analog
  sedd_03_poe2_sweep.py              # Phase-4 analog (10 pairs)
  sedd_04_poe3.py                    # Phase-6 analog (3 triplets)
  sedd_05_mu_sweep.py                # Phase-11 analog
```

**Isolation principle**: nothing in `src/sedd_composition/` imports from
`src/composition/`. The two stacks share only `src/eval/` (proxy energies,
joint satisfaction, scoring) — those work on text and are model-agnostic.

## 4. Three risks identified upfront

1. **Fused QKV in DDiT**: PEFT/LoRA `target_modules` must be set
   explicitly to `["attn_qkv", "attn_out", "mlp.0", "mlp.2"]`. We cannot
   LoRA Q/K/V independently without surgery — we accept fused QKV LoRA.

2. **MASK token at vocab index 50257**: SEDD uses GPT-2 + 1 absorbing
   token (vocab = 50258). Composition on log-scores is element-wise
   well-defined, but if any expert produces NaN at the MASK index for an
   unmasked token (because that transition has zero rate by construction),
   the PoE sum NaNs out. We special-case the absorbing column.

3. **Score representation = log-ratios in `graph.staggered_score`'s
   parameterization**, not plain logits. The composition formula
   `log s_PoE = log s_b + Σ λ_k (log s_k - log s_b)` is correct in this
   parameterization (verify upstream — see `model/utils.py:get_score_fn`,
   which exp's the model output for sampling). Sanity check via λ=0:
   composed forward must equal base forward token-for-token at fixed seed.

## 5. Experimental plan

| Step | Effort dev | Compute (USD) | Output |
|---|---:|---:|---|
| 0. Vendor SEDD | done | 0 | `external/sedd/` |
| 1. PAPER2 + arch design | done | 0 | this doc |
| 2. Smoke load + sample | 0.5 d | 0 (CPU/MPS for small batch) | `sedd_00_load_smoke.py` |
| 3. PoEScoreCompositionModel | 1 d | 0 | `src/sedd_composition/poe_score.py` |
| 4. Sampler wrapper | 1 d | 0 | `src/sedd_composition/sampler.py` |
| 5. λ=0 sanity check | 0.5 d | ~2 | sanity test |
| 6. LoRA training scaffold | 2 d | ~5 | `train_lora.py` |
| 7. Train 6 experts SEDD-small | 0.5 d | ~5 | 6 LoRA checkpoints |
| 8. PoE-2 sweep (10 pairs) | 0.5 d | ~6 | `sedd_poe2_sweep.json` |
| 9. PoE-3 (3 triplets) | 0.5 d | ~4 | `sedd_poe3.json` |
| 10. μ-sweep on stylistic triplet | 0.5 d | ~5 | `sedd_mu_sweep.json` |
| **Subtotal H1+H2+H3** | **~7 d** | **~30 USD** | core results |
| 11. (Optional) SEDD-medium repeat | 1 d | ~15 | H4 results |

Compared to Paper 1's ~95h pod time (~80 USD), Paper 2 should be
~$30–45 USD if reusing existing datasets and proxies (which we do).

## 6. What this gets us

- **If H2 confirmed (plateau lifts)**: Paper 2 has a clean
  methodological contribution — score-based composition is
  meaningfully better than logit-based for N=3 PoE on discrete diffusion.
  Venue: NeurIPS / ICLR / ICML main track.
- **If H2 falsified (plateau holds)**: μ-fix (Paper 1) generalizes
  beyond MDLM, and we have a strong negative result that motivates a
  *third* class of approaches (learned energies, classifier guidance).
  Venue: workshop or Findings, but the negative result is informative.
- **In all cases**: Paper 1 is unaffected — Paper 2 stands alone with
  its own framing.

## 7. Out of scope for Paper 2

- Anything that requires modifying SEDD's loss or noise schedule.
- N $\geq$ 4 — same scope-cap as Paper 1.
- Non-text discrete diffusion (images, code).
- Training SEDD from scratch — we adapt published checkpoints with LoRA.

## 8. Open questions to resolve before training

- Which proxy energies remain valid given SEDD's vocab=50258 vs MDLM's
  50258 (same shape — verify token IDs match)?
- LoRA rank: copy from Paper 1 (rank=8, alpha=16)? Or higher because DDiT
  blocks have different scale?
- Number of training steps per expert: target equivalent of Paper 1's
  "marginal recovers in solo composition at λ=1".

---

## 9. Results — first run on SEDD-small (2026-05-03)

End-to-end pipeline on RunPod A100 80GB, ~3.5h total compute (~$30):

* **Setup**: torch 2.11 + transformers 4.57 + peft 0.17. SEDD vendored
  with two upstream patches (flash_attn → SDPA fallback,
  `@torch.jit.script` decorators stripped — JIT crashes on torch 2.11).
* **Training**: 6 LoRA experts on SEDD-small (170M params with adapter,
  0.88M trainable), 2500 steps each at batch=16, seq_len=128. Final
  losses 380–430 across the 6 verticals (score-entropy units).
* **Eval**: $n=200$ samples per condition, seq_len=64, 128 sampling
  steps with the analytic τ-leaping predictor. Unconditional generation.

### 9.1 H1 — PoE-2 super-additivity: **falsified**

15 pairs out of $\binom{6}{2}$, ratios computed against
$m_a \times m_b$.

| | mean ratio | super-additive? |
|---|---:|---|
| **SEDD-small** (Paper 2) | **0.80** | **NO — sub-additive** |
| MDLM Paper 1 reference | 1.07 | YES |

The only super-additive pair is `positive × positive2` (ratio 1.68),
which is structurally trivial — both are sentiment proxies on the
same semantic axis. Worst pair: `formal × sports` at 0.11.

Plot: `artifacts/plots/sedd_poe2_bars.png`.

### 9.2 H2 — PoE-3 plateau lifts on SEDD: **falsified**

| Triplet | SEDD canonical | MDLM canonical | MDLM best μ-fix |
|---|---:|---:|---:|
| formal × positive × concrete (style) | **0.18** | 0.55 | 0.61 |
| formal × concrete × sports (mixed) | **0.00** | 0.46 | 1.23 |
| positive2 × concrete × sports (lex) | **0.84** | 3.23 | 3.23 |

SEDD is **worse than MDLM on all three triplets**, including the
purely lexical one that was strongly super-additive on MDLM (ratio
3.23). The mixed triplet collapses entirely (ratio 0.00).

Plot: `artifacts/plots/sedd_poe3_bars.png`.

### 9.3 H3 — μ-fix transports to SEDD: **falsified, in the inverse direction**

μ-sweep on `formal × positive × concrete` (the stylistic triplet
where Paper 1's μ-fix produced the largest gain on MDLM):

| μ | SEDD ratio | MDLM ratio (Paper 1) |
|---:|---:|---:|
| **−2 (canonical)** | **0.24** ⭐ | 0.55 |
| −1.5 | 0.12 | 0.54 |
| −1 | 0.06 | **0.61** ⭐ (+11%) |
| −0.5 | 0.00 | 0.38 |
| 0 | 0.00 | 0.38 |
| +0.5 | 0.00 | 0.38 |
| +1 | 0.00 | 0.31 |

On SEDD the canonical $\mu = 1-N = -2$ is **already optimal**, and any
relaxation drops the ratio to zero. This is the **inverse pattern**
of Paper 1 on MDLM, where μ-fix improved by +11–29 % over canonical.

Plot: `artifacts/plots/sedd_mu_sweep_bars.png`.

### 9.4 Synthesis — what the results tell us

Three crisp negative findings:

1. **Score-based composition does not lift the N=3 plateau.** The
   "exact at sequence level" theoretical argument we used to motivate
   Paper 2 (PoE-of-densities transports cleanly to log-scores) is
   *mathematically* correct but **does not translate into compositional
   capability gains** under our protocol on SEDD-small.
2. **Paper 1's μ-fix does not transport to SEDD.** Worse, the optimal
   direction is inverted — canonical is already optimal, relaxing
   destroys composition.
3. **MDLM > SEDD on all three triplets**, including the lexical one
   where MDLM's canonical PoE was already strongly super-additive.

This is informative. It says the PoE bottleneck is **not** simply the
per-position factorization implicit in the MDLM categorical sampler.
Some other property of the score-domain composition / τ-leaping
sampler is doing more harm than the factorization was avoiding.

### 9.5 Caveats / what we have not ruled out

- **LoRA undertrained**. 5 000 documents × 2500 steps. The training
  losses were still decreasing at the end. A longer run (10 000 steps,
  full datasets) might change the picture. **Cost to test**: ~12h pod
  + we'd need to re-run all evals (~$50 total). Probably worth doing
  before claiming the negative result is final.
- **Capacity asymmetry**. SEDD-small has 90M params; MDLM Paper 1's
  best results came from Qwen3-MDLM (596M). A SEDD-medium repeat
  (320M) would equalize for capacity. **Cost**: ~$15.
- **Sampling protocol**. Paper 1 used MDLM with a 12-token prompt
  prefix; Paper 2 uses unconditional SEDD generation. Within-paradigm
  ratios should be meaningful (canonical vs μ-fix on the same
  protocol), but the absolute baselines differ. **Cost to fix**:
  zero-effort (re-run with prompts), ~5h pod.
- **Score domain numerical issues**. We sanitize the absorbing-token
  column when summing log-scores; otherwise NaNs would corrupt the
  PoE sum. The sanitization is conservative (zero-out non-finite
  values). It might over- or under-mask in ways that hurt composition.

### 9.6 What we ship as Paper 2

A **negative-result paper** with three components:

1. **Methodology**: parallel score-based composition stack with
   identical algebra to MDLM PoE, plus the τ-leaping sampler.
2. **Hypotheses falsified**: H1 (super-additivity on N=2), H2 (plateau
   lifts on N=3), H3 (μ-fix transports). Tables 9.1–9.3.
3. **Discussion**: the score-domain composition is mathematically
   exact at the sequence level but empirically *worse* than the
   factorized MDLM PoE in our protocol. The PoE bottleneck is **not**
   just per-position factorization. This rules out a class of theoretical
   arguments and motivates explicitly-learned compositional energies
   (a third paper, not this one).

**Paper 1 stands unaffected.** Paper 2's negative finding does not
weaken Paper 1's μ-fix — it confirms that μ-fix is a property of
the MDLM-PoE *paradigm* and does not generalize to score-based PoE
in this implementation.

---

*Working draft updated 2026-05-03. Phase-9 results above are first-run
numbers; longer training and SEDD-medium repeat would tighten them.*
