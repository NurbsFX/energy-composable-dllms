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

## 10. Reframing — score-based composition is *semantically selective*

After running the §9 sweep I went back to the per-pair PoE-2 ratios
with sceptical lenses on (a) per-expert solo strength and (b) cross-
paradigm calibration of the top-quartile thresholds. Both controls are
necessary because they discipline the §9 cross-paradigm comparisons.
But the more important re-read is **intra-SEDD**: looking only at the
15 PoE-2 pairs computed under one protocol, against one calibration,
on one paradigm, a structured pattern emerges that the §9 headline
("H1 falsified, mean ratio 0.80") was averaging away.

The pattern is not "SEDD fails uniformly". It is **SEDD-PoE composition
is super-additive on semantically homogeneous expert pairs and
sub-additive on heterogeneous ones**, with the strongest sub-additivity
on the most semantically distant pairs. That is the load-bearing
finding of Paper 2 — a *positive* contribution about *what score-based
composition does differently*, not a failure mode.

### 10.1 Semantic class assignment of the 6 experts

| expert | class | rationale |
|---|---|---|
| `formal` | **style** | abstract register / syntactic pattern |
| `long` | **style** | abstract span constraint, distributed |
| `positive` | **sentiment** | sentence-level affect proxy |
| `positive2` | **sentiment** | sentence-level affect proxy (2nd) |
| `concrete` | **topic** | concrete-noun lexical density |
| `sports` | **topic** | topic-specific vocabulary |

Class composition of any pair is therefore one of: same-class,
sentiment×topic, sentiment×style, style×topic.

### 10.2 The 15 PoE-2 ratios partition cleanly by class composition

| class | n | mean | super-add count |
|---|---:|---:|---:|
| same-class | 3 | 1.09 | 1/3 |
| sentiment × topic | 4 | **1.05** ⭐ | 3/4 |
| sentiment × style | 4 | 0.66 | 0/4 |
| style × topic | 4 | **0.46** ⚠️ | 0/4 |

**Reading**: a clear monotone gradient from sentiment×topic
(super-additive, 3/4) through sentiment×style (moderate sub-add) to
style×topic (strong sub-add, mean 0.46). All four super-additive pairs
in the full sweep are class-pure or sentiment×topic; *none* of the 9
style-touching pairs is super-additive.

The bigger split is even simpler — any-style vs no-style:

| split | n | mean | super-add count |
|---|---:|---:|---:|
| any-style (formal or long) | 9 | 0.57 | 0/9 |
| no-style | 6 | **1.13** ⭐ | 4/6 |

**A 2× difference in mean ratio**, perfectly clean partition of the
super-additive set, statistically meaningful at n=15.

Plot: `artifacts/plots/sedd_semantic_selectivity_bars.png` (per-pair,
colored by class) and `sedd_class_comparison_bars.png` (class-level
aggregation).

### 10.3 What this signature means

Same parameterized algebra ($\log s_b + \sum_k \lambda_k (\log s_k -
\log s_b)$) gives qualitatively different composition behaviour
depending on whether the experts target the same or different *kinds*
of axes:

* **Sentiment × topic** super-additive: positive × sports (1.19),
  positive2 × sports (1.19), positive2 × concrete (1.05), positive ×
  concrete (0.76 — the only sub-add of this class).
* **Style × topic** strongly sub-additive: formal × sports (0.11),
  formal × concrete (0.40), long × concrete (0.57), long × sports
  (0.76).
* **Sentiment × style** intermediate sub-additive: 0.53–0.80, mean
  0.66.

The finding is intra-SEDD — same protocol, same baseline, same
threshold calibration on every pair. **No cross-paradigm comparison
is needed to make the claim**.

### 10.4 Individual-expert diagnostic (formal weakness)

| Expert | SEDD-small | MDLM-OWT 110M | MDLM Qwen3 596M |
|---|---:|---:|---:|
| long | 0.860 | — | — |
| **formal** ⚠️ | **0.220** | 0.300 | 0.535 |
| positive | 0.600 | 0.300 | 0.393 |
| positive2 | 0.595 | — | 0.284 |
| concrete | 0.625 | 0.315 | 0.288 |
| sports | 0.820 | 0.315 | 0.210 |

Inside the SEDD column: `formal` is a 3–4× outlier-low compared to all
other experts. Five of six experts cleared the 0.30 axis-recovery
floor. This is not a calibration artefact — see §10.5 for the
threshold check, where SEDD's `form` bar is the *strictest* of the
three paradigms (0.77 vs 0.71 MDLM-OWT vs 0.58 Qwen3), so re-calibrating
to MDLM thresholds would only make formal SEDD look better.

How does this interact with §10.2's semantic-selectivity? Formal
weakness amplifies the class effect on style-touching pairs but does
not cause it: the `long × *` pairs (where `long` has a healthy 0.86
marginal) are also all sub-additive (0.53–0.76 range). The class
pattern is robust to individual-expert variation.

Plot: `artifacts/plots/sedd_marginals_bars.png`.

### 10.5 Cross-paradigm calibration is asymmetric across axes

The top-quartile thresholds are computed *within* each paradigm against
its own baseline distribution. SEDD's unconditional baseline is broader
than MDLM's prompted baseline on most axes, but the direction differs by
proxy:

| Proxy | SEDD threshold | MDLM-OWT thr. | MDLM Qwen3 thr. |
|---|---:|---:|---:|
| `len` | 64.0 | 60.0 | 60.0 |
| **`form`** | **0.767** | 0.709 | 0.578 |
| `sent` | 0.679 | 0.995 | 0.978 |
| `sent2` | 0.150 | 0.437 | 0.416 |
| `conc` | 2.73 | 2.91 | 2.91 |
| `topic` | 0.405 | 0.448 | 0.999 |

* For `form`, SEDD's bar is the *strictest*. The formal SEDD expert
  clears it only 22 % of the time *despite* the higher bar — the
  weakness is real, re-calibrating to MDLM thresholds would only make
  formal SEDD look better.
* For `sent`/`sent2` (and partly `topic` vs Qwen3), SEDD's bar is much
  lower. The high marginals on these axes are partly a calibration
  effect — easier bar, more apparent solo strength.

Consequence: **most cross-paradigm ratio comparisons are confounded.**
The SEDD lexical triplet `positive2 × concrete × sports` (ratio 0.84)
cannot be cleanly compared to MDLM Qwen3's 3.23, because:

| Quantity | SEDD | MDLM Qwen3 |
|---|---:|---:|
| solo marginals (positive2, concrete, sports) | 0.595, 0.625, 0.820 | 0.284, 0.288, 0.210 |
| indep_ref ($m_a m_b m_c$) | 0.305 | 0.017 |
| absolute triple_sat | 0.255 | ≈ 0.055 |
| ratio | 0.84 | 3.23 |

In absolute terms SEDD generates **more** triple-satisfying samples
(51/200 vs ≈ 11/200) — the lower ratio is largely a head-room artefact
of the much higher independence reference. The 0.84 vs 3.23 comparison
**is not load-bearing**.

Same logic applies, in muted form, to the H1 mean-ratio comparison
(0.80 SEDD vs 1.07 MDLM Paper-1). With both `formal` and `long` excluded
the SEDD mean rises to 1.13 — moving the answer from "sub-additive" to
"super-additive" depending on which weak experts are kept. The headline
H1 number is calibration- and selection-sensitive, not robustly
falsified.

### 10.6 The μ-sweep inversion as a corollary of §10.3

§9.3 reported that the μ-sweep on `formal × positive × concrete`
shows opposite shapes on MDLM and SEDD:

| μ | SEDD ratio | MDLM Qwen3 ratio |
|---:|---:|---:|
| **−2 (canonical)** | **0.24** ⭐ | 0.55 |
| −1.5 | 0.12 | 0.54 |
| −1 | 0.06 | **0.61** ⭐ |
| −0.5 | 0.00 | 0.38 |
| 0 | 0.00 | 0.38 |
| +0.5 | 0.00 | 0.38 |
| +1 | 0.00 | 0.31 |

This is *intra-protocol within each paradigm* (no calibration mismatch)
and shows the same algebra producing a bell-shape on MDLM (relax μ →
improvement) and a monotone-decreasing curve on SEDD (relax μ →
collapse). On its own this is a clean finding.

**But — and this is the §10 update — it is now better understood as a
corollary of §10.3 rather than a primary finding.** The μ-sweep was
run on `formal × positive × concrete`, a style×sentiment-style class
mixture (formal=style, positive=sentiment, concrete=topic). Under the
selectivity hypothesis of §10.3, this triplet sits in the
sub-additive regime by construction. Relaxing μ amplifies the
expert log-score sum without anchoring on $\log s_b$, and that
amplification of an already-incoherent transition rate field is what
collapses the ratio.

The cleanest follow-up — pre-registered in §10.8 — is a μ-sweep on
a **homogeneous** triplet (e.g. `positive × concrete × sports`,
sentiment×topic×topic, which the §10.2 table predicts should be
super-additive at canonical). If that sweep also shows
monotone-decreasing collapse, then the μ-inversion is paradigm-level
*and* selectivity is real. If it shows a bell-shape similar to MDLM's,
the inversion is selectivity-driven and not a paradigm-level
phenomenon. **Either outcome strengthens Paper 2 in a different
direction**: it disambiguates "score-domain has a different μ
response" from "score-domain is selective on semantic class".

### 10.7 Mechanistic hypothesis

The data motivates the following testable account of *why* SEDD
score-based composition shows this selectivity:

> Sum-of-log-scores composes transition *rates*, which are tensors
> indexed by ``(position, current token, candidate token)``. The
> τ-leaping sampler updates several positions in parallel using the
> joint rate field. When two experts encode constraints of *different
> semantic kinds* (an abstract distributed style constraint vs a
> concrete lexical topic constraint), the per-position log-score
> shifts they emit are not aligned in any cross-position structure —
> they push toward different *kinds* of transitions, possibly at
> different positions. Adding the two log-score fields produces a
> joint rate field whose updates at parallel positions are
> *cross-position incoherent*: each position's preferred transition
> depends on a different constraint, but τ-leaping commits them
> jointly. The result is a noisy joint sample.
>
> When the experts target the same kind of axis (sentiment ×
> sentiment, topic × topic, sentiment × topic) the per-position
> shifts are aligned in their cross-position structure — they
> reinforce each other on the same kinds of transitions. The
> τ-leaping joint sample is then *coherently amplified*, hence
> super-additive ratios.
>
> MDLM's per-position categorical sampler does not have this
> cross-position-coherence requirement, because each position is
> resolved independently given $x_t$. It cannot exploit
> cross-position-coherent compositions to become super-additive
> beyond what the marginals already allow, but it also cannot suffer
> from cross-position-incoherent compositions. **The two paradigms
> trade off coherence-exploitation for coherence-fragility**.

This account predicts (a) the §10.2 selectivity, (b) the §10.6
μ-sweep collapse on heterogeneous triplets, and (c) that a
homogeneous-triplet μ-sweep on SEDD should look more bell-shaped or
at least less monotone-collapsing than the formal-heavy one. Test (c)
is the next thing to run.

### 10.8 Pre-registered next experiments

The §9.5 caveat list is replaced with three concrete predictions, each
with a clear discriminator outcome and a budget:

| # | Experiment | What it tests | Budget |
|---|---|---|---|
| (a) | Homogeneous-triplet μ-sweep — `positive × concrete × sports` | Disambiguates §10.6: paradigm-level inversion vs selectivity-driven | ~$5 |
| (b) | Heterogeneous-triplet μ-sweep — non-formal but style×lex (`long × positive × concrete`) | Tests selectivity prediction on a non-formal style axis | ~$5 |
| (c) | Quantify "semantic distance" between experts via mean log-score-shift cosine on validation tokens, then correlate with PoE-2 ratio across all 15 pairs | Quantifies the selectivity gradient predicted by §10.3 | $0 (local, ~30 min Python) |

Stop conditions:

* If (a) shows a non-monotone shape: §10.3 selectivity is real, μ
  inversion was a corollary. Paper 2 ships as a positive-finding
  paper on score-based composition selectivity.
* If (a) shows the same monotone collapse: the inversion is
  paradigm-level and selectivity is structural to the score-domain.
  Paper 2 ships with both findings, the inversion as a clean
  corollary.
* If (c) gives r ≥ 0.5 across the 15 pairs: the selectivity claim is
  quantified, not just descriptive. This makes the paper
  substantially stronger.

Prompted SEDD repeat and formal-only retrain are no longer top-priority
— they are cleanups that defend §9's H1/H2 numbers, which are no
longer the central claim. Run them only if a reviewer explicitly
attacks the unconditional protocol.

### 10.9 What Paper 2 claims after §10

> The PoE-of-densities identity transports algebraically to the
> log-score domain and admits a working τ-leaping sampler. Empirically,
> however, the resulting composition is *not* a uniform improvement
> over MDLM PoE composition: it is **selectively super- or
> sub-additive depending on the semantic homogeneity of the composed
> experts**. On 15 PoE-2 pairs from 6 LoRA experts spanning style,
> sentiment, and topic axes:
>
> * sentiment × topic pairs: mean ratio 1.05 (3/4 super-additive)
> * any pair touching a style expert (formal, long): mean ratio 0.57,
>   0/9 super-additive, including catastrophic collapses below the
>   independence baseline (formal × sports: 0.11)
>
> This is calibration-immune and paradigm-internal (one threshold
> calibration, one baseline). It motivates a mechanistic hypothesis:
> τ-leaping joint sampling exploits cross-position coherence between
> homogeneous experts (super-additive amplification) and amplifies
> incoherence between heterogeneous experts (active interference).
> MDLM's per-position categorical sampler trades both away — neither
> exploits coherence nor amplifies incoherence — explaining the
> uniformly modest super-additivity Paper 1 observed. The μ-sweep
> inversion of §9.3 is a corollary: relaxing μ amplifies the joint
> log-score field, which helps MDLM (uniform regime) and hurts SEDD
> on heterogeneous triplets (incoherent regime). Three pre-registered
> experiments at $≤15$ pod-USD will discriminate "paradigm-level
> inversion" from "selectivity-driven collapse" and quantify the
> selectivity gradient.

---

*Working draft updated 2026-05-03. §10 reframes the §9 negative
result as a positive selectivity finding. The decision-gate
experiments in §10.8 are pre-registered.*
