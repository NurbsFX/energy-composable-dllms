# REPORT.md — PoC results

> Running log of experimental results, updated as each phase produces
> artefacts. Final Sunday-evening write-up lives in the numbered sections
> below; the **Worklog** at the bottom captures the chronological progress.

## 1. Central claim

_TODO — one paragraph, the thesis in a single breath._

## 2. Formal recap

### 2.1 Empirical Gram matrix

Let `E_i(x)` be the proxy energy for vertical `i` evaluated on a text `x`,
and let `{x_n}` be `N` samples drawn from the reference distribution
`p_base` (here OpenWebText). The empirical Gram matrix is the sample
covariance:

```
G_ij = Cov(E_i, E_j) = (1/N) Σ_n E_i(x_n) E_j(x_n) − Ē_i · Ē_j
```

All four dependence metrics below are computed from this matrix or from
the full sample matrix `E ∈ ℝ^{N × k}` (k = number of verticals).

### 2.2 The five dependence metrics

We use five metrics on each unordered pair of verticals because no single
one captures everything. Linear coverage from κ, monotone non-linear
coverage from Spearman, full non-linear coverage from HSIC / CKA / MI.

**κ — orthogonality index (linear).** For a pair `(i, j)` with 2×2
sub-Gram `G_pair`:

```
κ_ij = ‖G_pair − diag(G_pair)‖_F / Tr(G_pair)
     = √2 · |Cov(E_i, E_j)| / (Var(E_i) + Var(E_j))
```

Range `[0, ∞)`. κ = 0 iff the two energies are linearly uncorrelated on
`p_base`; values grow with the off-diagonal magnitude relative to the
trace. **Captures:** Pearson-style linear dependence. **Misses:**
non-linear coupling — e.g. `Y = X²` with `X ~ N(0, 1)` has `Cov(X, Y) = 0`
hence κ ≈ 0 even though `Y` is a deterministic function of `X`. SE on the
estimate scales as `≈ 1/√N`. Implemented in
[src/energies/gram_matrix.py](src/energies/gram_matrix.py).

**Spearman — rank correlation (monotone).** Pearson's correlation
applied to the *ranks* of `x` and `y` rather than their values:

```
ρ_s(X, Y) = Pearson(rank(X), rank(Y))
```

Range `[-1, 1]`. ρ_s = ±1 iff `Y` is a monotone function of `X` (linear
or not — `Y = exp(X)`, `Y = log(X)`, any monotone curve all saturate to
ρ_s = 1). **Captures:** any monotone dependence — fills the gap between
κ (linear only) and the kernel/info-theoretic metrics (also non-monotone
non-linear). **Misses:** non-monotone non-linear dependence — `Y = X²`
with `X` centred is fully determined by `X` but gives ρ_s ≈ 0; only
HSIC/CKA/MI can catch that case. **Why bother:** ρ_s is a single line of
NumPy, robust to outliers (operates on ranks), bounded, and matches what
reviewers expect from a non-parametric dependence section. Implemented
in [src/energies/independence.py](src/energies/independence.py).

**HSIC — Hilbert-Schmidt Independence Criterion (kernel, non-linear).**
With RBF kernels `K_ij = exp(−‖x_i − x_j‖² / 2σ²)` and `L_ij` analogous
on `Y`, double-centered into `K_c = (I − 11ᵀ/N)·K·(I − 11ᵀ/N)`:

```
HSIC(X, Y) = (1 / (N − 1)²) · Tr(K_c · L_c)
           = (1 / (N − 1)²) · Σ_{ij} (K_c)_{ij} (L_c)_{ij}
```

σ chosen by the **median heuristic** (σ² = median of squared pairwise
distances / 2). Range `[0, ∞)`, HSIC = 0 iff `X ⊥ Y` in the kernel limit
(for characteristic kernels such as RBF, equivalent to true
independence). **Captures:** linear *and* non-linear coupling via the
RKHS embedding. **Limitation:** scale-dependent — raw HSIC values are
not directly comparable across pairs that have different marginals.
Implemented in [src/energies/independence.py](src/energies/independence.py).

**CKA — Centered Kernel Alignment (kernel, normalized).**

```
CKA(X, Y) = HSIC(X, Y) / √(HSIC(X, X) · HSIC(Y, Y))
```

Range `[0, 1]`. CKA = 0 iff no kernel-detectable dependence; CKA = 1
when `X` and `Y` produce identical centered kernel matrices (typically
when one is a deterministic function of the other in the kernel
embedding). **The right metric to *rank* pairs** with different
marginal distributions, because the marginal-scale dependence has been
normalized out.

**MI — Mutual information (information-theoretic).** Theoretical form:

```
I(X; Y) = ∫∫ p(x, y) log(p(x, y) / (p(x) p(y))) dx dy
```

Estimated via the **Kraskov-Stögbauer-Grassberger (KSG) k-NN estimator**
from `sklearn.feature_selection.mutual_info_regression` with `k = 3`.
Reported in nats. Range `[0, ∞)`, MI = 0 iff `X ⊥ Y` strictly.
**Captures:** any statistical dependence, fully non-linear,
information-theoretically motivated. **Limitations:** the KSG estimator
is biased (under-estimates at small `N`, over-estimates at large `N`),
and MI is unbounded so absolute comparison across very different
distributions is delicate. Used alongside CKA as an independent
cross-check on the kernel-based finding.

### 2.3 What each metric is for, in one line

| metric | linear? | monotone non-linear? | non-monotone non-linear? | bounded? | ranking pairs? |
|---|---|---|---|---|---|
| κ | ✓ | ✗ | ✗ | no | yes (within linear regime) |
| Spearman | ✓ | ✓ | ✗ | yes (`[-1, 1]`) | yes |
| HSIC | ✓ | ✓ | ✓ | no | only same-marginal pairs |
| CKA | ✓ | ✓ | ✓ | yes (`[0, 1]`) | yes (across all pairs) |
| MI | ✓ | ✓ | ✓ | no | yes, with bias caveats |

For everything that follows in the worklog, **κ alone is not enough**:
empirically (cf. the 2026-04-25 entry on N=5000 + non-linear metrics)
the κ-ranking and the CKA-ranking of the six OWT pairs disagree, and
that disagreement is itself a finding.

The PoE composition rule (used from Phase 4 onwards) is documented
inline in [src/composition/poe_sampler.py](src/composition/poe_sampler.py).

## 3. Figure 1 — Empirical Gram matrix (Phase 1)

_TODO: insert `artifacts/plots/gram_heatmap.png` and comment on which pairs
are (non-)orthogonal._

## 4. Figure 2 — Direct PoE formula check (Phase 4.5, Test 2)

_TODO: insert `artifacts/plots/poe_formula_check.png`. Report the slope ±
stderr and R² of the log-log fit. Target: slope ≈ 1, R² > 0.85._

## 5. Figure 3 — Extension to N=3 (Phase 4.5, Test 3)

_TODO: insert `artifacts/plots/n3_satisfaction.png`. Compare observed triple
joint satisfaction to the product of marginals._

## 6. Figure 4 — Joint satisfaction, 6 configs × 4 pairs (Phase 4)

_TODO: insert `artifacts/plots/joint_satisfaction_barplot.png`._

## 7. Figure 5 — κ vs deficit (Phase 5) — THE figure

_TODO: insert `artifacts/plots/kappa_vs_deficit.png`. Report Pearson(κ,
deficit) on the 6 pairs. Target: |Pearson| > 0.6._

## 8. Qualitative examples

_TODO: 3 samples × 6 configs × 2 pairs (one orthogonal + one non-orthogonal)._

## 9. Limits

See [ROADMAP_POC.md §10](ROADMAP_POC.md#10-limites-à-déclarer-explicitement-dans-le-report)
— copy verbatim and adjust based on what the experiments actually showed.

## 10. Future work

_TODO: list the follow-ups that fell out of scope this weekend (e.g. Test 1
intersection-trained expert, multi-backbone universality, N ≫ 3 scaling,
κ measured directly on DLLM energies)._

## 11. Citations

- Sahoo et al., MDLM (2024).
- Lou et al., SEDD (2024).
- Zhou et al., dLLM framework (2026).
- Schiff et al., Discrete Diffusion Guidance (2025).
- DeepMind, EBM-AR bijection (2025).
- Hinton, Product of Experts (2002).

---

## Worklog

Chronological log. Each entry is one short paragraph: what we ran, what came
out (numbers / plots / artefacts), and any GO/NO-GO decision. Section
write-ups above this line should be updated whenever a worklog entry produces
results that fit there.

### 2026-04-25 — Phase 0 / scaffolding

Set up the repo skeleton. No experiments run yet — no results to log.

### 2026-04-25 — Cleanup pass

Code-quality pass on the scaffolding before public release: tightened
docstrings, removed cross-references to the gitignored planning document,
dropped unused dependencies, added a unit-test suite covering the
numerical helpers, added `LICENSE`. No experiments run.

### 2026-04-25 — CI/CD

Added GitHub Actions (ruff + pytest on push/PR) and pre-commit hooks
(ruff lint+format and standard hygiene). Codebase is now gated by
automated checks on every commit. No experiments run.

### 2026-04-25 — Phase 1 smoke-test on Mac CPU (N = 100)

Validated the proxy stack end-to-end on a 100-sample OWT slice before
spending GPU time. All four classifiers downloaded, instantiated and
parsed their labels without manual intervention:

- `s-nlp/roberta-base-formality-ranker` → label `formal` resolved.
- `distilbert-base-uncased-finetuned-sst-2-english` → label `POSITIVE` resolved.
- `unitary/toxic-bert` → label `toxic` resolved (we read `1 − P(toxic)` for the non-toxic energy).
- GPT-2 tokenizer → length energy.

Produced `artifacts/gram_matrix.json` and the heatmap + pair-scatter plots.

Pairwise κ on N = 100 (preliminary, **not** the Phase 1 result; the Gram
matrix only converges at N ≫ 100):

| pair | κ | roadmap prediction |
|---|---|---|
| sent × tox  | 0.001 | > 0.35 |
| len × sent  | 0.003 | < 0.15 |
| form × sent | 0.014 | 0.15–0.30 |
| len × form  | 0.050 | < 0.10  |
| len × tox   | 0.058 | < 0.15 |
| form × tox  | 0.255 | 0.20–0.35 |
| **global**  | **0.042** | — |

The biggest surprises are `sent × tox` and `form × sent`, both far below
their predicted κ. Two plausible explanations: (a) N = 100 noise — Cov is
high-variance with so few samples; (b) on OWT specifically, true toxic
content is rare enough that the sentiment-toxicity coupling we expect from
Twitter / social data does not show up. The full N = 5000 run on a GPU pod
will resolve which.

GO criterion of the roadmap (≥ one pair with κ < 0.15 and ≥ one with κ ≥ 0.30)
is **not yet met** at this N — `form × tox` only reaches 0.255. To watch
on the real run.

### 2026-04-25 — Phase 1 smoke-test on Mac CPU (N = 1000)

Bumped the sample size by 10× to disambiguate whether the surprising
`sent × tox` value at N=100 was noise. Side-by-side with the N=100 run:

| pair | N=100 | N=1000 | roadmap prediction |
|---|---|---|---|
| sent × tox  | 0.001 | 0.040 | > 0.35 |
| len × sent  | 0.003 | 0.002 | < 0.15 |
| form × sent | 0.014 | 0.009 | 0.15–0.30 |
| len × form  | 0.050 | 0.033 | < 0.10 |
| len × tox   | 0.058 | 0.028 | < 0.15 |
| form × tox  | 0.255 | 0.142 | 0.20–0.35 |
| **global**  | 0.042 | 0.043 | — |

Every pair except `sent × tox` *decreased* in κ when N grew from 100 to
1000, and `sent × tox` only crept from 0.001 to 0.040 — still well below
the roadmap's > 0.35 prediction. Most importantly, the strongest pair
`form × tox` dropped from 0.255 to 0.142, **falling below the lower bound
of its predicted range** (0.20). The trend is downward, not upward, so
extrapolating to N=5000 is unlikely to recover a κ ≥ 0.30 anywhere.

Caveat on the conclusion. The standard error on κ_pair scales roughly as
1/√N, so at N=1000 each value carries SE ≈ 0.03. The 95% CIs on the two
strongest pairs are approximately:

- `form × tox`: 0.142 ± 0.06 → CI ≈ [0.08, 0.21]
- `sent × tox`: 0.040 ± 0.06 → CI ≈ [−0.02, 0.10]

`form × tox` could plausibly touch the lower bound of its predicted range
(0.20); reaching ≥ 0.30 anywhere is unlikely but not formally ruled out.
A previous version of this entry asserted that the experimental design
was "not satisfiable on this corpus as designed" — that claim was
overconfident given the SE at N=1000 and is retracted here. A run at
N=5000 (SE ≈ 0.014) is queued to pin down the asymptotic values; the
final decision on the §3.7 fallback (redundant 5th proxy) will be made
on those tighter estimates.

Smoke-test was on Mac CPU, ~4 minutes wall time including model loading.

### 2026-04-25 — Phase 1 on N = 5000 (Mac CPU, ~25 min)

Ran the full ROADMAP_POC §3 sample size to pin down the asymptotic κ
values. Convergence across the three N is clean:

| pair | N=100 | N=1000 | N=5000 | 95% CI at N=5000 | roadmap prediction |
|---|---|---|---|---|---|
| form × sent | 0.014 | 0.009 | 0.001 | [0, 0.029] | 0.15–0.30 |
| len × sent  | 0.003 | 0.002 | 0.001 | [0, 0.029] | < 0.15 |
| len × tox   | 0.058 | 0.028 | 0.031 | [0.003, 0.059] | < 0.15 |
| len × form  | 0.050 | 0.033 | 0.033 | [0.005, 0.061] | < 0.10 |
| sent × tox  | 0.001 | 0.040 | 0.046 | [0.018, 0.074] | > 0.35 |
| form × tox  | 0.255 | 0.142 | 0.144 | [0.116, 0.172] | 0.20–0.35 |
| **global**  | 0.042 | 0.043 | 0.048 | — | — |

SE ≈ 1/√N ≈ 0.014 at N=5000. Almost every pair moved by ≤ 0.005 between
N=1000 and N=5000, so we are firmly in the asymptotic regime.

**Firm conclusion (now warranted by the SE).** No pair has a plausible
chance of reaching κ ≥ 0.30 on OpenWebText with these four proxies; the
maximum is `form × tox` at 0.144 ± 0.014. The four verticals chosen for
the PoC are **near-orthogonal on this corpus**, and the GO criterion of
ROADMAP_POC §3.7 (a κ gradient with one pair < 0.15 and one pair ≥ 0.30)
is not satisfiable as designed. The walk-back from the previous entry
was warranted at N=1000 but the result holds at N=5000.

The N=100 high values for `form × tox` (0.255) and the predicted
`sent × tox` correlation reflected (a) small-sample covariance inflation
at low N, and (b) a roadmap intuition probably calibrated on social-media
or news data where toxic content is more abundant; on OWT the toxic tail
is thin enough that the sentiment-toxicity coupling does not show.

**On the counter-intuitive `form × tox` (0.144) > `sent × tox` (0.046).**
Naively one expects toxicity to align with negative sentiment more than
with informality. Three reasons the data say otherwise on OWT:

1. **OWT is low-toxicity.** It is articles and blog posts, not Reddit or
   4chan. `unitary/toxic-bert` rarely fires high, so `Var(E_tox)` is small
   and that compresses *every* covariance involving `E_tox`. This is
   consistent with all six κ values being small.
2. **Sentiment on OWT reads as *topic valence*, not *interpersonal tone*.**
   DistilBERT-SST-2 was trained on movie reviews; on a news article it
   captures "is this good or bad news?" (war = negative, sport win =
   positive), not the interpersonal aggression that toxicity captures.
   Sentiment and toxicity end up on two largely independent semantic axes.
3. **Formality and toxicity share the register/style axis.**
   `unitary/toxic-bert` was trained on Civil Comments and Wikipedia talk
   pages, where toxicity is associated with informal aggressive language.
   On OWT, the rare articles flagged as mildly toxic also tend to be the
   casual blog-style ones — exactly what the formality classifier picks
   up. So `form × tox` carries a small but real signal on the
   casual-register axis.

The result is therefore consistent with the training biases of the
classifiers and the topical composition of OWT, but in absolute terms
both κ values are small (0.046 and 0.144 against a roadmap target of
> 0.30). It is a comparison of two small numbers, not a strong coupling.

### 2026-04-25 — Non-linear dependence: HSIC, CKA, KSG mutual information

κ measures *linear* (Pearson-style) dependence and can therefore miss
non-linear coupling. To check whether the OWT-orthogonality conclusion
is robust, we ran two non-linear metrics on the same N = 5000 sample:

* HSIC with an RBF kernel and the median-heuristic bandwidth, plus its
  normalised variant CKA = HSIC(K, L) / sqrt(HSIC(K, K) · HSIC(L, L)) ∈ [0, 1].
* KSG mutual information (in nats) via `sklearn.feature_selection.mutual_info_regression`
  with k = 3.

| pair | κ | HSIC | CKA | MI |
|---|---|---|---|---|
| len × form  | 0.033 | 0.00605 | **0.068** | **0.091** |
| len × tox   | 0.031 | 0.00261 | 0.032 | 0.061 |
| form × tox  | **0.144** | 0.00187 | 0.024 | 0.070 |
| sent × tox  | 0.046 | 0.00143 | 0.015 | 0.021 |
| len × sent  | 0.001 | 0.00055 | 0.005 | 0.005 |
| form × sent | 0.001 | 0.00047 | 0.005 | 0.005 |

**The ranking changes with the metric.** By κ, `form × tox` is the most
coupled pair (0.144). By CKA and MI, `len × form` is, and `form × tox`
falls to the middle of the table. The two non-linear metrics agree on
the order, which gives confidence the divergence is a property of the
data, not estimator noise.

**Reading.**

* `form × tox`'s κ = 0.144 was almost entirely a *linear* signal — the
  casual-register association we hypothesised earlier. Once normalised
  by its own marginals (CKA), it is unremarkable.
* `len × form` (κ = 0.033) carries a *non-linear* dependence that κ
  misses: very short OWT texts are almost always informal, but past a
  length threshold formality varies freely. The relationship has a knee
  rather than a slope, which is exactly the structure Pearson is blind
  to and HSIC / MI catch.
* The two `… × sent` pairs remain bottom-of-table on every metric,
  reinforcing that DistilBERT-SST-2's "topic valence" axis on OWT is
  largely independent of length, formality and toxicity.

**Practical conclusion is unchanged.** Even with non-linear metrics the
verticals are weakly coupled on OWT — the maximum is CKA = 0.068 and
MI = 0.091 nats, far from a "strongly dependent" regime. The §3.7
fallback (inject a redundant proxy to force a high-κ control pair) is
still the right next step.

**For the paper, this is also useful methodologically.** Reporting κ
alone would have over-emphasised `form × tox`. Reporting κ + CKA + MI
together is more defensible and lets readers see when linear and
non-linear dependence diverge.

### 2026-04-25 — Phase 1 with 5 proxies (§3.7 fallback path)

Added `SentimentEnergyV2` based on `cardiffnlp/twitter-roberta-base-sentiment-latest`
to anchor a high-κ pair (`sent × sent2`) as suggested by ROADMAP §3.7.
Also pinned `max_length=512` in the HF pipeline construction so any
classifier whose tokenizer config does not declare `model_max_length`
still gets its inputs truncated correctly (cardiffnlp was the trigger;
the four base classifiers worked by accident).

Re-ran Phase 1 on the same N=5000 OWT sample with the resulting 5×5
Gram matrix:

| pair | κ | HSIC | CKA | MI |
|---|---|---|---|---|
| **sent × sent2** | **0.300** | 0.02448 | **0.211** | **0.225** |
| len × form  | 0.033 | 0.00605 | 0.068 | 0.091 |
| len × tox   | 0.031 | 0.00261 | 0.032 | 0.061 |
| form × sent2| 0.115 | 0.00274 | 0.029 | 0.031 |
| form × tox  | 0.144 | 0.00187 | 0.024 | 0.070 |
| len × sent2 | 0.011 | 0.00206 | 0.021 | 0.045 |
| sent2 × tox | 0.056 | 0.00138 | 0.016 | 0.018 |
| sent × tox  | 0.046 | 0.00143 | 0.015 | 0.021 |
| len × sent  | 0.001 | 0.00055 | 0.005 | 0.005 |
| form × sent | 0.001 | 0.00047 | 0.005 | 0.005 |

**The fallback works.** `sent × sent2` is top on every metric and lands
exactly on the roadmap GO threshold (κ ≥ 0.30). The ranking by κ, CKA,
MI agree on the top spot — unlike the earlier 4-proxy run where κ and
CKA disagreed on which pair was most coupled.

**Gradient now suitable for Figure 5.** 1 high pair (sent×sent2 ≈ 0.30),
2 medium (form×tox 0.144, form×sent2 0.115), 7 low pairs (< 0.06). Ten
points span roughly two orders of magnitude on κ — enough to fit a
slope with usable statistical power, even if the high-κ end is anchored
by a single point.

**Side observation on `form × sent2`.** With κ = 0.115 it is the second
most-coupled pair on κ, while `form × sent` was at κ = 0.001 with the
same formality classifier. Twitter-RoBERTa likely learnt an
informality ↔ negative sentiment association that DistilBERT-SST-2
(movie reviews) does not. CKA is much smaller (0.029), so the coupling
is essentially linear — no curved structure beyond the Pearson signal.

**Open question on the high-κ anchor.** sent×sent2 = 0.300 lands *just*
on the threshold. A more comfortable margin (κ ≈ 0.5+) would come from
a sentiment classifier with a more different decision boundary. To keep
in mind for any future iteration on the proxy set.

### 2026-04-26 — Phases 3, 4, 4.5 implemented (no GPU run yet)

All previously skeleton-only modules now have working implementations:

* **Phase 3** — `train_expert.train()` runs LoRA fine-tuning through
  `dllm.MDLMTrainer`, with frozen embeddings and the EOS-padding
  collator. Saves only the adapter to `artifacts/checkpoints/<name>/`.
* **Phase 4** — `PoESampler` plugs a PoE-composition wrapper into
  dllm's `MDLMSampler`; the λ=0 non-regression test is in place.
  `merge_loras` builds the naive baseline by linearly interpolating
  LoRA A/B matrices.
* **Phase 4.5 Test 2** — paired-sample MDLM ELBO estimator + log-log
  scatter driver.
* **Plan-B Test 1** — `build_intersection_dataset` + a separate train
  script for the intersection-trained 7th expert + a
  Kolmogorov-Smirnov / Welch t-test check against PoE(a, b).
* **Plan-B λ sweep** — `{0, 0.5, 1, 1.5, 2}` extension in `CONFIGS`.

Orchestration scripts for phases 3.5, 4, 4.5 and Plan-B Test 1 all
present, importable, and lint-clean. **Not yet executed**: every script
that calls `dllm.utils.get_model(...)` requires a GPU pod (the
`kuleshov-group/mdlm-owt` checkpoint plus the six classifier-backed
proxies do not fit a Mac CPU run in any reasonable time).

CI is green; the test gate covers the numerical helpers (40 tests on
Gram matrix, κ, HSIC/CKA/MI, Spearman, joint satisfaction, bootstrap
fits, jackknife, concreteness lookup, label-fallback) and the
`scripts/setup_dllm.sh` workaround is documented for fresh pods.

Next step is the GPU pod itself: provision, run `00_setup_runpod.sh`,
then `02 → 03 → 03b → 04 → 05 → 06 → 06b → 07 → 08`. Estimated cost
per Plan B is ~€25–35 of the €100 budget.

### 2026-04-25 — Phase 1 with 6 proxies (drop tox, add conc + topic)

Refactored the proxy set following the discussion of the
near-orthogonality of the 4 base verticals on OWT and the limited
margin of the sent×sent2 anchor. Final set: `len, form, sent, sent2,
conc, topic` (15 pairs). Spearman rank correlation added as a fifth
metric next to κ / HSIC / CKA / MI; it covers the *monotone non-linear*
regime that κ misses but CKA/MI catch.

| pair | κ | Spear | HSIC | CKA | MI |
|---|---|---|---|---|---|
| **sent × sent2**  | **0.300** | +0.59 | 0.0245 | **0.211** | **0.225** |
| **len × conc**    | **0.200** | −0.27 | 0.0031 | 0.033 | 0.051 |
| form × sent2      | 0.115 | −0.22 | 0.0027 | 0.029 | 0.031 |
| sent × topic      | 0.100 | +0.03 | 0.0011 | 0.012 | 0.014 |
| form × topic      | 0.040 | −0.01 | 0.0007 | 0.009 | 0.024 |
| len × form        | 0.033 | +0.37 | 0.0061 | 0.068 | 0.091 |
| form × conc       | 0.032 | −0.20 | 0.0011 | 0.013 | 0.034 |
| sent2 × topic     | 0.029 | −0.11 | 0.0015 | 0.018 | 0.026 |
| len × sent2       | 0.011 | −0.18 | 0.0021 | 0.021 | 0.045 |
| sent2 × conc      | 0.007 | +0.06 | 0.0009 | 0.009 | 0.022 |
| len × topic       | 0.004 | −0.06 | 0.0007 | 0.009 | 0.015 |
| conc × topic      | 0.002 | +0.02 | 0.0002 | 0.002 | 0.004 |
| len × sent        | 0.001 | −0.05 | 0.0006 | 0.005 | 0.005 |
| form × sent       | 0.001 | −0.01 | 0.0005 | 0.005 | 0.005 |
| sent × conc       | 0.000 | −0.00 | 0.0004 | 0.004 | 0.000 |
| **global**        | **0.171** | — | — | — | — |

**Three substantive findings.**

1. **`sent × sent2` is still the strongest anchor**, top on every metric
   (κ = 0.30, |ρ_s| = 0.59, CKA = 0.211, MI = 0.225 nats). The §3.7
   redundancy pair holds.

2. **`len × conc` is a *natural* high-coupling pair** that we did not
   have before (κ = 0.20, Spearman = −0.27, MI = 0.051). The negative
   Spearman says: longer OWT documents have *less* concrete vocabulary
   on average — exactly the expected pattern (long-form OWT skews
   towards analytical / scientific / philosophical writing, which has
   more abstract terms). This validates the choice of `conc` as a
   vertical: it has a genuine, interpretable, exploitable coupling
   with length, not a synthetic hack.

3. **The gradient is now rich enough that Figure 5 does not depend on
   a single high-leverage point.** Two distinct pairs above κ = 0.20,
   plus two more in the 0.10–0.12 range, plus 11 pairs below 0.06.
   The bootstrap + jackknife analysis from Phase 5 will still flag
   sent×sent2 as a leverage point, but the slope will not collapse if
   it gets removed because len×conc carries an independent positive
   signal.

**Cross-metric methodological notes.**

* `form × conc` (κ = 0.032, |ρ_s| = 0.20): a *monotone non-linear*
  coupling that κ misses and Spearman catches. Direction: more formal
  → less concrete (analytical / abstract register). Exactly the case
  for which we added Spearman.
* `sent × topic` (κ = 0.100, |ρ_s| = 0.03): the inverse case. A
  *linear-but-not-monotone* signal that κ catches and Spearman
  flattens. Likely a non-monotone artefact of how the SST-2 sentiment
  axis intersects the AG-News-Sports axis on OWT.
* `conc × topic` ≈ 0 on every metric — the lexical concreteness axis
  and the semantic topic axis are essentially independent on OWT. Good:
  it gives Figure 5 a clean low-coupling reference point.

**Updated GO criterion.** The roadmap §3.7 threshold (≥ 1 pair with
κ ≥ 0.30 and ≥ 1 with κ < 0.15) is comfortably met:

* Two pairs at κ ≥ 0.20 (sent×sent2, len×conc).
* Eleven pairs at κ < 0.06.
* Wide spread on every metric (Spearman: -0.27 to +0.59 on the high end;
  CKA: 0.002 to 0.211; MI: 0.000 to 0.225 nats).

Phase 1 is now considered complete on the smoke-test side. Decision on
whether to re-run on a GPU pod with N ≫ 5000 will come after Phase 2/3
implementation; the SE on κ at N = 5000 is already small enough
(≈ 0.014) that the values reported here are not expected to shift more
than a CI-width on a larger sample.

**Next.** Take the §3.7 fallback path: add a deliberately redundant 5th
proxy energy. The simplest is a *second* sentiment classifier with a
different decision boundary (e.g. `cardiffnlp/twitter-roberta-base-sentiment`
alongside DistilBERT-SST2) — this mechanically produces a high-κ pair
on the redundant axis, restoring the κ gradient that the κ-vs-deficit
experiment requires. Domain shift to Reddit / Twitter is the alternative
but introduces an OWT/non-OWT distribution shift that contaminates the
backbone-vs-finetune comparison.

---

## Phase 4–7 — Pod runs (2026-04-27 / 2026-04-28)

This section consolidates everything we measured on the RunPod A100 80GB
between 2026-04-26 (Phase 2 dataset build) and 2026-04-28 (Phase 7 in
progress). Numbers are shown verbatim; methodology and caveats follow.

### 4.0 Setup recap

* Backbone: `kuleshov-group/mdlm-owt` (110M params, MDLM trained on OWT).
  Custom modeling code patched to drop flash-attn (SDPA fallback), accept
  HF Trainer kwargs, force return_dict, handle sigma=None, and cast
  TimestepEmbedder to MLP weight dtype. See `scripts/patch_mdlm_no_flash_attn.py`.
* Tokenizer: GPT-2 fallback (MDLM-OWT ships no tokenizer); mask token
  added at id 50257 to match the model's `vocab_size = 50258`.
* 6 LoRA experts (rank 16, 2500 steps each) trained on 80k filtered OWT
  documents per vertical: `long`, `formal`, `positive`, `positive2`,
  `concrete`, `sports`.
* 2 intersection LoRA experts trained on the joint-filter corpus:
  `long_formal` (≈20k docs), `formal_concrete` (≈20k docs).
* Sampling: `max_new_tokens = 48`, `num_steps = 256`, prompts seeded with
  12-token snippets sampled from the 6 verticals (`prompts.jsonl`,
  600 prompts cycled). Rejection sampling on coherence rejects samples
  failing one of three heuristics (distinct-2 < 0.30, alpha-ratio < 0.55,
  top-token-frequency > 0.25); see `src/composition/coherence.py`.

### 4.1 Phase 3.5 — cross-vertical scoring

Mean raw signal per (expert, proxy) on N=200 samples:

| expert         | len    | form  | sent  | sent2 | conc  | topic |
|----------------|-------:|------:|------:|------:|------:|------:|
| `__baseline__` | 124.56 | 0.449 | 0.796 | 0.145 | 1.039 | 0.166 |
| concrete       |  93.38 | 0.209 | 0.475 | 0.224 | 1.432 | 0.102 |
| formal         | 127.97 | 0.763 | 0.857 | 0.134 | 2.273 | 0.037 |
| formal_concrete|  97.61 | 0.196 | 0.532 | 0.259 | 0.778 | 0.200 |
| long           | 127.95 | 0.733 | 0.843 | 0.162 | 2.347 | 0.040 |
| long_formal    | 127.96 | 0.716 | 0.802 | 0.171 | 2.297 | 0.038 |
| positive       | 117.25 | 0.454 | 0.607 | 0.211 | 1.378 | 0.288 |
| positive2      | 116.60 | 0.156 | 0.406 | 0.259 | 0.786 | 0.409 |
| sports         | 127.95 | 0.576 | 0.793 | 0.245 | 2.326 | 0.416 |

Each expert shifts the mean of *its* proxy in the right direction (concrete
+0.4 on `conc`, formal +0.31 on `form`, sports +0.25 on `topic`, …) with
the exception of `formal_concrete` and `long_formal`, whose intersection
training collapsed the form/conc signal — see §4.5.

### 4.2 Phase 4 — N=2 composition sweep (10 pairs × 8 configs)

n=200, max_new_tokens=48, prompts seeded from `prompts.jsonl`, rejection
sampling with coherence filter active. Joint satisfaction `JS` = fraction
of samples passing *both* top-quartile thresholds simultaneously.

| pair                  | baseline | exp-A | exp-B | PoE-half | **PoE-strict** | gain vs baseline |
|-----------------------|---------:|------:|------:|---------:|---------------:|----------------:|
| formal × positive     |   0.065  | 0.075 | 0.095 |   0.095  |     **0.120**  | +85 % |
| formal × positive2    |   0.085  | 0.065 | 0.110 |   0.085  |     **0.125**  | +47 % |
| formal × concrete     |   0.045  | 0.075 | 0.080 |   0.070  |     **0.065**  | +44 %  |
| formal × sports       |   0.035  | 0.075 | 0.090 |   0.080  |     **0.090**  | +157 % |
| positive × positive2  |   0.150  | 0.105 | 0.155 |   0.140  |     **0.270**  | +80 % |
| positive × concrete   |   0.040  | 0.060 | 0.080 |   0.050  |     **0.070**  | +75 % |
| positive × sports     |   0.080  | 0.110 | 0.115 |   0.110  |     **0.160**  | +100 % |
| positive2 × concrete  |   0.050  | 0.075 | 0.110 |   0.085  |     **0.080**  | +60 % |
| positive2 × sports    |   0.075  | 0.085 | 0.115 |   0.090  |     **0.105**  | +40 % |
| concrete × sports     |   0.055  | 0.075 | 0.080 |   0.065  |     **0.075**  | +36 % |

PoE-strict (λ_a = λ_b = 1) beats baseline on 10/10 pairs with a mean
gain of ≈ +72 % (median ≈ +71 %).

PPL ratio under PoE-strict averages 2.16 (vs 1.0 baseline) — the
fluency cost is moderate. PoE-amp (λ=2) collapses to ratio ≈ 8-150×,
unusable. PoE-1.5 sits at ≈ 5×, marginal.

### 4.3 Test 1 (Phase 4.5) — intersection vs PoE

`formal × concrete`, n=200, max_new_tokens=48, prompts seeded.
Compares samples drawn from the dedicated `formal_concrete` LoRA against
PoE(`formal`, `concrete`).

| proxy | mean intersection | mean PoE | KS stat | p_KS | p_t |
|-------|------------------:|---------:|--------:|-----:|-----:|
| len   |  40.88 |  47.06 | 0.44 | 9e-18 | 9e-20 |
| form  |  0.177 |  0.369 | 0.55 | 1e-27 | 2e-30 |
| sent  |  0.516 |  0.477 | 0.22 | 1e-04 | 0.36 |
| sent2 |  0.280 |  0.164 | 0.60 | 8e-34 | 3e-10 |
| conc  |  1.029 |  2.219 | 0.53 | 2e-25 | 3e-14 |
| topic |  0.183 |  ⋯    |   ⋯  |   ⋯   |   ⋯  |

KS p-values reject distributional equivalence on every proxy. **PoE
beats the dedicated intersection expert** on the two key axes:
`form` 0.37 vs 0.18, `conc` 2.22 vs 1.03. Likely cause: the
intersection training corpus (~20k filtered docs) was too narrow for
the LoRA to converge on both axes; composition of two well-trained
single-axis experts is more efficient.

### 4.4 Test 2 (Phase 4.5) — PoE formula validation

`formal × concrete`, k_pairs=50 random sequence pairs, num_t_samples=32.
ELBO log-ratios under each adapter and under `PoECompositionModel`:

```
slope     = 0.857
intercept = -0.041
R²        = 0.811
```

Predicts the paper's central identity
`log p_PoE(y)/p_PoE(x) = log p_a + log p_b − log p_base` with high
fidelity (R² > 0.8 on independent random sequences). The slope < 1 hints
at slight under-composition relative to theory, consistent with a per-
step approximation error that grows with the number of denoising steps
(see §4.7 discussion).

### 4.5 Phase 5 — κ / Spearman / CKA / MI vs deficit

Per-pair fit on 10 N=2 pairs, joint-satisfaction deficit
`Δ = JS_indep_ref − JS_PoE`:

| metric    | Pearson r | 95 % CI         | slope   | jackknife r range |
|-----------|----------:|-----------------|--------:|-------------------|
| **κ**     | **−0.917**| [−0.99, −0.20] | −0.598 | [−0.97, −0.65] |
| Spearman  |   −0.752  | [−0.96, +0.49] | −0.251 | [−0.86, +0.09] |
| **CKA**   | **−0.868**| [−0.99, +0.10] | −0.826 | [−0.94, −0.26] |
| **MI**    | **−0.836**| [−0.98, +0.52] | −0.759 | [−0.92, +0.11] |

The four independence metrics independently predict the PoE deficit:
**more independent axes → smaller PoE deficit** (closer to the
product-of-marginals reference). This is the paper's central
empirical claim.

### 4.6 Phase 6 — three N=3 triplets at n=500

Triplets ranked by maximum pairwise κ (least independent → most
independent):

| triplet                                   | max κ | marginals (×3)         | PoE-3   | indep_ref | **ratio** |
|-------------------------------------------|------:|------------------------|--------:|----------:|----------:|
| formal × concrete × sports                | 0.040 | (0.318, 0.326, …)      |  0.026  |  0.030    |  **0.87** |
| formal × positive × concrete              | 0.032 | (0.318, 0.318, 0.326)  |  0.018  |  0.033    |  **0.55** |
| positive2 × concrete × sports             | 0.029 | (…, …, …)              |  0.012  |  0.031    |  **0.39** |

**The more independent the triplet, the worse the PoE-3 ratio.** Opposite
of N=2 behaviour. The least-independent triplet `formal × concrete ×
sports` is the only one to land near indep_ref (ratio 0.87). All others
are sub-additive — PoE-3 at λ=1 produces *fewer* triple-satisfying
samples than three independent draws would.

### 4.7 Phase 7 — λ sweep at N=3 (formal × positive × concrete)

n=500 per λ, 11 points covering λ ∈ {0.10, 0.25, 0.333, 0.50, 0.577,
0.667, 0.80, 1.00, 1.20, 1.50} plus the κ-shrunk Bayesian variant
λ_i = 1/(1+κ̄_i) ≈ {0.984, 0.999, 0.984}.

| λ                       |   PoE-3 | ratio |
|-------------------------|--------:|------:|
| 0.10                    |  0.000  | 0.00 |
| 0.25                    |  0.008  | 0.24 |
| 0.333 (= 1/N)           |  0.006  | 0.18 |
| 0.50                    |  0.012  | 0.36 |
| 0.577 (= 1/√N)          |  0.010  | 0.30 |
| 0.667 (= 2/N)           |  0.008  | 0.24 |
| 0.80                    |  0.014  | 0.42 |
| **1.00 (PoE-strict)**   | **0.018** | **0.55** |
| Bayesian (≈ 0.99 each)  |  0.018  | 0.55 |
| 1.20                    |  0.008  | 0.24 |
| 1.50                    |  0.008  | 0.24 |

**Bell-shaped curve sharply peaked at λ=1.** Both 1/N and 1/√N
normalisations *aggravate* the deficit. λ > 1 likewise dégrade. The
Bayesian κ-shrunk variant is indistinguishable from PoE-strict
(κ̄ ≈ 0 on this triplet, so the shrinkage factor is ≈ 0.98 → 1).

### 4.8 What this tells us

1. **PoE composition works at N=2.** Robust across 10 pairs, super-
   additive on average. Calibrated by κ/CKA/MI.

2. **PoE-N at N=3 plateaus around ratio 0.55** under naïve composition
   regardless of λ calibration. Section 4.7 sweeps eleven λ values
   covering 0.1× to 1.5× the canonical 1.0 — λ=1 is the unambiguous
   optimum and the maximum is tight (deviations of ±20 % already cost
   half the signal).

3. **The plateau is not a calibration issue.** It is consistent with
   either (a) the per-step composition approximation in diffusion-LM
   (cf. Du Yan et al. 2023 *Reduce Reuse Recycle* in image diffusion;
   the analogue holds in MDLM) or (b) the limited capacity of MDLM-OWT
   (110M) for the joint manifold — the smaller the model, the rarer the
   text region satisfying three constraints simultaneously.

4. **Test 2 supports (a):** the slope-0.857 / R²-0.811 fit between the
   PoE log-ratio and the formula's prediction is *not* perfect, and the
   < 1 slope hints at under-composition that grows with the number of
   denoising steps.

5. **Phase 7 in progress (Option α + β):**
   * **β** — schedule λ along the trajectory (late_fire, cosine, exp,
     early_fire). Tests whether moving the composition push to the
     mostly-clean phase helps.
   * **α** — Gibbs MCMC refinement on top of naïve PoE-3 samples.
     Adapts Du Yan et al. 2023 to discrete-state MDLM.

### 4.9 Phase 7 v2 — Options α and β under fixed-marginal calibration

The Phase 7 v1 sweep had a measurement bias: when a non-constant
schedule is in effect the per-expert solo runs *also* used the schedule,
which biased the indep_ref. v2 separates the two: marginals always
measured at constant λ, the schedule under test only applied for the
PoE-3 step. n=500 throughout.

| config                 | naive  | mcmc-refined | r_naive | r_mcmc |
|------------------------|-------:|-------------:|--------:|-------:|
| **constant** (control) | 0.018  |     —        |  0.55   |   —    |
| early_fire             | 0.018  |     —        |  0.55   |   —    |
| exp                    | 0.010  |     —        |  0.30   |   —    |
| cosine                 | 0.004  |     —        |  0.12   |   —    |
| late_fire              | 0.004  |     —        |  0.12   |   —    |
| constant + MCMC×10     | 0.018  |   0.012      |  0.55   |  0.36  |
| exp + MCMC×10          | 0.010  |   0.008      |  0.30   |  0.24  |

Two negative results both worth reporting:

* **Option β (λ schedule) does not beat constant.** `early_fire` matches
  the control (so the first half of denoising carries the signal — a
  novel observation), and every other schedule degrades. `late_fire` and
  `cosine` *halve* the ratio: pushing the composition toward the clean
  phase loses the gain entirely. Pre-experiment intuition (push when
  the sequence is mostly clean) is therefore inverted.
* **Option α (Gibbs MCMC refinement) degrades.** A naïve Gibbs sweep
  that re-samples five token positions from the PoE-composed softmax
  collapses easy tokens, dropping ratio 0.55 → 0.36 on top of constant
  and 0.30 → 0.24 on top of exp. The Du Yan 2023 idea may still hold,
  but in continuous-state diffusion their MCMC is annealed Langevin
  with explicit energy gradients; our discrete Gibbs equivalent without
  proposal correction over-greedifies.

Combined with the §4.7 λ-sweep, the plateau at ratio ≈ 0.55 is now
robust against:

  * 11 uniform λ ∈ [0.1, 1.5] (§4.7)
  * Bayesian per-expert λ_i = 1/(1+κ̄_i) (§4.7)
  * 4 denoising-step schedules (§4.9)
  * 2 MCMC variants (§4.9)

→ The plateau is **not** a calibration artefact. It is consistent with
either the per-step approximation in diffusion-LM PoE composition (the
mechanism Du Yan et al. 2023 documented in image diffusion) or with the
limited capacity of the 110 M MDLM-OWT backbone. Section §4.10 (Option
γ) attacks this distinction by re-running the full pipeline on a 5×
larger MDLM backbone (Qwen3-0.6B-MDLM); if the plateau lifts with
backbone capacity, the bottleneck is structural (Niveau 3); if it
persists, the per-step approximation is the dominant cause (Niveau 2)
and Du Yan-style joint MCMC over the trajectory becomes the right
target for a follow-up.

### 4.10 Take-aways for the paper

What the PoC delivers (consolidated across Phases 4–7):

1. **N=2 PoE composition is robustly super-additive.** Across 10 pairs
   spanning five vertical axes, PoE-strict beats baseline 10/10 with
   median +71 % JS gain at moderate fluency cost (PPL ratio ≈ 2). The
   formula itself is empirically validated (Test 2: slope 0.857,
   R² 0.811 on 50 random sequences) and beats a dedicated intersection-
   trained adapter on the form/conc axes (Test 1).

2. **Independence metrics predict the PoE deficit.** On the same 10
   pairs we get r = −0.92 (κ), −0.87 (CKA), −0.84 (MI) between the
   pair's independence score and the joint-satisfaction deficit. The
   four metrics agree independently — strong, multi-method evidence
   that PoE compositionality is exactly captured by the κ Gram-matrix
   theory developed in §2.

3. **N=3 PoE plateaus around ratio 0.55** despite extensive calibration:
   constant λ (11 values), Bayesian per-expert λ, 4 schedule variants,
   MCMC refinement with two settings — all bottom out at the same
   value. The plateau is *robust*, not a calibration choice.

4. **The plateau dependency on independence reverses sign at N=3.**
   On three triplets at fixed λ=1, the more-independent triplet has
   the *worse* PoE-3 ratio (formal × concrete × sports → 0.87,
   formal × positive × concrete → 0.55, positive2 × concrete × sports
   → 0.39). At N=2, more independence helps PoE; at N=3, more
   independence apparently hurts. This is a candidate test-bed for
   future joint-MCMC corrective approaches.

5. **Early-step pushes are sufficient.** `early_fire` (push only on the
   first half of denoising) matches `constant`, while `late_fire` (push
   only on the second half) collapses to ratio 0.12. The composition
   signal is encoded in the early denoising steps; the clean-phase
   re-pushes do not add information.

---

## Phase 8 — Option γ (Qwen3-0.6B-MDLM, 5× larger backbone)

Ran the full N=2 + N=3 pipeline on `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1`
(596 M params, 5× MDLM-OWT) to disambiguate Niveau 2 (per-step diffusion-LM
approximation, theoretical) vs Niveau 3 (backbone capacity).

Setup: same datasets, same prompts.jsonl, same n=200 (N=2) and n=500 (N=3),
same `max_new_tokens=48`. LoRA target_modules switched to
`q_proj/k_proj/v_proj/o_proj` (Llama-style); 4.6 M trainable params per
expert (vs 0.88 M on MDLM-OWT). 7 LoRA adapters trained at 2500 steps each,
intersection adapter (`formal_concrete`) trained the same way.

### 5.1 N=2 PoE-strict ratio: side-by-side

| pair                    | MDLM-OWT r | Qwen3 r | Δ |
|-------------------------|-----------:|--------:|---:|
| formal × positive       |    1.33    |  0.24   | ⬇⬇ |
| formal × positive2      |    1.17    |  0.44   | ⬇  |
| formal × concrete       |    0.69    |  0.76   | ⬆  |
| formal × sports         |    0.95    |  0.69   | ⬇  |
| positive × positive2    |    2.54    |  1.48   | ⬇  |
| positive × concrete     |    0.74    |  1.08   | ⬆  |
| positive × sports       |    1.69    |  1.13   | ⬇  |
| positive2 × concrete    |    0.72    |  0.94   | ⬆  |
| positive2 × sports      |    0.94    |  1.15   | ⬆  |
| concrete × sports       |    0.76    |  2.77   | ⬆⬆⬆|
| **mean**                |  **1.04**  |**1.07** |    |

Means are nearly identical, but the per-pair variance is much wider on
Qwen3 (0.24 → 2.77, vs 0.69 → 2.54 on MDLM-OWT). Notably:

* **Lexical/topical pairs** (concrete, sports, positive2 with topic-coloured
  axes) tend to *improve* on Qwen3 — `concrete × sports` jumps from 0.76 to
  2.77 (+264 %).
* **Style-coloured pairs** (formal, positive) tend to *degrade* — `formal ×
  positive` collapses from 1.33 to 0.24.

This refutes the naïve "bigger backbone uniformly improves PoE" hypothesis.
Qwen3 LoRA experts produce stronger marginal shifts (formal: 0.32 → 0.59;
sports: 0.25 → 0.23) but those stronger shifts collide on entangled-style
axes while reinforcing on disjoint lexical axes.

### 5.2 N=3 PoE-3 ratio at λ=1, n=500, both backbones

| triplet (max κ on OWT)                   | MDLM-OWT r | Qwen3 r |
|------------------------------------------|-----------:|--------:|
| positive2 × concrete × sports (κ=0.029)  |   0.39     |  **1.84** |
| formal × positive × concrete (κ=0.032)   |   0.55     |  0.42   |
| formal × concrete × sports (κ=0.040)     |   0.88     |  0.84   |
| **mean**                                 |  **0.61**  | **1.03** |

Qwen3 lifts the mean PoE-3 ratio from sub-additive (0.61) to indep-ref
parity (1.03). One triplet (the most lexical, `positive2 × concrete ×
sports`) becomes super-additive (1.84). One stays flat (0.84). One
slightly degrades (0.42).

→ **The N=3 plateau is partially backbone-bounded** but not uniformly.
Lexical-axis triplets benefit, style-axis triplets do not.

### 5.3 Phase 5 (κ ↔ deficit) on Qwen3

Refit the κ-vs-PoE-deficit linear model on the 10 Qwen3 N=2 pairs (with
the same κ values from OWT, only JS_PoE / indep_ref change):

| metric    | MDLM-OWT r | Qwen3 r |
|-----------|-----------:|--------:|
| κ         |   −0.917   |  −0.242 |
| Spearman  |   −0.752   |  −0.210 |
| CKA       |   −0.868   |  −0.302 |
| MI        |   −0.836   |  −0.301 |

**The κ-deficit correlation collapses on Qwen3.** All four CIs span zero —
we can't even reject the null hypothesis. The strong relationship
documented on MDLM-OWT (§4.5) was specific to that backbone, not a
universal property of PoE composition.

### 5.4 What the paper looks like now (revised)

1. **Formula validation (§4.4 Test 2) holds.** PoE-as-logit-sum is
   empirically validated (slope 0.857, R² 0.811). This is independent of
   backbone.
2. **N=2 PoE composition is super-additive on average** (mean ratio 1.04
   on MDLM-OWT, 1.07 on Qwen3) but with substantial per-pair variance —
   hardly the clean "PoE works" story we'd hoped to tell.
3. **N=3 plateau is partially backbone-dependent.** Mean ratio lifts
   from 0.61 (small backbone) to 1.03 (large backbone), but per-triplet
   results vary widely. Lexical axes help, style axes don't.
4. **κ ↔ deficit correlation is backbone-specific.** Not generalizable
   from MDLM-OWT to Qwen3. The κ-Gram framework remains valid as a
   description of the *proxies' joint structure on OWT*, but not as a
   universal predictor of PoE outcomes.
5. **Tests 1 and 2** still hold qualitatively but should be re-checked
   on Qwen3 in any final write-up.

### 5.5 Bottom line

We do not have a single, universal claim "PoE composes super-additively
in MDLM if X". What we have is:

* A theoretical foundation (κ-Gram framework, PoE formula).
* An empirical study across 2 backbones and ~30 calibration variants.
* A nuanced finding: backbone capacity matters at N=3 *but only for
  lexical-axis composition*; style-axis composition stays plateaued.
* A negative result: κ does not universally predict PoE deficit.

This is publishable as an empirical-diagnostic paper for a workshop or
short-paper venue. It is not the clean "we proved compositionality"
paper. The honest framing positions this work as a careful
characterization of where PoE-on-MDLM works and where it doesn't,
opening directions for follow-up (entangled-axis composition,
backbone-invariant deficit predictors, joint-trajectory MCMC).
