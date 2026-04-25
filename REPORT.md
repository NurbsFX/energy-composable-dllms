# REPORT.md — PoC results

> Running log of experimental results, updated as each phase produces
> artefacts. Final Sunday-evening write-up lives in the numbered sections
> below; the **Worklog** at the bottom captures the chronological progress.

## 1. Central claim

_TODO — one paragraph, the thesis in a single breath._

## 2. Formal recap

See [THEORY.md](THEORY.md) for definitions, the ELBO bound used to evaluate
`log p_θ`, and the PoE composition rule.

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
