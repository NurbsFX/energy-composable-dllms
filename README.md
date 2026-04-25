# Composable DLLMs via Energy-Based Product of Experts

Empirical study of energy orthogonality as a predictor of Product-of-Experts
composition quality in masked diffusion language models.

## Claims

1. **Composition exists.** Two MDLMs LoRA-fine-tuned independently on two
   verticals can be composed at inference via PoE on the logits, and the
   sampler obeys
   `log p_PoE = log p_1 + log p_2 − log p_base`.
2. **Composition extends to N = 3 experts.**
3. **Quality is predicted by κ.** Energy orthogonality measured *before*
   training (Gram matrix of proxy energies on `p_base`) correlates with
   composition quality measured *after* (joint constraint satisfaction).

Results and figures: [REPORT.md](REPORT.md).

## Layout

- [src/energies/](src/energies/) — proxy energies (length, formality,
  sentiment, toxicity), Gram matrix and orthogonality index κ, plotting.
- [src/training/](src/training/) — LoRA expert fine-tuning loop on top of
  `dllm.MDLMTrainer`.
- [src/composition/](src/composition/) — Product-of-Experts sampler and the
  naive LoRA-merge baseline.
- [src/eval/](src/eval/) — joint satisfaction, direct PoE formula check,
  κ-vs-deficit fit and plot.
- [scripts/](scripts/) — numbered entry points, one per experimental phase.
- [tests/](tests/) — unit tests for the numerical helpers.
- [artifacts/](artifacts/) — produced data; large files gitignored, plots
  and small JSON metrics are kept under version control.

## Dependence metrics

The orthogonality assumption underlying the project is operationalised
by four complementary dependence metrics on the `(N, k)` matrix of
proxy energies sampled from `p_base`. We use all four because no single
metric captures everything.

| metric | formula | range | captures | code |
|---|---|---|---|---|
| **κ** | `‖G − diag(G)‖_F / Tr(G)` on the empirical covariance `G` | `[0, ∞)` | linear (Pearson) | [gram_matrix.py](src/energies/gram_matrix.py) |
| **HSIC** | `(1/(N−1)²) · Tr(K_c L_c)` with RBF kernels and the median heuristic for σ | `[0, ∞)` | linear + non-linear | [independence.py](src/energies/independence.py) |
| **CKA** | `HSIC(X,Y) / √(HSIC(X,X)·HSIC(Y,Y))` | `[0, 1]` | linear + non-linear, normalized | [independence.py](src/energies/independence.py) |
| **MI** | KSG k-NN estimator of `I(X;Y) = ∫∫ p(x,y) log(p(x,y)/(p(x)p(y))) dxdy` | `[0, ∞)` nats | any dependence, info-theoretic | [independence.py](src/energies/independence.py) |

In words:

- **κ** is the cheapest and the only one that has a closed-form, but it
  measures *only* linear (Pearson-style) covariance. Two variables can
  have κ ≈ 0 and still be deterministically related (`Y = X²` with
  centered `X` is the canonical example).
- **HSIC** lifts the variables into a reproducing-kernel Hilbert space
  before measuring covariance, so it sees non-linear coupling that κ
  misses. It vanishes iff the variables are independent in the kernel
  limit. Raw HSIC values are scale-dependent.
- **CKA** is HSIC normalized by its self-similarity terms; it lies in
  `[0, 1]` and is therefore directly comparable across pairs with
  different marginals. **This is the right metric for ranking pairs.**
- **MI** is the information-theoretic ground truth (`I(X;Y) = 0` iff
  the variables are strictly independent), estimated via k-nearest
  neighbours. Unbounded and slightly biased, but a useful cross-check
  on the kernel-based metrics.

For the central κ-vs-deficit experiment (Figure 5), κ is the variable
of theoretical interest because it ties back to the EBM Gram matrix.
But the experimental run also reports CKA and MI so that linear vs
non-linear effects can be disentangled.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Pre-training Gram matrix | [`01_compute_gram.py`](scripts/01_compute_gram.py) | `artifacts/gram_matrix.json`, heatmap |
| 2. Vertical sub-corpora | [`02_build_datasets.py`](scripts/02_build_datasets.py) | `artifacts/datasets/*.jsonl` |
| 3. LoRA expert fine-tuning | [`03_train_experts.py`](scripts/03_train_experts.py) | `artifacts/checkpoints/*` |
| 3.5. Cross-vertical validation | [`04_validate_experts.py`](scripts/04_validate_experts.py) | `artifacts/expert_validation.json` |
| 4. Composition sweep | [`05_run_composition.py`](scripts/05_run_composition.py) | `artifacts/joint_satisfaction.json` |
| 4.5a. Direct formula check | [`06_poe_formula_check.py`](scripts/06_poe_formula_check.py) | `artifacts/poe_formula_check.json` |
| 4.5b. N = 3 extension | [`07_n3_extension.py`](scripts/07_n3_extension.py) | `artifacts/n3_results.json` |
| 5. Final figure | [`08_final_plots.py`](scripts/08_final_plots.py) | `artifacts/plots/kappa_vs_deficit.png` |

## Setup

```bash
pip install -e ".[dev]"
./scripts/setup_dllm.sh   # workaround for an upstream packaging bug; see the script header
pre-commit install        # ruff lint + format on every commit
pytest
```

For a fresh GPU pod, [`scripts/00_setup_runpod.sh`](scripts/00_setup_runpod.sh)
provisions the environment, logs in to W&B and HuggingFace, and pre-fetches
the model weights used by the scripts.

## CI

GitHub Actions runs ruff (lint and format check) and the test suite on every
push and pull request to `main` — see
[.github/workflows/ci.yml](.github/workflows/ci.yml). The same checks run
locally on each commit via the [pre-commit](.pre-commit-config.yaml) hooks.

## License

MIT — see [LICENSE](LICENSE).

## Repo changelog

Running log of structural / dependency / convention changes. Per-phase
experimental results live in [REPORT.md](REPORT.md).

### 2026-04-26 — Phase 3 / 4 / 4.5 implementation + Plan-B Test 1

End-to-end implementations of all the previously skeleton-only paths so
the next step is purely "run on a GPU pod":

- [src/training/train_expert.py](src/training/train_expert.py): LoRA
  fine-tuning loop on top of `dllm.MDLMTrainer`, with frozen embeddings
  and the EOS-padding collator.
- [src/composition/poe_sampler.py](src/composition/poe_sampler.py):
  PoESampler reuses dllm's MDLMSampler with a small wrapper that returns
  PoE-composed logits (`base + Σ λ_i · (expert_i − base)`); ships the
  λ=0 non-regression test.
- [src/composition/baselines.py](src/composition/baselines.py):
  `merge_loras` builds a third adapter from a linear combo of LoRA A/B
  matrices for the naive-merge baseline.
- [src/eval/poe_formula_check.py](src/eval/poe_formula_check.py):
  paired-sample MDLM ELBO log-ratio estimator + `check_poe_formula`
  driver for Phase 4.5 Test 2.
- [src/eval/scoring.py](src/eval/scoring.py): `SampleScorer` runs the
  six proxies + GPT-2 PPL + distinct-2 in one call. SampleRecord is now
  schema-agnostic (proxy_scores is a dict, summarize() looks up by key).
- [src/data/build_datasets.py](src/data/build_datasets.py):
  `build_intersection_dataset` writes documents that pass *both*
  vertical filters; used by Plan-B Test 1.

Orchestration scripts (numbered to match the roadmap phases):
[`04_validate_experts.py`](scripts/04_validate_experts.py),
[`05_run_composition.py`](scripts/05_run_composition.py) (with the
Plan-B-extended λ sweep `{0, 0.5, 1, 1.5, 2}`),
[`06_poe_formula_check.py`](scripts/06_poe_formula_check.py),
[`07_n3_extension.py`](scripts/07_n3_extension.py),
[`03b_train_intersection_expert.py`](scripts/03b_train_intersection_expert.py)
and [`06b_test1_intersection_check.py`](scripts/06b_test1_intersection_check.py).

Tooling:
- [scripts/setup_dllm.sh](scripts/setup_dllm.sh) works around an
  upstream packaging bug in ZHZisZZ/dllm
  (`packages = ["dllm"]` drops every subpackage) and trims the eager
  imports of the optional pipelines (RL via trl, eval via lm-eval) we
  do not use. `00_setup_runpod.sh` chains it after `pip install`.
- CI install line gets `pandas` (Brysbaert TSV/XLSX loader) and
  `test_length_energy_zero_at_target` is now `pytest.importorskip` on
  `transformers` so the gate stays light.

### 2026-04-25 — Non-linear dependence metrics (HSIC, CKA, MI)

- New module [src/energies/independence.py](src/energies/independence.py)
  implementing HSIC with the median heuristic, the normalised CKA
  (HSIC / √(HSIC·HSIC)) and the KSG mutual-information estimator via
  scikit-learn.
- New script [scripts/01b_independence_metrics.py](scripts/01b_independence_metrics.py)
  reads the cached `E_matrix.npy` and prints κ / HSIC / CKA / MI
  side-by-side per pair.
- Patched [scripts/01_compute_gram.py](scripts/01_compute_gram.py) to dump
  `E_matrix.npy` alongside the JSON so non-linear metrics can be
  re-computed without re-running the proxy classifiers.
- Added 8 unit tests in [tests/test_independence.py](tests/test_independence.py),
  including the canonical `Y = X²` non-linear-but-Pearson-zero case.
- `.gitignore` now ignores `artifacts/*.npy` (binary intermediate).
- Full metric definitions added to [README.md](README.md) (this section
  above) and [REPORT.md](REPORT.md) §2.

### 2026-04-25 — Dependency bounds relaxed

- The `dllm` upstream pins `transformers==4.57.0`. Our previous
  `transformers>=4.40.0,<4.50.0` made resolution impossible. Dropped the
  upper bound on both `transformers` and `torch` in
  [pyproject.toml](pyproject.toml) and [requirements.txt](requirements.txt);
  let `dllm`'s pin drive the version. Caught locally before any GPU time.

### 2026-04-25 — CI/CD

- GitHub Actions workflow at [.github/workflows/ci.yml](.github/workflows/ci.yml):
  ruff lint + format check + pytest on every push and PR to `main`.
- Pre-commit hooks at [.pre-commit-config.yaml](.pre-commit-config.yaml):
  ruff (lint + format), trailing-whitespace, end-of-file-fixer,
  check-yaml/toml, check-merge-conflict, check-added-large-files.
- Added `pre-commit` to the `dev` extra and to `requirements.txt`.
- One-shot `ruff format` pass over the codebase so the format check is
  green on the first CI run.

### 2026-04-25 — Cleanup pass for public release

- Stripped local-doc cross-references (`ROADMAP_POC §X.Y`) from module
  docstrings; the file is gitignored and absent from the public repo.
- Tightened module and function docstrings; dropped `TODO(weekend)` markers
  in favour of plain `NotImplementedError` for unimplemented phases.
- Dropped unused dependencies (`rich`, `pyyaml`, `jsonlines`).
- Added [LICENSE](LICENSE) (MIT) and a [tests/](tests/) suite covering the
  numerical helpers (Gram matrix and κ, joint satisfaction, κ-vs-deficit
  fit).
- Extended [.gitignore](.gitignore) to cover AI-tooling traces
  (`.aider*`, `.cursor/`, `.windsurf*`, `.continue/`, project-local
  `.claude/`), tooling caches (`.ruff_cache/`, `.mypy_cache/`) and stray
  runtime files (`*.log`, `nohup.out`, `tmp/`, `scratch/`, `outputs/`).

### 2026-04-25 — Initial scaffolding

- Created `src/{data,energies,training,composition,eval}/`, numbered entry
  points in `scripts/`, `artifacts/{plots,datasets,checkpoints,samples}/`
  with `.gitkeep` placeholders.
- Pinned dependency stack: torch 2.1+, transformers 4.40+, peft, accelerate,
  datasets, `dllm` (git+ZHZisZZ), numpy/scipy/pandas/sklearn/einops,
  matplotlib/seaborn, wandb, typer.
- Implemented (no `NotImplementedError` left): proxy energies, Gram matrix
  + κ, plotting, joint-satisfaction aggregation, κ-vs-deficit fit and plot,
  scripts `01_compute_gram.py` and `08_final_plots.py`.
- Skeletons for the remaining phases: dataset builder, expert trainer, PoE
  sampler, naive-LoRA-merge baseline, ELBO-based formula check, scripts
  `02–07`.
