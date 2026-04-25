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
pre-commit install   # ruff lint + format on every commit
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
