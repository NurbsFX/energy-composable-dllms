"""Phase 12 — Auto-tune protocols for μ*.

Three protocols compared on the 6 setups for which we have known μ*
(from Phase 11 sweeps at n=200):

* **A — Quick grid sweep** at n=50 over a fixed set of μ candidates,
  pick the argmax ratio.
* **B — Bayesian optimization** with a Gaussian Process surrogate +
  Expected Improvement. Same n=50 per evaluation, K=4 iterations
  (3 BO + 1 initial point).
* **C — Structural predictor** (handled separately, in
  ``scripts/16_predict_mu.py``; computes features locally without
  any model forward).

We measure how often each protocol identifies the same μ* as the
expensive n=200 sweep, and how close the chosen ratio gets to the
ground truth optimum.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(add_completion=False)

EXPERT_TO_PROXY = {
    "long": "len",
    "formal": "form",
    "positive": "sent",
    "positive2": "sent2",
    "concrete": "conc",
    "sports": "topic",
}

# Default candidate μ values for protocol A
DEFAULT_CANDIDATES = "-2,-1,-0.5,0"


def _load_prompts(prompts_file: Path, n_samples: int, prompt_token_len: int, tokenizer):
    pool = [json.loads(line)["text"] for line in prompts_file.open() if line.strip()]
    if not pool:
        raise typer.BadParameter(f"empty prompts file: {prompts_file}")
    out = []
    for i in range(n_samples):
        ids = tokenizer.encode(pool[i % len(pool)], add_special_tokens=False)[:prompt_token_len]
        out.append(tokenizer.decode(ids, skip_special_tokens=True))
    return out


def _triple_sat(scored, triplet_keys, thresholds):
    return float(
        sum(all(r.proxy_scores[k] >= thresholds[k] for k in triplet_keys) for r in scored)
        / max(1, len(scored))
    )


def _sample_and_score(
    model, tokenizer, scorer, prompts, triplet, mu, num_steps, max_new_tokens, seed
):
    """Sample n prompts under PoE-N with given mu, return triple_sat-ready scored list."""
    import torch

    from src.composition.poe_sampler import PoEConfig, PoESampler

    cfg = PoEConfig(
        num_steps=num_steps,
        max_new_tokens=max_new_tokens,
        seed=seed,
        mu_base=None if mu is None else float(mu),
    )
    poe = PoESampler(model, tokenizer, cfg=cfg)
    torch.manual_seed(seed)
    texts = poe.sample(prompts, lambdas={n: 1.0 for n in triplet})
    return [scorer.score(s) for s in texts]


def _setup(
    triplet,
    backbone,
    checkpoints_dir,
    n_samples,
    prompts_file,
    prompt_token_len,
    num_steps,
    max_new_tokens,
    seed,
):
    """Load model + adapters, sample baseline + solo experts, return (model, tokenizer,
    scorer, prompts, thresholds, marginals, indep_ref, triplet_keys)."""
    import dllm
    import torch
    from peft import PeftModel

    from src.composition.poe_sampler import PoEConfig, PoESampler
    from src.energies import build_default_energies
    from src.eval.joint_satisfaction import compute_thresholds
    from src.eval.scoring import SampleScorer

    typer.echo(f"  loading {backbone} + adapters {triplet}...")
    model_args = dllm.utils.ModelArguments(model_name_or_path=backbone)
    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    for i, name in enumerate(triplet):
        path = checkpoints_dir / name
        if i == 0:
            model = PeftModel.from_pretrained(model, path, adapter_name=name)
        else:
            model.load_adapter(path, adapter_name=name)

    energies = build_default_energies()
    scorer = SampleScorer(energies=energies)
    energy_for = {n: EXPERT_TO_PROXY[n] for n in triplet}
    triplet_keys = [energy_for[n] for n in triplet]

    base_cfg = PoEConfig(num_steps=num_steps, max_new_tokens=max_new_tokens, seed=seed)
    poe_base = PoESampler(model, tokenizer, cfg=base_cfg)
    prompts = _load_prompts(prompts_file, n_samples, prompt_token_len, tokenizer)

    typer.echo("  baseline...")
    torch.manual_seed(seed)
    base_scored = [scorer.score(s) for s in poe_base.sample(prompts, lambdas={})]
    thresholds = compute_thresholds({k: [r.proxy_scores[k] for r in base_scored] for k in energies})

    marginals = {}
    for name in triplet:
        typer.echo(f"  solo {name}...")
        torch.manual_seed(seed)
        scored = [scorer.score(s) for s in poe_base.sample(prompts, lambdas={name: 1.0})]
        ek = energy_for[name]
        marginals[name] = float(
            sum(r.proxy_scores[ek] >= thresholds[ek] for r in scored) / max(1, len(scored))
        )

    indep_ref = 1.0
    for n in triplet:
        indep_ref *= marginals[n]

    return model, tokenizer, scorer, prompts, thresholds, marginals, float(indep_ref), triplet_keys


def protocol_A_quick_sweep(setup_args, candidates: list[float]) -> tuple[float, dict[float, float]]:
    """Evaluate every μ in candidates, return (best_mu, mu→ratio map)."""
    model, tokenizer, scorer, prompts, thresholds, _, indep_ref, triplet_keys = setup_args[:-1]
    triplet = setup_args[-1]  # last is triplet
    ratios = {}
    for mu in candidates:
        scored = _sample_and_score(
            model,
            tokenizer,
            scorer,
            prompts,
            triplet,
            mu,
            num_steps=256,
            max_new_tokens=48,
            seed=42,
        )
        ts = _triple_sat(scored, triplet_keys, thresholds)
        ratios[float(mu)] = ts / max(1e-9, indep_ref)
    best_mu = max(ratios, key=ratios.get)
    return best_mu, ratios


def protocol_B_bayes_opt(
    setup_args, search_range=(-2.5, 0.5), n_iters: int = 4
) -> tuple[float, list[tuple[float, float]]]:
    """Bayesian optimization over μ ∈ search_range using GP + EI.

    Starts at the canonical μ = 1−N (computed from the triplet length),
    then proposes 3 additional points via Expected Improvement on a GP.
    Returns (best_mu, history of (mu, ratio)).
    """
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.gaussian_process.kernels import ConstantKernel as C

    model, tokenizer, scorer, prompts, thresholds, _, indep_ref, triplet_keys = setup_args[:-1]
    triplet = setup_args[-1]
    N = len(triplet)
    canonical = 1 - N

    history: list[tuple[float, float]] = []

    def evaluate(mu):
        scored = _sample_and_score(
            model,
            tokenizer,
            scorer,
            prompts,
            triplet,
            mu,
            num_steps=256,
            max_new_tokens=48,
            seed=42,
        )
        ts = _triple_sat(scored, triplet_keys, thresholds)
        ratio = ts / max(1e-9, indep_ref)
        history.append((float(mu), float(ratio)))
        typer.echo(f"    BO eval: μ={mu:+.3f} → ratio={ratio:.3f}")
        return ratio

    # Seed point: canonical
    evaluate(canonical)

    # Add a contrasting point to give the GP some signal
    evaluate(0.0)

    # Iterative BO
    for _it in range(n_iters - 2):
        X = np.array([[m] for m, _ in history])
        y = np.array([r for _, r in history])
        kernel = C(1.0, (1e-3, 1e3)) * RBF(0.5, (1e-2, 10.0))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=4, alpha=0.05)
        try:
            gp.fit(X, y)
        except Exception as e:
            typer.echo(f"    GP fit failed: {e}; falling back to argmax of observed", err=True)
            break
        # Expected Improvement over a fine grid
        grid = np.linspace(search_range[0], search_range[1], 60).reshape(-1, 1)
        mu_pred, sigma = gp.predict(grid, return_std=True)
        f_best = max(y)
        with np.errstate(divide="ignore", invalid="ignore"):
            imp = mu_pred - f_best
            Z = imp / np.maximum(sigma, 1e-9)
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma < 1e-9] = 0.0
        next_mu = float(grid[np.argmax(ei), 0])
        evaluate(next_mu)

    best_mu, best_ratio = max(history, key=lambda t: t[1])
    typer.echo(f"  BO best: μ={best_mu:+.3f} ratio={best_ratio:.3f} after {len(history)} evals")
    return best_mu, history


@app.command()
def main(
    triplet: list[str] = typer.Option(["formal", "positive", "concrete"]),
    backbone: str = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
    checkpoints_dir: Path = Path("artifacts/checkpoints_qwen3"),
    n_samples: int = 50,
    max_new_tokens: int = 48,
    num_steps: int = 256,
    prompts_file: Path = Path("artifacts/datasets/prompts.jsonl"),
    prompt_token_len: int = 12,
    seed: int = 42,
    candidates: str = DEFAULT_CANDIDATES,
    bo_iters: int = 4,
    out_json: Path = Path("artifacts/auto_tune_mu.json"),
) -> None:
    if len(triplet) not in (2, 3):
        raise typer.BadParameter("need 2 or 3 expert names")

    setup_args = _setup(
        triplet,
        backbone,
        checkpoints_dir,
        n_samples,
        prompts_file,
        prompt_token_len,
        num_steps,
        max_new_tokens,
        seed,
    ) + (triplet,)

    typer.echo(f"\n=== A: quick grid sweep, candidates {candidates} ===")
    cands = [float(s) for s in candidates.split(",")]
    a_mu, a_ratios = protocol_A_quick_sweep(setup_args, cands)
    typer.echo(f"  A → best μ = {a_mu:+.2f}  (ratio={a_ratios[a_mu]:.3f})")

    typer.echo(f"\n=== B: Bayesian optimization, {bo_iters} evals ===")
    b_mu, b_history = protocol_B_bayes_opt(setup_args, n_iters=bo_iters)

    out = {
        "triplet": list(triplet),
        "backbone": backbone,
        "n_samples": n_samples,
        "indep_ref": setup_args[-3],
        "marginals": setup_args[-4],
        "protocol_A": {
            "candidates": cands,
            "ratios": a_ratios,
            "best_mu": a_mu,
            "best_ratio": a_ratios[a_mu],
            "n_evals": len(cands),
        },
        "protocol_B": {
            "n_iters": bo_iters,
            "history": b_history,
            "best_mu": b_mu,
            "best_ratio": max(r for _, r in b_history),
            "n_evals": len(b_history),
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    typer.echo(f"\nWrote {out_json}")


if __name__ == "__main__":
    app()
