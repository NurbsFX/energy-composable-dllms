#!/usr/bin/env python
"""Phase 7: advanced PoE-3 composition variants.

Two sub-experiments aimed at lifting the PoE-3 plateau (ratio ≈ 0.55) we
observed under naïve composition with constant λ:

* **Option β — λ schedule across denoising steps.** Modulate λ along the
  trajectory so the composition pushes harder once the sequence is mostly
  unmasked (late_fire / cosine / exp) or harder at the start (early_fire).
* **Option α — Gibbs MCMC refinement.** After naïve PoE-3 sampling, run
  a few Gibbs sweeps over token positions sampling from the PoE-composed
  conditional. Adaptation of Du Yan et al. 2023 to discrete-state MDLM.

Both experiments score samples against the same proxy thresholds + indep
reference as ``07_n3_extension.py``; results are written one JSON per
config.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


def _load_prompts(prompts_file: Path, n_samples: int, prompt_token_len: int, tokenizer):
    with prompts_file.open() as f:
        pool = [json.loads(line)["text"] for line in f if line.strip()]
    if not pool:
        raise typer.BadParameter(f"prompts file {prompts_file} is empty")
    prompts = []
    for i in range(n_samples):
        ids = tokenizer.encode(pool[i % len(pool)], add_special_tokens=False)[:prompt_token_len]
        prompts.append(tokenizer.decode(ids, skip_special_tokens=True))
    return prompts


@app.command()
def main(
    triplet: list[str] = typer.Option(["formal", "positive", "concrete"]),
    n_samples: int = 200,
    max_new_tokens: int = 48,
    num_steps: int = 256,
    lambda_each: float = 1.0,
    schedule: str = "constant",
    mcmc_iters: int = 0,
    mcmc_positions: int = 5,
    prompts_file: Path = Path("artifacts/datasets/prompts.jsonl"),
    prompt_token_len: int = 12,
    out_json: Path = Path("artifacts/n3_advanced.json"),
    checkpoints_dir: Path = Path("artifacts/checkpoints"),
    backbone: str = "kuleshov-group/mdlm-owt",
    seed: int = 42,
) -> None:
    import dllm
    import torch
    from peft import PeftModel

    from src.composition.mcmc_refine import gibbs_refine
    from src.composition.poe_sampler import PoECompositionModel, PoEConfig, PoESampler
    from src.energies import build_default_energies
    from src.eval.joint_satisfaction import compute_thresholds
    from src.eval.scoring import SampleScorer

    if len(triplet) != 3:
        raise typer.BadParameter(f"triplet must have 3 names, got {triplet}")

    typer.echo(f"Loading backbone {backbone} + adapters {triplet}...")
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
    # Two PoE handles: a constant-λ one for the baseline + per-expert
    # marginals (so the indep_ref stays comparable across schedule
    # variants) and a schedule-aware one used only for the actual PoE-3
    # step under test.
    cfg_const = PoEConfig(
        num_steps=num_steps,
        max_new_tokens=max_new_tokens,
        seed=seed,
        lambda_schedule=None,
    )
    cfg_poe3 = PoEConfig(
        num_steps=num_steps,
        max_new_tokens=max_new_tokens,
        seed=seed,
        lambda_schedule=schedule if schedule != "constant" else None,
    )
    poe_const = PoESampler(model, tokenizer, cfg=cfg_const)
    poe_sched = PoESampler(model, tokenizer, cfg=cfg_poe3)
    prompts = _load_prompts(prompts_file, n_samples, prompt_token_len, tokenizer)

    # Baseline → thresholds (always at constant schedule for fair comparison)
    typer.echo("Sampling baseline (constant schedule)...")
    torch.manual_seed(seed)
    base_texts = poe_const.sample(prompts, lambdas={})
    base_scored = [scorer.score(s) for s in base_texts]
    thresholds = compute_thresholds({k: [r.proxy_scores[k] for r in base_scored] for k in energies})

    # Per-expert solo marginals (always at constant schedule)
    from src.data.build_datasets import DEFAULT_VERTICAL_SPECS

    energy_for = {s.name: s.energy_key for s in DEFAULT_VERTICAL_SPECS}
    marginals: dict[str, float] = {}
    for name in triplet:
        typer.echo(f"Sampling solo {name} (constant schedule)...")
        torch.manual_seed(seed)
        scored = [scorer.score(s) for s in poe_const.sample(prompts, lambdas={name: 1.0})]
        ek = energy_for.get(name, name)
        marginals[name] = float(
            sum(r.proxy_scores[ek] >= thresholds[ek] for r in scored) / max(1, len(scored))
        )
        typer.echo(f"  marginal {name} = {marginals[name]:.3f}")

    # PoE-3 (with the schedule under test)
    typer.echo(f"Sampling PoE-3 (λ_each={lambda_each}, schedule={schedule})...")
    torch.manual_seed(seed)
    lambdas = {n: lambda_each for n in triplet}
    poe3_texts = poe_sched.sample(prompts, lambdas=lambdas)

    triplet_keys = [energy_for.get(n, n) for n in triplet]

    def _triple_sat(texts: list[str]) -> float:
        scored = [scorer.score(t) for t in texts]
        return float(
            sum(all(r.proxy_scores[k] >= thresholds[k] for k in triplet_keys) for r in scored)
            / max(1, len(scored))
        )

    triple_sat_naive = _triple_sat(poe3_texts)
    indep_ref = float(marginals[triplet[0]] * marginals[triplet[1]] * marginals[triplet[2]])

    typer.echo(
        f"PoE-3 (naïve): triple_sat={triple_sat_naive:.4f}  indep={indep_ref:.4f}  ratio={triple_sat_naive / indep_ref:.2f}"
    )

    # Optional MCMC refinement on the naïve samples
    triple_sat_mcmc = None
    mcmc_stats = None
    if mcmc_iters > 0:
        typer.echo(
            f"Running Gibbs MCMC refinement: {mcmc_iters} iters × {mcmc_positions} positions/sample..."
        )
        # Re-encode the texts back to token IDs for refinement
        sample_ids = [tokenizer.encode(t, add_special_tokens=False) for t in poe3_texts]
        # Build a PoE wrapper for refinement (fixed λ, no schedule)
        wrapper = PoECompositionModel(model, lambdas)
        refined_ids, mcmc_stats = gibbs_refine(
            sample_ids,
            wrapper,
            mask_id=tokenizer.mask_token_id,
            n_iters=mcmc_iters,
            positions_per_iter=mcmc_positions,
            seed=seed,
        )
        refined_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in refined_ids]
        triple_sat_mcmc = _triple_sat(refined_texts)
        typer.echo(
            f"PoE-3 (MCMC-refined): triple_sat={triple_sat_mcmc:.4f}  ratio={triple_sat_mcmc / indep_ref:.2f}"
        )
        typer.echo(f"  MCMC stats: {mcmc_stats}")

    out = {
        "triplet": triplet,
        "lambda_each": lambda_each,
        "schedule": schedule,
        "mcmc_iters": mcmc_iters,
        "mcmc_positions": mcmc_positions,
        "thresholds": thresholds,
        "marginals": marginals,
        "triple_satisfaction_poe3_naive": triple_sat_naive,
        "triple_satisfaction_poe3_mcmc": triple_sat_mcmc,
        "independence_reference": indep_ref,
        "mcmc_stats": mcmc_stats,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    typer.echo(f"Wrote {out_json}")


if __name__ == "__main__":
    app()
