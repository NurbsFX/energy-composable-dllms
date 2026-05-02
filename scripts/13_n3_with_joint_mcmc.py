"""Phase 10 — Joint MCMC corrector for N=3 PoE composition.

Implements the Du Yan et al. 2023 joint-MCMC idea adapted to discrete-state
MDLM. For each triplet, we compare:

* **naïve PoE-3** (current method, already in Phase 6 results)
* **PoE-3 + noise-then-denoise refinement** (block-Gibbs, K iters)
* **PoE-3 + MH token-swap refinement** (sequence-level ELBO acceptance,
  K iters; only run if --enable-mh because it's slow)

The hypothesis: the Test 2 slope of 0.857 < 1 is empirical evidence that
per-step PoE composition slightly under-shoots the true joint distribution.
A joint MCMC corrector should bring the slope (and the N=3 ratio) closer
to the theoretical maximum.

Inputs (from artifact_root):
* baseline samples + thresholds (used for marginals + indep_ref)
* LoRA checkpoints for the 6 single-axis experts (per backbone)

Outputs: ``artifacts/n3_mcmc_<triplet>.json`` with naïve / refined ratios
and a summary table.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)

PROXY_TO_EXPERT = {
    "len": "long",
    "form": "formal",
    "sent": "positive",
    "sent2": "positive2",
    "conc": "concrete",
    "topic": "sports",
}
EXPERT_TO_PROXY = {v: k for k, v in PROXY_TO_EXPERT.items()}


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


def _triple_sat(scored, triplet_keys, thresholds):
    """Fraction of samples satisfying all three top-quartile constraints."""
    return float(
        sum(all(r.proxy_scores[k] >= thresholds[k] for k in triplet_keys) for r in scored)
        / max(1, len(scored))
    )


@app.command()
def main(
    triplet: list[str] = typer.Option(["formal", "positive", "concrete"]),
    n_samples: int = 200,
    max_new_tokens: int = 48,
    num_steps: int = 256,
    prompts_file: Path = Path("artifacts/datasets/prompts.jsonl"),
    prompt_token_len: int = 12,
    backbone: str = "kuleshov-group/mdlm-owt",
    checkpoints_dir: Path = Path("artifacts/checkpoints"),
    out_json: Path = Path("artifacts/n3_mcmc_default.json"),
    seed: int = 42,
    # MCMC config
    mcmc_n_iters: int = typer.Option(3, help="Number of refinement iterations"),
    mcmc_mask_fraction: float = typer.Option(0.25, help="Fraction of positions to remask per iter"),
    mcmc_partial_steps: int = typer.Option(64, help="Sub-steps for partial denoising"),
    enable_mh: bool = typer.Option(False, help="Also run MH-token-swap (slow)"),
    mh_proposals: int = 8,
    mh_num_t_samples: int = 8,
) -> None:
    import dllm
    import torch
    from peft import PeftModel

    from src.composition.joint_mcmc import (
        MCMCRefineConfig,
        mh_token_swap,
        noise_then_denoise,
    )
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
    cfg = PoEConfig(num_steps=num_steps, max_new_tokens=max_new_tokens, seed=seed)
    poe = PoESampler(model, tokenizer, cfg=cfg)

    prompts = _load_prompts(prompts_file, n_samples, prompt_token_len, tokenizer)

    # 1. Baseline → thresholds
    typer.echo("Sampling baseline (constant schedule)...")
    torch.manual_seed(seed)
    base_texts = poe.sample(prompts, lambdas={})
    base_scored = [scorer.score(s) for s in base_texts]
    thresholds = compute_thresholds({k: [r.proxy_scores[k] for r in base_scored] for k in energies})

    # 2. Per-expert solo marginals
    energy_for = {n: EXPERT_TO_PROXY[n] for n in triplet}
    marginals: dict[str, float] = {}
    for name in triplet:
        typer.echo(f"Sampling solo {name} (constant)...")
        torch.manual_seed(seed)
        scored = [scorer.score(s) for s in poe.sample(prompts, lambdas={name: 1.0})]
        ek = energy_for[name]
        marginals[name] = float(
            sum(r.proxy_scores[ek] >= thresholds[ek] for r in scored) / max(1, len(scored))
        )
        typer.echo(f"  marginal {name} = {marginals[name]:.3f}")

    indep_ref = float(marginals[triplet[0]] * marginals[triplet[1]] * marginals[triplet[2]])

    # 3. Naïve PoE-3
    typer.echo("Sampling PoE-3 naïve...")
    torch.manual_seed(seed)
    naive_texts = poe.sample(prompts, lambdas={n: 1.0 for n in triplet})
    naive_scored = [scorer.score(s) for s in naive_texts]
    triplet_keys = [energy_for[n] for n in triplet]
    triple_sat_naive = _triple_sat(naive_scored, triplet_keys, thresholds)

    typer.echo(
        f"  naïve  triple_sat={triple_sat_naive:.4f}  indep={indep_ref:.4f}  "
        f"ratio={triple_sat_naive / max(1e-9, indep_ref):.2f}"
    )

    # 4. Joint MCMC: noise-then-denoise (block Gibbs)
    typer.echo(
        f"Running noise-then-denoise refinement: {mcmc_n_iters} iters × "
        f"{mcmc_mask_fraction} mask × {mcmc_partial_steps} sub-steps..."
    )
    naive_ids = [tokenizer.encode(t, add_special_tokens=False) for t in naive_texts]
    poe_wrapper = PoECompositionModel(model, {n: 1.0 for n in triplet})
    mcmc_cfg = MCMCRefineConfig(
        n_iters=mcmc_n_iters,
        mask_fraction=mcmc_mask_fraction,
        partial_denoise_steps=mcmc_partial_steps,
        seed=seed,
    )
    refined_ids, ntd_stats = noise_then_denoise(
        naive_ids,
        poe_wrapper,
        tokenizer,
        scheduler=poe.scheduler,
        cfg=mcmc_cfg,
        max_new_tokens=max_new_tokens,
    )
    refined_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in refined_ids]
    refined_scored = [scorer.score(s) for s in refined_texts]
    triple_sat_ntd = _triple_sat(refined_scored, triplet_keys, thresholds)
    typer.echo(
        f"  ntd    triple_sat={triple_sat_ntd:.4f}  ratio={triple_sat_ntd / max(1e-9, indep_ref):.2f}"
    )
    typer.echo(f"  ntd stats: {ntd_stats}")

    # 5. Optional MH refinement
    triple_sat_mh = None
    mh_stats = None
    if enable_mh:
        typer.echo(
            f"Running MH-token-swap refinement: {mcmc_n_iters} iters × "
            f"{mh_proposals} proposals × {mh_num_t_samples} t-samples..."
        )
        mh_cfg = MCMCRefineConfig(
            n_iters=mcmc_n_iters,
            mh_proposals_per_iter=mh_proposals,
            mh_num_t_samples=mh_num_t_samples,
            seed=seed,
        )
        mh_refined_ids, mh_stats = mh_token_swap(
            naive_ids,
            poe_wrapper,
            tokenizer,
            scheduler=poe.scheduler,
            cfg=mh_cfg,
            base_model_with_adapters=model,
            expert_a=triplet[0],
            expert_b=triplet[1],
            expert_c=triplet[2],
        )
        mh_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in mh_refined_ids]
        mh_scored = [scorer.score(s) for s in mh_texts]
        triple_sat_mh = _triple_sat(mh_scored, triplet_keys, thresholds)
        typer.echo(
            f"  mh     triple_sat={triple_sat_mh:.4f}  "
            f"ratio={triple_sat_mh / max(1e-9, indep_ref):.2f}"
        )
        typer.echo(f"  mh stats: {mh_stats}")

    # Save
    out = {
        "triplet": list(triplet),
        "backbone": backbone,
        "thresholds": thresholds,
        "marginals": marginals,
        "indep_ref": indep_ref,
        "triple_satisfaction": {
            "naive": triple_sat_naive,
            "noise_then_denoise": triple_sat_ntd,
            "mh_token_swap": triple_sat_mh,
        },
        "ratios": {
            "naive": triple_sat_naive / max(1e-9, indep_ref),
            "noise_then_denoise": triple_sat_ntd / max(1e-9, indep_ref),
            "mh_token_swap": (triple_sat_mh / max(1e-9, indep_ref)) if triple_sat_mh else None,
        },
        "ntd_stats": ntd_stats,
        "mh_stats": mh_stats,
        "config": {
            "n_samples": n_samples,
            "max_new_tokens": max_new_tokens,
            "num_steps": num_steps,
            "mcmc_n_iters": mcmc_n_iters,
            "mcmc_mask_fraction": mcmc_mask_fraction,
            "mcmc_partial_steps": mcmc_partial_steps,
            "enable_mh": enable_mh,
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    typer.echo(f"\nWrote {out_json}")


if __name__ == "__main__":
    app()
