"""Phase 11 — μ sweep on N=3 (decoupling the base coefficient).

Standard PoE composition:
    log p_PoE(x) = log p_base + Σ λ_i (log p_i − log p_base)
                 = (1 − Σ λ_i) log p_base + Σ λ_i log p_i

At N=3 with λ_i = 1, the coefficient on log p_base is 1 − 3 = −2: a strong
penalty against OWT-typical text. The hypothesis explored here is that this
coefficient is *too* punishing, and that decoupling it via

    log p_custom(x) = μ · log p_base(x) + Σ λ_i log p_i(x)

with tunable μ may yield better triple-satisfaction at N=3. We sweep
μ ∈ {-2, -1.5, -1, -0.5, 0, +0.5, +1} on the same triplet and backbone
that have failed under standard PoE so far (formal × positive × concrete on
Qwen3-0.6B-MDLM, ratio 0.42 in Phase 8 v1).
"""

from __future__ import annotations

import json
from pathlib import Path

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
    backbone: str = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
    checkpoints_dir: Path = Path("artifacts/checkpoints_qwen3"),
    out_json: Path = Path("artifacts/n3_mu_sweep.json"),
    seed: int = 42,
    mu_values: str = typer.Option(
        "-2,-1.5,-1,-0.5,0,0.5,1", help="Comma-separated μ values to sweep."
    ),
) -> None:
    import dllm
    import torch
    from peft import PeftModel

    from src.composition.poe_sampler import PoEConfig, PoESampler
    from src.energies import build_default_energies
    from src.eval.joint_satisfaction import compute_thresholds
    from src.eval.scoring import SampleScorer

    if len(triplet) not in (2, 3):
        raise typer.BadParameter(f"need 2 or 3 expert names (N=2 or N=3), got {triplet}")

    mu_list = [float(s.strip()) for s in mu_values.split(",") if s.strip()]
    typer.echo(f"Sweeping μ ∈ {mu_list} on triplet {triplet} ({backbone})")

    typer.echo("Loading backbone + adapters ...")
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

    # Baseline → thresholds (constant μ=None, standard PoE base)
    base_cfg = PoEConfig(num_steps=num_steps, max_new_tokens=max_new_tokens, seed=seed)
    poe_base = PoESampler(model, tokenizer, cfg=base_cfg)
    prompts = _load_prompts(prompts_file, n_samples, prompt_token_len, tokenizer)

    typer.echo("Sampling baseline (no adapter)...")
    torch.manual_seed(seed)
    base_texts = poe_base.sample(prompts, lambdas={})
    base_scored = [scorer.score(s) for s in base_texts]
    thresholds = compute_thresholds({k: [r.proxy_scores[k] for r in base_scored] for k in energies})

    # Per-expert solo marginals (constant standard sampling, μ=None)
    marginals: dict[str, float] = {}
    for name in triplet:
        typer.echo(f"Sampling solo {name} ...")
        torch.manual_seed(seed)
        scored = [scorer.score(s) for s in poe_base.sample(prompts, lambdas={name: 1.0})]
        ek = energy_for[name]
        marginals[name] = float(
            sum(r.proxy_scores[ek] >= thresholds[ek] for r in scored) / max(1, len(scored))
        )
        typer.echo(f"  marginal {name} = {marginals[name]:.3f}")
    indep_ref = 1.0
    for n in triplet:
        indep_ref *= marginals[n]
    indep_ref = float(indep_ref)

    # Naïve PoE-N (μ = None, standard) — for comparison
    N = len(triplet)
    standard_mu = 1 - N
    typer.echo(f"Sampling PoE-{N} (μ standard, ≡ 1 − Σλ = {standard_mu} at N={N}, λ=1)...")
    torch.manual_seed(seed)
    naive_scored = [
        scorer.score(s) for s in poe_base.sample(prompts, lambdas={n: 1.0 for n in triplet})
    ]
    triple_sat_standard = _triple_sat(naive_scored, triplet_keys, thresholds)
    typer.echo(
        f"  μ standard: triple_sat={triple_sat_standard:.4f}  "
        f"ratio={triple_sat_standard / max(1e-9, indep_ref):.2f}"
    )

    # Sweep over μ
    sweep_results: dict[str, dict] = {
        f"standard_{standard_mu:+g}": {
            "mu": float(standard_mu),
            "triple_sat": triple_sat_standard,
            "ratio": triple_sat_standard / max(1e-9, indep_ref),
        }
    }
    for mu in mu_list:
        typer.echo(f"\nSampling PoE-3 with μ = {mu} ...")
        cfg_mu = PoEConfig(
            num_steps=num_steps,
            max_new_tokens=max_new_tokens,
            seed=seed,
            mu_base=mu,
        )
        poe_mu = PoESampler(model, tokenizer, cfg=cfg_mu)
        torch.manual_seed(seed)
        texts = poe_mu.sample(prompts, lambdas={n: 1.0 for n in triplet})
        scored = [scorer.score(s) for s in texts]
        triple_sat = _triple_sat(scored, triplet_keys, thresholds)
        ratio = triple_sat / max(1e-9, indep_ref)
        sweep_results[f"mu_{mu:+.2f}"] = {
            "mu": mu,
            "triple_sat": triple_sat,
            "ratio": ratio,
        }
        typer.echo(f"  μ = {mu:+.2f}: triple_sat={triple_sat:.4f}  ratio={ratio:.2f}")

    out = {
        "triplet": list(triplet),
        "backbone": backbone,
        "thresholds": thresholds,
        "marginals": marginals,
        "indep_ref": indep_ref,
        "config": {
            "n_samples": n_samples,
            "max_new_tokens": max_new_tokens,
            "num_steps": num_steps,
            "mu_values": mu_list,
        },
        "sweep_results": sweep_results,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    typer.echo(f"\nWrote {out_json}")

    typer.echo("\n=== Summary ===")
    typer.echo(f"  {'config':<14s}  {'μ':>6s}  {'triple_sat':>10s}  {'ratio':>6s}")
    for name, r in sweep_results.items():
        typer.echo(f"  {name:<14s}  {r['mu']:>+6.2f}  {r['triple_sat']:>10.4f}  {r['ratio']:>6.2f}")


if __name__ == "__main__":
    app()
