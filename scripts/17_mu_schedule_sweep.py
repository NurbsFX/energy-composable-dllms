"""Phase 12d — μ schedule per-step sweep.

Phase 11 found a sweet-spot constant μ that rescues stylistic compositions.
This script asks the obvious follow-up: does the *optimal* μ vary along the
denoising trajectory? Concretely, it tests whether interpolating between two
μ endpoints — μ_start (early steps) and μ_end (late steps) — under several
schedule shapes can beat the best constant μ identified in Phase 11.

Schedules tested (reusing ``SCHEDULES`` from poe_sampler):
* **constant_mu_start** — control: μ ≡ μ_start across all steps. Phase 11.
* **constant_mu_end**   — control: μ ≡ μ_end across all steps. Phase 11.
* **linear**            — μ(p) = μ_start + (μ_end − μ_start) · p.
* **cosine**            — μ(p) = μ_start + (μ_end − μ_start) · 0.5(1 − cos(π p)).
* **late_fire**         — μ stays at μ_start until p = 0.5, then jumps to μ_end.
* **early_fire**        — μ stays at μ_end until p = 0.5, then drops to μ_start.

Default endpoints:
* **μ_start = canonical (1 − N)** — punitive on log p_base early.
* **μ_end   = best Phase-11 constant** (e.g. −1 on Qwen3 fpc) — relaxed late.

The hypothesis is that the early denoising phase benefits from canonical
suppression of OWT-typical text (clean base sampling), while the late phase
benefits from a relaxed μ that lets stylistic experts shape the surface.
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
    out_json: Path = Path("artifacts/mu_schedule_sweep.json"),
    seed: int = 42,
    mu_start: float = typer.Option(
        None,
        help="Early-step μ. Defaults to canonical 1−N (e.g. −2 at N=3).",
    ),
    mu_end: float = typer.Option(
        -1.0,
        help="Late-step μ. Default −1 (Phase-11 best on Qwen3 fpc).",
    ),
    schedules: str = typer.Option(
        "linear,cosine,late_fire,early_fire",
        help="Comma-separated schedule names from poe_sampler.SCHEDULES.",
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
        raise typer.BadParameter(f"need 2 or 3 expert names, got {triplet}")
    N = len(triplet)
    if mu_start is None:
        mu_start = float(1 - N)

    sched_names = [s.strip() for s in schedules.split(",") if s.strip()]

    typer.echo(
        f"μ-schedule sweep on triplet {triplet} ({backbone})  "
        f"μ_start={mu_start:+g}  μ_end={mu_end:+g}  schedules={sched_names}"
    )

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

    base_cfg = PoEConfig(num_steps=num_steps, max_new_tokens=max_new_tokens, seed=seed)
    poe_base = PoESampler(model, tokenizer, cfg=base_cfg)
    prompts = _load_prompts(prompts_file, n_samples, prompt_token_len, tokenizer)

    typer.echo("Sampling baseline (no adapter)...")
    torch.manual_seed(seed)
    base_scored = [scorer.score(s) for s in poe_base.sample(prompts, lambdas={})]
    thresholds = compute_thresholds({k: [r.proxy_scores[k] for r in base_scored] for k in energies})

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
    typer.echo(f"  indep_ref = {indep_ref:.5f}")

    results: dict[str, dict] = {}

    def _run(label: str, cfg: PoEConfig) -> dict:
        typer.echo(f"\n[{label}]")
        poe = PoESampler(model, tokenizer, cfg=cfg)
        torch.manual_seed(seed)
        texts = poe.sample(prompts, lambdas={n: 1.0 for n in triplet})
        scored = [scorer.score(s) for s in texts]
        ts = _triple_sat(scored, triplet_keys, thresholds)
        ratio = ts / max(1e-9, indep_ref)
        typer.echo(f"  triple_sat={ts:.4f}  ratio={ratio:.3f}")
        return {"triple_sat": ts, "ratio": ratio}

    # Baselines: constant μ at each endpoint (Phase-11 controls)
    results[f"constant_mu_start_{mu_start:+g}"] = {
        "mu_start": mu_start,
        "mu_end": mu_start,
        "schedule": None,
        **_run(
            f"constant μ = {mu_start:+g}",
            PoEConfig(
                num_steps=num_steps, max_new_tokens=max_new_tokens, seed=seed, mu_base=mu_start
            ),
        ),
    }
    results[f"constant_mu_end_{mu_end:+g}"] = {
        "mu_start": mu_end,
        "mu_end": mu_end,
        "schedule": None,
        **_run(
            f"constant μ = {mu_end:+g}",
            PoEConfig(
                num_steps=num_steps, max_new_tokens=max_new_tokens, seed=seed, mu_base=mu_end
            ),
        ),
    }

    # Schedules
    for sname in sched_names:
        cfg = PoEConfig(
            num_steps=num_steps,
            max_new_tokens=max_new_tokens,
            seed=seed,
            mu_base=mu_start,
            mu_base_end=mu_end,
            mu_schedule=sname,
        )
        results[f"sched_{sname}"] = {
            "mu_start": mu_start,
            "mu_end": mu_end,
            "schedule": sname,
            **_run(f"schedule = {sname}  (μ: {mu_start:+g} → {mu_end:+g})", cfg),
        }

    out = {
        "triplet": list(triplet),
        "backbone": backbone,
        "marginals": marginals,
        "indep_ref": indep_ref,
        "thresholds": thresholds,
        "config": {
            "n_samples": n_samples,
            "max_new_tokens": max_new_tokens,
            "num_steps": num_steps,
            "mu_start": mu_start,
            "mu_end": mu_end,
            "schedules": sched_names,
        },
        "results": results,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    typer.echo(f"\nWrote {out_json}")

    typer.echo("\n=== Summary ===")
    typer.echo(f"  {'config':<32s}  {'triple_sat':>10s}  {'ratio':>6s}")
    for name, r in results.items():
        typer.echo(f"  {name:<32s}  {r['triple_sat']:>10.4f}  {r['ratio']:>6.2f}")


if __name__ == "__main__":
    app()
