#!/usr/bin/env python
"""Extension to N=3 experts: triple satisfaction vs product of marginals."""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    triplet: list[str] = typer.Option(["formal", "positive", "concrete"]),
    n_samples: int = 200,
    # MDLM-OWT degenerates past ~30-50 tokens; see scripts/05_run_composition.py.
    max_new_tokens: int = 48,
    num_steps: int = 256,
    prompts_file: Path | None = None,
    prompt_token_len: int = 12,
    out_json: Path = Path("artifacts/n3_results.json"),
    out_png: Path = Path("artifacts/plots/n3_satisfaction.png"),
    checkpoints_dir: Path = Path("artifacts/checkpoints"),
    backbone: str = "kuleshov-group/mdlm-owt",
    seed: int = 42,
) -> None:
    import dllm
    import matplotlib.pyplot as plt
    import torch
    from peft import PeftModel

    from src.composition.poe_sampler import PoEConfig, PoESampler
    from src.energies import build_default_energies
    from src.eval.joint_satisfaction import compute_thresholds
    from src.eval.scoring import SampleScorer

    if len(triplet) != 3:
        raise typer.BadParameter(f"triplet must have exactly 3 names, got {triplet}")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Loading backbone {backbone} and adapters {triplet}...")
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

    if prompts_file is not None:
        with prompts_file.open() as f:
            pool = [json.loads(line)["text"] for line in f if line.strip()]
        if not pool:
            raise typer.BadParameter(f"prompts file {prompts_file} is empty")
        prompts = []
        for i in range(n_samples):
            ids = tokenizer.encode(pool[i % len(pool)], add_special_tokens=False)[:prompt_token_len]
            prompts.append(tokenizer.decode(ids, skip_special_tokens=True))
        typer.echo(f"Loaded {len(pool)} prompts from {prompts_file} (cycled to {n_samples}).")
    else:
        prompts = [""] * n_samples

    # Baseline → thresholds
    typer.echo("Sampling baseline...")
    torch.manual_seed(seed)
    base = [scorer.score(s) for s in poe.sample(prompts, lambdas={})]
    thresholds = compute_thresholds({k: [r.proxy_scores[k] for r in base] for k in energies})

    # Per-expert solo marginals
    marginals: dict[str, float] = {}
    for name in triplet:
        typer.echo(f"Sampling solo {name}...")
        torch.manual_seed(seed)
        scored = [scorer.score(s) for s in poe.sample(prompts, lambdas={name: 1.0})]
        # The expert ``name`` corresponds to a proxy whose key is the same name
        # only for the post-Phase-2 set ``{long: len, formal: form, positive: sent, …}``.
        # Use the energy_key mapping from build_datasets to be robust.
        from src.data.build_datasets import DEFAULT_VERTICAL_SPECS

        energy_key = next((s.energy_key for s in DEFAULT_VERTICAL_SPECS if s.name == name), name)
        marginals[name] = float(
            sum(r.proxy_scores[energy_key] >= thresholds[energy_key] for r in scored)
            / max(1, len(scored))
        )
        typer.echo(f"  marginal {name} = {marginals[name]:.3f}")

    # Triple PoE
    typer.echo("Sampling PoE-3...")
    torch.manual_seed(seed)
    scored = [scorer.score(s) for s in poe.sample(prompts, lambdas={n: 1.0 for n in triplet})]
    from src.data.build_datasets import DEFAULT_VERTICAL_SPECS

    triplet_energy_keys = []
    for n in triplet:
        ek = next((s.energy_key for s in DEFAULT_VERTICAL_SPECS if s.name == n), n)
        triplet_energy_keys.append(ek)
    triple_sat = float(
        sum(all(r.proxy_scores[k] >= thresholds[k] for k in triplet_energy_keys) for r in scored)
        / max(1, len(scored))
    )
    indep_ref = float(marginals[triplet[0]] * marginals[triplet[1]] * marginals[triplet[2]])

    typer.echo(f"Triple satisfaction (PoE-3): {triple_sat:.3f}")
    typer.echo(f"Independence reference   : {indep_ref:.3f}")

    out_json.write_text(
        json.dumps(
            {
                "triplet": triplet,
                "thresholds": thresholds,
                "marginals": marginals,
                "triple_satisfaction_poe3": triple_sat,
                "independence_reference": indep_ref,
            },
            indent=2,
        )
    )

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(["product of marginals", "PoE-3"], [indep_ref, triple_sat], color=["#999", "#1f77b4"])
    ax.set_ylabel("triple constraint satisfaction")
    ax.set_title(" × ".join(triplet))
    ax.set_ylim(0, max(indep_ref, triple_sat) * 1.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    typer.echo(f"Wrote {out_json} and {out_png}")


if __name__ == "__main__":
    app()
