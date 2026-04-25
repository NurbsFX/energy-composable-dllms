#!/usr/bin/env python
"""Cross-vertical validation of the trained LoRA experts.

For every checkpoint in ``checkpoints_dir`` we generate ``n_samples``
samples with that adapter active, score each sample on the six proxies,
and aggregate into a per-(expert, proxy) mean. The bare backbone is
included as the reference row. The resulting ``(N_experts + 1) ×
N_proxies`` table should be diagonal-dominant: each expert wins on its
own proxy and shows limited contamination on the others.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    checkpoints_dir: Path = Path("artifacts/checkpoints"),
    out_json: Path = Path("artifacts/expert_validation.json"),
    n_samples: int = 128,
    max_new_tokens: int = 128,
    num_steps: int = 256,
    backbone: str = "kuleshov-group/mdlm-owt",
    seed: int = 42,
) -> None:
    import dllm
    import torch
    from peft import PeftModel

    from src.composition.poe_sampler import PoEConfig, PoESampler
    from src.energies import build_default_energies
    from src.eval.scoring import SampleScorer

    out_json.parent.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Loading backbone {backbone} ...")
    model_args = dllm.utils.ModelArguments(model_name_or_path=backbone)
    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    expert_names: list[str] = []
    for sub in sorted(checkpoints_dir.iterdir()):
        if not sub.is_dir() or not (sub / "adapter_config.json").exists():
            continue
        name = sub.name
        typer.echo(f"  loading adapter {name}")
        if not expert_names:
            model = PeftModel.from_pretrained(model, sub, adapter_name=name)
        else:
            model.load_adapter(sub, adapter_name=name)
        expert_names.append(name)
    if not expert_names:
        raise typer.BadParameter(f"no adapter checkpoints found under {checkpoints_dir}")

    typer.echo(f"Found {len(expert_names)} experts: {expert_names}")
    energies = build_default_energies()
    scorer = SampleScorer(energies=energies)
    cfg = PoEConfig(num_steps=num_steps, max_new_tokens=max_new_tokens, seed=seed)
    poe_sampler = PoESampler(model, tokenizer, cfg=cfg)

    rows: dict[str, dict[str, float]] = {}
    prompts = [""] * n_samples  # unconditional generation

    def _row(name: str, lambdas: dict[str, float]) -> dict[str, float]:
        torch.manual_seed(seed)
        samples = poe_sampler.sample(prompts, lambdas=lambdas)
        scores = [scorer.score(s) for s in samples]
        agg: dict[str, float] = {}
        for proxy_key in energies:
            vals = [s.proxy_scores[proxy_key] for s in scores]
            agg[proxy_key] = float(sum(vals) / max(1, len(vals)))
        agg["ppl_gpt2"] = float(sum(s.ppl_gpt2 for s in scores) / max(1, len(scores)))
        agg["distinct_2"] = float(sum(s.distinct_2 for s in scores) / max(1, len(scores)))
        return agg

    typer.echo("Sampling from bare backbone (baseline row)...")
    rows["__baseline__"] = _row("__baseline__", lambdas={})

    for name in expert_names:
        typer.echo(f"Sampling from expert {name}...")
        rows[name] = _row(name, lambdas={name: 1.0})

    out_json.write_text(json.dumps({"expert_names": expert_names, "rows": rows}, indent=2))
    typer.echo(f"Wrote {out_json}")

    # Print diagonal-dominance summary.
    typer.echo("\nMean raw signal per (expert, proxy):")
    proxies = list(energies.keys())
    header = "  expert".ljust(15) + "".join(p.rjust(10) for p in proxies)
    typer.echo(header)
    for expert in ["__baseline__", *expert_names]:
        line = f"  {expert:<13}" + "".join(f"{rows[expert][p]:>10.3f}" for p in proxies)
        typer.echo(line)


if __name__ == "__main__":
    app()
