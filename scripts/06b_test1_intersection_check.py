#!/usr/bin/env python
"""Plan-B Test 1: distribution match between intersection-expert and PoE.

For one chosen pair (a, b) we compare two distributions:

* Samples from the LoRA expert trained on the intersection corpus
  (``a_b``), as built by ``scripts/03b_train_intersection_expert.py``.
* Samples from the Product-of-Experts composition of the two
  single-vertical experts ``a`` and ``b`` (``λ_a = λ_b = 1``).

Distributional equivalence is approximated metric-by-metric: for each
proxy, we compute the Kolmogorov-Smirnov statistic and a 2-sample
t-test on the proxy's raw signal across the two sample sets. A perfectly
exact PoE composition would give ``KS ≈ 0`` and ``p ≫ 0.05`` on every
proxy.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    pair: str = typer.Option("long:formal"),
    n_samples: int = 500,
    max_new_tokens: int = 128,
    num_steps: int = 256,
    out_json: Path = Path("artifacts/test1_intersection_check.json"),
    out_png: Path = Path("artifacts/plots/test1_intersection_check.png"),
    checkpoints_dir: Path = Path("artifacts/checkpoints"),
    backbone: str = "kuleshov-group/mdlm-owt",
    seed: int = 42,
) -> None:
    import dllm
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from peft import PeftModel
    from scipy.stats import ks_2samp, ttest_ind

    from src.composition.poe_sampler import PoEConfig, PoESampler
    from src.energies import build_default_energies
    from src.eval.scoring import SampleScorer

    a, b = (s.strip() for s in pair.split(":"))
    intersect_name = f"{a}_{b}"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Loading backbone + adapters {a}, {b}, {intersect_name}")
    model_args = dllm.utils.ModelArguments(model_name_or_path=backbone)
    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    for i, name in enumerate([a, b, intersect_name]):
        path = checkpoints_dir / name
        if i == 0:
            model = PeftModel.from_pretrained(model, path, adapter_name=name)
        else:
            model.load_adapter(path, adapter_name=name)

    energies = build_default_energies()
    scorer = SampleScorer(energies=energies)
    cfg = PoEConfig(num_steps=num_steps, max_new_tokens=max_new_tokens, seed=seed)
    poe = PoESampler(model, tokenizer, cfg=cfg)

    prompts = [""] * n_samples

    typer.echo("Sampling from intersection-trained expert...")
    torch.manual_seed(seed)
    samples_intersection = [
        scorer.score(s) for s in poe.sample(prompts, lambdas={intersect_name: 1.0})
    ]

    typer.echo("Sampling from PoE(a, b)...")
    torch.manual_seed(seed)
    samples_poe = [scorer.score(s) for s in poe.sample(prompts, lambdas={a: 1.0, b: 1.0})]

    results: dict[str, dict] = {}
    for proxy_key in energies:
        x = np.array([r.proxy_scores[proxy_key] for r in samples_intersection])
        y = np.array([r.proxy_scores[proxy_key] for r in samples_poe])
        ks = ks_2samp(x, y)
        tt = ttest_ind(x, y, equal_var=False)
        results[proxy_key] = {
            "mean_intersection": float(np.mean(x)),
            "mean_poe": float(np.mean(y)),
            "ks_stat": float(ks.statistic),
            "ks_p": float(ks.pvalue),
            "ttest_stat": float(tt.statistic),
            "ttest_p": float(tt.pvalue),
        }

    out_json.write_text(
        json.dumps(
            {
                "pair": [a, b],
                "intersection_expert": intersect_name,
                "n_samples": n_samples,
                "per_proxy": results,
            },
            indent=2,
        )
    )

    typer.echo("\n  proxy       mean_int  mean_poe   KS   p_KS  p_t")
    for k, r in results.items():
        typer.echo(
            f"  {k:<10} {r['mean_intersection']:>8.3f} {r['mean_poe']:>8.3f} "
            f"{r['ks_stat']:>5.3f} {r['ks_p']:>5.3f} {r['ttest_p']:>5.3f}"
        )

    proxies = list(results)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(proxies))
    width = 0.4
    ax.bar(
        x - width / 2,
        [results[p]["mean_intersection"] for p in proxies],
        width,
        label="intersection expert",
    )
    ax.bar(x + width / 2, [results[p]["mean_poe"] for p in proxies], width, label=f"PoE({a},{b})")
    ax.set_xticks(x)
    ax.set_xticklabels(proxies, rotation=30)
    ax.set_ylabel("mean raw signal")
    ax.set_title(f"Test 1: distribution match on ({a}, {b})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    typer.echo(f"\nWrote {out_json} and {out_png}")


if __name__ == "__main__":
    app()
