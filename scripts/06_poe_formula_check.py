#!/usr/bin/env python
"""Direct verification of the PoE composition formula on K sequence pairs."""

from __future__ import annotations

import json
import random
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


def _parse_pair(s: str) -> tuple[str, str]:
    a, b = s.split(":")
    return a.strip(), b.strip()


@app.command()
def main(
    pair: str = typer.Option("long:formal", help="Expert pair as 'name_a:name_b'."),
    k_pairs: int = 50,
    num_t_samples: int = 32,
    out_json: Path = Path("artifacts/poe_formula_check.json"),
    out_png: Path = Path("artifacts/plots/poe_formula_check.png"),
    checkpoints_dir: Path = Path("artifacts/checkpoints"),
    backbone: str = "kuleshov-group/mdlm-owt",
    seed: int = 42,
) -> None:
    import dllm
    import matplotlib.pyplot as plt
    from peft import PeftModel

    from src.eval.poe_formula_check import check_poe_formula

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    a, b = _parse_pair(pair)

    typer.echo(f"Loading backbone {backbone} and adapters {a}, {b}...")
    model_args = dllm.utils.ModelArguments(model_name_or_path=backbone)
    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    model = PeftModel.from_pretrained(model, checkpoints_dir / a, adapter_name=a)
    model.load_adapter(checkpoints_dir / b, adapter_name=b)

    # Build K=k_pairs (x, y) sequence pairs by single-token edits of base samples.
    # We sample base sequences from the backbone, then for each x make y by
    # randomly replacing one token with another vocab id.
    typer.echo(f"Building {k_pairs} sequence pairs (1-token edits)...")
    rng = random.Random(seed)
    seq_len = 64
    vocab_size = len(tokenizer)
    pairs: list[tuple[list[int], list[int]]] = []
    for _ in range(k_pairs):
        x = [rng.randrange(vocab_size) for _ in range(seq_len)]
        y = list(x)
        idx = rng.randrange(seq_len)
        new_tok = rng.randrange(vocab_size)
        while new_tok == y[idx]:
            new_tok = rng.randrange(vocab_size)
        y[idx] = new_tok
        pairs.append((x, y))

    typer.echo("Estimating ELBO log-ratios on each pair...")
    result = check_poe_formula(
        model,
        tokenizer,
        expert_a=a,
        expert_b=b,
        pairs=pairs,
        num_t_samples=num_t_samples,
        seed=seed,
    )

    out_json.write_text(
        json.dumps(
            {
                "pair": [a, b],
                "k_pairs": result.n_pairs,
                "num_t_samples": num_t_samples,
                "slope": result.slope,
                "intercept": result.intercept,
                "r2": result.r2,
            },
            indent=2,
        )
    )
    typer.echo(f"slope={result.slope:.3f}  intercept={result.intercept:.3f}  R²={result.r2:.3f}")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(result.log_ratios_predicted, result.log_ratios_observed, s=20, alpha=0.6)
    lo = min(result.log_ratios_predicted.min(), result.log_ratios_observed.min())
    hi = max(result.log_ratios_predicted.max(), result.log_ratios_observed.max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x (perfect)")
    ax.set_xlabel(r"predicted $\log p_1/p_b + \log p_2/p_b$")
    ax.set_ylabel(r"observed $\log p_{\mathrm{PoE}}/p_b$")
    ax.set_title(f"PoE formula on ({a}, {b}); slope={result.slope:.2f}, R²={result.r2:.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    typer.echo(f"Wrote {out_json} and {out_png}")


if __name__ == "__main__":
    app()
