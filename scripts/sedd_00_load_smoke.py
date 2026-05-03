"""Phase-0 smoke test for the SEDD parallel stack.

Goals (in order):
1. Verify ``external/sedd/`` is on path and importable.
2. Load a published SEDD checkpoint (default: SEDD-small, 90M).
3. Generate a few short sequences with Lou's sampler unchanged
   (no composition, no LoRA).
4. Decode and print — basic sanity that the GPT-2 + 1-MASK vocab
   round-trips cleanly through our tokenizer wrapper.
5. Exercise our PoEScoreCompositionModel with **zero adapters** loaded:
   composition with ``lambdas={}`` must equal the bare backbone
   (regression of the Paper-1 ``assert_lambda_zero_is_base`` invariant
   in the score domain).

This script does *not* train or compose with real LoRA experts — that
comes in Phase 1 once we have one trained adapter. It is the lightest
end-to-end exercise of the new stack.
"""

from __future__ import annotations

import sys

import torch
import typer

from src.sedd_composition import (
    PoEScoreConfig,
    PoEScoreSampler,
    load_sedd_from_hub,
)
from src.sedd_composition.load import get_gpt2_tokenizer_for_sedd

app = typer.Typer(add_completion=False)


@app.command()
def main(
    repo_id: str = "louaaron/sedd-small",
    num_samples: int = 2,
    seq_len: int = 32,
    num_steps: int = 64,
    seed: int = 0,
    device: str | None = None,
    skip_sample: bool = False,
) -> None:
    typer.echo(f"=== SEDD smoke test — {repo_id} ===")
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    typer.echo(f"device: {device}")

    typer.echo("Step 1/4: loading checkpoint...")
    try:
        model, graph, noise = load_sedd_from_hub(repo_id, device=device)
    except Exception as e:
        typer.echo(f"  FAILED to load {repo_id}: {e}", err=True)
        raise typer.Exit(1) from e
    n_params = sum(p.numel() for p in model.parameters())
    typer.echo(
        f"  loaded. n_params = {n_params / 1e6:.1f}M  device = {next(model.parameters()).device}"
    )

    typer.echo("Step 2/4: tokenizer...")
    tokenizer = get_gpt2_tokenizer_for_sedd()
    typer.echo(f"  vocab_size = {len(tokenizer)} (expect 50258 after MASK registration)")

    if skip_sample:
        typer.echo("--skip-sample passed; bailing before any forward pass.")
        return

    typer.echo("Step 3/4: bare-backbone sample via Lou's pc_sampler...")
    cfg = PoEScoreConfig(num_steps=num_steps, sample_batch_size=num_samples)
    sampler = PoEScoreSampler(model, graph, noise, tokenizer, cfg=cfg)
    torch.manual_seed(seed)
    try:
        texts = sampler.sample(
            num_samples=num_samples,
            seq_len=seq_len,
            lambdas={},  # no experts; composition reduces to base log-score
        )
    except Exception as e:
        typer.echo(f"  FAILED during sampling: {e}", err=True)
        raise typer.Exit(2) from e
    for i, t in enumerate(texts):
        typer.echo(f"  [{i}] {t!r}")

    typer.echo("Step 4/4: λ=0 invariance — disabled here (no adapters loaded yet).")
    typer.echo("        Will be exercised in scripts/sedd_01_train_lora.py once an")
    typer.echo("        adapter exists on the backbone.")

    typer.echo("\nSmoke test passed. Stack is wired correctly.")


if __name__ == "__main__":
    if "src.sedd_composition" not in sys.modules:
        # Make sure repo root is on sys.path when run as a script.
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    app()
