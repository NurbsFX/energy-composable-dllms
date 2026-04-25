#!/usr/bin/env python
"""Plan-B Test 1: train a single expert on the intersection corpus.

This script does two steps for one chosen pair of verticals (default
``long ∩ formal``):

1. Stream OpenWebText and build the intersection JSONL — only documents
   that pass both vertical filters end up in the file.
2. Fine-tune a 7th LoRA adapter on that JSONL using exactly the same
   hyperparameters as the six per-vertical experts.

The resulting adapter is saved next to the others under
``artifacts/checkpoints/<a>_<b>/`` and is used by Phase 4.5 Test 1 to
compare the intersection-trained expert distribution against the PoE
composition of the two single-vertical experts.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    pair: str = typer.Option("long:formal"),
    target_size: int = 20_000,
    max_examples_seen: int = 5_000_000,
    out_dir: Path = Path("artifacts/datasets"),
    checkpoints_dir: Path = Path("artifacts/checkpoints"),
    num_steps: int = 2500,
) -> None:
    from src.data.build_datasets import DEFAULT_VERTICAL_SPECS, build_intersection_dataset
    from src.energies import build_default_energies
    from src.training.train_expert import ExpertTrainingConfig, train

    a, b = (s.strip() for s in pair.split(":"))
    spec_by_name = {s.name: s for s in DEFAULT_VERTICAL_SPECS}
    if a not in spec_by_name or b not in spec_by_name:
        raise typer.BadParameter(f"unknown vertical name(s): {a}, {b}")

    intersect_name = f"{a}_{b}"
    jsonl_path = out_dir / f"{intersect_name}.jsonl"

    typer.echo(f"Building intersection dataset {a} ∩ {b} → {jsonl_path}")
    build_intersection_dataset(
        spec_by_name[a],
        spec_by_name[b],
        out_path=jsonl_path,
        energies=build_default_energies(),
        target_size=target_size,
        max_examples_seen=max_examples_seen,
    )

    typer.echo(f"Fine-tuning expert on {jsonl_path}")
    cfg = ExpertTrainingConfig(
        expert_name=intersect_name,
        train_jsonl=jsonl_path,
        output_dir=checkpoints_dir,
        num_steps=num_steps,
        wandb_run_name=f"expert-{intersect_name}",
    )
    ckpt = train(cfg)
    typer.echo(f"trained {intersect_name} → {ckpt}")


if __name__ == "__main__":
    app()
