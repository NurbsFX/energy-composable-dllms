#!/usr/bin/env python
"""Plan-B Test 1: train one or more experts on intersection corpora.

For each requested ``--pairs a:b``:

1. Stream OpenWebText (shared across all requested intersections in a
   single pass) and write a JSONL containing only documents that pass
   both vertical filters.
2. Fine-tune one LoRA adapter on each intersection JSONL with the same
   hyperparameters as the six per-vertical experts.

Resulting adapters are saved next to the per-vertical ones under
``artifacts/checkpoints/<a>_<b>/`` and are picked up by Phase 4.5
Test 1 (``scripts/06b_test1_intersection_check.py``).
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    pairs: list[str] = typer.Option(
        ["long:formal"], help="Intersection pair(s) as 'name_a:name_b'."
    ),
    target_size: int = 20_000,
    max_examples_seen: int = 5_000_000,
    out_dir: Path = Path("artifacts/datasets"),
    checkpoints_dir: Path = Path("artifacts/checkpoints"),
    num_steps: int = 2500,
) -> None:
    from src.data.build_datasets import (
        DEFAULT_VERTICAL_SPECS,
        IntersectionSpec,
        build_intersections,
    )
    from src.energies import build_default_energies
    from src.training.train_expert import ExpertTrainingConfig, train

    spec_by_name = {s.name: s for s in DEFAULT_VERTICAL_SPECS}
    intersection_specs: list[IntersectionSpec] = []
    for pair_str in pairs:
        a, b = (s.strip() for s in pair_str.split(":"))
        if a not in spec_by_name or b not in spec_by_name:
            raise typer.BadParameter(
                f"unknown vertical name(s) in {pair_str!r}; expected from {sorted(spec_by_name)}"
            )
        intersection_specs.append(
            IntersectionSpec(
                name=f"{a}_{b}",
                spec_a=spec_by_name[a],
                spec_b=spec_by_name[b],
                target_size=target_size,
            )
        )

    typer.echo(
        f"Building {len(intersection_specs)} intersection dataset(s) in one OWT pass: "
        f"{[s.name for s in intersection_specs]}"
    )
    paths = build_intersections(
        intersection_specs,
        out_dir=out_dir,
        energies=build_default_energies(),
        max_examples_seen=max_examples_seen,
    )

    for spec in intersection_specs:
        if spec.name not in paths:
            typer.echo(f"  ⚠ no documents accepted for {spec.name}; skipping training.", err=True)
            continue
        typer.echo(f"\nFine-tuning expert on {paths[spec.name]}")
        cfg = ExpertTrainingConfig(
            expert_name=spec.name,
            train_jsonl=paths[spec.name],
            output_dir=checkpoints_dir,
            num_steps=num_steps,
            wandb_run_name=f"expert-{spec.name}",
        )
        ckpt = train(cfg)
        typer.echo(f"trained {spec.name} → {ckpt}")


if __name__ == "__main__":
    app()
