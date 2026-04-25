#!/usr/bin/env python
"""Composition sweep: generate samples per (expert pair, λ config)."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


# The naive LoRA-merge baseline does not take per-expert λs; it is built by
# averaging adapter parameters before sampling (see src.composition.baselines).
CONFIGS: dict[str, dict[str, float] | None] = {
    "baseline": {"a": 0.0, "b": 0.0},
    "expert-A-only": {"a": 1.0, "b": 0.0},
    "expert-B-only": {"a": 0.0, "b": 1.0},
    "PoE-strict": {"a": 1.0, "b": 1.0},
    "PoE-amp": {"a": 2.0, "b": 2.0},
    "LoRA-merge": None,
}


@app.command()
def main(
    pairs: list[str] = typer.Option(
        ["len:form", "len:sent", "form:tox", "sent:tox"],
        help="Expert pairs as 'name_a:name_b'.",
    ),
    n_samples: int = 200,
    out_dir: Path = Path("artifacts/samples"),
    summary_json: Path = Path("artifacts/joint_satisfaction.json"),
) -> None:
    raise NotImplementedError("run after expert validation has passed")


if __name__ == "__main__":
    app()
