#!/usr/bin/env python
"""Extension to N=3 experts: triple satisfaction vs product of marginals."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    triplet: list[str] = typer.Option(["long", "formal", "nontoxic"]),
    n_samples: int = 200,
    out_json: Path = Path("artifacts/n3_results.json"),
    out_png: Path = Path("artifacts/plots/n3_satisfaction.png"),
) -> None:
    raise NotImplementedError("run after the composition sweep")


if __name__ == "__main__":
    app()
