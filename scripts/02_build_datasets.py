#!/usr/bin/env python
"""Stream OpenWebText and write one JSONL per vertical."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import typer

from src.data.build_datasets import DEFAULT_VERTICAL_SPECS, build_all
from src.energies import build_default_energies

app = typer.Typer(add_completion=False)


@app.command()
def main(
    out_dir: Path = Path("artifacts/datasets"),
    target_size: int = 80_000,
) -> None:
    energies = build_default_energies()
    specs = [dataclasses.replace(s, target_size=target_size) for s in DEFAULT_VERTICAL_SPECS]
    paths = build_all(out_dir=out_dir, energies=energies, specs=specs)
    for name, path in paths.items():
        typer.echo(f"  {name}: {path}")


if __name__ == "__main__":
    app()
