#!/usr/bin/env python
"""HSIC, CKA and KSG mutual information from the cached Phase-1 ``E_matrix``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from src.energies.gram_matrix import compute_gram
from src.energies.independence import compute_independence

app = typer.Typer(add_completion=False)


@app.command()
def main(
    e_matrix_npy: Path = Path("artifacts/E_matrix.npy"),
    gram_json: Path = Path("artifacts/gram_matrix.json"),
    out_json: Path = Path("artifacts/independence_metrics.json"),
) -> None:
    if not e_matrix_npy.exists():
        typer.echo(f"missing {e_matrix_npy}; run scripts/01_compute_gram.py first", err=True)
        raise typer.Exit(code=1)

    E = np.load(e_matrix_npy)
    energy_names = json.loads(gram_json.read_text())["energy_names"]
    typer.echo(f"E_matrix shape: {E.shape}, names: {energy_names}")

    typer.echo("Computing HSIC + CKA + MI ...")
    result = compute_independence(E, energy_names)
    gram = compute_gram(E, energy_names)

    out_json.write_text(json.dumps(result.to_json(), indent=2))

    typer.echo("\n  pair          |    κ     |   HSIC    |   CKA   |    MI   ")
    typer.echo("  " + "-" * 60)
    for pair in sorted(result.pair_cka.keys(), key=lambda p: -result.pair_cka[p]):
        a, b = pair
        typer.echo(
            f"  {a:>4} × {b:<6} | "
            f"{gram.pair_kappas[pair]:6.3f}  | "
            f"{result.pair_hsic[pair]:8.5f} | "
            f"{result.pair_cka[pair]:6.3f}  | "
            f"{result.pair_mi[pair]:6.3f}"
        )
    typer.echo(f"\nWrote {out_json}")


if __name__ == "__main__":
    app()
