#!/usr/bin/env python
"""Compute the empirical Gram matrix of the four proxy energies on OpenWebText."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from tqdm import tqdm

from src.energies import build_default_energies
from src.energies.gram_matrix import compute_gram
from src.energies.visualize import plot_gram_heatmap, plot_pair_scatters

app = typer.Typer(add_completion=False)


@app.command()
def main(
    n_samples: int = 5000,
    out_dir: Path = Path("artifacts"),
    text_max_chars: int = 1024,
) -> None:
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    energies = build_default_energies()
    energy_names = list(energies.keys())

    typer.echo(f"Streaming OpenWebText, gathering {n_samples} texts...")
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    texts: list[str] = []
    for ex in ds:
        if len(texts) >= n_samples:
            break
        t = ex["text"]
        if isinstance(t, str) and t.strip():
            texts.append(t[:text_max_chars])

    E = np.zeros((len(texts), len(energy_names)), dtype=np.float64)
    for j, name in enumerate(energy_names):
        typer.echo(f"Scoring with E_{name} ...")
        for i, t in enumerate(tqdm(texts)):
            E[i, j] = energies[name](t)

    result = compute_gram(E, energy_names)
    (out_dir / "gram_matrix.json").write_text(json.dumps(result.to_json(), indent=2))
    np.save(out_dir / "E_matrix.npy", E)

    plot_gram_heatmap(result.G, energy_names, out_dir / "plots" / "gram_heatmap.png")
    plot_pair_scatters(E, energy_names, out_dir / "plots" / "gram_pair_scatters.png")

    typer.echo("\nPairwise κ:")
    for (a, b), k in sorted(result.pair_kappas.items(), key=lambda kv: kv[1]):
        typer.echo(f"  {a:>4} × {b:<8}  κ = {k:.3f}")
    typer.echo(f"Global κ: {result.kappa_global:.3f}")


if __name__ == "__main__":
    app()
