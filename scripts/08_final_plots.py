#!/usr/bin/env python
"""Build the κ-vs-deficit figure from the artefacts of phases 1 and 4."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from src.eval.kappa_vs_quality import KappaDeficitPoint, fit, plot_kappa_vs_deficit

app = typer.Typer(add_completion=False)


@app.command()
def main(
    gram_json: Path = Path("artifacts/gram_matrix.json"),
    js_json: Path = Path("artifacts/joint_satisfaction.json"),
    out_dir: Path = Path("artifacts/plots"),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gram = json.loads(gram_json.read_text())
    js = json.loads(js_json.read_text())

    points: list[KappaDeficitPoint] = []
    for key, kappa in gram["pair_kappas"].items():
        if key not in js:
            continue
        a, b = key.split("|")
        points.append(
            KappaDeficitPoint(
                pair=(a, b),
                kappa=float(kappa),
                js_poe=float(js[key]["poe_strict"]),
                js_indep=float(js[key]["indep"]),
            )
        )

    if len(points) < 3:
        typer.echo(f"need ≥ 3 pairs to fit, got {len(points)}", err=True)
        raise typer.Exit(code=1)

    fit_result = fit(points)
    typer.echo(
        f"Pearson r = {fit_result.pearson_r:.3f}  "
        f"(p = {fit_result.pearson_p:.3f})  "
        f"slope = {fit_result.slope:.3f}"
    )

    out_png = out_dir / "kappa_vs_deficit.png"
    plot_kappa_vs_deficit(fit_result, str(out_png))
    typer.echo(f"wrote {out_png}")


if __name__ == "__main__":
    app()
