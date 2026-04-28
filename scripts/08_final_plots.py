#!/usr/bin/env python
"""Build the metric-vs-deficit figures from phases 1, 1b and 4 artefacts.

Reads:
* ``artifacts/gram_matrix.json``        — produced by ``01_compute_gram.py``
* ``artifacts/independence_metrics.json`` — produced by ``01b_independence_metrics.py``
* ``artifacts/joint_satisfaction.json`` — produced by ``05_run_composition.py``

Writes one panel per dependence metric (κ / Spearman / CKA / MI) plus the
combined four-panel figure, all with bootstrap-CI annotations and a
jackknife report on standard out.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from src.eval.kappa_vs_quality import (
    MetricDeficitPoint,
    fit,
    plot_all_metrics,
    plot_metric_vs_deficit,
)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    gram_json: Path = Path("artifacts/gram_matrix.json"),
    independence_json: Path = Path("artifacts/independence_metrics.json"),
    js_json: Path = Path("artifacts/joint_satisfaction.json"),
    out_dir: Path = Path("artifacts/plots"),
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    gram = json.loads(gram_json.read_text())
    indep = json.loads(independence_json.read_text())
    js = json.loads(js_json.read_text())

    # gram + independence keys are *proxy* pairs (e.g. "form|sent"); js keys
    # are *expert* pairs (e.g. "formal|positive"). Build the mapping from
    # the canonical vertical specs and then iterate by joint-satisfaction
    # entry so we only score the experiments that actually ran.
    from src.data.build_datasets import DEFAULT_VERTICAL_SPECS

    expert_to_proxy = {s.name: s.energy_key for s in DEFAULT_VERTICAL_SPECS}
    energy_order = list(gram["energy_names"])

    def proxy_key(a: str, b: str) -> str:
        pa, pb = expert_to_proxy[a], expert_to_proxy[b]
        # gram stores pairs by their order in `energy_names` (i < j), so
        # reorder to match.
        if energy_order.index(pa) > energy_order.index(pb):
            pa, pb = pb, pa
        return f"{pa}|{pb}"

    points: list[MetricDeficitPoint] = []
    for js_key, js_entry in js.items():
        a, b = js_key.split("|")
        if a not in expert_to_proxy or b not in expert_to_proxy:
            continue
        gk = proxy_key(a, b)
        if gk not in gram["pair_kappas"]:
            continue
        points.append(
            MetricDeficitPoint(
                pair=(a, b),
                kappa=float(gram["pair_kappas"][gk]),
                spearman_abs=abs(float(indep["pair_spearman"][gk])),
                cka=float(indep["pair_cka"][gk]),
                mi=float(indep["pair_mi"][gk]),
                js_poe=float(js_entry["__indep_reference__"]["poe_strict"]),
                js_indep=float(js_entry["__indep_reference__"]["indep"]),
            )
        )

    if len(points) < 3:
        typer.echo(f"need ≥ 3 pairs to fit, got {len(points)}", err=True)
        raise typer.Exit(code=1)

    fits = {
        m: fit(points, metric=m, n_bootstrap=n_bootstrap, seed=seed)
        for m in ("kappa", "spearman", "cka", "mi")
    }

    typer.echo(f"\nFit results on {len(points)} pairs:")
    typer.echo(f"  {'metric':<10} {'r':>6} {'CI95':>16} {'slope':>8} {'jackknife r range':>25}")
    for m, fr in fits.items():
        r_lo, r_hi = fr.pearson_r_ci95
        jk_lo, jk_hi = fr.jackknife_r_range
        typer.echo(
            f"  {m:<10} {fr.pearson_r:>6.3f} "
            f"[{r_lo:>+5.2f}, {r_hi:>+5.2f}] "
            f"{fr.slope:>8.3f} "
            f"[{jk_lo:>+5.2f}, {jk_hi:>+5.2f}]"
        )

    for m, fr in fits.items():
        out_png = out_dir / f"{m}_vs_deficit.png"
        plot_metric_vs_deficit(fr, str(out_png))
    combined = out_dir / "metrics_vs_deficit.png"
    plot_all_metrics(fits, str(combined))
    typer.echo(f"\nWrote per-metric plots and {combined}")


if __name__ == "__main__":
    app()
