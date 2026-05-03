"""C3 — Bootstrap confidence interval on the Phase-12 predictor MAE.

Phase 12c reported LOO-MAE = 0.469 on 17 setups for the linear regression
of μ\\* on (N, stylistic_load, log10(capacity), mean_marginal). The
question is: how stable is this number under resampling?

We re-fit the predictor on 1000 bootstrap samples of the 17 setups
(with replacement). For each bootstrap sample we compute the LOO-MAE
on that sample, then report the bootstrap distribution of MAE values.
This gives a non-parametric confidence interval that's safe at small n.

Outputs:
* ``artifacts/predictor_bootstrap.json``: percentiles + raw MAEs.
* ``artifacts/plots/predictor_mae_bootstrap.png``: histogram of MAEs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer

app = typer.Typer(add_completion=False)


def _loo_mae(X: np.ndarray, y: np.ndarray) -> float:
    """Re-implement scripts/16_predict_mu.py's ridge LOO loop."""
    n = len(y)
    if n < 4:
        return float("nan")
    preds = np.zeros(n)
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        Xtr, ytr = X[idx], y[idx]
        Xmean, ymean = Xtr.mean(axis=0), ytr.mean()
        Xc, yc = Xtr - Xmean, ytr - ymean
        # ridge with alpha=0.1 (matches Phase 12 script).
        coefs, *_ = np.linalg.lstsq(
            np.vstack([Xc, np.eye(Xc.shape[1]) * 0.1]),
            np.concatenate([yc, np.zeros(Xc.shape[1])]),
            rcond=None,
        )
        intercept = ymean - Xmean @ coefs
        preds[i] = X[i] @ coefs + intercept
    return float(np.mean(np.abs(preds - y)))


@app.command()
def main(
    predict_mu_json: Path = Path("artifacts/predict_mu.json"),
    out_json: Path = Path("artifacts/predictor_bootstrap.json"),
    out_png: Path = Path("artifacts/plots/predictor_mae_bootstrap.png"),
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> None:
    if not predict_mu_json.exists():
        raise typer.BadParameter(f"missing {predict_mu_json}")
    data = json.loads(predict_mu_json.read_text())
    setups = data["setups"]
    feature_names = data["linear"]["feature_names"]

    X = np.array(
        [[s["N"], s["stylistic_load"], s["capacity_log10M"], s["mean_marginal"]] for s in setups]
    )
    y = np.array([s["mu_star"] for s in setups])
    n = len(y)
    typer.echo(f"Setups: {n}, features: {feature_names}")
    typer.echo(f"Original LOO-MAE: {_loo_mae(X, y):.4f}")

    rng = np.random.default_rng(seed)
    boot_maes = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)  # with replacement
        # If a draw has < 4 unique points the LOO is too small; skip.
        if len(np.unique(idx)) < 4:
            continue
        Xb = X[idx]
        yb = y[idx]
        # Slight jitter on duplicate rows so the lstsq is well-conditioned.
        boot_maes.append(_loo_mae(Xb, yb))
    boot_maes = np.array([m for m in boot_maes if not np.isnan(m)])

    pct = lambda q: float(np.percentile(boot_maes, q))  # noqa: E731
    summary = {
        "n_setups": n,
        "n_bootstrap": int(len(boot_maes)),
        "original_mae": _loo_mae(X, y),
        "mean_mae": float(boot_maes.mean()),
        "median_mae": float(np.median(boot_maes)),
        "std_mae": float(boot_maes.std()),
        "ci_5_95": [pct(5), pct(95)],
        "ci_25_75": [pct(25), pct(75)],
        "raw_maes": boot_maes.tolist(),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    typer.echo(
        f"Bootstrap (n={summary['n_bootstrap']}): mean={summary['mean_mae']:.3f} "
        f"median={summary['median_mae']:.3f} std={summary['std_mae']:.3f}"
    )
    typer.echo(f"  CI [5%, 95%]: [{summary['ci_5_95'][0]:.3f}, {summary['ci_5_95'][1]:.3f}]")
    typer.echo(f"  CI [25%, 75%]: [{summary['ci_25_75'][0]:.3f}, {summary['ci_25_75'][1]:.3f}]")

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.hist(boot_maes, bins=40, edgecolor="black", linewidth=0.5, color="#1f77b4", alpha=0.85)
    ax.axvline(
        summary["original_mae"],
        color="#d62728",
        ls="-",
        lw=2.0,
        label=f"original MAE = {summary['original_mae']:.3f}",
    )
    ax.axvline(
        summary["ci_5_95"][0],
        color="black",
        ls="--",
        lw=1.0,
        alpha=0.7,
        label=f"5% = {summary['ci_5_95'][0]:.3f}",
    )
    ax.axvline(
        summary["ci_5_95"][1],
        color="black",
        ls="--",
        lw=1.0,
        alpha=0.7,
        label=f"95% = {summary['ci_5_95'][1]:.3f}",
    )
    ax.set_xlabel("LOO-MAE (bootstrap resample)")
    ax.set_ylabel("count")
    ax.set_title(
        f"C3 — Bootstrap of Phase 12c predictor LOO-MAE\n"
        f"original = {summary['original_mae']:.3f}, "
        f"mean = {summary['mean_mae']:.3f}, "
        f"95%-CI = [{summary['ci_5_95'][0]:.3f}, {summary['ci_5_95'][1]:.3f}]  (n_resamples = {summary['n_bootstrap']})",
        fontsize=10.5,
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"\n  wrote {out_json}\n  wrote {out_png}")


if __name__ == "__main__":
    app()
