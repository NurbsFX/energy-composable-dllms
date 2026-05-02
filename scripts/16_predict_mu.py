"""Phase 12 protocol C — Structural predictor of μ*.

Given the 6 (mainly) Phase-11 setups for which we have a known μ*, fit a
small regression model (μ_predicted vs structural features). Features:

* ``N`` — number of experts in the composition
* ``stylistic_load`` — fraction of axes from the *style* family
  (formal, positive, positive2). The remaining axes are "lexical"
  (concrete, sports, long).
* ``backbone_capacity`` — coarse log-scale proxy of backbone size:
  log10(num_params).
* ``mean_marginal`` — average solo-marginal of the experts (a proxy
  of how strongly each expert is pushing).
* ``leakage_mean`` — mean cross-axis leakage (for available pairs in
  the triplet, i.e., the off-diagonal of the leakage matrix).

These features are computed deterministically from existing data
(no model forward needed). We fit a simple model (linear regression
or k-NN) and report leave-one-out cross-validation.

The point is **not** to ship an exact predictor (we only have 6
data points!) but to investigate which features carry signal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(add_completion=False)

# Stylistic vs lexical axis classification. Length is treated as lexical
# since it acts as a span constraint, not as a register/sentiment signal.
STYLISTIC = {"formal", "positive", "positive2"}
LEXICAL = {"long", "concrete", "sports"}

# Backbone parameter counts (millions) — coarse proxies of capacity.
BACKBONE_CAPACITY = {
    "kuleshov-group/mdlm-owt": 110.0,
    "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1": 596.0,
}


def stylistic_load(experts: list[str]) -> float:
    n_styl = sum(1 for e in experts if e in STYLISTIC)
    return n_styl / max(1, len(experts))


def parse_setup_record(record: dict) -> dict:
    """Extract features and known μ* from a Phase-11 sweep result JSON."""
    triplet = record["triplet"]
    backbone = record["backbone"]
    marginals = record["marginals"]
    indep_ref = record["indep_ref"]
    sweep_results = record.get("sweep_results", {})

    # Identify μ* by the highest ratio in the sweep
    if not sweep_results:
        return {}
    best_entry = max(sweep_results.values(), key=lambda r: r["ratio"])
    mu_star = float(best_entry["mu"])
    best_ratio = float(best_entry["ratio"])

    # Features
    N = len(triplet)
    sl = stylistic_load(triplet)
    cap = BACKBONE_CAPACITY.get(backbone, 0.0)
    cap_log = float(np.log10(max(cap, 1.0)))
    mean_marginal = float(np.mean([marginals[t] for t in triplet]))

    return {
        "triplet": triplet,
        "backbone": backbone,
        "N": N,
        "stylistic_load": float(sl),
        "capacity_M": cap,
        "capacity_log10M": cap_log,
        "mean_marginal": mean_marginal,
        "indep_ref": indep_ref,
        "mu_star": mu_star,
        "best_ratio": best_ratio,
    }


def fit_linear_loo(setups: list[dict]) -> dict:
    """Fit μ* = a + b·N + c·stylistic_load + d·log10(capacity) and report LOO MAE."""
    if len(setups) < 4:
        return {"error": f"too few setups ({len(setups)})"}
    X = np.array(
        [[s["N"], s["stylistic_load"], s["capacity_log10M"], s["mean_marginal"]] for s in setups]
    )
    y = np.array([s["mu_star"] for s in setups])

    n = len(y)
    preds = np.zeros(n)
    coefs_history = []
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        Xtr, ytr = X[train_idx], y[train_idx]
        # Center
        Xmean, ymean = Xtr.mean(axis=0), ytr.mean()
        Xc, yc = Xtr - Xmean, ytr - ymean
        # Ridge to handle small n
        coefs, *_ = np.linalg.lstsq(
            np.vstack([Xc, np.eye(Xc.shape[1]) * 0.1]),
            np.concatenate([yc, np.zeros(Xc.shape[1])]),
            rcond=None,
        )
        intercept = ymean - Xmean @ coefs
        preds[i] = X[i] @ coefs + intercept
        coefs_history.append(coefs.tolist())

    mae = float(np.mean(np.abs(preds - y)))

    # Final fit on all data (for reporting coefficients)
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    coefs, *_ = np.linalg.lstsq(
        np.vstack([Xc, np.eye(Xc.shape[1]) * 0.1]),
        np.concatenate([yc, np.zeros(Xc.shape[1])]),
        rcond=None,
    )
    intercept = float(y.mean() - X.mean(axis=0) @ coefs)

    return {
        "feature_names": ["N", "stylistic_load", "capacity_log10M", "mean_marginal"],
        "coefficients": coefs.tolist(),
        "intercept": intercept,
        "loo_mae": mae,
        "loo_predictions": preds.tolist(),
        "ground_truth": y.tolist(),
    }


def fit_simple_rule(setups: list[dict]) -> dict:
    """Test the simple structural rule:
        μ* ≈ −(N − stylistic_load·N) − backbone_correction
    where backbone_correction = 1 for small (≤200M) backbones, 0 otherwise.

    This is the qualitative pattern observed in §13.4 of the paper.
    """
    preds = []
    for s in setups:
        bc = 1 if s["capacity_M"] <= 200 else 0
        pred = -(s["N"] - s["stylistic_load"] * s["N"]) - bc
        # Clip into realistic range
        pred = max(-2.5, min(0.5, pred))
        preds.append(pred)
    y = np.array([s["mu_star"] for s in setups])
    preds = np.array(preds)
    mae = float(np.mean(np.abs(preds - y)))

    return {
        "rule": "μ* ≈ −N(1−stylistic_load) − [1 if cap≤200M else 0]",
        "predictions": preds.tolist(),
        "ground_truth": y.tolist(),
        "mae": mae,
    }


@app.command()
def main(
    artifact_root: Path = Path.home() / "Documents/composable-dllms-artifacts",
    out_json: Path = Path("artifacts/predict_mu.json"),
) -> None:
    # Load all available μ-sweep records from local artifacts
    records: list[dict] = []
    candidate_files = (
        [
            artifact_root / "n3_mu_sweep.json",
        ]
        + sorted((artifact_root / "mu").glob("*.json"))
        if (artifact_root / "mu").exists()
        else [artifact_root / "n3_mu_sweep.json"]
    )
    for path in candidate_files:
        if not path.exists():
            continue
        try:
            records.append(json.loads(path.read_text()))
        except Exception as e:
            typer.echo(f"  skip {path}: {e}", err=True)

    setups = [parse_setup_record(r) for r in records if r]
    setups = [s for s in setups if s]
    typer.echo(f"Parsed {len(setups)} setups with known μ*:")
    for s in setups:
        typer.echo(
            f"  N={s['N']} sl={s['stylistic_load']:.2f} cap={s['capacity_M']:>5.0f}M "
            f"meanmarg={s['mean_marginal']:.3f}  μ*={s['mu_star']:+.2f}  r*={s['best_ratio']:.2f}"
        )

    # Linear regression LOO
    typer.echo("\n=== Linear regression LOO ===")
    lin = fit_linear_loo(setups)
    if "error" not in lin:
        typer.echo(f"  intercept = {lin['intercept']:+.3f}")
        for name, c in zip(lin["feature_names"], lin["coefficients"], strict=True):
            typer.echo(f"  coef[{name}] = {c:+.3f}")
        typer.echo(f"  LOO MAE = {lin['loo_mae']:.3f}")
        for s, pred, true in zip(setups, lin["loo_predictions"], lin["ground_truth"], strict=True):
            typer.echo(f"    {'×'.join(s['triplet']):<35s}  pred={pred:+.2f}  true={true:+.2f}")
    else:
        typer.echo(f"  {lin['error']}")

    # Simple structural rule
    typer.echo("\n=== Simple structural rule ===")
    rule = fit_simple_rule(setups)
    typer.echo(f"  rule: {rule['rule']}")
    typer.echo(f"  MAE = {rule['mae']:.3f}")
    for s, pred, true in zip(setups, rule["predictions"], rule["ground_truth"], strict=True):
        typer.echo(f"    {'×'.join(s['triplet']):<35s}  pred={pred:+.2f}  true={true:+.2f}")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps({"setups": setups, "linear": lin, "simple_rule": rule}, indent=2)
    )
    typer.echo(f"\nWrote {out_json}")


if __name__ == "__main__":
    app()
