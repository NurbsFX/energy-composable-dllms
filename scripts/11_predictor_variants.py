"""Phase 9b — Refined predictor variants.

Builds on `scripts/10_predictor_eval.py` outputs (`predictor_eval.json`)
and explores two refinements suggested by the post-hoc analysis:

* **2a — B variants**: signed and absolute deviation from baseline (0.25,
  the top-quartile reference). The hypothesis is that the *magnitude* of
  cross-axis leakage predicts deficit, regardless of direction. This
  could resolve the sign-flip we observed between backbones.

* **2c — Linear combinations of predictors**: optimise weights
  (alpha, beta, ...) on MDLM-OWT, evaluate on Qwen3 with the *same*
  weights. This is an out-of-sample validation of any meta-predictor.

Outputs ``artifacts/predictor_variants.json`` with full table.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import typer
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr

app = typer.Typer(add_completion=False)


def load_predictor_eval(path: Path) -> dict:
    return json.loads(path.read_text())


def correlate(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return {"n": len(x), "r": float("nan"), "p": float("nan")}
    r, p = pearsonr(x, y)
    return {"n": len(x), "r": float(r), "p": float(p)}


def aligned_arrays(predictor_vals: dict, deficits: dict, key_subset=None) -> tuple:
    """Return aligned numpy arrays (predictor, deficit) on common keys."""
    keys = sorted(set(predictor_vals) & set(deficits))
    if key_subset is not None:
        keys = [k for k in keys if k in key_subset]
    x = np.array([predictor_vals[k] for k in keys])
    y = np.array([deficits[k] for k in keys])
    return keys, x, y


# ---------------------------------------------------------------------------
# 2a — B variants
# ---------------------------------------------------------------------------


def compute_B_variants(predictor_eval: dict) -> dict:
    """Take the per-pair B leakage values and compute centred / absolute
    variants per backbone."""
    out = {}
    for bb_name, bb_data in predictor_eval["per_backbone"].items():
        if "B_leakage" not in bb_data:
            continue
        B_vals = bb_data["B_leakage"]["values"]
        deficits = predictor_eval["cross_backbone"]["B_leakage"]["deficits"]
        # Filter deficits to this backbone
        bb_deficits = {
            k.split("/", 1)[1]: v for k, v in deficits.items() if k.startswith(f"{bb_name}/")
        }

        variants = {}
        # ``baseline`` here is a closure-invariant constant (the top-quartile
        # reference, 0.25), so the standard B023 closure-binding warning does
        # not apply. Bind via default arg to silence the lint rule.
        for variant_name, fn in [
            ("B_raw", lambda b: b),
            ("B_centered", lambda b, baseline=0.25: b - baseline),
            ("B_abs_dev", lambda b, baseline=0.25: abs(b - baseline)),
        ]:
            vals = {k: fn(v) for k, v in B_vals.items()}
            keys, x, y = aligned_arrays(vals, bb_deficits)
            corr = correlate(x, y)
            variants[variant_name] = {"values": vals, "corr": corr}

        out[bb_name] = variants
    return out


# ---------------------------------------------------------------------------
# 2c — Linear combinations, optimised on one backbone, tested on the other
# ---------------------------------------------------------------------------


def best_linear_combo(
    pred_a_vals: dict[str, float],
    pred_b_vals: dict[str, float],
    deficits: dict[str, float],
):
    """Find weight ``w`` such that ``r(w * pred_a + (1-w) * pred_b, deficit)``
    is *maximised* in absolute value. ``w`` ∈ [-1, 2] (allows inversion).
    """
    keys = sorted(set(pred_a_vals) & set(pred_b_vals) & set(deficits))
    if len(keys) < 3:
        return None
    a = np.array([pred_a_vals[k] for k in keys])
    b = np.array([pred_b_vals[k] for k in keys])
    y = np.array([deficits[k] for k in keys])

    def neg_abs_r(w: float) -> float:
        x = w * a + (1 - w) * b
        if x.std() < 1e-12:
            return 0.0
        r, _ = pearsonr(x, y)
        return -abs(r)

    res = minimize_scalar(neg_abs_r, bounds=(-1.0, 2.0), method="bounded")
    return float(res.x)


def evaluate_combo(
    pred_a_vals: dict[str, float],
    pred_b_vals: dict[str, float],
    deficits: dict[str, float],
    weight: float,
) -> dict:
    keys = sorted(set(pred_a_vals) & set(pred_b_vals) & set(deficits))
    a = np.array([pred_a_vals[k] for k in keys])
    b = np.array([pred_b_vals[k] for k in keys])
    y = np.array([deficits[k] for k in keys])
    x = weight * a + (1 - weight) * b
    return correlate(x, y)


def evaluate_pair_cross_backbone(
    predictor_eval: dict, pred_name_a: str, pred_name_b: str
) -> dict | None:
    """Optimise (α, β) on MDLM-OWT using α·A + β·B; evaluate same coefficients on Qwen3."""
    bbs = predictor_eval["per_backbone"]
    if "mdlm_owt" not in bbs or "qwen3" not in bbs:
        return None
    mdlm = bbs["mdlm_owt"]
    qwen = bbs["qwen3"]
    if pred_name_a not in mdlm or pred_name_a not in qwen:
        return None
    if pred_name_b not in mdlm or pred_name_b not in qwen:
        return None

    # Per-backbone deficits (from cross_backbone block)
    cross_def = predictor_eval["cross_backbone"][pred_name_a]["deficits"]
    mdlm_def = {k.split("/", 1)[1]: v for k, v in cross_def.items() if k.startswith("mdlm_owt/")}
    qwen_def = {k.split("/", 1)[1]: v for k, v in cross_def.items() if k.startswith("qwen3/")}

    # Optimise weight on mdlm_owt
    w_opt = best_linear_combo(mdlm[pred_name_a]["values"], mdlm[pred_name_b]["values"], mdlm_def)
    if w_opt is None:
        return None

    # Evaluate on each backbone with same weight
    mdlm_corr = evaluate_combo(
        mdlm[pred_name_a]["values"], mdlm[pred_name_b]["values"], mdlm_def, w_opt
    )
    qwen_corr = evaluate_combo(
        qwen[pred_name_a]["values"], qwen[pred_name_b]["values"], qwen_def, w_opt
    )

    # Cross-backbone stacked
    all_a = {f"mdlm_owt/{k}": v for k, v in mdlm[pred_name_a]["values"].items()}
    all_a.update({f"qwen3/{k}": v for k, v in qwen[pred_name_a]["values"].items()})
    all_b = {f"mdlm_owt/{k}": v for k, v in mdlm[pred_name_b]["values"].items()}
    all_b.update({f"qwen3/{k}": v for k, v in qwen[pred_name_b]["values"].items()})
    all_def = cross_def
    cross_corr = evaluate_combo(all_a, all_b, all_def, w_opt)

    return {
        "predictor_a": pred_name_a,
        "predictor_b": pred_name_b,
        "weight_optimised_on_mdlm": w_opt,
        "mdlm_corr": mdlm_corr,
        "qwen_corr": qwen_corr,
        "cross_corr": cross_corr,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@app.command()
def main(
    in_json: Path = Path("artifacts/predictor_eval.json"),
    out_json: Path = Path("artifacts/predictor_variants.json"),
) -> None:
    pe = load_predictor_eval(in_json)

    # 2a — B variants
    typer.echo("\n=== 2a: B variants ===")
    typer.echo(f"  {'backbone':<10s}  {'variant':<14s}  {'n':>2s}  {'r':>7s}  {'p':>7s}")
    typer.echo("  " + "-" * 50)
    B_variants = compute_B_variants(pe)
    for bb_name in B_variants:
        for variant_name, vd in B_variants[bb_name].items():
            corr = vd["corr"]
            typer.echo(
                f"  {bb_name:<10s}  {variant_name:<14s}  {corr['n']:>2d}  "
                f"{corr['r']:>+7.3f}  {corr['p']:>7.3f}"
            )

    # 2c — Linear combos: all pairs of predictors
    typer.echo("\n=== 2c: Linear combos (weight optimised on MDLM-OWT, tested on Qwen3) ===")
    typer.echo(
        f"  {'pred_a + pred_b':<26s}  {'w*':>5s}  {'r_mdlm':>7s}  {'r_qwen':>7s}  {'r_cross':>8s}"
    )
    typer.echo("  " + "-" * 70)
    pred_names = sorted(set(pe["per_backbone"]["mdlm_owt"]) & set(pe["per_backbone"]["qwen3"]))
    combos_results = []
    for pa, pb in combinations(pred_names, 2):
        res = evaluate_pair_cross_backbone(pe, pa, pb)
        if res is None:
            continue
        combos_results.append(res)
        label = f"{pa} + {pb}"
        if len(label) > 26:
            label = label[:23] + "..."
        typer.echo(
            f"  {label:<26s}  {res['weight_optimised_on_mdlm']:>5.2f}  "
            f"{res['mdlm_corr']['r']:>+7.3f}  {res['qwen_corr']['r']:>+7.3f}  "
            f"{res['cross_corr']['r']:>+8.3f}"
        )

    # Summary: best combo on cross-backbone
    if combos_results:
        sorted_combos = sorted(combos_results, key=lambda r: -abs(r["cross_corr"]["r"]))
        typer.echo("\n=== Top 3 combos by |cross-backbone r| ===")
        for r in sorted_combos[:3]:
            typer.echo(
                f"  {r['predictor_a']} + {r['predictor_b']} (w={r['weight_optimised_on_mdlm']:.2f}) "
                f"→ cross r = {r['cross_corr']['r']:+.3f} (n={r['cross_corr']['n']})"
            )

    output = {
        "B_variants": B_variants,
        "linear_combos": combos_results,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2))
    typer.echo(f"\nWrote {out_json}")


if __name__ == "__main__":
    app()
