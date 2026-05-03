"""C4 — Linear probe regression of PoE-2 ratio on per-pair features.

For each PoE-2 pair (a, b) on each backbone, compute features:
* ``marg_a``, ``marg_b``: solo λ=1 marginals
* ``marg_diff``: |marg_a − marg_b| (asymmetry of expert strengths)
* ``marg_min``: min(marg_a, marg_b) (the weak link)
* ``marg_geom_mean``: sqrt(marg_a · marg_b) — captures both joint strength
* ``cos_dW``: cosine of LoRA-induced ΔW (from C1)
* ``kappa_ab``: |C[proxy_a, proxy_b]| from Paper-1 ``gram_matrix.json``
  (proxy correlation on OWT, model-free semantic distance)

Target: PoE-2 ratio = joint_sat / indep_ref.

Fit a ridge linear regression on ~30 pairs (10 pairs × 3 backbones if all
exist locally; here we have 2 MDLM backbones, so 20 pairs). Report
coefficients (standardized), R², and identify which feature(s) carry the
predictive signal.

Outputs:
* ``artifacts/linear_probe_deficit.json``
* ``artifacts/plots/linear_probe_deficit_coefs.png``
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer

app = typer.Typer(add_completion=False)


EXPERTS = ["long", "formal", "positive", "positive2", "concrete", "sports"]
EXPERT_TO_PROXY = {
    "long": "len",
    "formal": "form",
    "positive": "sent",
    "positive2": "sent2",
    "concrete": "conc",
    "sports": "topic",
}


def _load_adapter_delta(ckpt_dir: Path) -> np.ndarray | None:
    cfg_path = ckpt_dir / "adapter_config.json"
    safetensors_path = ckpt_dir / "adapter_model.safetensors"
    if not (cfg_path.exists() and safetensors_path.exists()):
        return None
    cfg = json.loads(cfg_path.read_text())
    rank = cfg.get("r", cfg.get("lora_rank", 8))
    alpha = cfg.get("lora_alpha", cfg.get("alpha", 16))
    scale = float(alpha) / float(rank)
    from safetensors import safe_open

    deltas = []
    with safe_open(safetensors_path, framework="np") as f:
        keys = list(f.keys())
        module_paths = sorted(
            {k.rsplit(".lora_A.weight", 1)[0] for k in keys if k.endswith(".lora_A.weight")}
        )
        for mp in module_paths:
            a = f.get_tensor(f"{mp}.lora_A.weight")
            b = f.get_tensor(f"{mp}.lora_B.weight")
            deltas.append((scale * (b @ a)).reshape(-1))
    if not deltas:
        return None
    return np.concatenate(deltas)


def _cos(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def _ridge(X: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> tuple[np.ndarray, float, float]:
    """Fit centered ridge, return (coefs, intercept, R²)."""
    Xmean, ymean = X.mean(axis=0), y.mean()
    Xc = X - Xmean
    yc = y - ymean
    coefs, *_ = np.linalg.lstsq(
        np.vstack([Xc, np.eye(Xc.shape[1]) * alpha]),
        np.concatenate([yc, np.zeros(Xc.shape[1])]),
        rcond=None,
    )
    intercept = float(ymean - Xmean @ coefs)
    yhat = X @ coefs + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - ymean) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return coefs, intercept, r2


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd_safe = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd_safe, mu, sd


def _ratios_for_backbone(js_json_path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Return per-pair {marg_a, marg_b, ratio}.

    Use solo λ=1 marginals from `expert-A-only` / `expert-B-only` (the
    `baseline` config has m_a=m_b≈0.25 by top-quartile construction —
    not informative for cross-pair variance).
    """
    if not js_json_path.exists():
        return {}
    js = json.loads(js_json_path.read_text())
    out: dict[tuple[str, str], dict[str, float]] = {}
    for label, configs in js.items():
        if "|" not in label:
            continue
        a, b = label.split("|")
        # Solo marginals from each expert running solo at λ=1
        a_only = configs.get("expert-A-only", {})
        b_only = configs.get("expert-B-only", {})
        m_a = float(a_only.get("marginal_a", 0.25))
        m_b = float(b_only.get("marginal_b", 0.25))
        indep = max(m_a * m_b, 1e-9)
        poe = configs.get("PoE-strict", {})
        if "joint_satisfaction" not in poe:
            continue
        out[(a, b)] = {
            "marg_a": m_a,
            "marg_b": m_b,
            "ratio": float(poe["joint_satisfaction"]) / indep,
        }
    return out


@app.command()
def main(
    mdlm_owt_ckpts: Path = Path.home() / "Documents/composable-dllms-artifacts/checkpoints",
    mdlm_qwen_ckpts: Path = Path.home() / "Documents/composable-dllms-artifacts/checkpoints_qwen3",
    mdlm_owt_js: Path = Path.home()
    / "Documents/composable-dllms-artifacts/joint_satisfaction.json",
    mdlm_qwen_js: Path = Path.home()
    / "Documents/composable-dllms-artifacts/qwen3_run/joint_satisfaction.json",
    gram_json: Path = Path("artifacts/gram_matrix.json"),
    out_json: Path = Path("artifacts/linear_probe_deficit.json"),
    out_png: Path = Path("artifacts/plots/linear_probe_deficit_coefs.png"),
) -> None:
    if not gram_json.exists():
        raise typer.BadParameter(f"missing {gram_json}")
    g = json.loads(gram_json.read_text())
    proxies = g["energy_names"]
    proxy_idx = {p: i for i, p in enumerate(proxies)}
    C = np.array(g["C"])

    backbones = {
        "MDLM-OWT 110M": (mdlm_owt_ckpts, mdlm_owt_js),
        "MDLM Qwen3 596M": (mdlm_qwen_ckpts, mdlm_qwen_js),
    }

    # Build the (n, k) design matrix.
    feature_names = [
        "marg_a",
        "marg_b",
        "marg_diff",
        "marg_min",
        "marg_geom_mean",
        "cos_dW",
        "kappa_ab",
    ]
    rows = []
    for bb_name, (ckpt_root, js_path) in backbones.items():
        if not ckpt_root.exists():
            continue
        deltas: dict[str, np.ndarray] = {}
        for e in EXPERTS:
            d = _load_adapter_delta(ckpt_root / e)
            if d is not None:
                deltas[e] = d
        ratios = _ratios_for_backbone(js_path)
        for a, b in itertools.combinations(EXPERTS, 2):
            r_info = ratios.get((a, b)) or ratios.get((b, a))
            if r_info is None or a not in deltas or b not in deltas:
                continue
            m_a = r_info["marg_a"]
            m_b = r_info["marg_b"]
            ratio = r_info["ratio"]
            cos_dW = _cos(deltas[a], deltas[b])
            kappa = abs(float(C[proxy_idx[EXPERT_TO_PROXY[a]], proxy_idx[EXPERT_TO_PROXY[b]]]))
            rows.append(
                {
                    "backbone": bb_name,
                    "a": a,
                    "b": b,
                    "marg_a": m_a,
                    "marg_b": m_b,
                    "marg_diff": abs(m_a - m_b),
                    "marg_min": min(m_a, m_b),
                    "marg_geom_mean": float(np.sqrt(m_a * m_b)),
                    "cos_dW": cos_dW,
                    "kappa_ab": kappa,
                    "ratio": ratio,
                }
            )

    if not rows:
        typer.echo("No data!", err=True)
        raise typer.Exit(1)

    typer.echo(f"=== C4 — linear probe on {len(rows)} pair × backbone observations ===")

    # Whole-data ridge (standardized).
    X = np.array([[r[f] for f in feature_names] for r in rows])
    y = np.array([r["ratio"] for r in rows])
    Xz, mu, sd = _standardize(X)

    # Univariate Pearson r per feature (informative independent of multi-collinearity)
    typer.echo("\nUnivariate Pearson r per feature (pooled n=20):")
    univariate = {}
    for i, name in enumerate(feature_names):
        xc = Xz[:, i] - Xz[:, i].mean()
        yc = y - y.mean()
        denom = float(np.sqrt((xc**2).sum() * (yc**2).sum()))
        r_uni = float((xc * yc).sum() / denom) if denom > 0 else float("nan")
        univariate[name] = r_uni
        typer.echo(f"  r({name:<14s}, ratio) = {r_uni:+.3f}")

    coefs_strong, intercept, r2 = _ridge(Xz, y, alpha=0.05)
    coefs_weak, _, r2_weak = _ridge(Xz, y, alpha=0.5)
    typer.echo(
        f"\nFull dataset (n={len(rows)}):  R² (α=0.05) = {r2:+.3f}, R² (α=0.5) = {r2_weak:+.3f}"
    )
    for name, c, c_w in zip(feature_names, coefs_strong.tolist(), coefs_weak.tolist(), strict=True):
        typer.echo(f"  std-coef[{name:<14s}] α=0.05: {c:+.3f}    α=0.5: {c_w:+.3f}")
    coefs = coefs_strong  # downstream use

    # Per-backbone re-fit for diagnostic.
    by_bb = {}
    for bb_name in {r["backbone"] for r in rows}:
        sub = [r for r in rows if r["backbone"] == bb_name]
        Xs = np.array([[r[f] for f in feature_names] for r in sub])
        ys = np.array([r["ratio"] for r in sub])
        Xz_s, _, _ = _standardize(Xs)
        coefs_s, intercept_s, r2_s = _ridge(Xz_s, ys, alpha=0.5)
        by_bb[bb_name] = {
            "n": len(sub),
            "coefs_std": dict(zip(feature_names, coefs_s.tolist(), strict=True)),
            "intercept": intercept_s,
            "r2": r2_s,
        }
        typer.echo(f"\n{bb_name} (n={len(sub)}): R² = {r2_s:+.3f}")
        for name, c in zip(feature_names, coefs_s.tolist(), strict=True):
            typer.echo(f"  std-coef[{name:<14s}] = {c:+.3f}")

    summary = {
        "feature_names": feature_names,
        "n_rows": len(rows),
        "rows": rows,
        "univariate_pearson_r": univariate,
        "full_dataset": {
            "n": len(rows),
            "coefs_std": dict(zip(feature_names, coefs.tolist(), strict=True)),
            "intercept": intercept,
            "r2": r2,
        },
        "per_backbone": by_bb,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    typer.echo(f"\n  wrote {out_json}")

    # --- Coefficient bar chart -----------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x_pos = np.arange(len(feature_names))
    w = 0.27
    by_bb_keys = list(by_bb.keys())
    n_bb = len(by_bb_keys)
    color_map = {"MDLM-OWT 110M": "#9467bd", "MDLM Qwen3 596M": "#1f77b4"}
    ax.bar(
        x_pos - w * (n_bb / 2),
        coefs,
        w,
        label=f"pooled (n={len(rows)}, R²={r2:+.2f})",
        color="#2ca02c",
        edgecolor="black",
        linewidth=0.5,
    )
    for i, bb in enumerate(by_bb_keys):
        offsets = w * (i + 1 - n_bb / 2)
        cs = [by_bb[bb]["coefs_std"][f] for f in feature_names]
        ax.bar(
            x_pos + offsets,
            cs,
            w,
            label=f"{bb} (n={by_bb[bb]['n']}, R²={by_bb[bb]['r2']:+.2f})",
            color=color_map.get(bb, "#777"),
            edgecolor="black",
            linewidth=0.5,
        )
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names, rotation=20, ha="right")
    ax.set_ylabel("standardized coefficient (ridge α=0.5)")
    ax.set_title(
        "C4 — Linear probe of PoE-2 ratio on per-pair features\n"
        "Standardized coefficients; positive = predicts higher ratio.",
        fontsize=10.5,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"  wrote {out_png}")


if __name__ == "__main__":
    app()
