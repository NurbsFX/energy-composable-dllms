"""C1 — Cosine ΔW × PoE-2 ratio cross-backbone.

For each LoRA expert, we have ``lora_A`` and ``lora_B`` matrices per target
module. The induced weight delta is

    ΔW = (alpha / rank) · B @ A

A natural geometric measure of "how similar are two experts" is the cosine
similarity between their flattened ΔW tensors (concatenated across modules).
This script:

1. Loads the 6 mono-axis LoRA adapters for each available backbone
   (MDLM-OWT 110M, MDLM Qwen3-0.6B 596M).
2. Computes ΔW per adapter, flattens, normalises.
3. Computes the 15 pairwise cosines per backbone.
4. Pulls the PoE-strict ratio (joint_sat / indep_ref) for each of the
   10 Phase-4 pairs and the Qwen3 equivalents from
   ``joint_satisfaction.json``.
5. Reports Pearson r between cos(ΔW_a, ΔW_b) and the PoE-2 ratio,
   per backbone and pooled cross-backbone.

The hypothesis (cf. Paper-1 §10.3): if cos(ΔW) captures something about
*how the LoRA experts modify the same parts of the backbone*, then it
should predict whether two experts compose well. A high cos means
"shifts are aligned" → either super- or sub-additive depending on sign
of effect; low cos means "shifts are orthogonal" → near-additive.
The expected sign of correlation is *either* positive (aligned shifts =
super-additive) or *negative* (aligned shifts = redundant, so the joint
push is wasted on the same vocabulary directions).

Outputs:
* ``artifacts/cos_lora_vs_ratio.json`` — per-pair cos + ratio + Pearson r.
* ``artifacts/plots/cos_lora_vs_ratio.png`` — scatter, two backbones colour-coded.
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


def _load_adapter_delta(ckpt_dir: Path) -> np.ndarray | None:
    """Load LoRA adapter at ``ckpt_dir`` and return the flattened ΔW vector
    (concatenated across all target modules, normalised by alpha/rank)."""
    cfg_path = ckpt_dir / "adapter_config.json"
    safetensors_path = ckpt_dir / "adapter_model.safetensors"
    if not cfg_path.exists() or not safetensors_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())
    rank = cfg.get("r", cfg.get("lora_rank", 8))
    alpha = cfg.get("lora_alpha", cfg.get("alpha", 16))
    scale = float(alpha) / float(rank)

    # Lazy-import safetensors only when actually loading.
    from safetensors import safe_open

    deltas = []
    with safe_open(safetensors_path, framework="np") as f:
        keys = list(f.keys())
        # Group keys by module path: we expect "...lora_A.weight" and
        # "...lora_B.weight" pairs.
        module_paths = sorted(
            {k.rsplit(".lora_A.weight", 1)[0] for k in keys if k.endswith(".lora_A.weight")}
        )
        for mp in module_paths:
            a = f.get_tensor(f"{mp}.lora_A.weight")  # (r, in)
            b = f.get_tensor(f"{mp}.lora_B.weight")  # (out, r)
            dw = scale * (b @ a)  # (out, in)
            deltas.append(dw.reshape(-1))
    if not deltas:
        return None
    return np.concatenate(deltas)


def _cos(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def _ratios_for_backbone(js_json_path: Path) -> dict[tuple[str, str], float]:
    """Pull the PoE-strict joint_sat / indep_ref ratio for each pair."""
    if not js_json_path.exists():
        return {}
    js = json.loads(js_json_path.read_text())
    out: dict[tuple[str, str], float] = {}
    for label, configs in js.items():
        if "|" not in label:
            continue
        a, b = label.split("|")
        baseline = configs.get("baseline", {})
        m_a = baseline.get("marginal_a", 0.25)
        m_b = baseline.get("marginal_b", 0.25)
        # The joint_satisfaction.json uses paradigm-specific marginals;
        # we use baseline marginals to compute indep_ref. For Paper 1 the
        # indep_ref should match — re-derive consistent with §4.
        indep = float(m_a * m_b) if m_a and m_b else 1e-9
        poe = configs.get("PoE-strict", {})
        js_val = poe.get("joint_satisfaction")
        if js_val is None:
            continue
        ratio = float(js_val) / max(indep, 1e-9)
        out[(a, b)] = ratio
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    denom = float(np.sqrt((xc**2).sum() * (yc**2).sum()))
    if denom == 0:
        return float("nan"), float("nan")
    r = float((xc * yc).sum() / denom)
    n = len(x)
    if abs(r) >= 1.0:
        return r, 0.0
    t = r * np.sqrt((n - 2) / (1 - r**2))
    from math import erf, sqrt

    p = 1 - erf(abs(t) / sqrt(2))
    return r, float(p)


@app.command()
def main(
    mdlm_owt_ckpts: Path = Path.home() / "Documents/composable-dllms-artifacts/checkpoints",
    mdlm_qwen_ckpts: Path = Path.home() / "Documents/composable-dllms-artifacts/checkpoints_qwen3",
    mdlm_owt_js: Path = Path.home()
    / "Documents/composable-dllms-artifacts/joint_satisfaction.json",
    mdlm_qwen_js: Path = Path.home()
    / "Documents/composable-dllms-artifacts/qwen3_run/joint_satisfaction.json",
    out_json: Path = Path("artifacts/cos_lora_vs_ratio.json"),
    out_png: Path = Path("artifacts/plots/cos_lora_vs_ratio.png"),
) -> None:
    backbones = {
        "MDLM-OWT 110M": (mdlm_owt_ckpts, mdlm_owt_js),
        "MDLM Qwen3 596M": (mdlm_qwen_ckpts, mdlm_qwen_js),
    }
    summary: dict[str, dict] = {}
    pooled_cos = []
    pooled_ratio = []
    pooled_color = []

    typer.echo("=== C1 — cosine ΔW × PoE-2 ratio ===")
    for bb_name, (ckpt_root, js_path) in backbones.items():
        if not ckpt_root.exists():
            typer.echo(f"[skip] {bb_name}: no checkpoints at {ckpt_root}")
            continue
        deltas: dict[str, np.ndarray] = {}
        for e in EXPERTS:
            delta = _load_adapter_delta(ckpt_root / e)
            if delta is None:
                typer.echo(f"  [skip] {bb_name} / {e}: missing adapter")
                continue
            deltas[e] = delta
        # Diagnose dims: experts trained on the same backbone should produce
        # ΔW of identical flat-length.
        lens = {e: d.size for e, d in deltas.items()}
        if len(set(lens.values())) > 1:
            typer.echo(f"  [warn] {bb_name}: heterogeneous ΔW lengths {lens}")

        ratios = _ratios_for_backbone(js_path)
        rows = []
        for a, b in itertools.combinations(EXPERTS, 2):
            if a not in deltas or b not in deltas:
                continue
            # Try both orderings (a|b) and (b|a) for the ratio key.
            r = ratios.get((a, b), ratios.get((b, a)))
            if r is None:
                continue
            c = _cos(deltas[a], deltas[b])
            rows.append({"a": a, "b": b, "cos": c, "ratio": r})

        cos_vals = np.array([r["cos"] for r in rows])
        ratio_vals = np.array([r["ratio"] for r in rows])
        pr, pp = _pearson(cos_vals, ratio_vals)
        summary[bb_name] = {
            "n_pairs": len(rows),
            "rows": rows,
            "pearson_r": pr,
            "pearson_p": pp,
        }
        typer.echo(f"\n{bb_name}: n={len(rows)} pairs")
        for r in rows:
            typer.echo(
                f"  {r['a']:<10s} × {r['b']:<10s}  cos={r['cos']:+.3f}  ratio={r['ratio']:.3f}"
            )
        typer.echo(f"  Pearson r = {pr:+.3f}  (p ≈ {pp:.4f})")

        pooled_cos.extend(cos_vals.tolist())
        pooled_ratio.extend(ratio_vals.tolist())
        pooled_color.extend([bb_name] * len(rows))

    # Cross-backbone pooled correlation.
    if pooled_cos:
        pr_pool, pp_pool = _pearson(np.array(pooled_cos), np.array(pooled_ratio))
        summary["pooled"] = {
            "n_pairs": len(pooled_cos),
            "pearson_r": pr_pool,
            "pearson_p": pp_pool,
        }
        typer.echo(
            f"\nPooled cross-backbone: n={len(pooled_cos)}  r={pr_pool:+.3f}  p≈{pp_pool:.4f}"
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    typer.echo(f"\n  wrote {out_json}")

    # --- Scatter plot --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    color_map = {"MDLM-OWT 110M": "#9467bd", "MDLM Qwen3 596M": "#1f77b4"}
    for bb_name, info in summary.items():
        if bb_name == "pooled":
            continue
        rows = info["rows"]
        xs = [r["cos"] for r in rows]
        ys = [r["ratio"] for r in rows]
        ax.scatter(
            xs,
            ys,
            s=110,
            color=color_map.get(bb_name, "#777"),
            edgecolor="black",
            linewidth=0.5,
            alpha=0.92,
            label=f"{bb_name}  (r={info['pearson_r']:+.2f}, n={info['n_pairs']})",
        )
        for r in rows:
            ax.annotate(
                f"{r['a'][:3]}×{r['b'][:3]}",
                xy=(r["cos"], r["ratio"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
                color="black",
                alpha=0.85,
            )

    ax.axhline(1.0, color="grey", ls=":", lw=0.9, alpha=0.7)
    ax.set_xlabel("cos(ΔW_a, ΔW_b) — flattened LoRA-induced delta similarity")
    ax.set_ylabel("PoE-2 ratio (joint_sat / indep_ref)")
    pooled_r = summary.get("pooled", {}).get("pearson_r", float("nan"))
    pooled_p = summary.get("pooled", {}).get("pearson_p", float("nan"))
    ax.set_title(
        "C1 — cos(ΔW) vs PoE-2 ratio cross-backbone\n"
        f"pooled r = {pooled_r:+.3f}, p ≈ {pooled_p:.4f}",
        fontsize=10.5,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"  wrote {out_png}")


if __name__ == "__main__":
    app()
