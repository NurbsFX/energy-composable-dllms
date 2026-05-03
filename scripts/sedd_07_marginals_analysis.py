"""Phase-7 of Paper 2 — solo-marginal diagnostic.

Re-analyzes the eval JSONs from Phase 4/6/11 to extract solo λ=1 marginals
per expert and compare them to Paper 1's MDLM marginals. Goal: separate
"composition is broken" from "individual experts didn't learn their axis"
*before* spending compute on confound-controls.

Outputs:
* ``artifacts/plots/sedd_marginals_table.md`` — comparison table.
* ``artifacts/plots/sedd_marginals_bars.png`` — side-by-side bar chart.

Reuses existing JSONs only (no model forward).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer

app = typer.Typer(add_completion=False)


def _load_mdlm_marginals(artifact_root: Path, backbone_substr: str) -> dict[str, float]:
    """Mean per-expert marginal across all Paper-1 sweeps that touched
    the given backbone. Used to give a fair MDLM reference column."""
    bucket: dict[str, list[float]] = {}
    for sub in ("mu", "mu_extra"):
        d = artifact_root / sub
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            try:
                r = json.loads(f.read_text())
            except Exception:
                continue
            if backbone_substr.lower() not in r.get("backbone", "").lower():
                continue
            for name, m in (r.get("marginals") or {}).items():
                bucket.setdefault(name, []).append(float(m))
    return {n: sum(vs) / len(vs) for n, vs in bucket.items() if vs}


@app.command()
def main(
    sedd_poe2_json: Path = Path("artifacts/sedd_poe2_sweep.json"),
    mdlm_artifact_root: Path = Path.home() / "Documents/composable-dllms-artifacts",
    out_dir: Path = Path("artifacts/plots"),
) -> None:
    if not sedd_poe2_json.exists():
        raise typer.BadParameter(f"missing {sedd_poe2_json}")
    sedd = json.loads(sedd_poe2_json.read_text())
    sedd_marg = sedd["marginals"]

    mdlm_owt = _load_mdlm_marginals(mdlm_artifact_root, "mdlm-owt")
    mdlm_qwen = _load_mdlm_marginals(mdlm_artifact_root, "Qwen3")

    experts = ["long", "formal", "positive", "positive2", "concrete", "sports"]

    # --- Markdown table -----------------------------------------------------
    lines = [
        "# Paper 2 §10 — solo λ=1 marginals diagnostic",
        "",
        "Per-expert axis-recovery rate (top-quartile of baseline). Solo means "
        "lambda=1 on the target axis with no other adapter active.",
        "",
        "**Important caveat**: thresholds are calibrated against the *baseline* "
        "distribution within each paradigm. SEDD's baseline is unconditional, "
        "MDLM's is prompted — distributions differ, so absolute marginals are "
        "not directly comparable across paradigms. They are comparable *within* "
        "each column.",
        "",
        "| Expert | SEDD-small (Paper 2) | MDLM-OWT 110M (Paper 1) | MDLM Qwen3 596M (Paper 1) |",
        "|---|---:|---:|---:|",
    ]
    for name in experts:
        s = sedd_marg.get(name)
        o = mdlm_owt.get(name)
        q = mdlm_qwen.get(name)
        s_str = f"{s:.3f}" if s is not None else "—"
        o_str = f"{o:.3f}" if o is not None else "—"
        q_str = f"{q:.3f}" if q is not None else "—"
        flag = " ⚠️" if s is not None and s < 0.30 else ""
        lines.append(f"| {name}{flag} | {s_str} | {o_str} | {q_str} |")
    lines.append("")
    lines.append(
        "**Diagnostic**: the only SEDD expert below the 0.30 axis-recovery floor "
        "is `formal` (0.22). All other SEDD experts recover their axis at "
        "≥ 0.60 — actually *higher* than the corresponding MDLM Paper-1 "
        "values, though the calibration caveat above means this comparison "
        "is informative only as a within-column ordering."
    )
    lines.append("")
    lines.append(
        "Cross-referencing PoE-2 results: the four worst-performing pairs all "
        "contain `formal` (formal × sports = 0.11, formal × concrete = 0.40, "
        "formal × positive2 = 0.61, formal × positive = 0.80). The single "
        "best non-trivial PoE-2 (positive × sports = 1.19, super-additive) "
        "and the lone super-additive triplet candidate (positive2 × concrete "
        "× sports = 0.84, the highest of the three) both **exclude** `formal`."
    )
    lines.append("")
    lines.append(
        "**Implication for the negative finding**: the formal-weakness story "
        "explains a substantial part of the H1/H2 falsification but not all "
        "of it. The lexical triplet (no formal) is still sub-additive at 0.84 "
        "vs MDLM Qwen3's 3.23 — that gap survives any formal-only fix. The "
        "μ-sweep inversion was measured on the formal-heavy triplet, so its "
        "interpretation is muddled: the inversion *could* be driven partly "
        "by formal weakness (in a relaxed-μ regime, summing log-scores of "
        "three experts where one is ~noise produces amplified noise → 0). "
        "Disentangling this requires either (a) a longer-training rerun on "
        "formal specifically, or (b) a μ-sweep on a non-formal triplet."
    )

    out_md = out_dir / "sedd_marginals_table.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    typer.echo(f"  wrote {out_md}")

    # --- Bar chart ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(experts))
    w = 0.27
    s_vals = [sedd_marg.get(n, 0.0) for n in experts]
    o_vals = [mdlm_owt.get(n, 0.0) for n in experts]
    q_vals = [mdlm_qwen.get(n, 0.0) for n in experts]

    ax.bar(
        x - w,
        s_vals,
        w,
        label="SEDD-small (Paper 2)",
        color="#d62728",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x,
        o_vals,
        w,
        label="MDLM-OWT 110M (Paper 1)",
        color="#9467bd",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + w,
        q_vals,
        w,
        label="MDLM Qwen3 596M (Paper 1)",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axhline(0.30, color="grey", ls=":", lw=0.9, alpha=0.7, label="0.30 axis-recovery floor")
    ax.axhline(0.25, color="black", ls=":", lw=0.6, alpha=0.4, label="indep top-quartile")

    for i, v in enumerate(s_vals):
        if 0 < v < 0.30:
            ax.text(
                i - w,
                v + 0.02,
                "weak",
                color="#d62728",
                ha="center",
                fontsize=8.5,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(experts)
    ax.set_ylabel("solo λ=1 marginal (axis recovery rate)")
    ax.set_title(
        "Paper 2 §10 — solo expert marginals: SEDD-small vs MDLM Paper-1 reference\n"
        "`formal` is the only SEDD expert below the 0.30 floor; all others are healthy.",
        fontsize=10.5,
    )
    ax.set_ylim(0, max(max(s_vals), max(o_vals + [0]), max(q_vals + [0])) * 1.15)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    out_png = out_dir / "sedd_marginals_bars.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"  wrote {out_png}")

    typer.echo("\nDone.")


if __name__ == "__main__":
    if not os.environ.get("HF_HUB_DISABLE_TELEMETRY"):
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    app()
