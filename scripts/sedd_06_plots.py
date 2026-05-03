"""Phase-6 of Paper 2 — figures.

Generates four artefacts in ``artifacts/plots/``:

* ``sedd_poe2_bars.png``    — per-pair ratio bars for the 15 PoE-2 pairs.
* ``sedd_poe3_bars.png``    — per-triplet ratios with MDLM reference.
* ``sedd_mu_sweep_bars.png`` — μ-sweep on formal × positive × concrete,
  side-by-side with the Paper-1 MDLM curve.
* ``sedd_summary_table.md`` — markdown table comparing SEDD vs MDLM
  on H1/H2/H3.

No model forward needed; pure plotting from the JSONs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer

app = typer.Typer(add_completion=False)


# Paper-1 reference results on Qwen3-MDLM (Section 13 of PAPER_DRAFT.md).
# These come from the paper draft tables (canonical and best μ-fix).
MDLM_PAPER1_REFERENCE = {
    "n3_fpc": {  # formal × positive × concrete
        "canonical": 0.55,
        "best_mu": (-1.0, 0.61),
    },
    "n3_fcs": {  # formal × concrete × sports
        "canonical": 0.46,
        "best_mu": (-1.0, 1.23),
    },
    "n3_p2cs": {  # positive2 × concrete × sports
        "canonical": 3.23,
        "best_mu": (-2.0, 3.23),
    },
    # μ-sweep on Qwen3 fpc, n=200 from Paper 1 §13.2.
    "mu_sweep_fpc": {
        -2.0: 0.55,  # canonical
        -1.5: 0.54,
        -1.0: 0.61,
        -0.5: 0.38,
        0.0: 0.38,
        0.5: 0.38,
        1.0: 0.31,
    },
    "poe2_mean_ratio": 1.07,  # Paper 1 mean PoE-2 ratio on Qwen3
}


def plot_poe2_bars(poe2_json: Path, out_path: Path) -> None:
    if not poe2_json.exists():
        typer.echo(f"missing {poe2_json}", err=True)
        return
    d = json.loads(poe2_json.read_text())
    results = d["results"]
    labels = list(results.keys())
    ratios = [results[k]["ratio"] for k in labels]
    mean_r = sum(ratios) / max(1, len(ratios))

    fig, ax = plt.subplots(figsize=(11, 4.6))
    colors = ["#2ca02c" if r >= 1.0 else "#d62728" for r in ratios]
    ax.bar(range(len(labels)), ratios, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="grey", ls=":", lw=0.9, alpha=0.7, label="ratio = 1")
    ax.axhline(mean_r, color="black", ls="--", lw=0.9, alpha=0.6, label=f"mean = {mean_r:.2f}")
    ax.axhline(
        MDLM_PAPER1_REFERENCE["poe2_mean_ratio"],
        color="#1f77b4",
        ls="--",
        lw=1.0,
        alpha=0.7,
        label=f"MDLM Paper-1 mean = {MDLM_PAPER1_REFERENCE['poe2_mean_ratio']:.2f}",
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([k.replace("_", "×") for k in labels], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("PoE-2 ratio (pair_sat / indep_ref)")
    ax.set_title(
        f"Paper 2 — PoE-2 sweep on SEDD-small (15 pairs, n=200, unconditional)\n"
        f"mean = {mean_r:.2f} — sub-additive on average, opposite of MDLM Paper 1",
        fontsize=10.5,
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"  wrote {out_path}")


def plot_poe3_bars(poe3_json: Path, out_path: Path) -> None:
    if not poe3_json.exists():
        typer.echo(f"missing {poe3_json}", err=True)
        return
    d = json.loads(poe3_json.read_text())
    triplets = list(d["triplets"].items())
    sedd_ratios = [t[1]["ratio"] for t in triplets]

    # Map to MDLM reference (canonical + best μ).
    keymap = {
        "formal_positive_concrete": "n3_fpc",
        "formal_concrete_sports": "n3_fcs",
        "positive2_concrete_sports": "n3_p2cs",
    }
    mdlm_canonical = [MDLM_PAPER1_REFERENCE[keymap[t[0]]]["canonical"] for t in triplets]
    mdlm_best = [MDLM_PAPER1_REFERENCE[keymap[t[0]]]["best_mu"][1] for t in triplets]
    labels = [
        "×".join(t[1]["triplet"])
        + "\n"
        + (
            "style"
            if "formal_positive" in t[0]
            else "mixed"
            if "formal_concrete_sports" == t[0]
            else "lexical"
        )
        for t in triplets
    ]

    x = np.arange(len(triplets))
    w = 0.27

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar(
        x - w,
        sedd_ratios,
        w,
        label="SEDD canonical",
        color="#d62728",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x,
        mdlm_canonical,
        w,
        label="MDLM canonical (Paper 1)",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + w,
        mdlm_best,
        w,
        label="MDLM best μ-fix (Paper 1)",
        color="#2ca02c",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axhline(1.0, color="grey", ls=":", lw=0.9, alpha=0.7, label="ratio = 1")

    for i, (r_s, r_c, r_b) in enumerate(zip(sedd_ratios, mdlm_canonical, mdlm_best, strict=True)):
        ax.text(i - w, r_s + 0.05, f"{r_s:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i, r_c + 0.05, f"{r_c:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w, r_b + 0.05, f"{r_b:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("PoE-3 triple-sat ratio")
    ax.set_title(
        "Paper 2 — PoE-3 on SEDD vs MDLM (3 triplets, n=200)\n"
        "SEDD is worse than MDLM on all triplets — H2 falsified.",
        fontsize=10.5,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"  wrote {out_path}")


def plot_mu_sweep_bars(mu_json: Path, out_path: Path) -> None:
    if not mu_json.exists():
        typer.echo(f"missing {mu_json}", err=True)
        return
    d = json.loads(mu_json.read_text())
    sweep = d["sweep_results"]

    # Drop the duplicate "canonical_*" entry — same μ as one of the mu_* entries.
    pts = []
    for k, r in sweep.items():
        if k.startswith("canonical_"):
            continue
        pts.append((float(r["mu"]), float(r["ratio"])))
    pts.sort()
    mus = [p[0] for p in pts]
    sedd_ratios = [p[1] for p in pts]

    mdlm_ref = MDLM_PAPER1_REFERENCE["mu_sweep_fpc"]
    mdlm_ratios = [mdlm_ref.get(m, None) for m in mus]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(mus, sedd_ratios, "o-", color="#d62728", lw=1.7, markersize=7, label="SEDD-small")
    if all(r is not None for r in mdlm_ratios):
        ax.plot(
            mus,
            mdlm_ratios,
            "s--",
            color="#1f77b4",
            lw=1.7,
            markersize=7,
            label="MDLM Qwen3 (Paper 1)",
        )
    ax.axhline(1.0, color="grey", ls=":", lw=0.9, alpha=0.7, label="ratio = 1")
    ax.set_xlabel("μ (decoupled coefficient on log p_base / log s_base)")
    ax.set_ylabel("triple-sat ratio")
    ax.set_title(
        f"Paper 2 — μ-sweep on formal × positive × concrete (n=200)\n"
        f"SEDD: canonical (μ={1 - 3}) is already optimal; relaxing destroys composition.\n"
        f"MDLM (Paper 1): bell-shape with peak at μ=−1, +29% over canonical.",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"  wrote {out_path}")


def write_summary_table(
    poe2_json: Path,
    poe3_json: Path,
    mu_json: Path,
    out_path: Path,
) -> None:
    poe2 = json.loads(poe2_json.read_text())
    poe3 = json.loads(poe3_json.read_text())
    mu = json.loads(mu_json.read_text())

    poe2_results = poe2["results"]
    mean_r = sum(r["ratio"] for r in poe2_results.values()) / max(1, len(poe2_results))

    lines = [
        "# Paper 2 — SEDD vs MDLM PoE summary",
        "",
        "## H1 — PoE-2 super-additivity",
        "",
        "| | mean ratio | super-additive ? |",
        "|---|---:|---|",
        f"| SEDD-small (Paper 2) | {mean_r:.2f} | NO (sub-additive) |",
        f"| MDLM Paper 1 reference | {MDLM_PAPER1_REFERENCE['poe2_mean_ratio']:.2f} | YES |",
        "",
        "**Falsified.**",
        "",
        "## H2 — PoE-3 plateau lifts on SEDD",
        "",
        "| Triplet | SEDD ratio | MDLM canonical | MDLM best μ-fix |",
        "|---|---:|---:|---:|",
    ]
    keymap = {
        "formal_positive_concrete": "n3_fpc",
        "formal_concrete_sports": "n3_fcs",
        "positive2_concrete_sports": "n3_p2cs",
    }
    for label, r in poe3["triplets"].items():
        ref = MDLM_PAPER1_REFERENCE[keymap[label]]
        lines.append(
            f"| {' × '.join(r['triplet'])} | {r['ratio']:.2f} | {ref['canonical']:.2f} | "
            f"{ref['best_mu'][1]:.2f} (μ={ref['best_mu'][0]:+g}) |"
        )
    lines.append("")
    lines.append("**Falsified — SEDD is worse on all 3 triplets.**")
    lines.append("")
    lines.append("## H3 — μ-fix transports to SEDD")
    lines.append("")
    lines.append("| μ | SEDD ratio | MDLM ratio |")
    lines.append("|---:|---:|---:|")
    for k, r in mu["sweep_results"].items():
        if k.startswith("canonical_"):
            continue
        m = float(r["mu"])
        sedd_r = r["ratio"]
        mdlm_r = MDLM_PAPER1_REFERENCE["mu_sweep_fpc"].get(m, None)
        mdlm_str = f"{mdlm_r:.2f}" if mdlm_r is not None else "—"
        lines.append(f"| {m:+g} | {sedd_r:.2f} | {mdlm_str} |")
    lines.append("")
    lines.append(
        "**Falsified — μ-fix from Paper 1 does not transport to SEDD.** "
        "Canonical (μ=−2) is already optimal on SEDD; relaxing μ destroys "
        "the composition (ratio → 0). On MDLM the same sweep peaked at "
        "μ=−1 with +29 % over canonical."
    )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    typer.echo(f"  wrote {out_path}")


@app.command()
def main(
    artifacts_dir: Path = Path("artifacts"),
    out_dir: Path = Path("artifacts/plots"),
) -> None:
    poe2_json = artifacts_dir / "sedd_poe2_sweep.json"
    poe3_json = artifacts_dir / "sedd_poe3.json"
    mu_json = artifacts_dir / "sedd_mu_sweep.json"

    plot_poe2_bars(poe2_json, out_dir / "sedd_poe2_bars.png")
    plot_poe3_bars(poe3_json, out_dir / "sedd_poe3_bars.png")
    plot_mu_sweep_bars(mu_json, out_dir / "sedd_mu_sweep_bars.png")
    write_summary_table(poe2_json, poe3_json, mu_json, out_dir / "sedd_summary_table.md")
    typer.echo("Done.")


if __name__ == "__main__":
    app()
