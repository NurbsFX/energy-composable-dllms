"""Phase-8 of Paper 2 — semantic-selectivity analysis of PoE-2 ratios.

Re-reads ``artifacts/sedd_poe2_sweep.json`` and partitions the 15 PoE-2
pairs by semantic-class composition. Hypothesis (post-hoc, motivated
by the §10 critique): SEDD score-based PoE composition is *selectively*
super-/sub-additive depending on whether the two experts target
semantically homogeneous or heterogeneous axes.

Class assignment:
* **style** (abstract / distributed): ``formal``, ``long``
* **sentiment** (sentence-level affect): ``positive``, ``positive2``
* **topic** (concrete / lexical): ``concrete``, ``sports``

Pair classes:
* same-class (style/style, sentiment/sentiment, topic/topic)
* style × sentiment
* style × topic
* sentiment × topic

Outputs:
* ``artifacts/plots/sedd_semantic_selectivity_table.md`` — per-pair table
  with class assignment + class-aggregated means.
* ``artifacts/plots/sedd_semantic_selectivity_bars.png`` — color-coded
  bar chart.
* ``artifacts/plots/sedd_class_comparison_bars.png`` — class-level
  aggregation comparing mean ratios.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer

app = typer.Typer(add_completion=False)


CLASS_OF = {
    "long": "style",
    "formal": "style",
    "positive": "sentiment",
    "positive2": "sentiment",
    "concrete": "topic",
    "sports": "topic",
}
CLASS_COLOR = {
    "same": "#2ca02c",
    "sentiment×topic": "#1f77b4",
    "sentiment×style": "#ff7f0e",
    "style×topic": "#d62728",
}


def _pair_class(a: str, b: str) -> str:
    ca, cb = CLASS_OF[a], CLASS_OF[b]
    if ca == cb:
        return "same"
    pair = tuple(sorted([ca, cb]))
    return f"{pair[0]}×{pair[1]}"


@app.command()
def main(
    poe2_json: Path = Path("artifacts/sedd_poe2_sweep.json"),
    out_dir: Path = Path("artifacts/plots"),
) -> None:
    if not poe2_json.exists():
        raise typer.BadParameter(f"missing {poe2_json}")
    d = json.loads(poe2_json.read_text())
    results = d["results"]

    # Annotate each pair with its class and ratio.
    by_class: dict[str, list[tuple[str, float]]] = defaultdict(list)
    rows = []
    for label, r in results.items():
        a, b = r["pair"]
        cls = _pair_class(a, b)
        ratio = r["ratio"]
        rows.append((label, a, b, cls, ratio))
        by_class[cls].append((label, ratio))

    # Per-class aggregates.
    class_summary: dict[str, dict] = {}
    for cls, items in by_class.items():
        ratios = [r for _, r in items]
        class_summary[cls] = {
            "n": len(ratios),
            "mean": float(np.mean(ratios)),
            "median": float(np.median(ratios)),
            "min": float(np.min(ratios)),
            "max": float(np.max(ratios)),
            "n_super_add": sum(1 for r in ratios if r >= 1.0),
        }

    # --- Table -------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Paper 2 §10 — semantic selectivity analysis of PoE-2 ratios",
        "",
        "Class assignment of each expert:",
        "",
        "| expert | class |",
        "|---|---|",
    ]
    for n in ["long", "formal", "positive", "positive2", "concrete", "sports"]:
        lines.append(f"| {n} | {CLASS_OF[n]} |")
    lines += [
        "",
        "## Per-pair table (sorted by ratio)",
        "",
        "| pair | class composition | ratio | regime |",
        "|---|---|---:|---|",
    ]
    rows_sorted = sorted(rows, key=lambda r: r[4])
    for _label, a, b, cls, ratio in rows_sorted:
        regime = (
            "STRONG sub-add" if ratio < 0.5 else "moderate sub-add" if ratio < 1.0 else "super-add"
        )
        lines.append(f"| {a} × {b} | {cls} | {ratio:.3f} | {regime} |")

    lines += [
        "",
        "## Class-level aggregation",
        "",
        "| class | n | mean | median | min | max | super-add count |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    cls_order = ["same", "sentiment×topic", "sentiment×style", "style×topic"]
    for cls in cls_order:
        if cls not in class_summary:
            continue
        s = class_summary[cls]
        lines.append(
            f"| {cls} | {s['n']} | {s['mean']:.3f} | {s['median']:.3f} | "
            f"{s['min']:.3f} | {s['max']:.3f} | {s['n_super_add']}/{s['n']} |"
        )
    # Style-touching vs no-style aggregation (the bigger split).
    has_style = [r for r in rows if "style" in (CLASS_OF[r[1]], CLASS_OF[r[2]])]
    no_style = [r for r in rows if "style" not in (CLASS_OF[r[1]], CLASS_OF[r[2]])]
    hs_ratios = [r[4] for r in has_style]
    ns_ratios = [r[4] for r in no_style]
    lines += [
        "",
        "## Bigger split: style-containing vs no-style pairs",
        "",
        "| split | n | mean | super-add count |",
        "|---|---:|---:|---:|",
        f"| any-style (formal or long) | {len(hs_ratios)} | "
        f"{np.mean(hs_ratios):.3f} | {sum(1 for r in hs_ratios if r >= 1.0)}/{len(hs_ratios)} |",
        f"| no-style | {len(ns_ratios)} | "
        f"{np.mean(ns_ratios):.3f} | {sum(1 for r in ns_ratios if r >= 1.0)}/{len(ns_ratios)} |",
        "",
        "**Reading**: the super-additive pairs cluster among the no-style ones; "
        "all 9 style-containing pairs are sub-additive. The split corresponds to "
        "a 2× difference in mean ratio (~1.13 vs ~0.57).",
    ]
    out_md = out_dir / "sedd_semantic_selectivity_table.md"
    out_md.write_text("\n".join(lines))
    typer.echo(f"  wrote {out_md}")

    # --- Per-pair bar chart, colored by class ---------------------------
    rows_sorted_for_plot = sorted(rows, key=lambda r: r[4])
    labels = [f"{a}×{b}" for _, a, b, _, _ in rows_sorted_for_plot]
    ratios = [r[4] for r in rows_sorted_for_plot]
    classes = [r[3] for r in rows_sorted_for_plot]
    colors = [CLASS_COLOR[c] for c in classes]

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.bar(range(len(labels)), ratios, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="grey", ls=":", lw=0.9, alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("PoE-2 ratio")
    ax.set_ylim(0, max(ratios) * 1.15)

    # Legend — Patch handles, one per class actually present.
    from matplotlib.patches import Patch as _Patch

    handles = [
        _Patch(facecolor=CLASS_COLOR[cls], edgecolor="black", linewidth=0.5, label=cls)
        for cls in cls_order
        if cls in class_summary
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper left")

    ax.set_title(
        "Paper 2 §10 — PoE-2 ratios on SEDD-small, colored by semantic class composition\n"
        "All super-additive pairs (ratio ≥ 1) exclude `style` (formal/long); all 9 style-containing pairs are sub-additive.",
        fontsize=10.5,
    )
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_png = out_dir / "sedd_semantic_selectivity_bars.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"  wrote {out_png}")

    # --- Class-level mean bar chart ------------------------------------
    cls_present = [c for c in cls_order if c in class_summary]
    means = [class_summary[c]["mean"] for c in cls_present]
    ns = [class_summary[c]["n"] for c in cls_present]
    bar_colors = [CLASS_COLOR[c] for c in cls_present]

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.bar(cls_present, means, color=bar_colors, edgecolor="black", linewidth=0.5)
    for i, (m, n) in enumerate(zip(means, ns, strict=True)):
        ax.text(i, m + 0.02, f"{m:.2f}\n(n={n})", ha="center", va="bottom", fontsize=9)
    ax.axhline(1.0, color="grey", ls=":", lw=0.9, alpha=0.7, label="ratio = 1")
    ax.set_ylabel("mean PoE-2 ratio")
    ax.set_ylim(0, max(means) * 1.30)
    ax.set_title(
        "Paper 2 §10 — Class-level aggregation: mean PoE-2 ratio by pair class\n"
        "sentiment×topic super-additive; any pair touching `style` (formal, long) sub-additive.",
        fontsize=10.5,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_png2 = out_dir / "sedd_class_comparison_bars.png"
    fig.savefig(out_png2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"  wrote {out_png2}")

    typer.echo("\nDone.")


if __name__ == "__main__":
    app()
