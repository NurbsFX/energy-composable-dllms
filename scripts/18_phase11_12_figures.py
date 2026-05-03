"""Phase 11/12 final figures.

Generates four artefacts in ``artifacts/plots/``:

* ``mu_sweep_curves.png`` — ratio vs μ for each available sweep, on one panel
  per setup (bell-shape diagnostic).
* ``predictor_loo_scatter.png`` — predicted vs ground-truth μ\\* on the 17
  setups of Phase 12c, with y=x guide and per-point residuals.
* ``mu_schedule_bar.png`` — Phase 12d bar chart comparing 2 constant-μ
  controls and 4 schedule shapes on Qwen3 fpc.
* ``phase11_gains_table.md`` — concise markdown table summarizing best μ,
  best ratio, canonical ratio, and gain (%) for each setup.

Inputs (read from ``artifact_root``):
* ``mu/*.json`` and ``mu_extra/*.json`` — μ-sweep records (PoE-N).
* ``n3_mu_sweep.json`` — the initial Phase 11 sweep.
* ``predict_mu.json`` — predictor LOO output (in repo's ``artifacts/`` by
  default; we read from there).
* ``artifacts/mu_schedule_qwen3_fpc.json`` — Phase 12d schedule sweep.

No model forward needed; pure plot from existing JSONs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer

app = typer.Typer(add_completion=False)

BACKBONE_LABEL = {
    "kuleshov-group/mdlm-owt": "MDLM-OWT (110M)",
    "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1": "Qwen3-0.6B-MDLM (596M)",
}


def _label(record: dict) -> str:
    triplet = "×".join(record["triplet"])
    bb = BACKBONE_LABEL.get(record["backbone"], record["backbone"])
    return f"{triplet}\n{bb}"


def _short_label(record: dict) -> str:
    triplet = "×".join(record["triplet"])
    bb = "Qwen3" if "Qwen3" in record["backbone"] else "MDLM"
    return f"{triplet} · {bb}"


def _load_records(artifact_root: Path) -> list[dict]:
    records: list[dict] = []
    candidates = [artifact_root / "n3_mu_sweep.json"]
    for sub in ("mu", "mu_extra"):
        sub_dir = artifact_root / sub
        if sub_dir.exists():
            candidates.extend(sorted(sub_dir.glob("*.json")))
    seen = set()
    for path in candidates:
        if not path.exists():
            continue
        try:
            r = json.loads(path.read_text())
        except Exception as e:
            typer.echo(f"  skip {path}: {e}", err=True)
            continue
        # Dedupe by (triplet, backbone) — n3_mu_sweep.json is also in mu/
        key = (tuple(r["triplet"]), r["backbone"])
        if key in seen:
            continue
        seen.add(key)
        records.append(r)
    return records


def _curve_points(record: dict) -> tuple[list[float], list[float]]:
    """Return (sorted μ list, ratio list) — drops the duplicate
    ``standard_<μ>`` entry that 14_mu_sweep.py emits as a sanity check."""
    points: dict[float, float] = {}
    for k, v in record.get("sweep_results", {}).items():
        if k.startswith("standard_"):
            continue
        points[float(v["mu"])] = float(v["ratio"])
    mus = sorted(points.keys())
    return mus, [points[m] for m in mus]


def plot_mu_sweep_curves(records: list[dict], out_path: Path) -> None:
    n = len(records)
    if n == 0:
        typer.echo("No records — skipping mu_sweep_curves figure.", err=True)
        return
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for i, record in enumerate(records):
        ax = axes[i // cols, i % cols]
        ax.set_visible(True)
        mus, ratios = _curve_points(record)
        if not mus:
            continue

        N = len(record["triplet"])
        canonical_mu = 1 - N
        canonical_ratio = next(
            (r for m, r in zip(mus, ratios, strict=True) if m == canonical_mu), None
        )
        best_mu = mus[int(np.argmax(ratios))]
        best_ratio = max(ratios)

        ax.plot(mus, ratios, "o-", color="#1f77b4", lw=1.6, markersize=5.5)
        ax.axhline(1.0, color="grey", ls=":", lw=0.8, alpha=0.6, label="ratio = 1")
        if canonical_ratio is not None:
            ax.scatter(
                [canonical_mu],
                [canonical_ratio],
                s=70,
                facecolors="none",
                edgecolors="#d62728",
                lw=1.8,
                label=f"canonical μ={canonical_mu:+g}",
                zorder=5,
            )
        ax.scatter(
            [best_mu],
            [best_ratio],
            s=90,
            marker="*",
            color="#2ca02c",
            label=f"best μ={best_mu:+g}",
            zorder=6,
        )

        ax.set_title(_short_label(record), fontsize=9.5)
        ax.set_xlabel("μ")
        ax.set_ylabel("triple-sat ratio")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="best", framealpha=0.85)

    fig.suptitle(
        "Phase 11 — μ sweep curves: ratio vs decoupled coefficient on log p_base",
        fontsize=11,
        y=1.00,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"  wrote {out_path}")


def plot_predictor_loo_scatter(predict_mu_json: Path, out_path: Path) -> None:
    if not predict_mu_json.exists():
        typer.echo(f"Missing {predict_mu_json}; skipping LOO scatter.", err=True)
        return
    data = json.loads(predict_mu_json.read_text())
    setups = data["setups"]
    lin = data["linear"]
    rule = data["simple_rule"]

    y_true = np.array(lin["ground_truth"])
    y_lin = np.array(lin["loo_predictions"])

    bbs = ["Qwen3" if "Qwen3" in s["backbone"] else "MDLM" for s in setups]
    Ns = [s["N"] for s in setups]
    color_map = {"Qwen3": "#1f77b4", "MDLM": "#d62728"}
    marker_map = {2: "o", 3: "s"}

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    lims = (-2.6, 0.6)
    ax.plot(lims, lims, "k--", lw=1.0, alpha=0.5, label="y = x")

    for bb_key, color in color_map.items():
        for n_key, marker in marker_map.items():
            idx = [i for i in range(len(setups)) if bbs[i] == bb_key and Ns[i] == n_key]
            if not idx:
                continue
            ax.scatter(
                y_true[idx],
                y_lin[idx],
                marker=marker,
                s=78,
                color=color,
                edgecolors="black",
                linewidth=0.5,
                label=f"{bb_key}, N={n_key}",
                alpha=0.92,
            )

    # Residual segments (predicted vs true)
    for i in range(len(setups)):
        ax.plot([y_true[i], y_true[i]], [y_true[i], y_lin[i]], color="grey", lw=0.6, alpha=0.4)

    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("ground-truth μ\\*")
    ax.set_ylabel("LOO-predicted μ̂  (linear regression)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    mae_lin = lin["loo_mae"]
    mae_rule = rule["mae"]
    ax.set_title(
        f"Phase 12c — Predictor C: LOO scatter on {len(setups)} setups\n"
        f"linear LOO-MAE = {mae_lin:.3f}   |   simple rule MAE = {mae_rule:.3f}",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"  wrote {out_path}")


def write_gains_table(records: list[dict], out_path: Path) -> None:
    """Markdown table: per setup, list canonical / best μ / best ratio / gain."""
    rows = []
    for record in records:
        triplet = record["triplet"]
        N = len(triplet)
        canonical_mu = 1 - N
        bb = "Qwen3" if "Qwen3" in record["backbone"] else "MDLM-OWT"

        mus, ratios = _curve_points(record)
        if not mus:
            continue
        canonical_ratio = next(
            (r for m, r in zip(mus, ratios, strict=True) if m == canonical_mu), None
        )
        best_idx = int(np.argmax(ratios))
        best_mu, best_ratio = mus[best_idx], ratios[best_idx]

        if canonical_ratio is None or canonical_ratio == 0:
            gain_str = "—"
        else:
            gain = (best_ratio / canonical_ratio - 1) * 100
            gain_str = f"{gain:+.0f}%"

        rows.append(
            {
                "triplet": "×".join(triplet),
                "backbone": bb,
                "N": N,
                "canonical_mu": canonical_mu,
                "canonical_ratio": canonical_ratio if canonical_ratio is not None else float("nan"),
                "best_mu": best_mu,
                "best_ratio": best_ratio,
                "gain_str": gain_str,
            }
        )

    rows.sort(key=lambda r: (r["N"], r["backbone"], r["triplet"]))

    lines = [
        "# Phase 11/12 — μ-fix gains summary",
        "",
        f"Records: {len(rows)} setups (Phase 11 sweeps + Phase 12c additional N=2 sweeps).",
        "",
        "| Setup | Backbone | N | canonical μ | canonical ratio | best μ | best ratio | gain |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        cr = "—" if np.isnan(r["canonical_ratio"]) else f"{r['canonical_ratio']:.2f}"
        lines.append(
            f"| {r['triplet']} | {r['backbone']} | {r['N']} | "
            f"{r['canonical_mu']:+g} | {cr} | "
            f"{r['best_mu']:+g} | {r['best_ratio']:.2f} | "
            f"**{r['gain_str']}** |"
        )
    lines.append("")
    lines.append(
        "Gain = (best_ratio / canonical_ratio − 1) × 100. "
        "Pattern: stylistic-heavy compositions on Qwen3 yield the largest gains "
        "(formal×positive N=2: +220%, formal×positive×concrete N=3: +300%, "
        "formal×concrete×sports N=3 mixed: +167%). Purely lexical compositions "
        "(positive×concrete, positive2×concrete on Qwen3, positive2×concrete×sports N=3) "
        "show ≈0% — canonical 1−N is already optimal. MDLM-OWT sweeps are noisier "
        "(small backbone, n=200 limit) but show a consistent direction: best μ is "
        "always less punitive than canonical."
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    typer.echo(f"  wrote {out_path}")


def plot_mu_schedule_bar(schedule_json: Path, out_path: Path) -> None:
    """Phase 12d: bar chart of 6 conditions on Qwen3 fpc."""
    if not schedule_json.exists():
        typer.echo(f"Missing {schedule_json}; skipping schedule bar.", err=True)
        return
    data = json.loads(schedule_json.read_text())
    triplet = "×".join(data["triplet"])
    bb = "Qwen3" if "Qwen3" in data["backbone"] else "MDLM"

    # Order: control μ_start, control μ_end, then schedules
    order = [
        ("constant μ=−2  (canonical)", "#888888"),
        ("constant μ=−1  (Phase 11 best)", "#2ca02c"),
        ("linear  (−2 → −1)", "#1f77b4"),
        ("cosine  (−2 → −1)", "#1f77b4"),
        ("late_fire  (−2 → −1)", "#1f77b4"),
        ("early_fire  (−1 → −2)", "#ff7f0e"),
    ]
    keys = [
        "constant_mu_start_-2",
        "constant_mu_end_-1",
        "sched_linear",
        "sched_cosine",
        "sched_late_fire",
        "sched_early_fire",
    ]
    results = data["results"]
    ratios = [results[k]["ratio"] for k in keys]
    labels = [o[0] for o in order]
    colors = [o[1] for o in order]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    bars = ax.bar(range(len(labels)), ratios, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(1.0, color="grey", ls=":", lw=0.9, alpha=0.6, label="ratio = 1")

    for bar, r in zip(bars, ratios, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{r:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8.5)
    ax.set_ylabel("triple-sat ratio")
    ax.set_ylim(0, max(ratios) * 1.25)
    ax.set_title(
        f"Phase 12d — μ-schedule per-step on {triplet}  ({bb}, n=200)\n"
        "Early-step μ determines the outcome; late switch has no effect.",
        fontsize=10.5,
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"  wrote {out_path}")


@app.command()
def main(
    artifact_root: Path = Path.home() / "Documents/composable-dllms-artifacts",
    predict_mu_json: Path = Path("artifacts/predict_mu.json"),
    schedule_json: Path = Path("artifacts/mu_schedule_qwen3_fpc.json"),
    out_dir: Path = Path("artifacts/plots"),
) -> None:
    typer.echo(f"Loading μ-sweep records from {artifact_root}...")
    records = _load_records(artifact_root)
    typer.echo(f"  {len(records)} unique (triplet, backbone) records.")

    typer.echo("Generating mu_sweep_curves.png ...")
    plot_mu_sweep_curves(records, out_dir / "mu_sweep_curves.png")

    typer.echo("Generating predictor_loo_scatter.png ...")
    plot_predictor_loo_scatter(predict_mu_json, out_dir / "predictor_loo_scatter.png")

    typer.echo("Generating mu_schedule_bar.png ...")
    plot_mu_schedule_bar(schedule_json, out_dir / "mu_schedule_bar.png")

    typer.echo("Writing phase11_gains_table.md ...")
    write_gains_table(records, out_dir / "phase11_gains_table.md")

    typer.echo("Done.")


if __name__ == "__main__":
    app()
