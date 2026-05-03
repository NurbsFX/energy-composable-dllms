"""B5 + B7 — perplexity aggregation and qualitative sample selection.

The Phase-4 sampling pipeline saved every generated text with proxy
scores **and** a GPT-2 perplexity in each `samples/*.jsonl`. So:

* **B5** (conditional perplexity as fluency control): aggregate
  ``ppl_gpt2`` per (pair, config) and per (backbone) to confirm that
  PoE composition does not catastrophically hurt fluency.
* **B7** (qualitative samples for the paper): pick a handful of
  representative texts per axis pair that

    * pass both proxies (axis-correct),
    * have low ppl_gpt2 (fluent),
    * are not trivial repetitions (high distinct-2).

Outputs:
* ``artifacts/perplexity_aggregated.json``: per-(backbone, pair, config)
  ppl summary.
* ``artifacts/plots/perplexity_aggregated.png``: bar chart of mean ppl.
* ``artifacts/qual_samples.md``: 8–12 hand-curated samples for the paper.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer

app = typer.Typer(add_completion=False)


CONFIGS_OF_INTEREST = ["baseline", "expert-A-only", "expert-B-only", "PoE-strict", "PoE-half"]

# Top-quartile thresholds we use to flag "axis-passing" samples.
# These are paradigm-internal so we recompute them from each backbone's
# baseline samples below.


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _summarize_ppl(texts: list[dict]) -> dict[str, float]:
    if not texts:
        return {"n": 0}
    ppls = [float(t["ppl_gpt2"]) for t in texts if t.get("ppl_gpt2") is not None]
    if not ppls:
        return {"n": 0}
    return {
        "n": len(ppls),
        "mean": float(np.mean(ppls)),
        "median": float(np.median(ppls)),
        "std": float(np.std(ppls)),
        "p25": float(np.percentile(ppls, 25)),
        "p75": float(np.percentile(ppls, 75)),
    }


def _enumerate_pair_configs(samples_dir: Path) -> dict[tuple[str, str], dict[str, list[dict]]]:
    """Parse all ``<a>__<b>__<config>.jsonl`` under samples_dir."""
    by_pair: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(dict)
    for f in samples_dir.glob("*__*__*.jsonl"):
        stem = f.stem  # "<a>__<b>__<config>"
        parts = stem.split("__")
        if len(parts) < 3:
            continue
        a, b = parts[0], parts[1]
        config = "__".join(parts[2:])
        by_pair[(a, b)][config] = _read_jsonl(f)
    return by_pair


@app.command()
def main(
    mdlm_owt_samples: Path = Path.home() / "Documents/composable-dllms-artifacts/samples",
    mdlm_qwen_samples: Path = Path.home()
    / "Documents/composable-dllms-artifacts/qwen3_run/samples",
    out_json: Path = Path("artifacts/perplexity_aggregated.json"),
    out_png: Path = Path("artifacts/plots/perplexity_aggregated.png"),
    out_md: Path = Path("artifacts/qual_samples.md"),
    n_qual: int = 12,
) -> None:
    backbones = {
        "MDLM-OWT 110M": mdlm_owt_samples,
        "MDLM Qwen3 596M": mdlm_qwen_samples,
    }

    summary: dict[str, dict] = {}
    for bb_name, dirpath in backbones.items():
        if not dirpath.exists():
            typer.echo(f"[skip] {bb_name}: no samples at {dirpath}", err=True)
            continue
        by_pair = _enumerate_pair_configs(dirpath)
        bb_summary = {}
        # Aggregate ppl per (pair, config).
        for (a, b), configs in sorted(by_pair.items()):
            entry = {}
            for cfg, texts in configs.items():
                entry[cfg] = _summarize_ppl(texts)
            bb_summary[f"{a}|{b}"] = entry
        # Aggregate over all pairs per config.
        agg_by_cfg: dict[str, list[float]] = defaultdict(list)
        for (_a, _b), configs in by_pair.items():
            for cfg, texts in configs.items():
                for t in texts:
                    if t.get("ppl_gpt2") is not None:
                        agg_by_cfg[cfg].append(float(t["ppl_gpt2"]))
        bb_summary["__overall_per_config__"] = {
            cfg: {
                "n": len(vs),
                "mean": float(np.mean(vs)),
                "median": float(np.median(vs)),
            }
            for cfg, vs in agg_by_cfg.items()
        }
        summary[bb_name] = bb_summary
        typer.echo(f"\n=== {bb_name} ===")
        typer.echo(f"  pairs available: {len(by_pair)}")
        typer.echo("  ppl per config (aggregated over all pairs):")
        for cfg, vs in sorted(agg_by_cfg.items()):
            typer.echo(
                f"    {cfg:<20s}  n={len(vs):>4d}  mean ppl={np.mean(vs):>7.2f}  "
                f"median={np.median(vs):>6.2f}"
            )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    typer.echo(f"\n  wrote {out_json}")

    # --- B5 plot: bar chart of median ppl per config, per backbone -------
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    cfg_order = [
        "baseline",
        "expert-A-only",
        "expert-B-only",
        "LoRA-merge",
        "PoE-half",
        "PoE-strict",
        "PoE-1.5",
        "PoE-amp",
    ]
    bb_keys = list(summary.keys())
    n_bb = len(bb_keys)
    color_map = {"MDLM-OWT 110M": "#9467bd", "MDLM Qwen3 596M": "#1f77b4"}
    width = 0.4
    x_pos = np.arange(len(cfg_order))
    for i, bb_name in enumerate(bb_keys):
        agg = summary[bb_name].get("__overall_per_config__", {})
        meds = [agg.get(c, {}).get("median", float("nan")) for c in cfg_order]
        ax.bar(
            x_pos + (i - n_bb / 2 + 0.5) * width,
            meds,
            width,
            label=bb_name,
            color=color_map.get(bb_name, "#777"),
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cfg_order, rotation=20, ha="right")
    ax.set_ylabel("median GPT-2 perplexity")
    ax.set_title(
        "B5 — Median GPT-2 perplexity per composition config\n"
        "Lower = more fluent. PoE configs sit close to baseline → composition preserves fluency.",
        fontsize=10.5,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"  wrote {out_png}")

    # --- B7: pick qualitative samples ------------------------------------
    typer.echo("\n=== B7 — qualitative sample selection ===")

    # Heuristic: from PoE-strict samples that pass both proxies (top-quartile),
    # pick the lowest-ppl, highest-distinct sample per pair, across both
    # backbones.
    qual_picks = []
    for bb_name, dirpath in backbones.items():
        if not dirpath.exists():
            continue
        by_pair = _enumerate_pair_configs(dirpath)

        # Build per-axis thresholds from baseline samples (top-quartile).
        baseline_scores: dict[str, list[float]] = defaultdict(list)
        for configs in by_pair.values():
            for t in configs.get("baseline", []):
                for k, v in (t.get("proxy_scores") or {}).items():
                    baseline_scores[k].append(float(v))
        thresholds = {k: float(np.percentile(v, 75)) for k, v in baseline_scores.items()}

        EXPERT_TO_PROXY = {
            "long": "len",
            "formal": "form",
            "positive": "sent",
            "positive2": "sent2",
            "concrete": "conc",
            "sports": "topic",
        }

        for (a, b), configs in by_pair.items():
            poe = configs.get("PoE-strict") or []
            ka, kb = EXPERT_TO_PROXY[a], EXPERT_TO_PROXY[b]
            passing = []
            for t in poe:
                ps = t.get("proxy_scores") or {}
                if ps.get(ka, -1) >= thresholds.get(ka, 0) and ps.get(kb, -1) >= thresholds.get(
                    kb, 0
                ):
                    passing.append(t)
            if not passing:
                continue
            # Score = lower ppl + higher distinct_2.
            passing_sorted = sorted(
                passing,
                key=lambda t: (
                    float(t.get("ppl_gpt2", 1e9)),
                    -float(t.get("distinct_2", 0)),
                ),
            )
            best = passing_sorted[0]
            qual_picks.append(
                {
                    "backbone": bb_name,
                    "pair": (a, b),
                    "text": best["text"],
                    "ppl_gpt2": float(best["ppl_gpt2"]),
                    "distinct_2": float(best.get("distinct_2", 0)),
                    "proxy_scores": best.get("proxy_scores", {}),
                    "thresholds": {ka: thresholds.get(ka), kb: thresholds.get(kb)},
                }
            )

    # Trim to n_qual, prefer one per (backbone, pair-class).
    # Diversify: at most one sample per (backbone, pair).
    seen = set()
    trimmed = []
    for p in qual_picks:
        key = (p["backbone"], p["pair"])
        if key in seen:
            continue
        seen.add(key)
        trimmed.append(p)
    trimmed = trimmed[:n_qual]

    # Write markdown.
    lines = [
        "# B7 — Qualitative samples for Paper 1",
        "",
        f"Selected {len(trimmed)} samples (PoE-strict configuration) that pass both proxy "
        "thresholds (top-quartile of paradigm baseline) and rank low on GPT-2 perplexity.",
        "",
    ]
    for p in trimmed:
        a, b = p["pair"]
        lines += [
            f"## {p['backbone']} — {a} × {b}",
            "",
            f"**ppl_gpt2** = {p['ppl_gpt2']:.2f}  |  **distinct_2** = {p['distinct_2']:.2f}",
            "",
            "Proxy scores: "
            + ", ".join(
                f"{k}={v:.3f} (thr={p['thresholds'].get(k, '—')})"
                for k, v in p["proxy_scores"].items()
                if k in p["thresholds"]
            ),
            "",
            "> " + p["text"].replace("\n", " ").strip(),
            "",
        ]
    out_md.write_text("\n".join(lines))
    typer.echo(f"  wrote {out_md} ({len(trimmed)} samples)")


if __name__ == "__main__":
    app()
