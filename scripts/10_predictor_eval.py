"""Phase 9 — Cross-backbone predictor evaluation.

Tests four candidate predictors of the PoE composition deficit
(``Δ = JS_indep − JS_PoE``) on the 10 N=2 pairs across two backbones
(MDLM-OWT 110M and Qwen3-0.6B-MDLM 596M).

The four candidates (rationale in PAPER_DRAFT.md §10.3, refined
by external Claude review):

* **B** — *Cross-axis leakage*. For each pair ``(a, b)``, the fraction
  of ``expert-A-only`` samples that already pass the threshold for
  axis *b*. Pure post-hoc on existing samples; no model needed.

* **F-js** — *Distributional distance variant of "KL conditional"*.
  Empirical Jensen-Shannon divergence between the per-axis proxy-score
  distributions under expert-A-only vs expert-B-only. Captures whether
  the two adapters drive their shared proxy in similar/different ways.
  Sample-based, no model needed.

* **A'** — *Logit-shift alignment*. ``E_x [cos(Δℓ_a(x), Δℓ_b(x))]``
  averaged over a fixed pool of pivot prompts, with ``Δℓ_a(x) =
  logits_a(x) − logits_base(x)``. Requires the model + adapters loaded.

* **E** — *Spatial overlap of position norms*. For the same shifts,
  compute the cosine between the position-wise norms ``‖Δℓ_a(x, t)‖``
  and ``‖Δℓ_b(x, t)‖``. Operationalises the "lexical pushes broadly
  vs stylistic pushes locally" hypothesis from §9.

Output: ``artifacts/predictor_eval.json`` with per-predictor values and
cross-backbone Pearson correlations against the observed PoE deficit.

Run modes:

* ``--no-model`` → only computes B and F-js (fast, runs anywhere).
* ``--device cpu|mps|cuda`` → computes A' and E too. Requires the
  patched MDLM-OWT modeling code and Qwen3 weights to be importable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(add_completion=False)

PROXY_TO_EXPERT = {
    "len": "long",
    "form": "formal",
    "sent": "positive",
    "sent2": "positive2",
    "conc": "concrete",
    "topic": "sports",
}
EXPERT_TO_PROXY = {v: k for k, v in PROXY_TO_EXPERT.items()}

PAIRS = [
    ("formal", "positive"),
    ("formal", "positive2"),
    ("formal", "concrete"),
    ("formal", "sports"),
    ("positive", "positive2"),
    ("positive", "concrete"),
    ("positive", "sports"),
    ("positive2", "concrete"),
    ("positive2", "sports"),
    ("concrete", "sports"),
]


# ---------------------------------------------------------------------------
# Loaders for the existing artifacts
# ---------------------------------------------------------------------------


def load_joint_satisfaction(path: Path) -> dict:
    """Returns dict[pair_key] -> dict with PoE-strict / __indep_reference__ entries."""
    return json.loads(path.read_text())


def load_thresholds(samples_dir: Path) -> dict:
    """Re-derive top-quartile thresholds from baseline samples in this run.

    We use the *first* baseline jsonl we find to read off the proxy distribution
    and compute q75 — this matches the convention used by 05_run_composition.
    """
    # Use formal__positive__baseline.jsonl by convention (always present).
    baseline_path = next(samples_dir.glob("*__baseline.jsonl"), None)
    if baseline_path is None:
        raise FileNotFoundError(f"no baseline jsonl in {samples_dir}")
    samples = [json.loads(line) for line in baseline_path.open()]
    out = {}
    for proxy in EXPERT_TO_PROXY.values():
        vals = np.array([s["proxy_scores"][proxy] for s in samples])
        out[proxy] = float(np.quantile(vals, 0.75))
    return out


def load_expert_samples(samples_dir: Path, a: str, b: str, kind: str) -> list[dict]:
    """Load the JSONL of records for one (pair, config). Returns [] if missing."""
    path = samples_dir / f"{a}__{b}__{kind}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open()]


# ---------------------------------------------------------------------------
# Predictor B — cross-axis leakage
# ---------------------------------------------------------------------------


def predictor_B_leakage(samples_dir: Path, thresholds: dict) -> dict:
    """For each pair (a, b), the marginal of `expert-A-only` samples on axis b."""
    out: dict[str, float] = {}
    for a, b in PAIRS:
        samples_a = load_expert_samples(samples_dir, a, b, "expert-A-only")
        if not samples_a:
            continue
        proxy_b = EXPERT_TO_PROXY[b]
        marginal_b_under_a = sum(
            1 for s in samples_a if s["proxy_scores"][proxy_b] >= thresholds[proxy_b]
        ) / len(samples_a)
        # Symmetric view: also expert-B leaking onto a-axis
        samples_b = load_expert_samples(samples_dir, a, b, "expert-B-only")
        if samples_b:
            proxy_a = EXPERT_TO_PROXY[a]
            marginal_a_under_b = sum(
                1 for s in samples_b if s["proxy_scores"][proxy_a] >= thresholds[proxy_a]
            ) / len(samples_b)
        else:
            marginal_a_under_b = float("nan")
        # Combined leakage: average of both directions
        combined = (marginal_b_under_a + marginal_a_under_b) / 2
        out[f"{a}|{b}"] = float(combined)
    return out


# ---------------------------------------------------------------------------
# Predictor F — sample-based distributional distance
# ---------------------------------------------------------------------------


def predictor_F_js(samples_dir: Path) -> dict:
    """For each pair (a, b), JS-divergence between expert-A-only and expert-B-only
    on each of the two relevant proxies, averaged.

    Captures whether the two adapters actually drive different distributions
    on the *same* axes — high JS means the two experts diverge structurally."""
    from scipy.spatial.distance import jensenshannon

    out: dict[str, float] = {}
    for a, b in PAIRS:
        samples_a = load_expert_samples(samples_dir, a, b, "expert-A-only")
        samples_b = load_expert_samples(samples_dir, a, b, "expert-B-only")
        if not samples_a or not samples_b:
            continue
        proxy_a, proxy_b = EXPERT_TO_PROXY[a], EXPERT_TO_PROXY[b]
        # JS on each axis
        js_per_axis = []
        for proxy in (proxy_a, proxy_b):
            scores_a = np.array([s["proxy_scores"][proxy] for s in samples_a])
            scores_b = np.array([s["proxy_scores"][proxy] for s in samples_b])
            # Common range, 30 bins
            lo = float(min(scores_a.min(), scores_b.min()))
            hi = float(max(scores_a.max(), scores_b.max()))
            if hi <= lo:
                js_per_axis.append(0.0)
                continue
            bins = np.linspace(lo, hi, 31)
            h_a, _ = np.histogram(scores_a, bins=bins, density=False)
            h_b, _ = np.histogram(scores_b, bins=bins, density=False)
            h_a = h_a / max(h_a.sum(), 1)
            h_b = h_b / max(h_b.sum(), 1)
            js = jensenshannon(h_a, h_b)
            if math.isnan(js):
                js = 0.0
            js_per_axis.append(float(js))
        out[f"{a}|{b}"] = float(np.mean(js_per_axis))
    return out


# ---------------------------------------------------------------------------
# Predictor A' and E — logit-shift based, model required
# ---------------------------------------------------------------------------


def load_pivot_prompts(prompts_jsonl: Path, n_prompts: int = 64, max_tokens: int = 12) -> list[str]:
    """Load a small fixed set of pivot prompts for shift evaluation."""
    if not prompts_jsonl.exists():
        raise FileNotFoundError(f"prompts file not found: {prompts_jsonl}")
    pool = [json.loads(line)["text"] for line in prompts_jsonl.open() if line.strip()]
    return pool[:n_prompts]


def compute_logit_shifts_per_expert(
    model,
    tokenizer,
    prompts: list[str],
    expert_names: list[str],
    device: str,
    max_seq: int = 32,
):
    """Compute Δℓ_a(x) for each (prompt, expert).

    Returns dict[expert] -> list of np.ndarray of shape [T, V] (cpu, fp32).
    """
    import torch

    base_logits_per_prompt = []
    for txt in prompts:
        ids = tokenizer.encode(txt, add_special_tokens=False)[:max_seq]
        if len(ids) < 2:
            continue
        ids_t = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad(), model.disable_adapter():
            out = model(input_ids=ids_t)
            logits = out.logits if hasattr(out, "logits") else out
        base_logits_per_prompt.append(logits.float().cpu().squeeze(0).numpy())

    shifts: dict[str, list[np.ndarray]] = {name: [] for name in expert_names}
    for name in expert_names:
        model.set_adapter(name)
        i = 0
        for txt in prompts:
            ids = tokenizer.encode(txt, add_special_tokens=False)[:max_seq]
            if len(ids) < 2:
                continue
            ids_t = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(input_ids=ids_t)
                logits = out.logits if hasattr(out, "logits") else out
            shift = logits.float().cpu().squeeze(0).numpy() - base_logits_per_prompt[i]
            shifts[name].append(shift)
            i += 1

    return shifts


def predictor_A_prime(shifts: dict[str, list[np.ndarray]]) -> dict:
    """E_x[cos(flatten(Δℓ_a(x)), flatten(Δℓ_b(x)))]."""
    out: dict[str, float] = {}
    for a, b in PAIRS:
        if a not in shifts or b not in shifts:
            continue
        cosines = []
        for sa, sb in zip(shifts[a], shifts[b], strict=True):
            va = sa.flatten()
            vb = sb.flatten()
            denom = np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12
            cosines.append(float(np.dot(va, vb) / denom))
        out[f"{a}|{b}"] = float(np.mean(cosines))
    return out


def predictor_E_spatial(shifts: dict[str, list[np.ndarray]]) -> dict:
    """E_x[cos(||Δℓ_a(x, ·)||, ||Δℓ_b(x, ·)||)] across positions."""
    out: dict[str, float] = {}
    for a, b in PAIRS:
        if a not in shifts or b not in shifts:
            continue
        cosines = []
        for sa, sb in zip(shifts[a], shifts[b], strict=True):
            norm_a = np.linalg.norm(sa, axis=-1)  # [T]
            norm_b = np.linalg.norm(sb, axis=-1)
            denom = np.linalg.norm(norm_a) * np.linalg.norm(norm_b) + 1e-12
            cosines.append(float(np.dot(norm_a, norm_b) / denom))
        out[f"{a}|{b}"] = float(np.mean(cosines))
    return out


# ---------------------------------------------------------------------------
# Correlations vs deficit
# ---------------------------------------------------------------------------


def compute_deficits(joint_sat: dict) -> dict:
    """Δ = JS_indep − JS_PoE-strict for each pair."""
    out = {}
    for pair_key, entry in joint_sat.items():
        if "PoE-strict" not in entry or "__indep_reference__" not in entry:
            continue
        js_poe = entry["PoE-strict"]["joint_satisfaction"]
        js_indep = entry["__indep_reference__"]["indep"]
        out[pair_key] = float(js_indep - js_poe)
    return out


def correlate(predictor_vals: dict, deficits: dict) -> dict:
    """Pearson r between predictor and deficit on overlapping pairs."""
    keys = sorted(set(predictor_vals) & set(deficits))
    if len(keys) < 3:
        return {"n": len(keys), "r": float("nan"), "p": float("nan")}
    x = np.array([predictor_vals[k] for k in keys])
    y = np.array([deficits[k] for k in keys])
    if x.std() < 1e-12 or y.std() < 1e-12:
        return {"n": len(keys), "r": float("nan"), "p": float("nan")}
    from scipy.stats import pearsonr

    r, p = pearsonr(x, y)
    return {"n": len(keys), "r": float(r), "p": float(p)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@app.command()
def main(
    artifact_root: Path = Path.home() / "Documents/composable-dllms-artifacts",
    no_model: bool = typer.Option(
        False, help="Skip A' and E (which require the model). Faster, runs anywhere."
    ),
    device: str = typer.Option("cpu", help="cpu | mps | cuda for A'/E."),
    n_prompts: int = 64,
    max_seq: int = 32,
    out_json: Path = Path("artifacts/predictor_eval.json"),
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)

    backbones = {
        "mdlm_owt": {
            "samples_dir": artifact_root / "samples",
            "joint_sat": artifact_root / "joint_satisfaction.json",
            "checkpoints": artifact_root / "checkpoints",
            "model_name": "kuleshov-group/mdlm-owt",
            "lora_targets": ("attn_qkv", "attn_out"),
        },
        "qwen3": {
            "samples_dir": artifact_root / "qwen3_run/samples",
            "joint_sat": artifact_root / "qwen3_run/joint_satisfaction.json",
            "checkpoints": artifact_root / "checkpoints_qwen3",
            "model_name": "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
            "lora_targets": ("q_proj", "k_proj", "v_proj", "o_proj"),
        },
    }

    results: dict = {"per_backbone": {}, "cross_backbone": {}}

    # Per-backbone evaluation
    all_predictors: dict[str, dict[str, dict[str, float]]] = {}
    all_deficits: dict[str, dict[str, float]] = {}

    for bb_name, bb in backbones.items():
        typer.echo(f"\n=== Backbone: {bb_name} ===")
        joint = load_joint_satisfaction(bb["joint_sat"])
        thresholds = load_thresholds(bb["samples_dir"])
        deficits = compute_deficits(joint)
        all_deficits[bb_name] = deficits

        per_predictor: dict[str, dict[str, float]] = {}

        # B — leakage
        per_predictor["B_leakage"] = predictor_B_leakage(bb["samples_dir"], thresholds)
        # F-js — distributional distance
        per_predictor["F_js"] = predictor_F_js(bb["samples_dir"])

        # A' and E if model loadable
        if not no_model:
            try:
                shifts = _try_compute_shifts(
                    bb,
                    n_prompts=n_prompts,
                    max_seq=max_seq,
                    device=device,
                    artifact_root=artifact_root,
                )
                if shifts:
                    per_predictor["A_prime"] = predictor_A_prime(shifts)
                    per_predictor["E_spatial"] = predictor_E_spatial(shifts)
            except Exception as e:
                typer.echo(f"  ⚠ A'/E skipped on {bb_name}: {e}", err=True)

        # Per-predictor correlations vs deficit
        bb_results: dict[str, dict] = {}
        for pred_name, pred_vals in per_predictor.items():
            corr = correlate(pred_vals, deficits)
            typer.echo(
                f"  {pred_name:<12s}  n={corr['n']:>2d}  r={corr['r']:+.3f}  p={corr['p']:.3f}"
            )
            bb_results[pred_name] = {"values": pred_vals, "corr": corr}

        results["per_backbone"][bb_name] = bb_results
        all_predictors[bb_name] = per_predictor

    # Cross-backbone correlation: stack the two backbones' values
    typer.echo("\n=== Cross-backbone (stacked, 20 points) ===")
    pred_names = set()
    for d in all_predictors.values():
        pred_names |= set(d)
    for pred_name in sorted(pred_names):
        merged_pred: dict[str, float] = {}
        merged_def: dict[str, float] = {}
        for bb_name in all_predictors:
            for k, v in all_predictors[bb_name].get(pred_name, {}).items():
                merged_pred[f"{bb_name}/{k}"] = v
                if k in all_deficits[bb_name]:
                    merged_def[f"{bb_name}/{k}"] = all_deficits[bb_name][k]
        corr = correlate(merged_pred, merged_def)
        typer.echo(f"  {pred_name:<12s}  n={corr['n']:>2d}  r={corr['r']:+.3f}  p={corr['p']:.3f}")
        results["cross_backbone"][pred_name] = {
            "predictor_values": merged_pred,
            "deficits": merged_def,
            "corr": corr,
        }

    out_json.write_text(json.dumps(results, indent=2))
    typer.echo(f"\nWrote {out_json}")


def _try_compute_shifts(
    bb: dict, *, n_prompts: int, max_seq: int, device: str, artifact_root: Path
):
    """Attempt to load the backbone + adapters and compute logit shifts.

    Wrapped in a separate function so the import errors are caught cleanly.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    typer.echo(f"  loading backbone {bb['model_name']} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(bb["model_name"], trust_remote_code=True)
    except (ValueError, OSError):
        # MDLM-OWT ships no tokenizer; fall back to GPT-2 + mask token.
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.add_special_tokens({"mask_token": "<|mdlm_mask|>"})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|pad|>"
    model = (
        AutoModelForMaskedLM.from_pretrained(
            bb["model_name"],
            trust_remote_code=True,
            dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        )
        .to(device)
        .eval()
    )

    # Load adapters
    expert_names = ["formal", "positive", "positive2", "concrete", "sports"]
    first = expert_names[0]
    model = PeftModel.from_pretrained(model, bb["checkpoints"] / first, adapter_name=first)
    for name in expert_names[1:]:
        model.load_adapter(bb["checkpoints"] / name, adapter_name=name)

    prompts = load_pivot_prompts(
        artifact_root / "datasets/prompts.jsonl", n_prompts=n_prompts, max_tokens=max_seq
    )
    typer.echo(f"  computing shifts on {len(prompts)} prompts × {len(expert_names)} experts ...")
    return compute_logit_shifts_per_expert(
        model, tokenizer, prompts, expert_names, device=device, max_seq=max_seq
    )


if __name__ == "__main__":
    app()
