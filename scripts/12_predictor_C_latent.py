"""Phase 9c — Predictor C: κ on the model's latent activations.

The 4th candidate from PAPER_DRAFT.md §10.3, never tested empirically.

Intuition: κ on raw OWT measures whether the *proxy energies* are correlated
on natural text. But what matters for composition is *where in the model's
representation space the axes live*. Two axes that activate the same
sub-space fight; two axes that activate orthogonal sub-spaces compose.

Procedure (per backbone):
  1. Pass the baseline samples through the backbone, extract the mean-pooled
     last-hidden-state activation ``h_x ∈ R^d`` for each sample.
  2. Fit a linear probe per axis: ``s_a(x) ≈ w_a^T h_x + b_a``
     (where ``s_a(x)`` is the proxy score on axis a).
  3. Compute κ_act(a, b) ≈ √2 · |⟨w_a, w_b⟩| / (‖w_a‖² + ‖w_b‖²).

Cross-backbone evaluation: same as scripts/10_predictor_eval.py — Pearson r
of κ_act values against per-pair PoE deficits, on each backbone separately
and stacked across both.

Output: ``artifacts/predictor_C_latent.json``.
"""

from __future__ import annotations

import json
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


def load_baseline_samples(samples_dir: Path) -> list[dict]:
    """Load baseline samples (200 per pair, but baseline is shared across pairs).

    We use the first baseline jsonl we find — the baseline samples themselves
    don't depend on the pair, only the threshold calibration does.
    """
    baseline_path = next(samples_dir.glob("*__baseline.jsonl"), None)
    if baseline_path is None:
        raise FileNotFoundError(f"no baseline jsonl in {samples_dir}")
    return [json.loads(line) for line in baseline_path.open()]


def extract_mean_hidden(model, tokenizer, texts: list[str], device: str, max_seq: int = 64):
    """For each text, return the mean-pooled last-hidden-state vector."""
    import torch

    model.eval()
    out = []
    for txt in texts:
        ids = tokenizer.encode(txt, add_special_tokens=False)[:max_seq]
        if len(ids) < 2:
            out.append(None)
            continue
        ids_t = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            with model.disable_adapter():
                outputs = model(input_ids=ids_t, output_hidden_states=True)
            # Use last hidden state for both backbones (both expose this).
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                h = outputs.hidden_states[-1]  # [1, T, D]
            elif hasattr(outputs, "last_hidden_state"):
                h = outputs.last_hidden_state
            else:
                # MDLM-OWT custom model: hidden_states list might not be there
                h = outputs.logits  # fallback proxy (logit space)
            mean_h = h.float().mean(dim=1).squeeze(0).cpu().numpy()
        out.append(mean_h)
    return out


def fit_linear_probes(hiddens: list[np.ndarray], scores_per_axis: dict[str, np.ndarray]):
    """For each axis, fit a linear regression: s = w^T h + b.

    Returns dict[axis] -> w (np.ndarray of length d).
    """
    from sklearn.linear_model import Ridge

    H = np.stack([h for h in hiddens if h is not None])  # [n, d]
    valid_mask = np.array([h is not None for h in hiddens])
    weights: dict[str, np.ndarray] = {}
    for axis, scores in scores_per_axis.items():
        s = scores[valid_mask]
        # Standardise features (helps when d is large)
        H_centered = H - H.mean(axis=0, keepdims=True)
        s_centered = s - s.mean()
        ridge = Ridge(alpha=1.0)
        ridge.fit(H_centered, s_centered)
        weights[axis] = ridge.coef_
    return weights


def kappa_on_weights(w_a: np.ndarray, w_b: np.ndarray) -> float:
    """κ-style orthogonality between two probe-weight vectors.

    Mirrors the κ formula on energies: κ = √2 · |Cov| / (Var(a) + Var(b)),
    here adapted as κ = √2 · |⟨w_a, w_b⟩| / (‖w_a‖² + ‖w_b‖²).
    """
    inner = float(np.abs(np.dot(w_a, w_b)))
    norms = float(np.dot(w_a, w_a) + np.dot(w_b, w_b))
    if norms < 1e-12:
        return float("nan")
    return float(np.sqrt(2) * inner / norms)


def cosine_on_weights(w_a: np.ndarray, w_b: np.ndarray) -> float:
    """Pure cosine similarity between probe-weight vectors (sign-preserving)."""
    denom = float(np.linalg.norm(w_a) * np.linalg.norm(w_b))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(w_a, w_b) / denom)


def correlate(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return {"n": len(x), "r": float("nan"), "p": float("nan")}
    from scipy.stats import pearsonr

    r, p = pearsonr(x, y)
    return {"n": len(x), "r": float(r), "p": float(p)}


def compute_deficits(joint_sat_path: Path) -> dict:
    js = json.loads(joint_sat_path.read_text())
    out = {}
    for pair_key, entry in js.items():
        if "PoE-strict" not in entry or "__indep_reference__" not in entry:
            continue
        out[pair_key] = float(
            entry["__indep_reference__"]["indep"] - entry["PoE-strict"]["joint_satisfaction"]
        )
    return out


def evaluate_backbone(
    bb: dict, *, device: str, max_seq: int, max_samples: int, artifact_root: Path
) -> dict:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    typer.echo(f"  loading backbone {bb['model_name']}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(bb["model_name"], trust_remote_code=True)
    except (ValueError, OSError):
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

    # Need to load *some* adapter so disable_adapter() can be called
    from peft import PeftModel

    expert_names = ["formal", "positive", "positive2", "concrete", "sports"]
    first = expert_names[0]
    model = PeftModel.from_pretrained(model, bb["checkpoints"] / first, adapter_name=first)

    samples = load_baseline_samples(bb["samples_dir"])[:max_samples]
    texts = [s["text"] for s in samples]
    typer.echo(f"  extracting last-hidden activations on {len(texts)} baseline samples...")
    hiddens = extract_mean_hidden(model, tokenizer, texts, device=device, max_seq=max_seq)

    # Per-axis proxy scores
    scores_per_axis: dict[str, np.ndarray] = {}
    for proxy in EXPERT_TO_PROXY.values():
        scores_per_axis[proxy] = np.array([s["proxy_scores"][proxy] for s in samples])

    typer.echo("  fitting linear probes per axis...")
    probe_weights = fit_linear_probes(hiddens, scores_per_axis)

    # Compute κ_act (and cosine) for each pair
    kappa_act: dict[str, float] = {}
    cos_act: dict[str, float] = {}
    for a, b in PAIRS:
        proxy_a, proxy_b = EXPERT_TO_PROXY[a], EXPERT_TO_PROXY[b]
        if proxy_a not in probe_weights or proxy_b not in probe_weights:
            continue
        w_a, w_b = probe_weights[proxy_a], probe_weights[proxy_b]
        kappa_act[f"{a}|{b}"] = kappa_on_weights(w_a, w_b)
        cos_act[f"{a}|{b}"] = cosine_on_weights(w_a, w_b)

    # Per-backbone deficits + correlations
    deficits = compute_deficits(bb["joint_sat"])
    keys = sorted(set(kappa_act) & set(deficits))
    kappa_arr = np.array([kappa_act[k] for k in keys])
    cos_arr = np.array([cos_act[k] for k in keys])
    def_arr = np.array([deficits[k] for k in keys])
    return {
        "kappa_act": kappa_act,
        "cos_act": cos_act,
        "deficits": deficits,
        "corr_kappa_act_vs_deficit": correlate(kappa_arr, def_arr),
        "corr_cos_act_vs_deficit": correlate(cos_arr, def_arr),
    }


@app.command()
def main(
    artifact_root: Path = Path.home() / "Documents/composable-dllms-artifacts",
    device: str = typer.Option("cpu", help="cpu | mps | cuda"),
    max_seq: int = 64,
    max_samples: int = 200,
    out_json: Path = Path("artifacts/predictor_C_latent.json"),
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)

    backbones = {
        "mdlm_owt": {
            "samples_dir": artifact_root / "samples",
            "joint_sat": artifact_root / "joint_satisfaction.json",
            "checkpoints": artifact_root / "checkpoints",
            "model_name": "kuleshov-group/mdlm-owt",
        },
        "qwen3": {
            "samples_dir": artifact_root / "qwen3_run/samples",
            "joint_sat": artifact_root / "qwen3_run/joint_satisfaction.json",
            "checkpoints": artifact_root / "checkpoints_qwen3",
            "model_name": "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
        },
    }

    results: dict = {"per_backbone": {}, "cross_backbone": {}}

    for bb_name, bb in backbones.items():
        typer.echo(f"\n=== Backbone: {bb_name} ===")
        try:
            r = evaluate_backbone(
                bb,
                device=device,
                max_seq=max_seq,
                max_samples=max_samples,
                artifact_root=artifact_root,
            )
        except Exception as e:
            typer.echo(f"  ⚠ failed: {e}", err=True)
            continue
        results["per_backbone"][bb_name] = r
        ck = r["corr_kappa_act_vs_deficit"]
        cc = r["corr_cos_act_vs_deficit"]
        typer.echo(f"  κ_act vs deficit: n={ck['n']}  r={ck['r']:+.3f}  p={ck['p']:.3f}")
        typer.echo(f"  cos_act vs deficit: n={cc['n']}  r={cc['r']:+.3f}  p={cc['p']:.3f}")

    # Cross-backbone
    typer.echo("\n=== Cross-backbone (stacked, n=20) ===")
    for metric_name in ("kappa_act", "cos_act"):
        merged_x: dict[str, float] = {}
        merged_y: dict[str, float] = {}
        for bb_name, r in results["per_backbone"].items():
            for k, v in r[metric_name].items():
                merged_x[f"{bb_name}/{k}"] = v
                if k in r["deficits"]:
                    merged_y[f"{bb_name}/{k}"] = r["deficits"][k]
        keys = sorted(set(merged_x) & set(merged_y))
        x = np.array([merged_x[k] for k in keys])
        y = np.array([merged_y[k] for k in keys])
        corr = correlate(x, y)
        typer.echo(f"  {metric_name:<10s} r={corr['r']:+.3f}  p={corr['p']:.3f}  n={corr['n']}")
        results["cross_backbone"][metric_name] = {
            "predictor_values": merged_x,
            "deficits": merged_y,
            "corr": corr,
        }

    out_json.write_text(json.dumps(results, indent=2))
    typer.echo(f"\nWrote {out_json}")


if __name__ == "__main__":
    app()
