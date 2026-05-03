"""Phase-11 analog for Paper 2 — μ-sweep on SEDD-small.

For a target triplet (default: formal × positive × concrete, the
stylistic-heavy setup that maxed out at ratio 0.55 with canonical PoE on
MDLM, then jumped to 0.71 with μ=0 in Paper 1 §13.3.2), sweep μ ∈
{−2, −1.5, −1, −0.5, 0, +0.5, +1} under decoupled mixture-PoE:

    log s_custom = μ · log s_b + Σ λ_k · log s_k

Compares to canonical PoE (μ = 1−N = −2 at N=3). Outputs per-μ
triple_sat + ratio. The discriminator hypothesis (H3 in PAPER2_SEDD.md):
under exact score-based composition, the bell shape Paper 1 saw on MDLM
should be milder — varying μ may have a smaller effect, suggesting the
canonical was less catastrophic to begin with.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import typer

from src.energies import build_default_energies
from src.eval.joint_satisfaction import compute_thresholds
from src.eval.scoring import SampleScorer
from src.sedd_composition import PoEScoreConfig, PoEScoreSampler, load_sedd_from_hub
from src.sedd_composition.load import get_gpt2_tokenizer_for_sedd

app = typer.Typer(add_completion=False)


EXPERT_TO_PROXY = {
    "long": "len",
    "formal": "form",
    "positive": "sent",
    "positive2": "sent2",
    "concrete": "conc",
    "sports": "topic",
}


def _triple_sat(scored, triplet_keys, thresholds) -> float:
    return float(
        sum(all(r.proxy_scores[k] >= thresholds[k] for k in triplet_keys) for r in scored)
        / max(1, len(scored))
    )


@app.command()
def main(
    triplet: list[str] = typer.Option(["formal", "positive", "concrete"]),
    backbone: str = "louaaron/sedd-small",
    checkpoints_dir: Path = Path("artifacts/sedd_checkpoints"),
    n_samples: int = 200,
    seq_len: int = 64,
    num_steps: int = 128,
    out_json: Path = Path("artifacts/sedd_mu_sweep.json"),
    seed: int = 42,
    mu_values: str = typer.Option(
        "-2,-1.5,-1,-0.5,0,0.5,1",
        help="Comma-separated μ values to sweep.",
    ),
) -> None:
    from peft import PeftModel

    if len(triplet) not in (2, 3):
        raise typer.BadParameter(f"need 2 or 3 expert names, got {triplet}")

    mu_list = [float(s.strip()) for s in mu_values.split(",") if s.strip()]
    typer.echo(f"=== SEDD μ-sweep — triplet={triplet}  μ∈{mu_list} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load backbone + adapters.
    score_model, graph, noise = load_sedd_from_hub(backbone, device=device)
    score_model = PeftModel.from_pretrained(
        score_model, checkpoints_dir / triplet[0], adapter_name=triplet[0]
    )
    for name in triplet[1:]:
        score_model.load_adapter(checkpoints_dir / name, adapter_name=name)

    tokenizer = get_gpt2_tokenizer_for_sedd()
    energies = build_default_energies()
    scorer = SampleScorer(energies=energies)
    triplet_keys = [EXPERT_TO_PROXY[n] for n in triplet]

    base_cfg = PoEScoreConfig(num_steps=num_steps, sample_batch_size=min(32, n_samples))
    sampler = PoEScoreSampler(score_model, graph, noise, tokenizer, cfg=base_cfg)

    # 2. Baseline + marginals.
    typer.echo("Sampling baseline (no adapter) ...")
    torch.manual_seed(seed)
    base_texts = sampler.sample(num_samples=n_samples, seq_len=seq_len, lambdas={})
    base_scored = [scorer.score(s) for s in base_texts]
    thresholds = compute_thresholds({k: [r.proxy_scores[k] for r in base_scored] for k in energies})

    marginals: dict[str, float] = {}
    for name in triplet:
        typer.echo(f"Sampling solo {name} ...")
        torch.manual_seed(seed)
        texts = sampler.sample(num_samples=n_samples, seq_len=seq_len, lambdas={name: 1.0})
        scored = [scorer.score(s) for s in texts]
        ek = EXPERT_TO_PROXY[name]
        marginals[name] = float(
            sum(r.proxy_scores[ek] >= thresholds[ek] for r in scored) / max(1, len(scored))
        )
        typer.echo(f"  marginal {name} = {marginals[name]:.3f}")
    indep_ref = 1.0
    for n in triplet:
        indep_ref *= marginals[n]
    indep_ref = float(indep_ref)

    # 3. Canonical (μ=None ≡ canonical PoE).
    N = len(triplet)
    canonical_mu = 1 - N
    typer.echo(f"Sampling canonical PoE-{N} (μ ≡ {canonical_mu}) ...")
    torch.manual_seed(seed)
    canon_texts = sampler.sample(
        num_samples=n_samples, seq_len=seq_len, lambdas={n: 1.0 for n in triplet}
    )
    canon_scored = [scorer.score(s) for s in canon_texts]
    canon_ts = _triple_sat(canon_scored, triplet_keys, thresholds)
    sweep_results: dict[str, dict] = {
        f"canonical_{canonical_mu:+g}": {
            "mu": float(canonical_mu),
            "triple_sat": canon_ts,
            "ratio": canon_ts / max(1e-9, indep_ref),
        }
    }

    # 4. μ sweep.
    for mu in mu_list:
        typer.echo(f"Sampling μ = {mu:+.2f} ...")
        cfg_mu = PoEScoreConfig(
            num_steps=num_steps,
            sample_batch_size=min(32, n_samples),
            mu_base=mu,
        )
        sampler_mu = PoEScoreSampler(score_model, graph, noise, tokenizer, cfg=cfg_mu)
        torch.manual_seed(seed)
        texts = sampler_mu.sample(
            num_samples=n_samples, seq_len=seq_len, lambdas={n: 1.0 for n in triplet}
        )
        scored = [scorer.score(s) for s in texts]
        ts = _triple_sat(scored, triplet_keys, thresholds)
        ratio = ts / max(1e-9, indep_ref)
        sweep_results[f"mu_{mu:+.2f}"] = {
            "mu": mu,
            "triple_sat": ts,
            "ratio": ratio,
        }
        typer.echo(f"  μ = {mu:+.2f}: triple_sat={ts:.4f}  ratio={ratio:.2f}")

    out = {
        "triplet": list(triplet),
        "backbone": backbone,
        "thresholds": thresholds,
        "marginals": marginals,
        "indep_ref": indep_ref,
        "config": {
            "n_samples": n_samples,
            "seq_len": seq_len,
            "num_steps": num_steps,
            "mu_values": mu_list,
            "seed": seed,
        },
        "sweep_results": sweep_results,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    typer.echo(f"\nWrote {out_json}")

    typer.echo("\n=== Summary ===")
    typer.echo(f"  {'config':<14s}  {'μ':>6s}  {'triple_sat':>10s}  {'ratio':>6s}")
    for name, r in sweep_results.items():
        typer.echo(f"  {name:<14s}  {r['mu']:>+6.2f}  {r['triple_sat']:>10.4f}  {r['ratio']:>6.2f}")


if __name__ == "__main__":
    app()
