"""Phase-6 analog for Paper 2 — PoE-3 on SEDD-small.

Evaluates PoE-3 on three triplets (mirroring Paper 1 §6):
* formal × positive × concrete (stylistic-heavy)
* formal × concrete × sports (mixed)
* positive2 × concrete × sports (lexical)

For each triplet, computes:
    triple_sat = JS_PoE-3 / (m_a × m_b × m_c)   (canonical λ=1, μ canonical)
    + ratio against indep_ref

The discriminator hypothesis (H2 in PAPER2_SEDD.md): if SEDD lifts the
plateau on the stylistic triplet (Paper 1 ratio = 0.55 with canonical μ),
score-based composition is meaningfully better than logit-based.
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

DEFAULT_TRIPLETS = [
    ["formal", "positive", "concrete"],
    ["formal", "concrete", "sports"],
    ["positive2", "concrete", "sports"],
]


def _triple_sat(scored, triplet_keys, thresholds) -> float:
    return float(
        sum(all(r.proxy_scores[k] >= thresholds[k] for k in triplet_keys) for r in scored)
        / max(1, len(scored))
    )


@app.command()
def main(
    backbone: str = "louaaron/sedd-small",
    checkpoints_dir: Path = Path("artifacts/sedd_checkpoints"),
    n_samples: int = 200,
    seq_len: int = 64,
    num_steps: int = 128,
    out_json: Path = Path("artifacts/sedd_poe3.json"),
    seed: int = 42,
    triplets_str: str | None = typer.Option(
        None,
        help="Override triplets, e.g. 'formal,positive,concrete;long,positive2,sports'",
    ),
) -> None:
    from peft import PeftModel

    typer.echo(f"=== SEDD PoE-3 — {backbone} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    triplets = (
        DEFAULT_TRIPLETS
        if triplets_str is None
        else [list(t.strip().split(",")) for t in triplets_str.split(";") if t.strip()]
    )

    # 1. Load backbone + all 6 adapters.
    score_model, graph, noise = load_sedd_from_hub(backbone, device=device)
    expert_names = list(EXPERT_TO_PROXY.keys())
    score_model = PeftModel.from_pretrained(
        score_model, checkpoints_dir / expert_names[0], adapter_name=expert_names[0]
    )
    for name in expert_names[1:]:
        score_model.load_adapter(checkpoints_dir / name, adapter_name=name)
    typer.echo(f"  loaded adapters: {expert_names}")

    tokenizer = get_gpt2_tokenizer_for_sedd()
    energies = build_default_energies()
    scorer = SampleScorer(energies=energies)

    cfg = PoEScoreConfig(num_steps=num_steps, sample_batch_size=min(32, n_samples))
    sampler = PoEScoreSampler(score_model, graph, noise, tokenizer, cfg=cfg)

    # 2. Baseline + thresholds.
    typer.echo("Sampling baseline (no adapter) ...")
    torch.manual_seed(seed)
    base_texts = sampler.sample(num_samples=n_samples, seq_len=seq_len, lambdas={})
    base_scored = [scorer.score(s) for s in base_texts]
    thresholds = compute_thresholds({k: [r.proxy_scores[k] for r in base_scored] for k in energies})

    # 3. Solo marginals (computed once, shared across triplets).
    marginals: dict[str, float] = {}
    for name in set(n for t in triplets for n in t):
        typer.echo(f"Sampling solo {name} ...")
        torch.manual_seed(seed)
        texts = sampler.sample(num_samples=n_samples, seq_len=seq_len, lambdas={name: 1.0})
        scored = [scorer.score(s) for s in texts]
        ek = EXPERT_TO_PROXY[name]
        marginals[name] = float(
            sum(r.proxy_scores[ek] >= thresholds[ek] for r in scored) / max(1, len(scored))
        )
        typer.echo(f"  marginal {name} = {marginals[name]:.3f}")

    # 4. PoE-3 per triplet.
    triplet_results: dict[str, dict] = {}
    for triplet in triplets:
        if any(n not in EXPERT_TO_PROXY for n in triplet):
            typer.echo(f"  skip unknown triplet {triplet}", err=True)
            continue
        label = "_".join(triplet)
        triplet_keys = [EXPERT_TO_PROXY[n] for n in triplet]
        indep = 1.0
        for n in triplet:
            indep *= marginals[n]

        typer.echo(f"Sampling PoE-3 for {triplet} ...")
        torch.manual_seed(seed)
        texts = sampler.sample(
            num_samples=n_samples,
            seq_len=seq_len,
            lambdas={n: 1.0 for n in triplet},
        )
        scored = [scorer.score(s) for s in texts]
        ts = _triple_sat(scored, triplet_keys, thresholds)
        ratio = ts / max(1e-9, indep)
        triplet_results[label] = {
            "triplet": list(triplet),
            "indep_ref": indep,
            "triple_sat": ts,
            "ratio": ratio,
        }
        typer.echo(f"  {label:<32s}: triple_sat={ts:.4f}  indep={indep:.4f}  ratio={ratio:.2f}")

    out = {
        "backbone": backbone,
        "config": {
            "n_samples": n_samples,
            "seq_len": seq_len,
            "num_steps": num_steps,
            "seed": seed,
        },
        "thresholds": thresholds,
        "marginals": marginals,
        "triplets": triplet_results,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    typer.echo(f"\nWrote {out_json}")


if __name__ == "__main__":
    app()
