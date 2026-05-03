"""Phase-4 analog for Paper 2 — PoE-2 sweep on SEDD-small.

Loads the 6 trained SEDD LoRA experts and evaluates PoE-2 on 10 pairs
(C(6,2) minus duplicates), measuring per-pair joint-satisfaction ratios:

    ratio = JS_PoE-2(pair) / (marginal_a × marginal_b)

with marginals computed from solo λ=1 sampling. Same protocol as Paper 1
Phase 4 (`scripts/05_run_composition.py`), but on SEDD scores +
unconditional Tweedie τ-leaping.

Note: SEDD's pc_sampler generates *unconditional* sequences (no prompt
prefix), so the empirical baseline differs from Paper 1's prompted MDLM
baseline. Within-paradigm ratios remain meaningful.
"""

from __future__ import annotations

import itertools
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


def _pair_sat(scored, pair_keys, thresholds) -> float:
    return float(
        sum(all(r.proxy_scores[k] >= thresholds[k] for k in pair_keys) for r in scored)
        / max(1, len(scored))
    )


@app.command()
def main(
    backbone: str = "louaaron/sedd-small",
    checkpoints_dir: Path = Path("artifacts/sedd_checkpoints"),
    n_samples: int = 200,
    seq_len: int = 64,
    num_steps: int = 128,
    out_json: Path = Path("artifacts/sedd_poe2_sweep.json"),
    seed: int = 42,
    pairs: str | None = None,  # e.g. "formal,positive;long,sports"
) -> None:
    from peft import PeftModel

    typer.echo(f"=== SEDD PoE-2 sweep — {backbone} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # 2. Build the proxy scorer.
    energies = build_default_energies()
    scorer = SampleScorer(energies=energies)

    # 3. Baseline (no adapter) for thresholds.
    cfg = PoEScoreConfig(num_steps=num_steps, sample_batch_size=min(32, n_samples))
    sampler = PoEScoreSampler(score_model, graph, noise, tokenizer, cfg=cfg)
    typer.echo("Sampling baseline (no adapter) ...")
    torch.manual_seed(seed)
    base_texts = sampler.sample(num_samples=n_samples, seq_len=seq_len, lambdas={})
    base_scored = [scorer.score(s) for s in base_texts]
    thresholds = compute_thresholds({k: [r.proxy_scores[k] for r in base_scored] for k in energies})

    # 4. Solo marginals.
    marginals: dict[str, float] = {}
    for name in expert_names:
        typer.echo(f"Sampling solo {name} ...")
        torch.manual_seed(seed)
        texts = sampler.sample(num_samples=n_samples, seq_len=seq_len, lambdas={name: 1.0})
        scored = [scorer.score(s) for s in texts]
        ek = EXPERT_TO_PROXY[name]
        marginals[name] = float(
            sum(r.proxy_scores[ek] >= thresholds[ek] for r in scored) / max(1, len(scored))
        )
        typer.echo(f"  marginal {name} = {marginals[name]:.3f}")

    # 5. PoE-2 on selected pairs.
    if pairs is None:
        pair_list = list(itertools.combinations(expert_names, 2))
    else:
        pair_list = [tuple(p.strip().split(",")) for p in pairs.split(";") if p.strip()]
    typer.echo(f"  running {len(pair_list)} pairs ...")

    results: dict[str, dict] = {}
    for a, b in pair_list:
        if a not in expert_names or b not in expert_names:
            typer.echo(f"  skip unknown pair {a}×{b}", err=True)
            continue
        label = f"{a}_{b}"
        ka, kb = EXPERT_TO_PROXY[a], EXPERT_TO_PROXY[b]
        indep = marginals[a] * marginals[b]
        torch.manual_seed(seed)
        texts = sampler.sample(
            num_samples=n_samples,
            seq_len=seq_len,
            lambdas={a: 1.0, b: 1.0},
        )
        scored = [scorer.score(s) for s in texts]
        ts = _pair_sat(scored, [ka, kb], thresholds)
        ratio = ts / max(1e-9, indep)
        results[label] = {
            "pair": [a, b],
            "marginal_a": marginals[a],
            "marginal_b": marginals[b],
            "indep_ref": indep,
            "pair_sat": ts,
            "ratio": ratio,
        }
        typer.echo(f"  {label:<22s}: pair_sat={ts:.4f}  ratio={ratio:.2f}")

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
        "results": results,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    typer.echo(f"\nWrote {out_json}")

    typer.echo("\n=== Summary ===")
    typer.echo(f"  {'pair':<22s}  {'pair_sat':>10s}  {'indep':>6s}  {'ratio':>6s}")
    for k, r in results.items():
        typer.echo(
            f"  {k:<22s}  {r['pair_sat']:>10.4f}  {r['indep_ref']:>6.3f}  {r['ratio']:>6.2f}"
        )
    if results:
        mean_ratio = sum(r["ratio"] for r in results.values()) / len(results)
        typer.echo(f"  mean ratio = {mean_ratio:.3f}")


if __name__ == "__main__":
    app()
