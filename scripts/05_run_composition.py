#!/usr/bin/env python
"""Composition sweep: generate samples per (expert pair, λ config).

For every pair of experts and every λ configuration in :data:`CONFIGS`,
we generate ``n_samples`` samples through the PoE sampler (or the naive
LoRA-merge baseline) and score them on the six proxies plus GPT-2 PPL
and distinct-2. The aggregated table feeds Phase 5.

Plan B λ sweep is ``{0, 0.5, 1, 1.5, 2}`` per expert symmetric pairs
plus a baseline, plus single-expert controls and the LoRA-merge baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


# Plan-B extended λ sweep. Symmetric (λ_a = λ_b) for each step.
CONFIGS: dict[str, dict[str, float] | None] = {
    "baseline": {"a": 0.0, "b": 0.0},
    "expert-A-only": {"a": 1.0, "b": 0.0},
    "expert-B-only": {"a": 0.0, "b": 1.0},
    "PoE-half": {"a": 0.5, "b": 0.5},
    "PoE-strict": {"a": 1.0, "b": 1.0},
    "PoE-1.5": {"a": 1.5, "b": 1.5},
    "PoE-amp": {"a": 2.0, "b": 2.0},
    # The naive LoRA-merge baseline averages adapter parameters and
    # samples through a single (merged) adapter; handled separately.
    "LoRA-merge": None,
}


def _parse_pair(s: str) -> tuple[str, str]:
    a, b = s.split(":")
    return a.strip(), b.strip()


@app.command()
def main(
    pairs: list[str] = typer.Option(
        ["long:formal", "long:positive", "formal:positive", "positive:positive2"],
        help="Expert pairs as 'name_a:name_b'.",
    ),
    n_samples: int = 500,
    max_new_tokens: int = 128,
    num_steps: int = 256,
    out_dir: Path = Path("artifacts/samples"),
    summary_json: Path = Path("artifacts/joint_satisfaction.json"),
    checkpoints_dir: Path = Path("artifacts/checkpoints"),
    backbone: str = "kuleshov-group/mdlm-owt",
    seed: int = 42,
) -> None:
    import dllm
    import torch
    from peft import PeftModel

    from src.composition.baselines import MergedSampler, NaiveLoRAMergeConfig, merge_loras
    from src.composition.poe_sampler import PoEConfig, PoESampler
    from src.energies import build_default_energies
    from src.eval.joint_satisfaction import compute_thresholds, summarize
    from src.eval.scoring import SampleScorer

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    from src.data.build_datasets import DEFAULT_VERTICAL_SPECS

    expert_to_proxy: dict[str, str] = {s.name: s.energy_key for s in DEFAULT_VERTICAL_SPECS}

    pair_tuples = [_parse_pair(p) for p in pairs]
    needed_experts = sorted({n for ab in pair_tuples for n in ab})
    for n in needed_experts:
        if n not in expert_to_proxy:
            raise typer.BadParameter(
                f"unknown expert name {n!r}; expected one of {sorted(expert_to_proxy)}"
            )

    typer.echo(f"Loading backbone {backbone} and adapters {needed_experts}...")
    model_args = dllm.utils.ModelArguments(model_name_or_path=backbone)
    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    for i, name in enumerate(needed_experts):
        path = checkpoints_dir / name
        if i == 0:
            model = PeftModel.from_pretrained(model, path, adapter_name=name)
        else:
            model.load_adapter(path, adapter_name=name)

    energies = build_default_energies()
    scorer = SampleScorer(energies=energies)
    cfg = PoEConfig(num_steps=num_steps, max_new_tokens=max_new_tokens, seed=seed)
    poe = PoESampler(model, tokenizer, cfg=cfg)

    # --- Critical non-regression: λ=0 must reproduce the bare backbone.
    typer.echo("Running λ=0 non-regression check...")
    poe.assert_lambda_zero_is_base(prompts=[""])

    summary: dict[str, dict[str, dict]] = {}
    prompts = [""] * n_samples

    # --- baseline once (independent of pair) ------------------------------
    typer.echo("Sampling baseline (no adapter active)...")
    torch.manual_seed(seed)
    base_samples = poe.sample(prompts, lambdas={})
    base_scores = [scorer.score(s) for s in base_samples]
    distinct_2_base = float(sum(s.distinct_2 for s in base_scores) / max(1, len(base_scores)))
    ppl_base = float(sum(s.ppl_gpt2 for s in base_scores) / max(1, len(base_scores)))
    thresholds = compute_thresholds({k: [s.proxy_scores[k] for s in base_scores] for k in energies})
    typer.echo(f"Top-quartile thresholds: {thresholds}")

    # --- per-pair sweep ---------------------------------------------------
    for a, b in pair_tuples:
        typer.echo(f"\n=== pair {a} × {b} ===")
        pair_summary: dict[str, dict] = {}

        proxy_a, proxy_b = expert_to_proxy[a], expert_to_proxy[b]
        for cfg_name, ab in CONFIGS.items():
            if cfg_name == "LoRA-merge":
                merged_name = merge_loras(model, NaiveLoRAMergeConfig(a, b))
                merged_sampler = MergedSampler(model, tokenizer, cfg=cfg)
                torch.manual_seed(seed)
                samples = merged_sampler.sample(prompts, merged_name=merged_name)
                model.delete_adapter(merged_name)
            else:
                lambdas = {a: ab["a"], b: ab["b"]}
                torch.manual_seed(seed)
                samples = poe.sample(prompts, lambdas=lambdas)

            scores = [scorer.score(s) for s in samples]
            sample_path = out_dir / f"{a}__{b}__{cfg_name}.jsonl"
            with sample_path.open("w") as f:
                for s in scores:
                    f.write(json.dumps(s.__dict__, ensure_ascii=False) + "\n")

            cs = summarize(
                scores,
                score_keys=(proxy_a, proxy_b),
                thresholds=thresholds,
                baseline_distinct_2=distinct_2_base,
                baseline_ppl=ppl_base,
                config_name=cfg_name,
            )
            pair_summary[cfg_name] = {
                "joint_satisfaction": cs.joint_satisfaction,
                "marginal_a": cs.marginal_a,
                "marginal_b": cs.marginal_b,
                "mode_collapse_ratio": cs.mode_collapse_ratio,
                "fluency_ratio": cs.fluency_ratio,
            }
            typer.echo(
                f"  {cfg_name:<14} JS={cs.joint_satisfaction:.3f} "
                f"a={cs.marginal_a:.3f} b={cs.marginal_b:.3f} "
                f"ppl_ratio={cs.fluency_ratio:.3f}"
            )

        # Independence reference: P(A) · P(B) computed from the marginals
        # of expert-A-only and expert-B-only configs.
        marginal_a = pair_summary["expert-A-only"]["marginal_a"]
        marginal_b = pair_summary["expert-B-only"]["marginal_b"]
        pair_summary["__indep_reference__"] = {
            "marginal_a": marginal_a,
            "marginal_b": marginal_b,
            "indep": float(marginal_a * marginal_b),
            "poe_strict": pair_summary["PoE-strict"]["joint_satisfaction"],
        }
        summary[f"{a}|{b}"] = pair_summary

    summary_json.write_text(json.dumps(summary, indent=2))
    typer.echo(f"\nWrote {summary_json}")


if __name__ == "__main__":
    app()
