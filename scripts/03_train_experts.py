#!/usr/bin/env python
"""Fine-tune the six Plan-B LoRA experts (sequentially or one at a time)."""

from __future__ import annotations

from pathlib import Path

import typer

from src.training.train_expert import ExpertTrainingConfig, train

app = typer.Typer(add_completion=False)


# Plan B: dropped `nontoxic` (correlated with formality, weak HSIC signal)
# and added redundant proxies (`positive2`) plus orthogonal axes (`concrete`,
# `sports`) to expose stronger and more diverse independence patterns.
EXPERTS: dict[str, str] = {
    "long": "artifacts/datasets/long.jsonl",
    "formal": "artifacts/datasets/formal.jsonl",
    "positive": "artifacts/datasets/positive.jsonl",
    "positive2": "artifacts/datasets/positive2.jsonl",
    "concrete": "artifacts/datasets/concrete.jsonl",
    "sports": "artifacts/datasets/sports.jsonl",
}


@app.command()
def main(
    only: str | None = typer.Option(None, help="Train one expert by name."),
    output_dir: Path = Path("artifacts/checkpoints"),
    num_steps: int = 2500,
    force: bool = typer.Option(False, help="Retrain even if a checkpoint already exists."),
    backbone: str = typer.Option(
        "kuleshov-group/mdlm-owt", help="MDLM backbone to fine-tune on top of."
    ),
    lora_target: str = typer.Option(
        "attn_qkv,attn_out",
        help="Comma-separated LoRA target_modules. MDLM-OWT uses 'attn_qkv,attn_out'; "
        "Qwen3/Llama backbones use 'q_proj,k_proj,v_proj,o_proj'.",
    ),
) -> None:
    if only is not None and only not in EXPERTS:
        raise typer.BadParameter(f"unknown expert: {only}")
    todo = {only: EXPERTS[only]} if only else EXPERTS

    target_modules = tuple(s.strip() for s in lora_target.split(",") if s.strip())

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, train_jsonl in todo.items():
        ckpt_dir = output_dir / name
        if not force and (ckpt_dir / "adapter_model.safetensors").exists():
            typer.echo(f"skipped {name} (adapter already at {ckpt_dir})")
            continue
        cfg = ExpertTrainingConfig(
            expert_name=name,
            train_jsonl=train_jsonl,
            output_dir=output_dir,
            num_steps=num_steps,
            wandb_run_name=f"expert-{name}",
            backbone=backbone,
            lora_target_modules=target_modules,
        )
        ckpt = train(cfg)
        typer.echo(f"trained {name} → {ckpt}")


if __name__ == "__main__":
    app()
