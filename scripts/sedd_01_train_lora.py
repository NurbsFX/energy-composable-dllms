"""Phase-1 SEDD LoRA training, single expert (CLI shim).

Mirrors `scripts/03_train_experts.py` (which trains MDLM experts) but
calls our SEDD trainer. Use this for one-off training runs; for the
full 6-expert sweep see `scripts/sedd_02_train_all_experts.sh`.
"""

from __future__ import annotations

from pathlib import Path

import typer

from src.sedd_composition.train_lora import SeddExpertTrainingConfig, train

app = typer.Typer(add_completion=False)


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
    expert: str = typer.Option(...),
    output_dir: Path = Path("artifacts/sedd_checkpoints"),
    backbone: str = "louaaron/sedd-small",
    num_steps: int = 2500,
    batch_size: int = 16,
    sequence_length: int = 256,
    learning_rate: float = 3e-4,
    warmup_steps: int = 150,
    save_every: int = 500,
    log_every: int = 25,
    seed: int = 42,
    train_jsonl: Path | None = None,
) -> None:
    if expert not in EXPERTS:
        raise typer.BadParameter(f"unknown expert: {expert}; choices: {list(EXPERTS)}")
    train_path = train_jsonl or Path(EXPERTS[expert])
    cfg = SeddExpertTrainingConfig(
        expert_name=expert,
        train_jsonl=train_path,
        backbone=backbone,
        output_dir=output_dir,
        num_steps=num_steps,
        batch_size=batch_size,
        sequence_length=sequence_length,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        save_every=save_every,
        log_every=log_every,
        seed=seed,
    )
    out = train(cfg)
    typer.echo(f"\nSaved adapter at: {out}")


if __name__ == "__main__":
    app()
