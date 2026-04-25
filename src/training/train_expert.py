"""Single-expert LoRA fine-tuning loop on top of ``dllm.MDLMTrainer``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExpertTrainingConfig:
    expert_name: str
    train_jsonl: str | Path
    val_jsonl: str | Path | None = None

    backbone: str = "kuleshov-group/mdlm-owt"

    learning_rate: float = 3e-4
    batch_size: int = 32
    grad_accum_steps: int = 1
    sequence_length: int = 256
    num_steps: int = 2_500
    warmup_steps: int = 150
    lr_schedule: str = "cosine"
    precision: str = "bf16"
    grad_checkpointing: bool = True

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    # Required for PoE coherence: if the embeddings drift across experts the
    # per-position sum of logits used by the composed sampler is no longer
    # mathematically meaningful.
    freeze_embeddings: bool = True

    output_dir: str | Path = "artifacts/checkpoints"
    save_every: int = 500
    eval_every: int = 500
    wandb_project: str = "composable-dllms"
    wandb_run_name: str | None = None
    seed: int = 42

    eval_generation_count: int = 32
    eval_max_new_tokens: int = 128
    eval_proxy_keys: tuple[str, ...] = field(default_factory=lambda: ("len", "form", "sent", "tox"))


def train(cfg: ExpertTrainingConfig) -> Path:
    """Run a LoRA fine-tuning and return the saved adapter path."""
    raise NotImplementedError
