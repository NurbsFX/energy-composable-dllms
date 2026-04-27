"""Single-expert LoRA fine-tuning loop on top of ``dllm.MDLMTrainer``.

The expert is a LoRA adapter that nudges a frozen MDLM backbone towards a
single vertical (long, formal, positive, …). Training data is one of the
JSONL files produced by Phase 2; loss is the masked-diffusion ELBO from
``dllm.core.trainers.MDLMTrainer``.

Embedding parameters must stay frozen so that the per-position sum of
logits used by the Phase-4 PoE sampler remains mathematically coherent
across experts (cf. ROADMAP §2.2). LoRA only touches the modules listed
in ``cfg.lora_target_modules``; the default is the attention projections,
but the actual module names depend on the backbone and are printed at
training startup so they can be verified.
"""

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
    grad_checkpointing: bool = False  # MDLM-OWT does not implement it

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # MDLM-OWT uses combined `attn_qkv` + `attn_out` (not Llama-style q/k/v/o_proj).
    lora_target_modules: tuple[str, ...] = ("attn_qkv", "attn_out")
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
    """Run a LoRA fine-tuning and return the saved adapter path.

    The function follows the example layout in ``dllm/examples/bert/pt.py``
    and reuses dllm utilities (``get_model``, ``get_tokenizer``,
    ``tokenize_and_group``, ``NoAttentionMaskWrapper``,
    ``MDLMConfig``/``MDLMTrainer``) so that we benefit from the upstream
    treatment of the masking schedule, the loss weighting and the
    EOS-padding collator.
    """
    import functools
    import os

    import dllm
    import transformers
    from datasets import load_dataset

    output_dir = Path(cfg.output_dir) / cfg.expert_name
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)

    # ---- 1. Backbone + tokenizer (LoRA wrapping is delegated to dllm) ---
    target_modules = ",".join(cfg.lora_target_modules)
    model_args = dllm.utils.ModelArguments(
        model_name_or_path=cfg.backbone,
        dtype="bfloat16" if cfg.precision == "bf16" else "float32",
        lora=True,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=None,  # don't accidentally make embeddings trainable
    )
    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # Belt-and-braces: explicitly freeze embedding-like parameters even
    # though peft already locks everything that is not a LoRA adapter.
    if cfg.freeze_embeddings:
        for name, param in model.named_parameters():
            lname = name.lower()
            if ("embed" in lname or "wte" in lname or "wpe" in lname) and "lora" not in lname:
                param.requires_grad = False

    # ---- 2. Dataset (JSONL → tokenised, grouped into fixed-length blocks) ---
    raw = load_dataset("json", data_files=str(cfg.train_jsonl), split="train")
    dataset = raw.map(
        functools.partial(
            dllm.utils.tokenize_and_group,
            tokenizer=tokenizer,
            text_field="text",
            seq_length=cfg.sequence_length,
            insert_eos=True,
            drop_tail=True,
        ),
        batched=True,
        remove_columns=raw.column_names,
        desc=f"tokenize_and_group ({cfg.expert_name})",
    )

    # ---- 3. Training arguments ----------------------------------------------
    training_args = dllm.core.trainers.MDLMConfig(
        output_dir=str(output_dir),
        max_steps=cfg.num_steps,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=max(1, cfg.batch_size // cfg.grad_accum_steps),
        gradient_accumulation_steps=cfg.grad_accum_steps,
        warmup_steps=cfg.warmup_steps,
        lr_scheduler_type=cfg.lr_schedule,
        bf16=cfg.precision == "bf16",
        gradient_checkpointing=cfg.grad_checkpointing,
        save_steps=cfg.save_every,
        eval_strategy="no",
        save_strategy="steps",
        save_only_model=True,
        seed=cfg.seed,
        report_to="wandb",
        run_name=cfg.wandb_run_name or f"expert-{cfg.expert_name}",
        logging_steps=10,
        # MDLM.forward does not declare `labels` (it derives them from input_ids
        # via the masking step), so HF Trainer's default would drop the column.
        remove_unused_columns=False,
    )

    # ---- 4. Trainer ----------------------------------------------------------
    collator = dllm.utils.NoAttentionMaskWrapper(
        transformers.DataCollatorForSeq2Seq(
            tokenizer,
            return_tensors="pt",
            padding=True,
            label_pad_token_id=tokenizer.pad_token_id,
        ),
    )
    trainer = dllm.core.trainers.MDLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=collator,
    )

    # Surface the actual module structure once so target_modules can be
    # verified / corrected against the real backbone (DiT / BERT / …).
    trainer.accelerator.print(
        f"[{cfg.expert_name}] LoRA target_modules={cfg.lora_target_modules}; "
        f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):_}"
    )

    trainer.train()

    # ---- 5. Persist the adapter only (LoRA weights, not the backbone) ------
    model.save_pretrained(output_dir)
    return output_dir
