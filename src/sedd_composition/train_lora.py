"""LoRA fine-tuning on SEDD with score-entropy loss.

Paper 2's training equivalent of ``src/training/train_expert.py`` (which
trains MDLM LoRA experts under the masked-diffusion ELBO). Here we use
the score-entropy loss from Lou et al. 2024, lifted from
``external/sedd/losses.py::get_loss_fn``.

Design choices (intentionally minimal):

* No HF Trainer. Lou's repo is pure PyTorch; integrating with HF
  Trainer would mean wrapping `model(x, sigma) -> log_score` in a
  module that produces a `loss` field, plus a custom data collator —
  ~2 days of plumbing for negligible gain at our scale.
* Manual training loop with Adam + linear warmup + cosine decay +
  gradient clipping — same recipe as Lou's `train.py`.
* LoRA target_modules = ("attn_qkv", "attn_out"), the same shape as
  the MDLM-OWT experts in Paper 1. Embeddings frozen for the same
  PoE-coherence reason.
* Tokenization: chunked GPT-2 tokens to fixed `sequence_length`. EOS
  is appended per source document then chunks are split — same as the
  MDLM training.

Usage:

    python -m src.sedd_composition.train_lora \\
        --expert-name formal \\
        --train-jsonl artifacts/datasets/formal.jsonl \\
        --output-dir artifacts/sedd_checkpoints \\
        --num-steps 2500
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import typer

from .load import _ensure_sedd_on_path, load_sedd_from_hub

app = typer.Typer(add_completion=False)


@dataclass
class SeddExpertTrainingConfig:
    expert_name: str
    train_jsonl: str | Path
    backbone: str = "louaaron/sedd-small"

    learning_rate: float = 3e-4
    batch_size: int = 16
    grad_accum_steps: int = 1
    sequence_length: int = 256
    num_steps: int = 2500
    warmup_steps: int = 150
    lr_schedule: str = "cosine"  # cosine | linear | constant
    grad_clip: float = 1.0
    weight_decay: float = 0.0

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("attn_qkv", "attn_out")
    freeze_embeddings: bool = True

    output_dir: str | Path = "artifacts/sedd_checkpoints"
    save_every: int = 500
    log_every: int = 25
    seed: int = 42

    # Loss-side knobs (defaults match Lou's training).
    sampling_eps: float = 1e-3

    extra_meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _load_text_lines(path: str | Path) -> list[str]:
    """Read a JSONL file and return the ``text`` field of each line."""
    out: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("text")
            if t:
                out.append(t)
    return out


def _tokenize_and_chunk(
    texts: list[str],
    tokenizer,
    seq_length: int,
    insert_eos: bool = True,
) -> list[list[int]]:
    """Concatenate documents (with EOS separators) and split into fixed
    ``seq_length`` chunks. Mirrors what dllm.utils.tokenize_and_group does
    for MDLM training, manually so we don't import dllm here."""
    eos = tokenizer.eos_token_id
    if eos is None:
        # GPT-2 sets eos_token to <|endoftext|>; guard anyway.
        raise RuntimeError("tokenizer has no eos_token_id; cannot chunk safely")
    buf: list[int] = []
    chunks: list[list[int]] = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if insert_eos:
            ids.append(eos)
        buf.extend(ids)
        while len(buf) >= seq_length:
            chunks.append(buf[:seq_length])
            buf = buf[seq_length:]
    return chunks


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def _build_lr_schedule(name: str, base_lr: float, warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return base_lr * (step + 1) / max(1, warmup)
        if name == "constant":
            return base_lr
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(1.0, max(0.0, progress))
        if name == "cosine":
            return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        if name == "linear":
            return base_lr * (1.0 - progress)
        raise ValueError(f"unknown lr_schedule: {name}")

    return f


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train(cfg: SeddExpertTrainingConfig) -> Path:
    """Run a LoRA fine-tuning of SEDD on one vertical and save the adapter."""
    _ensure_sedd_on_path()

    import losses  # type: ignore  # Lou's losses.py
    from peft import LoraConfig, get_peft_model

    from .load import get_gpt2_tokenizer_for_sedd

    output_dir = Path(cfg.output_dir) / cfg.expert_name
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    typer.echo(f"=== SEDD LoRA training: {cfg.expert_name} on {cfg.backbone} ===")
    typer.echo(f"  device={device}  num_steps={cfg.num_steps}  lr={cfg.learning_rate}")

    # 1. Backbone + graph + noise (the latter two come with the checkpoint).
    score_model, graph, noise = load_sedd_from_hub(cfg.backbone, device=device)

    # 2. Apply LoRA. Embeddings are not in target_modules; PEFT will leave
    #    them frozen. We additionally enforce that explicitly below.
    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=list(cfg.lora_target_modules),
    )
    score_model = get_peft_model(score_model, lora_cfg)

    if cfg.freeze_embeddings:
        for name, param in score_model.named_parameters():
            lname = name.lower()
            if ("embed" in lname or "wte" in lname or "wpe" in lname) and "lora" not in lname:
                param.requires_grad = False

    n_trainable = sum(p.numel() for p in score_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in score_model.parameters())
    typer.echo(f"  trainable params: {n_trainable / 1e6:.2f}M / {n_total / 1e6:.1f}M")

    # 3. Tokenize the dataset.
    tokenizer = get_gpt2_tokenizer_for_sedd()
    typer.echo(f"  loading {cfg.train_jsonl}...")
    texts = _load_text_lines(cfg.train_jsonl)
    typer.echo(f"  {len(texts)} documents loaded; tokenizing ...")
    chunks = _tokenize_and_chunk(texts, tokenizer, seq_length=cfg.sequence_length, insert_eos=True)
    typer.echo(f"  {len(chunks)} chunks of length {cfg.sequence_length}")
    if not chunks:
        raise RuntimeError(f"empty dataset for {cfg.expert_name}")

    # Stack into a tensor for cheap shuffled iteration.
    chunks_t = torch.tensor(chunks, dtype=torch.long)

    # 4. Build the loss closure (Lou's loss_fn — closes over noise, graph).
    loss_fn = losses.get_loss_fn(noise, graph, train=True, sampling_eps=cfg.sampling_eps)

    # 5. Optimiser + LR schedule.
    optim = torch.optim.AdamW(
        [p for p in score_model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )
    lr_fn = _build_lr_schedule(cfg.lr_schedule, cfg.learning_rate, cfg.warmup_steps, cfg.num_steps)

    # 6. Training loop.
    score_model.train()
    n_per_epoch = max(1, chunks_t.shape[0] // cfg.batch_size)
    rng = torch.Generator(device="cpu").manual_seed(cfg.seed)

    losses_window: list[float] = []
    step = 0
    accum = 0
    optim.zero_grad(set_to_none=True)

    # Resume hook: if a checkpoint already exists, fail early so the user
    # has to rm it (we don't want silent re-training of a finished expert).
    if (output_dir / "adapter_model.safetensors").exists():
        typer.echo(f"  ⚠ adapter already exists at {output_dir}; skipping training.")
        return output_dir

    while step < cfg.num_steps:
        # Build a fresh shuffled epoch.
        perm = torch.randperm(chunks_t.shape[0], generator=rng)
        for ep_idx in range(n_per_epoch):
            if step >= cfg.num_steps:
                break
            idx = perm[ep_idx * cfg.batch_size : (ep_idx + 1) * cfg.batch_size]
            if idx.numel() < cfg.batch_size:
                continue
            batch = chunks_t[idx].to(device)

            # Forward (loss_fn samples its own t and perturbs the batch).
            loss = loss_fn(score_model, batch).mean()
            loss = loss / cfg.grad_accum_steps
            loss.backward()
            accum += 1

            if accum >= cfg.grad_accum_steps:
                # LR + clip + step.
                lr = lr_fn(step)
                for pg in optim.param_groups:
                    pg["lr"] = lr
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in score_model.parameters() if p.requires_grad],
                        max_norm=cfg.grad_clip,
                    )
                optim.step()
                optim.zero_grad(set_to_none=True)
                accum = 0

                losses_window.append(float(loss.detach().cpu()) * cfg.grad_accum_steps)
                if step % cfg.log_every == 0:
                    avg = sum(losses_window[-cfg.log_every :]) / max(
                        1, len(losses_window[-cfg.log_every :])
                    )
                    typer.echo(
                        f"  step {step:5d}/{cfg.num_steps}  "
                        f"loss={loss.item() * cfg.grad_accum_steps:.4f}  "
                        f"avg{cfg.log_every}={avg:.4f}  lr={lr:.2e}"
                    )

                if cfg.save_every > 0 and step > 0 and step % cfg.save_every == 0:
                    score_model.save_pretrained(output_dir)
                    typer.echo(f"  saved checkpoint to {output_dir} at step {step}")
                step += 1

    # Final save.
    score_model.save_pretrained(output_dir)
    # Pin the meta so we can audit which backbone was used for this LoRA.
    (output_dir / "training_meta.json").write_text(
        json.dumps(
            {
                "expert_name": cfg.expert_name,
                "backbone": cfg.backbone,
                "num_steps": cfg.num_steps,
                "sequence_length": cfg.sequence_length,
                "batch_size": cfg.batch_size,
                "lora_rank": cfg.lora_rank,
                "lora_alpha": cfg.lora_alpha,
                "lora_target_modules": list(cfg.lora_target_modules),
                "n_chunks": int(chunks_t.shape[0]),
                "n_documents": len(texts),
                "final_loss": losses_window[-1] if losses_window else None,
                **cfg.extra_meta,
            },
            indent=2,
        )
    )
    typer.echo(f"  saved final adapter to {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    expert_name: str = typer.Option(...),
    train_jsonl: Path = typer.Option(...),
    backbone: str = "louaaron/sedd-small",
    output_dir: Path = Path("artifacts/sedd_checkpoints"),
    num_steps: int = 2500,
    batch_size: int = 16,
    sequence_length: int = 256,
    learning_rate: float = 3e-4,
    warmup_steps: int = 150,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    save_every: int = 500,
    log_every: int = 25,
    seed: int = 42,
) -> None:
    cfg = SeddExpertTrainingConfig(
        expert_name=expert_name,
        train_jsonl=train_jsonl,
        backbone=backbone,
        output_dir=output_dir,
        num_steps=num_steps,
        batch_size=batch_size,
        sequence_length=sequence_length,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        save_every=save_every,
        log_every=log_every,
        seed=seed,
    )
    train(cfg)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    app()
