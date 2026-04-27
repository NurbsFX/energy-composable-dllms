"""Baselines compared against PoE — currently only naive LoRA-parameter merging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn


@dataclass
class NaiveLoRAMergeConfig:
    expert_a: str
    expert_b: str
    w_a: float = 0.5
    w_b: float = 0.5
    merged_name: str = "merged"


class MergedSampler:
    """Sampler that activates a single, parameter-merged adapter."""

    def __init__(self, base_with_merged_adapter, tokenizer, scheduler=None, cfg=None):
        from src.composition.poe_sampler import PoEConfig

        self.base = base_with_merged_adapter
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.cfg = cfg or PoEConfig()

    def sample(self, prompts: list[str], merged_name: str = "merged") -> list[str]:
        from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig
        from dllm.core.schedulers import LinearAlphaScheduler

        from .coherence import sample_with_rejection

        self.base.set_adapter(merged_name)
        sampler = MDLMSampler(
            model=self.base,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler or LinearAlphaScheduler(),
        )
        config = MDLMSamplerConfig(
            max_new_tokens=self.cfg.max_new_tokens,
            steps=self.cfg.num_steps,
            temperature=self.cfg.temperature,
            block_size=self.cfg.block_size or self.cfg.max_new_tokens,
        )
        prompt_tokens = [self.tokenizer.encode(p, add_special_tokens=False) for p in prompts]
        bs = max(1, self.cfg.sample_batch_size)

        def _one_attempt(chunk: list[list[int]]) -> list[list[int]]:
            out: list = []
            for start in range(0, len(chunk), bs):
                out.extend(sampler.sample(chunk[start : start + bs], config=config))
            return out

        if not self.cfg.coherence_filter:
            return [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in _one_attempt(prompt_tokens)
            ]

        texts, _ = sample_with_rejection(
            _one_attempt,
            lambda ids: self.tokenizer.decode(ids, skip_special_tokens=True),
            prompt_tokens,
            seed=self.cfg.seed,
            max_attempts=self.cfg.max_resample_attempts,
            label=f"merged:{merged_name}",
        )
        return texts


def merge_loras(base_model: nn.Module, cfg: NaiveLoRAMergeConfig) -> str:
    """Linearly interpolate two adapters into a single ``cfg.merged_name`` adapter.

    Both adapters must already be loaded on ``base_model`` (a ``PeftModel``
    instance with both ``cfg.expert_a`` and ``cfg.expert_b`` registered).
    The merge happens directly on the LoRA A/B matrices: for each layer,

        merged.A = w_a · A_a + w_b · A_b
        merged.B = w_a · B_a + w_b · B_b

    The combined effective update is therefore

        ΔW_merged = (w_a A_a + w_b A_b)(w_a B_a + w_b B_b)ᵀ

    which is *not* the same as ``w_a · ΔW_a + w_b · ΔW_b`` (the cross
    terms ``w_a w_b A_a B_b + w_a w_b A_b B_a`` are exactly what makes
    naive LoRA parameter averaging a non-trivial baseline against PoE on
    logits).

    Returns the name of the registered merged adapter; the caller is
    responsible for building a :class:`MergedSampler` with its own
    tokenizer and for ``base_model.delete_adapter(merged_name)`` once
    the adapter is no longer needed.
    """
    import torch
    from peft import LoraConfig
    from peft.tuners.lora import LoraLayer

    state_a = _adapter_state(base_model, cfg.expert_a)
    state_b = _adapter_state(base_model, cfg.expert_b)
    if set(state_a) != set(state_b):
        raise ValueError(
            "merged adapters must share the exact same module set; "
            f"a={sorted(state_a)} b={sorted(state_b)}"
        )

    # Register the merged adapter using the config of expert A as the
    # template (same r, alpha, target_modules, …).
    template = base_model.peft_config[cfg.expert_a]
    base_model.add_adapter(cfg.merged_name, LoraConfig(**template.to_dict()))

    # Overwrite its tensors with the linear combo.
    with torch.no_grad():
        for _name, module in base_model.named_modules():
            if not isinstance(module, LoraLayer):
                continue
            if cfg.merged_name not in module.lora_A:
                continue  # not a target_module for this adapter
            a = module.lora_A
            b = module.lora_B
            a[cfg.merged_name].weight.copy_(
                cfg.w_a * a[cfg.expert_a].weight + cfg.w_b * a[cfg.expert_b].weight
            )
            b[cfg.merged_name].weight.copy_(
                cfg.w_a * b[cfg.expert_a].weight + cfg.w_b * b[cfg.expert_b].weight
            )

    return cfg.merged_name


def _adapter_state(model, adapter_name: str) -> dict[str, tuple]:
    """Return the dict of LoRA tensors for ``adapter_name`` keyed by module path."""
    from peft.tuners.lora import LoraLayer

    out: dict[str, tuple] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer) and adapter_name in module.lora_A:
            out[name] = (module.lora_A[adapter_name], module.lora_B[adapter_name])
    return out
