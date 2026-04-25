"""Baselines compared against PoE — currently only naive LoRA-parameter merging."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NaiveLoRAMergeConfig:
    expert_a: str
    expert_b: str
    w_a: float = 0.5
    w_b: float = 0.5


class MergedSampler:
    def __init__(self, base_model, tokenizer):
        self.base = base_model
        self.tokenizer = tokenizer

    def sample(self, prompt_tokens) -> str:
        raise NotImplementedError


def merge_loras(base_model, cfg: NaiveLoRAMergeConfig) -> MergedSampler:
    """Linearly interpolate two adapters into a single 'merged' adapter."""
    raise NotImplementedError
