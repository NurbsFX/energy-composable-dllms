"""Product-of-Experts sampler over a shared MDLM backbone with LoRA adapters.

Composition rule on logits:

    logits_PoE(x_t) = logits_base(x_t)
                    + Σ_i λ_i · (logits_i(x_t) − logits_base(x_t))

All adapters share the backbone (one model in memory); we swap adapters via
``peft.set_adapter`` / ``disable_adapter`` between forwards. Embeddings must
have been frozen during fine-tuning for the per-position sum of logits to
remain mathematically coherent.

The implementation reuses ``dllm.core.samplers.MDLMSampler`` as the
denoising backbone — we wrap our peft-multi-adapter model in a small
``nn.Module`` whose ``forward`` returns the PoE-composed logits, and let
the dllm sampler drive the iterative unmasking loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn


@dataclass
class PoEConfig:
    num_steps: int = 256
    max_new_tokens: int = 128
    temperature: float = 1.0
    block_size: int | None = None  # default: full max_new_tokens (one block)
    seed: int | None = None


class PoECompositionModel:
    """``forward(input_ids, attention_mask)`` → PoE-composed logits.

    Acts as a drop-in for a regular HuggingFace MaskedLM model from the
    dllm sampler's perspective: one ``forward`` call returns an object
    with a ``.logits`` attribute. Internally it does ``len(lambdas) + 1``
    forward passes through the shared backbone (one base, one per active
    expert).
    """

    def __init__(
        self,
        base_with_adapters: nn.Module,
        lambdas: dict[str, float],
    ):
        self.base = base_with_adapters
        self.lambdas = {k: float(v) for k, v in lambdas.items()}
        # Used by some HF utilities.
        self.device = next(base_with_adapters.parameters()).device

    def parameters(self):  # pragma: no cover — duck-typed for the dllm sampler
        return self.base.parameters()

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        import torch
        from transformers.modeling_outputs import MaskedLMOutput

        with torch.no_grad():
            with self.base.disable_adapter():
                logits_base = self.base(input_ids=input_ids, attention_mask=attention_mask).logits

            logits_delta = torch.zeros_like(logits_base)
            for name, lam in self.lambdas.items():
                if lam == 0.0:
                    continue
                self.base.set_adapter(name)
                logits_i = self.base(input_ids=input_ids, attention_mask=attention_mask).logits
                logits_delta = logits_delta + lam * (logits_i - logits_base)

        return MaskedLMOutput(logits=logits_base + logits_delta)


class PoESampler:
    """Compose multiple LoRA experts at inference time on top of an MDLM backbone."""

    def __init__(
        self,
        base_with_adapters: nn.Module,
        tokenizer,
        scheduler=None,
        cfg: PoEConfig | None = None,
    ):
        self.base = base_with_adapters
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.cfg = cfg or PoEConfig()

    def _build_dllm_sampler(self, lambdas: dict[str, float]):
        from dllm.core.samplers import MDLMSampler
        from dllm.core.schedulers import LinearAlphaScheduler

        scheduler = self.scheduler or LinearAlphaScheduler()
        wrapped = PoECompositionModel(self.base, lambdas)
        return MDLMSampler(model=wrapped, tokenizer=self.tokenizer, scheduler=scheduler)

    def sample(
        self,
        prompts: list[str],
        lambdas: dict[str, float],
    ) -> list[str]:
        """Generate one sample per prompt with PoE-composed logits."""
        import torch
        from dllm.core.samplers import MDLMSamplerConfig

        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)

        sampler = self._build_dllm_sampler(lambdas)
        config = MDLMSamplerConfig(
            max_new_tokens=self.cfg.max_new_tokens,
            steps=self.cfg.num_steps,
            temperature=self.cfg.temperature,
            block_size=self.cfg.block_size or self.cfg.max_new_tokens,
        )
        prompt_tokens = [self.tokenizer.encode(p, add_special_tokens=False) for p in prompts]
        out = sampler.sample(prompt_tokens, config=config)
        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in out]

    def assert_lambda_zero_is_base(self, prompts: list[str], *, n_trials: int = 3) -> None:
        """λ=0 must reproduce the bare backbone token-for-token at fixed seed.

        If this fails the composition formula has a sign / accumulation bug
        and Phase-4 measurements cannot be trusted.
        """
        import torch

        zero_lambdas = {name: 0.0 for name in self.lambdas_keys_for_base()}

        for trial in range(n_trials):
            seed = (self.cfg.seed or 0) + trial

            torch.manual_seed(seed)
            base_only_cfg = PoEConfig(**{**self.cfg.__dict__, "seed": seed})
            base_sampler = PoESampler(self.base, self.tokenizer, self.scheduler, base_only_cfg)
            base_out = base_sampler.sample(prompts, lambdas={})

            torch.manual_seed(seed)
            poe_cfg = PoEConfig(**{**self.cfg.__dict__, "seed": seed})
            poe_sampler = PoESampler(self.base, self.tokenizer, self.scheduler, poe_cfg)
            poe_out = poe_sampler.sample(prompts, lambdas=zero_lambdas)

            if base_out != poe_out:
                raise AssertionError(
                    f"λ=0 regression failed on trial {trial}: base={base_out!r} vs poe={poe_out!r}"
                )

    def lambdas_keys_for_base(self) -> list[str]:
        """Return the adapter names installed on the backbone, used by λ=0 test."""
        try:
            return list(self.base.peft_config.keys())
        except AttributeError:  # pragma: no cover — backbone without peft
            return []
