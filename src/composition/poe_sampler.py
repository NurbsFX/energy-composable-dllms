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
    sample_batch_size: int = 32
    coherence_filter: bool = True
    max_resample_attempts: int = 5
    # Step-aware λ schedule (Option β). ``None`` ≡ constant λ across all
    # denoising steps (vanilla PoE). Otherwise the chosen schedule modulates
    # the effective λ per denoising step — late_fire/cosine/exp let the
    # composition wait until the mostly-clean phase to push hard.
    lambda_schedule: str | None = None
    # Decoupled coefficient on log p_base in the composition. ``None`` ≡
    # standard PoE (coefficient = 1 − Σ λ_i). Setting an explicit value
    # turns the composition into a "mixture-PoE" of the form
    #   logits_custom = mu_base · logits_base + Σ λ_i · logits_i
    # which lets us decouple how strongly OWT-typical text is penalised
    # from how strongly experts push.
    mu_base: float | None = None


class PoECompositionModel:
    """``forward(input_ids, attention_mask)`` → PoE-composed logits.

    Acts as a drop-in for a regular HuggingFace MaskedLM model from the
    dllm sampler's perspective: one ``forward`` call returns an object
    with a ``.logits`` attribute. Internally it does ``len(lambdas) + 1``
    forward passes through the shared backbone (one base, one per active
    expert).

    When ``lambda_schedule_fn`` is set, the per-expert λ at call ``k`` is
    multiplied by ``lambda_schedule_fn(k / max(1, total_steps - 1))``. The
    schedule lets late-step calls push harder than early ones — useful for
    diffusion-LM PoE-N composition where the per-step approximation error
    cumulates over the denoising trajectory.
    """

    def __init__(
        self,
        base_with_adapters: nn.Module,
        lambdas: dict[str, float],
        lambda_schedule_fn=None,
        total_steps: int = 256,
        mu_base: float | None = None,
    ):
        self.base = base_with_adapters
        self.lambdas = {k: float(v) for k, v in lambdas.items()}
        self.lambda_schedule_fn = lambda_schedule_fn
        self.total_steps = max(1, int(total_steps))
        self.step_count = 0
        # ``mu_base`` decouples the coefficient on log p_base from (1 - Σλ_i).
        # Standard PoE: mu_base = None → coefficient is (1 - Σλ_i), matching
        #   logits_PoE = logits_base + Σ λ_i (logits_i − logits_base).
        # Custom mixture-PoE: mu_base = float → coefficient on logits_base
        #   becomes that value independently, allowing
        #   logits_custom = mu_base · logits_base + Σ λ_i · logits_i
        # (so e.g. mu_base = 0 yields a pure sum-of-experts; mu_base = -1 a
        # half-strength PoE penalty at N=3 instead of the canonical -2).
        self.mu_base = mu_base
        # Used by some HF utilities.
        self.device = next(base_with_adapters.parameters()).device

    def reset_step_count(self) -> None:
        self.step_count = 0

    def parameters(self):  # pragma: no cover — duck-typed for the dllm sampler
        return self.base.parameters()

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        import torch
        from transformers.modeling_outputs import MaskedLMOutput

        if self.lambda_schedule_fn is not None:
            progress = self.step_count / max(1, self.total_steps - 1)
            scale = float(self.lambda_schedule_fn(progress))
            effective = {k: v * scale for k, v in self.lambdas.items()}
        else:
            effective = self.lambdas
        self.step_count += 1

        with torch.no_grad():
            with self.base.disable_adapter():
                logits_base = self.base(input_ids=input_ids, attention_mask=attention_mask).logits

            if self.mu_base is None:
                # Standard PoE form: logits_PoE = logits_base + Σ λ_i (logits_i − logits_base)
                logits_delta = torch.zeros_like(logits_base)
                for name, lam in effective.items():
                    if lam == 0.0:
                        continue
                    self.base.set_adapter(name)
                    logits_i = self.base(input_ids=input_ids, attention_mask=attention_mask).logits
                    logits_delta = logits_delta + lam * (logits_i - logits_base)
                composed = logits_base + logits_delta
            else:
                # Decoupled mixture-PoE: logits_custom = mu_base · logits_base + Σ λ_i · logits_i
                composed = float(self.mu_base) * logits_base
                for name, lam in effective.items():
                    if lam == 0.0:
                        continue
                    self.base.set_adapter(name)
                    logits_i = self.base(input_ids=input_ids, attention_mask=attention_mask).logits
                    composed = composed + lam * logits_i

        return MaskedLMOutput(logits=composed)


# Library of λ schedules over the denoising trajectory. Each takes
# ``progress ∈ [0, 1]`` (0 = first denoising step, 1 = last) and returns the
# multiplicative scale to apply to the base λ.


def schedule_constant(_progress: float) -> float:
    return 1.0


def schedule_linear(progress: float) -> float:
    """Ramp from 0 → 1 over the trajectory."""
    return progress


def schedule_late_fire(progress: float) -> float:
    """Stay at 0 until midpoint, jump to 1 — keeps the early base diffusion clean."""
    return 0.0 if progress < 0.5 else 1.0


def schedule_cosine(progress: float) -> float:
    """Smooth ramp 0 → 1 via half-cosine."""
    import math

    return 0.5 * (1 - math.cos(math.pi * progress))


def schedule_exp(progress: float) -> float:
    """Exponential ramp — late steps get most of the push."""
    import math

    return (math.exp(progress) - 1) / (math.e - 1)


def schedule_early_fire(progress: float) -> float:
    """Inverse of late-fire: push hard early, fade out — diagnostic counterpart."""
    return 1.0 if progress < 0.5 else 0.0


SCHEDULES = {
    "constant": schedule_constant,
    "linear": schedule_linear,
    "late_fire": schedule_late_fire,
    "cosine": schedule_cosine,
    "exp": schedule_exp,
    "early_fire": schedule_early_fire,
}


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
        sched_fn = SCHEDULES[self.cfg.lambda_schedule] if self.cfg.lambda_schedule else None
        wrapped = PoECompositionModel(
            self.base,
            lambdas,
            lambda_schedule_fn=sched_fn,
            total_steps=self.cfg.num_steps,
            mu_base=self.cfg.mu_base,
        )
        self._wrapped_model = wrapped  # held so we can reset_step_count between attempts
        return MDLMSampler(model=wrapped, tokenizer=self.tokenizer, scheduler=scheduler)

    def sample(
        self,
        prompts: list[str],
        lambdas: dict[str, float],
    ) -> list[str]:
        """Generate one sample per prompt with PoE-composed logits."""
        from dllm.core.samplers import MDLMSamplerConfig

        from .coherence import sample_with_rejection

        sampler = self._build_dllm_sampler(lambdas)
        config = MDLMSamplerConfig(
            max_new_tokens=self.cfg.max_new_tokens,
            steps=self.cfg.num_steps,
            temperature=self.cfg.temperature,
            block_size=self.cfg.block_size or self.cfg.max_new_tokens,
        )
        prompt_tokens = [self.tokenizer.encode(p, add_special_tokens=False) for p in prompts]
        bs = max(1, self.cfg.sample_batch_size)

        def _one_attempt(prompt_chunk: list[list[int]]) -> list[list[int]]:
            out: list = []
            for start in range(0, len(prompt_chunk), bs):
                if hasattr(self, "_wrapped_model") and self._wrapped_model is not None:
                    self._wrapped_model.reset_step_count()
                out.extend(sampler.sample(prompt_chunk[start : start + bs], config=config))
            return out

        if not self.cfg.coherence_filter:
            return [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in _one_attempt(prompt_tokens)
            ]

        label = "+".join(f"{k}={v:.2f}" for k, v in sorted(lambdas.items())) or "base"
        texts, _ = sample_with_rejection(
            _one_attempt,
            lambda ids: self.tokenizer.decode(ids, skip_special_tokens=True),
            prompt_tokens,
            seed=self.cfg.seed,
            max_attempts=self.cfg.max_resample_attempts,
            label=label,
        )
        return texts

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
