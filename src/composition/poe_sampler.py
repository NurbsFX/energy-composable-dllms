"""Product-of-Experts sampler over a shared MDLM backbone with LoRA adapters.

Composition rule on logits:

    logits_PoE(x_t) = logits_base(x_t)
                    + Σ_i λ_i · (logits_i(x_t) − logits_base(x_t))

All adapters share the backbone (one model in memory); we swap adapters via
``peft.set_adapter`` / ``peft.disable_adapter`` between forwards.  Embeddings
must have been frozen during fine-tuning for the per-position sum of logits to
remain mathematically coherent.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PoEConfig:
    num_steps: int = 256
    max_new_tokens: int = 128
    temperature: float = 1.0
    seed: int | None = None


class PoESampler:
    def __init__(
        self,
        base_model,
        experts: dict[str, str],  # expert name -> peft adapter id
        tokenizer,
        cfg: PoEConfig | None = None,
    ):
        self.base = base_model
        self.experts = experts
        self.tokenizer = tokenizer
        self.cfg = cfg or PoEConfig()

    def sample(self, prompt_tokens, lambdas: dict[str, float]) -> str:
        raise NotImplementedError

    def _init_masked(self, length: int):
        raise NotImplementedError

    def _denoise_step(self, x_t, logits, t: float):
        raise NotImplementedError

    def assert_lambda_zero_is_base(self, prompt_tokens, *, n_trials: int = 3) -> None:
        """λ=0 must reproduce the bare backbone token-for-token at fixed seed."""
        raise NotImplementedError
