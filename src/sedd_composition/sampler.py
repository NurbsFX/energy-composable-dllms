"""High-level sampling API for the score-based composition stack.

Wraps Lou's ``sampling.get_pc_sampler`` with our composed-score model.
The composition is exact at the sequence level (see ``poe_score.py``);
the predictor-corrector loop with the analytic (τ-leaping) predictor
is the canonical SEDD sampler — we reuse it unchanged.

Usage:

    base = peft_sedd_with_adapters
    cfg = PoEScoreConfig(num_steps=128)
    sampler = PoEScoreSampler(base, graph, noise, tokenizer, cfg=cfg)
    texts = sampler.sample(
        num_samples=8,
        seq_len=64,
        lambdas={"formal": 1.0, "positive": 1.0},
    )
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .load import _ensure_sedd_on_path
from .poe_score import PoEScoreCompositionModel, PoEScoreConfig


@dataclass
class _SamplerArtifacts:
    """Convenience bundle so callers don't juggle (graph, noise, tokenizer)."""

    graph: object
    noise: object
    tokenizer: object


class PoEScoreSampler:
    """Compose multiple SEDD LoRA experts at inference time."""

    def __init__(
        self,
        base_with_adapters,
        graph,
        noise,
        tokenizer,
        cfg: PoEScoreConfig | None = None,
    ):
        self.base = base_with_adapters
        self.artifacts = _SamplerArtifacts(graph=graph, noise=noise, tokenizer=tokenizer)
        self.cfg = cfg or PoEScoreConfig()

    def _build_composition_model(self, lambdas: dict[str, float]) -> PoEScoreCompositionModel:
        return PoEScoreCompositionModel(
            self.base,
            lambdas=lambdas,
            cfg=self.cfg,
            total_steps=self.cfg.num_steps,
        )

    def _device(self) -> torch.device:
        return next(self.base.parameters()).device

    def _build_pc_sampler(self, batch_dims: tuple[int, ...]):
        """Build Lou's predictor-corrector sampler bound to graph+noise."""
        _ensure_sedd_on_path()
        from sampling import get_pc_sampler  # type: ignore

        device = self._device()
        sampler_fn = get_pc_sampler(
            graph=self.artifacts.graph,
            noise=self.artifacts.noise,
            batch_dims=batch_dims,
            predictor="analytic",  # τ-leaping
            steps=self.cfg.num_steps,
            denoise=True,
            eps=self.cfg.eps,
            device=device,
        )
        return sampler_fn

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        seq_len: int,
        lambdas: dict[str, float],
    ) -> list[str]:
        """Generate `num_samples` sequences of length `seq_len`.

        Each pass through the sampler runs ``num_steps`` predictor-corrector
        updates; the composition model performs ``len(active lambdas) + 1``
        backbone forwards per step. Coherence-filter / rejection logic is
        deliberately omitted at this layer (SEDD samples tend to be more
        coherent than MDLM at equal compute); add it later if needed.
        """
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")

        bs = max(1, self.cfg.sample_batch_size)
        out: list[str] = []
        for start in range(0, num_samples, bs):
            chunk = min(bs, num_samples - start)
            sampler_fn = self._build_pc_sampler(batch_dims=(chunk, seq_len))
            wrapped = self._build_composition_model(lambdas)
            wrapped.reset_step_count()
            ids = sampler_fn(wrapped)  # (chunk, seq_len) of token ids
            for row in ids.tolist():
                # Strip the absorbing/MASK token id if it ever leaks through
                # (shouldn't after Denoiser; defensive).
                clean = [t for t in row if t != self.cfg.absorbing_vocab_index]
                out.append(self.artifacts.tokenizer.decode(clean, skip_special_tokens=True))
        return out

    @torch.no_grad()
    def assert_lambda_zero_is_base(
        self,
        num_samples: int = 4,
        seq_len: int = 32,
        seed: int = 0,
        adapter_names: list[str] | None = None,
    ) -> None:
        """λ=0 ≡ disabled adapters: composition with all λ_k = 0 must
        produce identical samples to the bare backbone at fixed seed.

        Failing this means the composition formula has a sign / offset
        bug — Phase 4 measurements would not be trustable.
        """
        if adapter_names is None:
            adapter_names = self._adapter_names()
        zero_lambdas = {n: 0.0 for n in adapter_names}

        def _seeded_sample(lambdas: dict[str, float]) -> list[str]:
            torch.manual_seed(seed)
            try:
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            except Exception:  # pragma: no cover
                pass
            return self.sample(num_samples=num_samples, seq_len=seq_len, lambdas=lambdas)

        base_out = _seeded_sample({})
        composed_out = _seeded_sample(zero_lambdas)

        if base_out != composed_out:
            mismatches = [(b, c) for b, c in zip(base_out, composed_out, strict=True) if b != c]
            raise AssertionError(
                f"λ=0 regression failed on {len(mismatches)}/{num_samples} samples. "
                f"First mismatch:\n  base:    {mismatches[0][0]!r}\n  composed:{mismatches[0][1]!r}"
            )

    def _adapter_names(self) -> list[str]:
        """Return adapter names installed on the backbone."""
        try:
            return list(self.base.peft_config.keys())
        except AttributeError:  # pragma: no cover — backbone w/o peft
            return []
