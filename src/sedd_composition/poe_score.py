"""PoE-score composition for SEDD.

Mirrors ``src.composition.poe_sampler.PoECompositionModel`` (which composes
MDLM logits) but operates on **log-scores** as defined in SEDD:

    log s_theta(x, t)_y = log [ p_t(y) / p_t(x) ]   for y != x

The PoE-of-densities identity in log-space gives:

    log p_PoE(x) = log p_b(x) + Σ_k λ_k (log p_k(x) − log p_b(x))

which transports cleanly to log-scores:

    log s_PoE = log s_b + Σ_k λ_k (log s_k − log s_b)
              = (1 − Σ λ_k) · log s_b + Σ λ_k · log s_k

This identity is **exact at the sequence level** because scores capture
transition ratios that already encode joint structure — unlike MDLM's
factorized per-position categoricals.

Decoupled mixture-PoE (the Paper-1 μ-fix) translates verbatim:

    log s_custom = mu_base · log s_b + Σ λ_k · log s_k

with mu_base = (1 − Σ λ_k) recovering vanilla PoE.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PoEScoreConfig:
    """Composition + sampling config for the score-based stack."""

    # Sampling
    num_steps: int = 256
    eps: float = 1e-5
    sample_batch_size: int = 32

    # Composition
    # Decouple the coefficient on log s_base from the canonical (1 − Σλ_k).
    # None ≡ standard PoE. Set to a float to enable Paper-1's μ-fix on
    # the score domain. Identical semantics to PoEConfig.mu_base.
    mu_base: float | None = None

    # Step-aware schedule on mu_base. If both mu_schedule and mu_base_end
    # are set, μ varies between mu_base (start) and mu_base_end (end)
    # along progress p ∈ [0, 1]. Schedule names mirror the MDLM stack
    # ("linear", "cosine", "late_fire", "early_fire"); supplied via
    # the SCHEDULES dict below.
    mu_schedule: str | None = None
    mu_base_end: float | None = None

    # Numerical guard for the absorbing-token column. SEDD parameterizes
    # transition log-ratios; the absorbing index (vocab − 1) corresponds
    # to "stay masked" rates and must remain self-consistent across
    # experts. We zero out the per-expert delta at this index by default.
    sanitize_absorbing_index: bool = True
    absorbing_vocab_index: int = 50257  # GPT-2 + 1 absorbing


# Reuse the same shape library as the MDLM stack for consistency.
def _schedule_constant(_p: float) -> float:
    return 1.0


def _schedule_linear(p: float) -> float:
    return float(p)


def _schedule_late_fire(p: float) -> float:
    return 0.0 if p < 0.5 else 1.0


def _schedule_early_fire(p: float) -> float:
    return 1.0 if p < 0.5 else 0.0


def _schedule_cosine(p: float) -> float:
    import math

    return 0.5 * (1.0 - math.cos(math.pi * p))


SCHEDULES = {
    "constant": _schedule_constant,
    "linear": _schedule_linear,
    "cosine": _schedule_cosine,
    "late_fire": _schedule_late_fire,
    "early_fire": _schedule_early_fire,
}


class PoEScoreCompositionModel(nn.Module):
    """Drop-in for the SEDD sampler: ``forward(x, sigma) -> log_score``.

    Internally performs ``len(lambdas) + 1`` forwards through the shared
    backbone (one base, one per active expert), then composes log-scores
    on the same shape ``(batch, seq, vocab)``. Returns log-scores; the
    sampler's ``get_score_fn(..., sampling=True)`` will exp() it.

    The shared backbone is a ``peft.PeftModel`` carrying multiple LoRA
    adapters (the experts). Adapters are swapped in/out via
    ``set_adapter`` / ``disable_adapter``.

    Parameters
    ----------
    base_with_adapters : nn.Module
        SEDD score model with ≥ 1 LoRA adapter loaded.
    lambdas : dict[str, float]
        Adapter name → λ_k.
    cfg : PoEScoreConfig
    total_steps : int
        Used for step-aware μ schedules.
    """

    def __init__(
        self,
        base_with_adapters: nn.Module,
        lambdas: dict[str, float],
        cfg: PoEScoreConfig | None = None,
        total_steps: int = 256,
    ):
        super().__init__()
        self.base = base_with_adapters
        self.lambdas = {k: float(v) for k, v in lambdas.items()}
        self.cfg = cfg or PoEScoreConfig()
        self.total_steps = max(1, int(total_steps))
        self.step_count = 0

        # Schedule fn lookup
        if self.cfg.mu_schedule is not None:
            if self.cfg.mu_schedule not in SCHEDULES:
                raise ValueError(
                    f"unknown mu_schedule '{self.cfg.mu_schedule}'; choices: {list(SCHEDULES)}"
                )
            self._mu_schedule_fn = SCHEDULES[self.cfg.mu_schedule]
        else:
            self._mu_schedule_fn = None

    # ---- bookkeeping ---------------------------------------------------

    def reset_step_count(self) -> None:
        self.step_count = 0

    @property
    def device(self):  # pragma: no cover — duck-typed for sampler
        return next(self.base.parameters()).device

    # ---- composition ---------------------------------------------------

    def _resolve_mu(self) -> float | None:
        """Return the effective μ for the current step, or None for the
        canonical (1 − Σλ_k) coefficient."""
        cfg = self.cfg
        if cfg.mu_base is None:
            return None
        if self._mu_schedule_fn is None or cfg.mu_base_end is None:
            return float(cfg.mu_base)
        progress = self.step_count / max(1, self.total_steps - 1)
        w = float(self._mu_schedule_fn(progress))
        return float(cfg.mu_base) + (float(cfg.mu_base_end) - float(cfg.mu_base)) * w

    def _sanitize(self, log_score: torch.Tensor) -> torch.Tensor:
        """Replace NaN/Inf at the absorbing column with 0 to keep the PoE
        sum well-defined. SEDD's score at the absorbing index is
        structurally ``-inf`` for unmasked tokens; an unweighted PoE sum
        of ``-inf`` would propagate. We keep that column as the base
        model's value and zero out per-expert deltas there."""
        if not self.cfg.sanitize_absorbing_index:
            return log_score
        idx = self.cfg.absorbing_vocab_index
        if idx >= log_score.shape[-1]:
            return log_score
        # Replace non-finite at the absorbing index with 0.
        col = log_score[..., idx]
        bad = ~torch.isfinite(col)
        if bad.any():
            log_score = log_score.clone()
            log_score[..., idx] = torch.where(bad, torch.zeros_like(col), col)
        return log_score

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Compose log-scores from the base + active LoRA adapters.

        Returns
        -------
        log_score : Tensor of shape (batch, seq_len, vocab)
            Log of the SEDD score (transition log-ratios). The sampler's
            ``get_score_fn(..., sampling=True)`` will exp() this.
        """
        self.step_count += 1

        # Base log-score: backbone with all adapters disabled.
        with torch.no_grad():
            with self.base.disable_adapter():
                log_s_base = self.base(x, sigma)

            mu_eff = self._resolve_mu()

            if mu_eff is None:
                # Canonical PoE form:
                #   log s_PoE = log s_b + Σ λ_k (log s_k − log s_b)
                delta = torch.zeros_like(log_s_base)
                for name, lam in self.lambdas.items():
                    if lam == 0.0:
                        continue
                    self.base.set_adapter(name)
                    log_s_k = self.base(x, sigma)
                    delta = delta + lam * (log_s_k - log_s_base)
                composed = log_s_base + delta
            else:
                # Decoupled mixture-PoE:
                #   log s_custom = mu_eff · log s_b + Σ λ_k · log s_k
                composed = float(mu_eff) * log_s_base
                for name, lam in self.lambdas.items():
                    if lam == 0.0:
                        continue
                    self.base.set_adapter(name)
                    log_s_k = self.base(x, sigma)
                    composed = composed + lam * log_s_k

        return self._sanitize(composed)

    # The sampler calls model(x, sigma); nn.Module wires that to forward().
    def __call__(self, x, sigma):  # pragma: no cover — explicit for clarity
        return self.forward(x, sigma)
