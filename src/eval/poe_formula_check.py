"""Direct verification of the PoE composition formula on K sequence pairs.

For every pair (x, y) we estimate the four log-ratios via the MDLM ELBO bound
and check that

    log p_PoE(y)/p_PoE(x)  ≈  log p_1(y)/p_1(x)
                            + log p_2(y)/p_2(x)
                            − log p_base(y)/p_base(x).

A successful fit on the resulting log-log scatter has slope ≈ 1, intercept
≈ 0, and R² ≳ 0.85.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FormulaCheckResult:
    log_ratios_predicted: np.ndarray  # log p_1/p_base + log p_2/p_base
    log_ratios_observed: np.ndarray  # log p_PoE/p_base
    slope: float
    intercept: float
    r2: float
    n_pairs: int


def estimate_log_ratio_elbo(model, x, y, *, num_t_samples: int = 32) -> float:
    """Estimate log p_θ(y) − log p_θ(x) by Monte-Carlo over MDLM denoising losses."""
    raise NotImplementedError


def check_poe_formula(
    base_model,
    expert_a,
    expert_b,
    pairs: list[tuple[str, str]],
    *,
    num_t_samples: int = 32,
) -> FormulaCheckResult:
    raise NotImplementedError
