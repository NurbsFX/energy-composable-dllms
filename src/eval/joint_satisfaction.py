"""Per-config aggregation of the composition sweep."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SampleRecord:
    text: str
    length: int
    score_len: float
    score_form: float
    score_sent: float
    score_tox: float
    ppl_gpt2: float
    distinct_2: float


def compute_distinct_2(tokens: list[str]) -> float:
    """Fraction of unique bigrams; mode-collapse signal."""
    if len(tokens) < 2:
        return 0.0
    bigrams = list(zip(tokens[:-1], tokens[1:], strict=True))
    return len(set(bigrams)) / len(bigrams)


def compute_thresholds(scores_baseline: dict[str, np.ndarray], q: float = 0.75) -> dict[str, float]:
    """Per-vertical threshold = q-quantile of the baseline distribution."""
    return {k: float(np.quantile(v, q)) for k, v in scores_baseline.items()}


@dataclass
class ConfigSummary:
    config_name: str
    n_samples: int
    joint_satisfaction: float
    marginal_a: float
    marginal_b: float
    mode_collapse_ratio: float
    fluency_ratio: float


def summarize(
    samples: list[SampleRecord],
    *,
    score_keys: tuple[str, str],
    thresholds: dict[str, float],
    baseline_distinct_2: float,
    baseline_ppl: float,
    config_name: str = "",
) -> ConfigSummary:
    if not samples:
        return ConfigSummary(config_name, 0, 0.0, 0.0, 0.0, float("nan"), float("nan"))

    a_key, b_key = score_keys
    sat_a = np.array([getattr(s, f"score_{a_key}") > thresholds[a_key] for s in samples])
    sat_b = np.array([getattr(s, f"score_{b_key}") > thresholds[b_key] for s in samples])

    distinct_2 = float(np.mean([s.distinct_2 for s in samples]))
    ppl = float(np.mean([s.ppl_gpt2 for s in samples]))

    return ConfigSummary(
        config_name=config_name,
        n_samples=len(samples),
        joint_satisfaction=float(np.mean(sat_a & sat_b)),
        marginal_a=float(np.mean(sat_a)),
        marginal_b=float(np.mean(sat_b)),
        mode_collapse_ratio=distinct_2 / baseline_distinct_2
        if baseline_distinct_2
        else float("nan"),
        fluency_ratio=ppl / baseline_ppl if baseline_ppl else float("nan"),
    )
