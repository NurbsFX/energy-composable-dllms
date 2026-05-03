"""Score-based composition stack for SEDD (Lou et al. 2024).

Parallel to ``src.composition`` (which implements PoE-logits on MDLM),
this package implements PoE-scores on SEDD. The two stacks share only
``src.eval`` for proxy scoring; they do not import from each other.

Public surface:
    load_sedd_from_hub      — fetch SEDD checkpoint via Lou's load_model_hf
    PoEScoreCompositionModel — analog of poe_sampler.PoECompositionModel
    PoEScoreSampler          — wraps Lou's pc_sampler with composed scores
"""

from __future__ import annotations

from .load import load_sedd_from_hub
from .poe_score import PoEScoreCompositionModel, PoEScoreConfig
from .sampler import PoEScoreSampler

__all__ = [
    "PoEScoreCompositionModel",
    "PoEScoreConfig",
    "PoEScoreSampler",
    "load_sedd_from_hub",
]
