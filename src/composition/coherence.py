"""Coherence filter for MDLM-OWT samples — used for rejection sampling.

The 110M MDLM-OWT backbone, while serviceable, frequently slides into one of
three degenerate patterns past ~30 tokens, which all defeat naive distinct-2:

1. *Phrase repetition.* "Spartan Stadium Stadium Spartan Stadium..." — caught
   by distinct-2 < 0.30 (most bigrams are duplicates).
2. *Single-token flood.* "thick, thick, thick, thick..." — distinct-2 stays
   high if the surrounding context varies, but a single token saturates the
   sequence. Caught by max-token-frequency > 0.25.
3. *Digit/punctuation flood.* "0, 1, 2, 3, 4, 5..." — every token is unique
   so distinct-2 = 1.0, but it's noise. Caught by alphabetic-char-ratio < 0.55.

Combining the three rejects ~53% of empty-prompt MDLM samples on this
backbone; what remains is qualitatively usable for measuring relative
proxy shifts between PoE configurations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class CoherenceConfig:
    distinct_2_min: float = 0.30
    alpha_min: float = 0.55
    top_token_max: float = 0.25
    min_tokens: int = 8


def is_coherent(text: str, cfg: CoherenceConfig | None = None) -> bool:
    """Return True if the sample looks like real text rather than degenerate noise."""
    cfg = cfg or CoherenceConfig()
    tokens = text.split()
    if len(tokens) < cfg.min_tokens:
        return False

    bigrams = list(zip(tokens[:-1], tokens[1:], strict=True))
    if bigrams and len(set(bigrams)) / len(bigrams) < cfg.distinct_2_min:
        return False

    alpha = sum(c.isalpha() for c in text)
    if alpha / max(1, len(text)) < cfg.alpha_min:
        return False

    counter = Counter(tokens)
    if counter.most_common(1)[0][1] / len(tokens) > cfg.top_token_max:
        return False

    return True


def sample_with_rejection(
    sample_one_attempt,
    decode,
    prompt_tokens: list[list[int]],
    *,
    seed: int | None,
    max_attempts: int,
    coh_cfg: CoherenceConfig | None = None,
    label: str = "",
) -> tuple[list[str], dict]:
    """Rejection-sampling loop over an arbitrary token-level sampler.

    ``sample_one_attempt(prompts: list[list[int]]) -> list[list[int]]`` is
    called once per attempt with a *shifted* RNG seed so that previously
    rejected slots get a fresh draw. Slots that never converge after
    ``max_attempts`` keep their last draw (logged as ``forced_fallback``).

    Parameters
    ----------
    sample_one_attempt : callable
        Closure that runs the underlying sampler on a list of prompts and
        returns aligned token-id lists. Must respect torch's global RNG
        state — we call ``torch.manual_seed`` before invoking it.
    decode : callable
        ``list[int] -> str`` — usually ``tokenizer.decode(..., skip_special_tokens=True)``.
    """
    import torch

    target_n = len(prompt_tokens)
    accepted_ids: list[list[int] | None] = [None] * target_n
    fallback_ids: list[list[int]] = [[] for _ in range(target_n)]
    rejection_count = 0

    for attempt in range(max_attempts):
        todo = [i for i, x in enumerate(accepted_ids) if x is None]
        if not todo:
            break
        if seed is not None:
            torch.manual_seed(seed + attempt * 10_007)
        candidates = sample_one_attempt([prompt_tokens[i] for i in todo])
        for idx, cand in zip(todo, candidates, strict=True):
            text = decode(cand)
            if is_coherent(text, coh_cfg):
                accepted_ids[idx] = cand
            else:
                fallback_ids[idx] = cand
                rejection_count += 1

    forced = 0
    for i, x in enumerate(accepted_ids):
        if x is None:
            accepted_ids[i] = fallback_ids[i]
            forced += 1

    stats = {
        "target_n": target_n,
        "rejection_count": rejection_count,
        "forced_fallback": forced,
        "label": label,
    }
    if rejection_count or forced:
        suffix = f" [{label}]" if label else ""
        print(
            f"  rejection sampling{suffix}: {rejection_count} rejected, "
            f"{forced} forced-fallback (target {target_n})"
        )

    return [decode(ids) for ids in accepted_ids], stats
