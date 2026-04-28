"""Gibbs-style MCMC refinement of PoE samples.

Du Yan et al. 2023 ("Reduce Reuse Recycle") show that the per-step
composition error in compositional diffusion models can be reduced by
running additional MCMC steps that target the true PoE-composed
distribution. They do this with annealed Langevin in continuous-state
diffusion. We adapt the idea to MDLM (discrete-state) by running Gibbs
sweeps over the token positions, with the sampling distribution at each
position given by the PoE-composed logits *evaluated at that position*
while the others are held fixed.

This is an approximation: a true joint MCMC over the full sequence would
require iterating over all positions in a coordinated way (block Gibbs)
or Metropolis-Hastings with a full-sequence likelihood (intractable for
MDLM without an ELBO estimator at each step). The single-position Gibbs
sweep below is the cheapest non-trivial correction we can run on top of
naïve PoE samples and still target the same stationary distribution.

Algorithm per refinement iteration:

  for each sample s in batch:
      pick a random subset of positions (size ``positions_per_iter``)
      mask those positions in s
      forward s through PoECompositionModel → logits at masked positions
      sample new tokens from softmax(logits) and substitute back
"""

from __future__ import annotations

import random


def gibbs_refine(
    samples_ids: list[list[int]],
    poe_model,  # PoECompositionModel-compatible callable
    mask_id: int,
    *,
    n_iters: int = 5,
    positions_per_iter: int = 5,
    batch_size: int = 32,
    seed: int = 0,
) -> tuple[list[list[int]], dict]:
    """Run Gibbs refinement on a batch of PoE-naïve samples.

    Returns the refined samples plus a small stats dict.
    """
    import torch

    rng = random.Random(seed)
    device = next(poe_model.parameters()).device
    refined: list[list[int]] = [list(s) for s in samples_ids]
    seq_lens = [len(s) for s in refined]
    total_changes = 0
    total_attempts = 0

    for _ in range(n_iters):
        for start in range(0, len(refined), batch_size):
            chunk_idx = list(range(start, min(start + batch_size, len(refined))))
            chunk = [refined[i] for i in chunk_idx]
            chunk_lens = [seq_lens[i] for i in chunk_idx]
            max_len = max(chunk_lens)

            # Pad to a common length with mask_id (the trailing pad won't be
            # touched because we only sample positions within each sample's
            # original length).
            input_ids = torch.full((len(chunk), max_len), mask_id, dtype=torch.long, device=device)
            for i, s in enumerate(chunk):
                input_ids[i, : chunk_lens[i]] = torch.tensor(s, dtype=torch.long, device=device)

            # Pick positions to mask per sample
            positions: list[list[int]] = []
            for li in chunk_lens:
                k = min(positions_per_iter, li)
                positions.append(rng.sample(range(li), k))

            # Apply mask
            for i, plist in enumerate(positions):
                for p in plist:
                    input_ids[i, p] = mask_id

            # Forward through PoE
            with torch.no_grad():
                # PoECompositionModel itself increments its internal step
                # counter; reset before this MCMC sweep so any active schedule
                # treats Gibbs as a single trailing "high-progress" step.
                if hasattr(poe_model, "reset_step_count"):
                    poe_model.reset_step_count()
                # Force schedule progress to 1.0 for refinement (treat MCMC as
                # the "final clean phase"). We do this by faking total_steps=1.
                if hasattr(poe_model, "total_steps"):
                    saved_total = poe_model.total_steps
                    poe_model.total_steps = 1
                    try:
                        out = poe_model(input_ids=input_ids).logits  # [b, L, V]
                    finally:
                        poe_model.total_steps = saved_total
                else:
                    out = poe_model(input_ids=input_ids).logits

                # Sample new tokens at the masked positions
                for i, plist in enumerate(positions):
                    for p in plist:
                        probs = torch.softmax(out[i, p].float(), dim=-1)
                        new_tok = int(torch.multinomial(probs, 1).item())
                        old_tok = refined[chunk_idx[i]][p]
                        refined[chunk_idx[i]][p] = new_tok
                        total_attempts += 1
                        if new_tok != old_tok:
                            total_changes += 1

    stats = {
        "n_iters": n_iters,
        "positions_per_iter": positions_per_iter,
        "n_samples": len(samples_ids),
        "total_attempts": total_attempts,
        "total_changes": total_changes,
        "change_rate": total_changes / max(1, total_attempts),
    }
    return refined, stats
