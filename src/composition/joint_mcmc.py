"""Joint MCMC corrector for PoE composition à la Du Yan et al. 2023.

Du Yan et al. ("Reduce Reuse Recycle", NeurIPS 2023) showed that the
per-step composition error in compositional diffusion models can be
reduced by running MCMC steps that target the *true* PoE-composed
distribution. They use annealed Langevin in continuous-state diffusion.

This module adapts the idea to **discrete-state MDLM**. Two strategies
are implemented and can be combined:

1. ``noise_then_denoise`` (re-injection refinement). Take a naïve PoE
   sample, mask K random positions (re-noise), then re-run a partial
   PoE denoising trajectory to fill them back in. The intuition: each
   re-injection lets the model "fix" positions where the cumulative
   per-step approximation error pushed it away from the true joint.
   This corresponds to a **block Gibbs sweep** under the PoE
   conditional distribution at the chosen masking level.

2. ``mh_token_swap`` (Metropolis-Hastings token swap). For each token
   position, propose a substitution from the PoE-composed softmax,
   accept with the MH ratio
       α = min(1, p_PoE(seq_new) / p_PoE(seq_old))
   where the sequence-level p_PoE is approximated via the per-step
   ELBO estimator from `src.eval.poe_formula_check`. This is the
   discrete-state analogue of the Langevin step in Du Yan et al. 2023.

The first strategy is much cheaper (no ELBO call per swap) and tends
to be more stable on single-position degeneracy. The second is
theoretically more correct but slower; we expose both so the runner
can pick.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MCMCRefineConfig:
    n_iters: int = 5
    """Number of refinement iterations (block-Gibbs sweeps or MH passes)."""

    mask_fraction: float = 0.25
    """For ``noise_then_denoise``: fraction of positions to re-mask each iter."""

    partial_denoise_steps: int = 64
    """For ``noise_then_denoise``: number of denoising sub-steps used to
    fill the re-masked positions back. Smaller is faster, larger gives
    a better local refinement."""

    mh_proposals_per_iter: int = 8
    """For ``mh_token_swap``: number of MH proposals per iteration."""

    mh_num_t_samples: int = 8
    """For ``mh_token_swap``: paired-MC ELBO sample count for the
    sequence-level acceptance ratio. Smaller is faster but noisier."""

    seed: int = 0


def noise_then_denoise(
    sample_ids: list[list[int]],
    poe_model,
    tokenizer,
    scheduler,
    *,
    cfg: MCMCRefineConfig,
    max_new_tokens: int,
):
    """Block-Gibbs refinement via re-noising then re-denoising.

    ``poe_model`` should be a ``PoECompositionModel``-compatible callable
    (input_ids → MaskedLMOutput with PoE-composed logits).

    For each iteration:
      1. Pick ``mask_fraction × seq_len`` random positions per sample.
      2. Replace those positions with the mask token.
      3. Run ``partial_denoise_steps`` of MDLM-style denoising on the
         partially-masked input, using the PoE-composed logits to fill
         the masked positions.

    The partial denoising re-uses MDLM's masking schedule: at iteration
    ``s``, we've already unmasked ``s/partial_denoise_steps`` of the
    re-masked positions (in expectation).

    Returns the refined sample_ids and a stats dict.
    """
    import random

    import torch

    rng = random.Random(cfg.seed)
    device = next(poe_model.parameters()).device
    refined: list[list[int]] = [list(s) for s in sample_ids]
    seq_lens = [len(s) for s in refined]
    mask_id = tokenizer.mask_token_id

    n_changed_total = 0
    n_attempts_total = 0

    for _it in range(cfg.n_iters):
        for idx, ids in enumerate(refined):
            seq_len = seq_lens[idx]
            n_to_mask = max(1, int(cfg.mask_fraction * seq_len))
            mask_positions = sorted(rng.sample(range(seq_len), n_to_mask))

            # 1. Re-noise: replace selected positions with mask
            input_ids = list(ids)
            for p in mask_positions:
                input_ids[p] = mask_id

            # 2. Partial denoising trajectory
            # We mimic MDLM by progressively un-masking through K sub-steps.
            # At each sub-step, sample new tokens for the still-masked positions
            # from the PoE-composed softmax.
            still_masked = set(mask_positions)
            t_grid = [
                1.0 - (step + 1) / cfg.partial_denoise_steps
                for step in range(cfg.partial_denoise_steps)
            ]
            # ratio of positions to unmask at each sub-step
            for _sub_step, t in enumerate(t_grid):
                if not still_masked:
                    break
                ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    if hasattr(poe_model, "reset_step_count"):
                        poe_model.reset_step_count()
                    out = poe_model(input_ids=ids_t)
                    logits = out.logits if hasattr(out, "logits") else out
                # How many of the still-masked to unmask at this sub-step?
                # We unmask at rate proportional to (1 - t) - so by the end
                # (t→0) all positions are unmasked.
                n_remaining = len(still_masked)
                # Expected fraction to unmask now
                target_keep = max(0, int(n_remaining * t))
                n_unmask_now = n_remaining - target_keep
                if n_unmask_now <= 0:
                    continue
                positions_to_unmask = rng.sample(list(still_masked), n_unmask_now)
                for p in positions_to_unmask:
                    probs = torch.softmax(logits[0, p].float(), dim=-1)
                    new_tok = int(torch.multinomial(probs, 1).item())
                    input_ids[p] = new_tok
                    still_masked.discard(p)

            # In case any positions remain masked at the end (rounding), unmask greedily
            if still_masked:
                ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    if hasattr(poe_model, "reset_step_count"):
                        poe_model.reset_step_count()
                    out = poe_model(input_ids=ids_t)
                    logits = out.logits if hasattr(out, "logits") else out
                for p in still_masked:
                    probs = torch.softmax(logits[0, p].float(), dim=-1)
                    new_tok = int(torch.multinomial(probs, 1).item())
                    input_ids[p] = new_tok

            # Count changes wrt the input ids
            for p in mask_positions:
                n_attempts_total += 1
                if input_ids[p] != ids[p]:
                    n_changed_total += 1
            refined[idx] = input_ids

    return refined, {
        "method": "noise_then_denoise",
        "n_iters": cfg.n_iters,
        "mask_fraction": cfg.mask_fraction,
        "partial_denoise_steps": cfg.partial_denoise_steps,
        "n_samples": len(sample_ids),
        "n_attempts": n_attempts_total,
        "n_changes": n_changed_total,
        "change_rate": n_changed_total / max(1, n_attempts_total),
    }


def mh_token_swap(
    sample_ids: list[list[int]],
    poe_model,
    tokenizer,
    scheduler,
    *,
    cfg: MCMCRefineConfig,
    base_model_with_adapters,
    expert_a: str,
    expert_b: str,
    expert_c: str | None = None,
):
    """Metropolis-Hastings refinement using sequence-level ELBO ratios.

    For each iteration, we propose ``mh_proposals_per_iter`` token swaps
    and accept each with the MH ratio computed via the paired-MC
    sequence-level ELBO estimator under the PoE-composed distribution.

    This is more rigorous than ``noise_then_denoise`` but considerably
    slower (each MH step requires one ELBO call ≈ ``mh_num_t_samples``
    forward passes per expert).
    """
    import random

    import torch

    from src.eval.poe_formula_check import estimate_log_ratio_elbo_poe

    rng = random.Random(cfg.seed)
    device = next(poe_model.parameters()).device
    refined: list[list[int]] = [list(s) for s in sample_ids]
    seq_lens = [len(s) for s in refined]

    lambdas = {expert_a: 1.0, expert_b: 1.0}
    if expert_c is not None:
        lambdas[expert_c] = 1.0

    n_accept_total = 0
    n_attempts_total = 0

    for it in range(cfg.n_iters):
        for idx, ids in enumerate(refined):
            seq_len = seq_lens[idx]
            for _ in range(cfg.mh_proposals_per_iter):
                pos = rng.randrange(seq_len)
                old_tok = ids[pos]
                # Get PoE softmax at this position by forward
                ids_t = torch.tensor([ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    if hasattr(poe_model, "reset_step_count"):
                        poe_model.reset_step_count()
                    out = poe_model(input_ids=ids_t)
                    logits = out.logits if hasattr(out, "logits") else out
                probs = torch.softmax(logits[0, pos].float(), dim=-1)
                proposal = int(torch.multinomial(probs, 1).item())
                if proposal == old_tok:
                    continue
                # Sequence-level ELBO ratio: log p_PoE(new) - log p_PoE(old)
                ids_new = list(ids)
                ids_new[pos] = proposal
                # Use estimate_log_ratio_elbo_poe with x=old, y=new
                log_ratio = estimate_log_ratio_elbo_poe(
                    base_model_with_adapters,
                    tokenizer,
                    ids,
                    ids_new,
                    lambdas=lambdas,
                    num_t_samples=cfg.mh_num_t_samples,
                    seed=cfg.seed + it * 1000 + pos,
                )
                accept = (log_ratio >= 0) or (rng.random() < pow(2.71828, log_ratio))
                n_attempts_total += 1
                if accept:
                    ids[pos] = proposal
                    refined[idx] = ids
                    n_accept_total += 1

    return refined, {
        "method": "mh_token_swap",
        "n_iters": cfg.n_iters,
        "mh_proposals_per_iter": cfg.mh_proposals_per_iter,
        "n_attempts": n_attempts_total,
        "n_accepted": n_accept_total,
        "accept_rate": n_accept_total / max(1, n_attempts_total),
    }
