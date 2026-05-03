# `src/sedd_composition/` — score-based composition stack

This package is the SEDD-side mirror of `src/composition/` (which holds
the MDLM PoE-logits stack). It is **isolated** by design: nothing here
imports from `src/composition/`, and vice-versa. The shared surface is
`src/eval/` (proxy energies, joint satisfaction) — those operate on
text and are model-agnostic.

## Why a parallel stack

See `PAPER2_SEDD.md` for the science. In short: MDLM's PoE composition
factorizes per position (loses cross-position correlations); SEDD's
score-based composition is exact at the sequence level. The two stacks
let us A/B-test that hypothesis on otherwise-comparable backbones.

## Files

| File | Role |
|---|---|
| `load.py` | `load_sedd_from_hub(repo_id, device)` — fetch `(model, graph, noise)`. Tokenizer helper. |
| `poe_score.py` | `PoEScoreCompositionModel` — composes log-scores from the base + LoRA experts. Decoupled-μ supported (Paper-1's μ-fix transports verbatim). |
| `sampler.py` | `PoEScoreSampler` — wraps Lou's `pc_sampler` with the composition model. `assert_lambda_zero_is_base()` for the sanity check. |
| `train_lora.py` | _(TODO)_ Score-entropy LoRA training. |

## Quick reference — composition formula

```
log s_PoE = log s_base + Σ_k λ_k (log s_k − log s_base)
          = (1 − Σ λ_k) · log s_base + Σ λ_k · log s_k
```

with the decoupled variant (Paper 1 §13):

```
log s_custom = mu_base · log s_base + Σ λ_k · log s_k
```

Same algebra as MDLM logits — different *interpretation* (transition
log-ratios) and different *sampler* (τ-leaping).

## How to use (typical session)

```python
from src.sedd_composition import (
    load_sedd_from_hub,
    PoEScoreSampler,
    PoEScoreConfig,
)
from src.sedd_composition.load import get_gpt2_tokenizer_for_sedd
from peft import PeftModel

# 1. Base backbone + LoRA experts
model, graph, noise = load_sedd_from_hub("louaaron/sedd-small")
model = PeftModel.from_pretrained(model, "<path>/formal", adapter_name="formal")
model.load_adapter("<path>/positive", adapter_name="positive")
tokenizer = get_gpt2_tokenizer_for_sedd()

# 2. Compose
cfg = PoEScoreConfig(num_steps=128, sample_batch_size=8)
sampler = PoEScoreSampler(model, graph, noise, tokenizer, cfg=cfg)
texts = sampler.sample(
    num_samples=8,
    seq_len=64,
    lambdas={"formal": 1.0, "positive": 1.0},
)

# 3. Sanity check (cheap, run once per session)
sampler.assert_lambda_zero_is_base(num_samples=4, seq_len=32)
```

## Status

- ✅ Architecture in place, vendored SEDD at `external/sedd/`
- ✅ Composition + sampler wrappers
- ☐ Training (LoRA + score-entropy loss)
- ☐ End-to-end smoke test (requires a GPU pod for non-trivial samples)
- ☐ Phase-4 / Phase-6 / Phase-11 analogs

## Three risks logged in `PAPER2_SEDD.md` §4

1. Fused QKV in DDiT (PEFT target_modules need explicit naming).
2. Vocab=50258 with absorbing token at index 50257 (PoE sum sanitized in `_sanitize`).
3. Score representation = log-ratios in `staggered_score`'s parameterization
   (verified via `model/utils.py:get_score_fn`, which `.exp()`s for sampling).
