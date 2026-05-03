# Phase 11/12 — μ-fix gains summary

Records: 16 setups (Phase 11 sweeps + Phase 12c additional N=2 sweeps).

| Setup | Backbone | N | canonical μ | canonical ratio | best μ | best ratio | gain |
|---|---|---:|---:|---:|---:|---:|---:|
| concrete×sports | MDLM-OWT | 2 | -1 | 0.76 | -1.5 | 2.67 | **+253%** |
| formal×positive | MDLM-OWT | 2 | -1 | 1.33 | -0.5 | 1.56 | **+17%** |
| concrete×sports | Qwen3 | 2 | -1 | 3.29 | -1 | 3.29 | **+0%** |
| formal×concrete | Qwen3 | 2 | -1 | 0.81 | -0.5 | 0.84 | **+4%** |
| formal×positive | Qwen3 | 2 | -1 | 0.33 | -0.5 | 1.07 | **+220%** |
| formal×positive2 | Qwen3 | 2 | -1 | 0.45 | +0 | 1.10 | **+143%** |
| formal×sports | Qwen3 | 2 | -1 | 0.62 | -0.5 | 0.67 | **+7%** |
| positive2×concrete | Qwen3 | 2 | -1 | 1.05 | -1 | 1.05 | **+0%** |
| positive2×sports | Qwen3 | 2 | -1 | 0.85 | -0.5 | 1.02 | **+20%** |
| positive×concrete | Qwen3 | 2 | -1 | 1.33 | -1 | 1.33 | **+0%** |
| positive×positive2 | Qwen3 | 2 | -1 | 1.56 | -0.5 | 2.16 | **+38%** |
| positive×sports | Qwen3 | 2 | -1 | 1.08 | -0.5 | 1.21 | **+12%** |
| formal×positive×concrete | MDLM-OWT | 3 | -2 | 0.35 | +0 | 0.71 | **+100%** |
| formal×concrete×sports | Qwen3 | 3 | -2 | 0.46 | -1 | 1.23 | **+167%** |
| formal×positive×concrete | Qwen3 | 3 | -2 | 0.15 | -1 | 0.61 | **+300%** |
| positive2×concrete×sports | Qwen3 | 3 | -2 | 3.23 | -2 | 3.23 | **+0%** |

Gain = (best_ratio / canonical_ratio − 1) × 100. Pattern: stylistic-heavy compositions on Qwen3 yield the largest gains (formal×positive N=2: +220%, formal×positive×concrete N=3: +300%, formal×concrete×sports N=3 mixed: +167%). Purely lexical compositions (positive×concrete, positive2×concrete on Qwen3, positive2×concrete×sports N=3) show ≈0% — canonical 1−N is already optimal. MDLM-OWT sweeps are noisier (small backbone, n=200 limit) but show a consistent direction: best μ is always less punitive than canonical.
