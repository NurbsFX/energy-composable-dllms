# Paper 2 §10 — solo λ=1 marginals diagnostic

Per-expert axis-recovery rate (top-quartile of baseline). Solo means lambda=1 on the target axis with no other adapter active.

**Important caveat**: thresholds are calibrated against the *baseline* distribution within each paradigm. SEDD's baseline is unconditional, MDLM's is prompted — distributions differ, so absolute marginals are not directly comparable across paradigms. They are comparable *within* each column.

| Expert | SEDD-small (Paper 2) | MDLM-OWT 110M (Paper 1) | MDLM Qwen3 596M (Paper 1) |
|---|---:|---:|---:|
| long | 0.860 | — | — |
| formal ⚠️ | 0.220 | 0.300 | 0.535 |
| positive | 0.600 | 0.300 | 0.393 |
| positive2 | 0.595 | — | 0.284 |
| concrete | 0.625 | 0.315 | 0.288 |
| sports | 0.820 | 0.315 | 0.210 |

**Diagnostic**: the only SEDD expert below the 0.30 axis-recovery floor is `formal` (0.22). All other SEDD experts recover their axis at ≥ 0.60 — actually *higher* than the corresponding MDLM Paper-1 values, though the calibration caveat above means this comparison is informative only as a within-column ordering.

Cross-referencing PoE-2 results: the four worst-performing pairs all contain `formal` (formal × sports = 0.11, formal × concrete = 0.40, formal × positive2 = 0.61, formal × positive = 0.80). The single best non-trivial PoE-2 (positive × sports = 1.19, super-additive) and the lone super-additive triplet candidate (positive2 × concrete × sports = 0.84, the highest of the three) both **exclude** `formal`.

**Implication for the negative finding**: the formal-weakness story explains a substantial part of the H1/H2 falsification but not all of it. The lexical triplet (no formal) is still sub-additive at 0.84 vs MDLM Qwen3's 3.23 — that gap survives any formal-only fix. The μ-sweep inversion was measured on the formal-heavy triplet, so its interpretation is muddled: the inversion *could* be driven partly by formal weakness (in a relaxed-μ regime, summing log-scores of three experts where one is ~noise produces amplified noise → 0). Disentangling this requires either (a) a longer-training rerun on formal specifically, or (b) a μ-sweep on a non-formal triplet.
