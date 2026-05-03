# Paper 2 §10.8/(c) — semantic-distance correlation with PoE-2 ratio

Distance is computed from the OWT proxy correlation matrix `C` (Paper 1, artifacts/gram_matrix.json), 5000-document sample:

    distance_abs(a, b) = 1 - |C[proxy(a), proxy(b)]|

Small distance = correlated axes (same nature). Large distance = uncorrelated axes (heterogeneous).

| pair | proxies | C | distance_abs | PoE-2 ratio |
|---|---|---:|---:|---:|
| positive × positive2 | sent × sent2 | +0.592 | 0.408 | 1.681 |
| long × concrete | len × conc | -0.307 | 0.693 | 0.567 |
| long × formal | len × form | +0.301 | 0.699 | 0.661 |
| formal × concrete | form × conc | -0.199 | 0.801 | 0.400 |
| formal × positive2 | form × sent2 | -0.171 | 0.829 | 0.611 |
| positive × sports | sent × topic | +0.142 | 0.858 | 1.189 |
| long × positive2 | len × sent2 | -0.141 | 0.859 | 0.713 |
| long × sports | len × topic | -0.139 | 0.861 | 0.759 |
| formal × sports | form × topic | -0.108 | 0.892 | 0.111 |
| positive2 × sports | sent2 × topic | +0.061 | 0.939 | 1.189 |
| positive2 × concrete | sent2 × conc | +0.057 | 0.943 | 1.049 |
| concrete × sports | conc × topic | +0.050 | 0.950 | 0.937 |
| long × positive | len × sent | -0.026 | 0.974 | 0.533 |
| positive × concrete | sent × conc | -0.004 | 0.996 | 0.760 |
| formal × positive | form × sent | -0.002 | 0.998 | 0.795 |

## Pearson correlations across n=15 pairs

* **`ratio` vs `C` (signed)**: r = +0.740 using d_signed = 1−C, equivalently r(ratio, C) = +0.740, p ≈ 0.0001
* `ratio` vs `distance_abs` (1 - |C|): r = -0.374, p ≈ 0.1462

**Strong positive correlation between PoE-2 ratio and proxy correlation C** (r = +0.740, p ≈ 0.0001, n=15). Pairs whose proxy energies are positively correlated on OWT compose super-additively under SEDD; pairs whose proxies are negatively correlated compose sub-additively. The §10.3 selectivity gradient is now **quantified**: ratio is approximately a linear function of OWT proxy correlation. The §10.7 mechanistic hypothesis (cross-position-coherence sensitivity of score-domain composition) gains quantitative support: positive C means the proxies pick out overlapping OWT segments — their experts shift logits in compatible directions. Negative C means the proxies pick out anti-aligned segments — composition amplifies destructive interference. The unsigned distance |C| is a noisier predictor (r = -0.374) because it loses the directionality of the interference.
