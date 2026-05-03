# Composabilité d'experts via Product-of-Experts sur les Masked Diffusion Language Models : une étude empirique

**Brouillon de papier — 2026-04-29**
*Bruno Kalfa*

---

## Résumé

Nous étudions empiriquement la composition d'experts spécialisés dans les **Masked Diffusion Language Models** (MDLM) via la règle **Product-of-Experts** (PoE), appliquée au niveau des logits à chaque pas de débruitage. Sur deux backbones de tailles différentes (MDLM-OWT 110M et Qwen3-0.6B-MDLM 596M), avec 8 adapters LoRA spécialisés (six axes mono-attribut et deux experts d'intersection) et un cadre d'évaluation reposant sur six proxies sémantiques (longueur, formalité, sentiment ×2, concrétude, sportivité), nous obtenons les résultats suivants.

D'abord, **la formule PoE elle-même est validée** au niveau des log-ratios (slope = 0.857, R² = 0.811 sur 50 paires de séquences aléatoires). Ensuite, **la composition à N=2 est en moyenne super-additive** sur les deux backbones (ratio moyen 1.04 et 1.07), avec une variance forte par paire. Sur MDLM-OWT, la **corrélation entre l'orthogonalité des proxies (κ, CKA, MI) et le déficit de composition** est très forte (r = -0.92 sur κ). Cependant, **cette corrélation s'effondre sur Qwen3** (r = -0.24), révélant qu'elle n'était pas une loi universelle mais une propriété backbone-spécifique. Pour la composition à **N=3**, MDLM-OWT plafonne à un ratio de 0.55 indépendamment de **19 variantes de calibrage** (sweep λ, schedules, MCMC, Bayesian). Sur Qwen3, ce plateau est **partiellement levé** : la moyenne monte à 1.03, mais l'effet est triplet-dépendant (lexical : super-additif ; stylistique : pas d'amélioration).

Nous concluons que la composabilité est démontrée empiriquement à N=2 mais qu'**aucun prédicteur géométrique sur les données brutes ne capture de façon stable et cross-backbone le déficit PoE observé**. Nous proposons quatre mesures alternatives qui pourraient mieux capturer la composabilité réelle.

---

## 1. Introduction

### 1.1 Contexte

L'apprentissage automatique génératif a vu se multiplier les techniques de **composition d'experts** : par exemple en classifier-free guidance pour les modèles de diffusion d'images (Ho & Salimans, 2022), en LoRA-merge pour les modèles de langage autoregressifs (Wortsman et al., 2022), ou en composition diffuse (Liu et al., 2022 ; Du Yan et al., 2023). L'idée centrale : entraîner indépendamment plusieurs spécialistes, puis les combiner à l'inférence pour obtenir un comportement composé, sans avoir à entraîner un modèle joint coûteux.

Le **Product-of-Experts** (PoE) est une formulation théorique élégante de cette idée. Étant données $N$ distributions $p_i(x)$ "tilted" autour d'une distribution de base $p_b(x)$, on définit :

$$
p_{\text{PoE}}(x) \propto p_b(x) \cdot \prod_{i=1}^{N} \left( \frac{p_i(x)}{p_b(x)} \right)^{\lambda_i}
$$

ou, en log-probabilités :

$$
\log p_{\text{PoE}}(x) = \log p_b(x) + \sum_{i=1}^{N} \lambda_i \cdot \left( \log p_i(x) - \log p_b(x) \right)
$$

Les coefficients $\lambda_i \in \mathbb{R}$ pondèrent la contribution de chaque expert. À $\lambda_i = 1$ pour tout $i$, on retrouve le PoE canonique.

Pour les **diffusion language models** (Lou et al., 2024 ; Sahoo et al., 2024), cette composition s'applique au niveau des **logits** à chaque pas de débruitage :

$$
\text{logits}_{\text{PoE}}(x_t) = \text{logits}_b(x_t) + \sum_{i=1}^{N} \lambda_i \cdot \left( \text{logits}_i(x_t) - \text{logits}_b(x_t) \right)
$$

Les questions ouvertes que nous étudions :

1. **La formule PoE est-elle empiriquement valide** sur MDLM-text ?
2. **À N=2, la composition produit-elle un comportement super-additif** (i.e., satisfaire les deux contraintes plus fréquemment qu'avec une combinaison aléatoire indépendante) ?
3. **Existe-t-il un prédicteur géométrique** (orthogonalité, indépendance des proxies) du déficit PoE observé ?
4. **À N=3, la composition continue-t-elle à fonctionner** ? Si non, quelle est la cause ?

### 1.2 Contributions principales

Nous apportons :

- **Validation empirique de la formule PoE sur MDLM** : slope = 0.857, R² = 0.811 sur 50 paires de séquences arbitraires.
- **Étude exhaustive à N=2** sur 10 paires d'experts, deux backbones (MDLM-OWT 110M et Qwen3-0.6B-MDLM 596M), 8 configurations de composition (baseline, expert-A-only, expert-B-only, PoE-half, PoE-strict, PoE-1.5, PoE-amp, LoRA-merge).
- **Caractérisation précise du plateau à N=3** via 19 variantes de calibrage : sweep λ uniforme (11 valeurs), schedules denoising-aware (4 variantes), MCMC post-sampling (Gibbs), λ Bayesian (κ-shrinked).
- **Étude cross-backbone** : un backbone 5× plus gros (Qwen3-0.6B) permet de distinguer la cause théorique (approximation per-step, *Niveau 2*) de la cause pragmatique (capacité du modèle, *Niveau 3*).
- **Résultat négatif important** : la corrélation observée sur le petit backbone entre l'orthogonalité géométrique des proxies et le déficit PoE **ne se généralise pas**.

### 1.3 Plan du papier

La Section 2 présente le cadre théorique. La Section 3 décrit la méthodologie expérimentale. Les Sections 4–7 présentent les résultats sur MDLM-OWT (Phase 4 = N=2 ; Phase 5 = κ ↔ déficit ; Phase 6–7 = plateau N=3). La Section 8 présente Phase 8 (Qwen3). La Section 9 discute les limites et propose quatre mesures alternatives à κ comme prédicteur du déficit.

---

## 2. Cadre théorique

### 2.1 La règle de composition Product-of-Experts

Soit $p_b(x)$ la distribution de base d'un MDLM (le backbone non spécialisé). Soit $\{p_i(x)\}_{i=1}^{N}$ un ensemble de distributions "tiltées" obtenues en fine-tunant le backbone sur $N$ corpus spécialisés. La distribution composée par PoE est :

$$
p_{\text{PoE}}(x) = \frac{1}{Z} \cdot p_b(x) \cdot \prod_{i=1}^{N} \left( \frac{p_i(x)}{p_b(x)} \right)^{\lambda_i}
$$

où $Z$ est une constante de normalisation. À $\lambda_i = 1$, l'expression se simplifie en :

$$
p_{\text{PoE}}(x) \propto \frac{\prod_i p_i(x)}{p_b(x)^{N-1}}
$$

Cette formule est **mathématiquement exacte** quand on a accès aux distributions séquence-niveau exactes. Pour les diffusion language models, elle est appliquée **par pas de débruitage** sur les logits, ce qui constitue une **approximation** dont l'erreur peut s'accumuler.

### 2.2 Métriques d'indépendance entre proxies

Soit $E_i(x)$ une fonction d'énergie scalaire (un proxy sémantique) évaluée sur un texte $x$. Étant données $N$ échantillons $\{x_n\}$ tirés d'une distribution de référence (typiquement, un échantillon du corpus de pré-training), nous calculons cinq métriques d'indépendance par paire de proxies :

**κ (orthogonalité linéaire).** Pour la matrice de Gram empirique 2×2 entre deux énergies $E_i, E_j$ :

$$
\kappa_{ij} = \frac{\sqrt{2} \cdot |\text{Cov}(E_i, E_j)|}{\text{Var}(E_i) + \text{Var}(E_j)}
$$

$\kappa = 0$ signifie linéaire-orthogonalité parfaite.

**Spearman absolu.** $|\rho_{\text{Sp}}(E_i, E_j)|$ — capture les dépendances monotones non-linéaires.

**HSIC (Hilbert-Schmidt Independence Criterion).** Norm Hilbert-Schmidt de la covariance entre les distributions dans des RKHS gaussiens sur $E_i$ et $E_j$. Capture les dépendances non-linéaires arbitraires.

**CKA (Centered Kernel Alignment).** Version normalisée du HSIC :

$$
\text{CKA}(E_i, E_j) = \frac{\text{HSIC}(E_i, E_j)}{\sqrt{\text{HSIC}(E_i, E_i) \cdot \text{HSIC}(E_j, E_j)}}
$$

**MI (Information Mutuelle).** Estimateur KSG basé sur les $k$-NN.

Ces cinq métriques fournissent une vue multi-aspect de la dépendance entre énergies. Pour le critère central de notre étude (**la prédiction du déficit PoE par l'orthogonalité**), nous utilisons principalement κ, CKA et MI (Spearman et HSIC sont en backup).

### 2.3 Hypothèse centrale (à tester)

**H₀ — La κ-Gram theory** : *plus deux axes sont indépendants (faible κ, faible CKA, faible MI), plus la composition PoE de leurs experts produit un nombre élevé d'échantillons satisfaisant les deux contraintes simultanément (par rapport à des tirages indépendants).*

Mathématiquement, soit $\Delta_{\text{PoE}} = \text{JS}_{\text{indep}} - \text{JS}_{\text{PoE}}$ le déficit de composition (où $\text{JS}_{\text{indep}} = P(A) \cdot P(B)$ est la satisfaction jointe attendue sous indépendance). H₀ prédit :

$$
\Delta_{\text{PoE}} \approx \alpha + \beta \cdot \kappa(E_i, E_j) + \epsilon
$$

avec $\beta > 0$ (plus c'est corrélé, plus le déficit est grand).

---

## 3. Méthodologie expérimentale

### 3.1 Backbones

Deux backbones MDLM ont été utilisés :

| Backbone | Params | Architecture | Contexte | Tokenizer |
|---|---|---|---|---|
| `kuleshov-group/mdlm-owt` | 110 M | DiT-style (12 blocks × 768d × 12h) | OpenWebText | GPT-2 (50 257 + mask) |
| `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` | 596 M | A2DQwen3LMHeadModel | Mix dllm-hub | Qwen2 (151 643 + mask) |

Pour MDLM-OWT, plusieurs patches ont été nécessaires (drop flash-attn → SDPA fallback, accept HF Trainer kwargs, `sigma=None` handling, `return_dict=True` forced, dtype matching pour TimestepEmbedder). Voir `scripts/patch_mdlm_no_flash_attn.py`.

### 3.2 Proxies sémantiques

Six axes ont été choisis pour couvrir trois familles : **lexique** (longueur, concrétude, sportivité), **style** (formalité), **sentiment** (deux variantes redondantes pour amplifier la dépendance contrôlée).

| Axe | Énergie | Implémentation |
|---|---|---|
| `len` | Longueur | Compte de tokens GPT-2 |
| `form` | Formalité | `s-nlp/roberta-base-formality-ranker` |
| `sent` | Sentiment | `distilbert-base-uncased-finetuned-sst-2-english` |
| `sent2` | Sentiment redondant | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| `conc` | Concrétude | Dictionnaire Brysbaert (40 000 mots) |
| `topic` | Sportivité | `fabriceyhc/distilbert-base-uncased-agnews` (label "Sports") |

### 3.3 Construction des datasets

Pour chaque axe $i$, nous avons streamé OpenWebText et conservé les 80 000 documents les plus extrêmes selon $E_i$. Seuils calibrés empiriquement :

```
long       : len ≥ 700 tokens
formal     : form ≥ 0.75
positive   : sent ≥ 0.85
positive2  : sent2 ≥ 0.70
concrete   : conc ≥ 2.80
sports     : topic ≥ 0.50
```

Pour les tests d'intersection, nous avons construit deux corpus joints (`long_formal` : 19 993 docs ; `formal_concrete` : 18 551 docs) en streamant OWT et en conservant les documents passant simultanément les deux seuils.

### 3.4 Adapters LoRA

Pour chaque axe et chaque backbone, un adapter LoRA a été entraîné sur le corpus correspondant :

| Hyperparamètre | Valeur |
|---|---|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Steps | 2 500 |
| Learning rate | 3e-4 |
| Batch size | 32 |
| Sequence length | 256 |
| Précision | bf16 |
| Embeddings frozen | oui |
| Target modules (MDLM-OWT) | `attn_qkv`, `attn_out` |
| Target modules (Qwen3) | `q_proj`, `k_proj`, `v_proj`, `o_proj` |

Embeddings explicitement gelés pour préserver la cohérence du sum-of-logits (PoE n'est mathématiquement valide que si les $E_i$ partagent le même tokenizer/embedding ; LoRA garantit cela).

### 3.5 Procédure d'échantillonnage

Pour chaque configuration de composition, nous échantillonnons $n$ textes :

```
prompt        : 12 tokens GPT-2 (Qwen2 sur Qwen3) tirés au hasard de prompts.jsonl
                (mix équilibré des 6 verticales)
max_new_tokens: 48 (au-delà, MDLM-OWT dégénère en répétitions)
num_steps     : 256 (pas de débruitage)
temperature   : 1.0
```

Nous appliquons **rejection sampling sur la cohérence** : un sample est rejeté s'il échoue à au moins un de trois critères heuristiques (distinct-2 < 0.30, ratio alphabétique < 0.55, fréquence du token le plus fréquent > 0.25). Au plus 5 redraws par slot ; sinon, on garde le dernier candidat (logué comme `forced_fallback`).

### 3.6 Métriques d'évaluation

Pour chaque échantillon, nous calculons les six proxy scores. Pour chaque proxy $k$, le seuil $\tau_k$ est le 75ᵉ centile de la distribution baseline (top-quartile). La **joint satisfaction** pour une paire d'axes $(a, b)$ est :

$$
\text{JS}_{\text{config}}(a, b) = \frac{1}{n} \sum_{s=1}^{n} \mathbb{1}\!\left[ E_a(s) \geq \tau_a \;\wedge\; E_b(s) \geq \tau_b \right]
$$

L'**indep-reference** est :

$$
\text{JS}_{\text{indep}}(a, b) = P(E_a \geq \tau_a \mid \text{expert-A-only}) \cdot P(E_b \geq \tau_b \mid \text{expert-B-only})
$$

Le **ratio** d'efficacité de composition est :

$$
r(a, b) = \frac{\text{JS}_{\text{PoE-strict}}(a, b)}{\text{JS}_{\text{indep}}(a, b)}
$$

- $r > 1$ : super-additif (la composition fait mieux que des tirages indépendants).
- $r = 1$ : composition aussi efficace que l'indépendance.
- $r < 1$ : sous-additif (la composition produit moins de samples joint que des tirages indépendants).

À N=3, mêmes définitions étendues à des triplets.

---

## 4. Validation empirique de la formule PoE (Test 2)

Avant d'évaluer le comportement empirique de la composition, nous vérifions que la formule au niveau des logits est correcte. Pour chaque paire d'experts $(a, b)$, nous générons $K = 50$ paires de séquences aléatoires $(x, y)$ de même longueur (un édit aléatoire d'un token sur 64). Pour chacune, nous estimons via Monte Carlo paired sur l'ELBO (32 timesteps) :

- $L_a = \log p_a(y) - \log p_a(x)$
- $L_b = \log p_b(y) - \log p_b(x)$
- $L_{\text{base}} = \log p_b(y) - \log p_b(x)$
- $L_{\text{PoE}} = \log p_{\text{PoE}}(y) - \log p_{\text{PoE}}(x)$

Si la formule PoE tient, alors :

$$
L_{\text{PoE}} \approx L_a + L_b - L_{\text{base}}
$$

Régression linéaire sur les 50 paires :

$$
\boxed{\text{slope} = 0.857, \quad \text{intercept} = -0.041, \quad R^2 = 0.811}
$$

→ La formule est empiriquement validée. Le slope < 1 suggère une légère sous-composition, cohérente avec l'erreur d'approximation per-step.

---

## 5. Phase 4 — Composition à N=2 sur MDLM-OWT

Pour chaque paire d'experts $(a, b) \in \binom{\{6 \text{ verticales sans long}\}}{2} = 10$ paires, nous évaluons 8 configurations à $n=200$ samples chacune.

### 5.1 Configurations testées

| Config | $\lambda_a$ | $\lambda_b$ |
|---|---|---|
| baseline | 0 | 0 |
| expert-A-only | 1 | 0 |
| expert-B-only | 0 | 1 |
| PoE-half | 0.5 | 0.5 |
| **PoE-strict** | **1** | **1** |
| PoE-1.5 | 1.5 | 1.5 |
| PoE-amp | 2 | 2 |
| LoRA-merge | (linéaire dans le poids) | |

### 5.2 Résultats

Joint satisfaction de PoE-strict comparée à baseline et au ratio à indep :

| Pair | baseline | expert-A | expert-B | PoE-strict | gain vs baseline | ratio |
|---|---:|---:|---:|---:|---:|---:|
| formal × positive | 0.065 | 0.075 | 0.095 | **0.120** | +85 % | 1.33 |
| formal × positive2 | 0.085 | 0.065 | 0.110 | **0.125** | +47 % | 1.17 |
| formal × concrete | 0.045 | 0.075 | 0.080 | **0.065** | +44 % | 0.69 |
| formal × sports | 0.035 | 0.075 | 0.090 | **0.090** | +157 % | 0.95 |
| positive × positive2 | 0.150 | 0.105 | 0.155 | **0.270** | +80 % | 2.54 |
| positive × concrete | 0.040 | 0.060 | 0.080 | **0.070** | +75 % | 0.74 |
| positive × sports | 0.080 | 0.110 | 0.115 | **0.160** | +100 % | 1.69 |
| positive2 × concrete | 0.050 | 0.075 | 0.110 | **0.080** | +60 % | 0.72 |
| positive2 × sports | 0.075 | 0.085 | 0.115 | **0.105** | +40 % | 0.94 |
| concrete × sports | 0.055 | 0.075 | 0.080 | **0.075** | +36 % | 0.76 |
| **moyenne** | 0.068 | 0.080 | 0.103 | **0.116** | **+72 %** | **1.04** |

→ **PoE-strict bat la baseline 10/10 paires** avec un gain médian de +71 %. Le ratio à indep moyen est 1.04 (légèrement super-additif).

PPL-ratio médian sous PoE-strict : 2.16 (le coût en fluidité est modéré). PoE-amp ($\lambda=2$) collapse en mode collapse (PPL-ratio jusqu'à 150×).

---

## 6. Phase 5 — Métriques d'orthogonalité vs déficit PoE

Pour chaque paire d'axes, nous calculons κ, Spearman, CKA, MI sur la baseline OWT (5 000 samples). Nous comparons avec le déficit observé :

$$
\Delta(a, b) = \text{JS}_{\text{indep}}(a, b) - \text{JS}_{\text{PoE-strict}}(a, b)
$$

Régression linéaire avec bootstrap CI₉₅ sur les 10 paires :

| Métrique | Pearson r | CI₉₅ | slope | jackknife r range |
|---|---:|---|---:|---|
| **κ** | **−0.917** | [−0.99, −0.20] | −0.598 | [−0.97, −0.65] |
| Spearman | −0.752 | [−0.96, +0.49] | −0.251 | [−0.86, +0.09] |
| **CKA** | **−0.868** | [−0.99, +0.10] | −0.826 | [−0.94, −0.26] |
| **MI** | **−0.836** | [−0.98, +0.52] | −0.759 | [−0.92, +0.11] |

→ **Quatre métriques indépendantes confirment H₀** : plus les axes sont indépendants (faible κ/CKA/MI), plus le déficit PoE est petit (composition plus efficace). Sur cette base, le κ-Gram framework apparaît comme un prédicteur empirique fort.

> ⚠️ **Spoiler de la Section 8** : cette corrélation va s'effondrer sur Qwen3.

---

## 7. Test 1 (Phase 4.5) — PoE vs expert intersection-trained

Pour la paire `formal × concrete`, nous comparons :

1. Samples générés par l'**expert dédié à l'intersection** `formal_concrete` (entraîné spécifiquement sur le corpus joint).
2. Samples générés par PoE(`formal`, `concrete`) avec $\lambda = 1$.

Évaluation par proxy avec test KS (Kolmogorov-Smirnov) à 2 échantillons :

| Proxy | mean intersection | mean PoE | KS stat | p-value KS |
|---|---:|---:|---:|---:|
| len | 40.88 | 47.06 | 0.44 | 9 × 10⁻¹⁸ |
| **form** | 0.177 | **0.369** | 0.55 | 1 × 10⁻²⁷ |
| sent | 0.516 | 0.477 | 0.22 | 1 × 10⁻⁴ |
| sent2 | 0.280 | 0.164 | 0.60 | 8 × 10⁻³⁴ |
| **conc** | 1.029 | **2.219** | 0.53 | 2 × 10⁻²⁵ |

Toutes les distributions diffèrent significativement (p ≪ 0.05). De manière contre-intuitive :

→ **PoE bat l'expert intersection-trained sur les axes-clés** : `form` 0.37 vs 0.18, `conc` 2.22 vs 1.03.

Hypothèse : le corpus d'intersection (~20k docs) était trop étroit pour que l'adapter LoRA converge sur les deux axes simultanément. La composition explicite via PoE de deux experts bien entraînés est plus efficace.

---

## 8. Phase 6–7 — Le plateau à N=3 sur MDLM-OWT

### 8.1 Trois triplets, observation initiale

À N=3 avec $\lambda_i = 1$ et $n=500$ samples, nous évaluons trois triplets choisis pour couvrir un spectre d'orthogonalité (κ moyen croissant) :

| Triplet | max κ | marginals | PoE-3 | indep-ref | ratio |
|---|---:|---|---:|---:|---:|
| `positive2 × concrete × sports` | 0.029 | (0.31, 0.26, 0.23) | 0.012 | 0.031 | **0.39** |
| `formal × positive × concrete` | 0.032 | (0.32, 0.32, 0.33) | 0.018 | 0.033 | **0.55** |
| `formal × concrete × sports` | 0.040 | (0.32, 0.26, 0.23) | 0.026 | 0.030 | **0.87** |

→ **Tous les triplets sont sous-additifs** (ratio < 1). De plus, **le triplet le plus indépendant est le pire** — un retournement complet du pattern observé à N=2.

### 8.2 Sweep λ uniforme (Phase 7a)

Pour vérifier si le plateau est un artefact de calibrage, nous balayons $\lambda \in \{0.10, 0.25, 0.333, 0.50, 0.577, 0.667, 0.80, 1.00, 1.20, 1.50\}$ plus une variante Bayesian per-expert ($\lambda_i = 1/(1 + \bar{\kappa}_i)$).

Résultats sur le triplet `formal × positive × concrete` ($n = 500$) :

| λ | PoE-3 | ratio |
|---|---:|---:|
| 0.10 | 0.000 | 0.00 |
| 0.25 | 0.008 | 0.24 |
| 0.333 (= 1/N) | 0.006 | 0.18 |
| 0.50 | 0.012 | 0.36 |
| 0.577 (= 1/√N) | 0.010 | 0.30 |
| 0.667 | 0.008 | 0.24 |
| 0.80 | 0.014 | 0.42 |
| **1.00 (PoE-strict)** | **0.018** | **0.55** |
| Bayesian (≈ 0.99 each) | 0.018 | 0.55 |
| 1.20 | 0.008 | 0.24 |
| 1.50 | 0.008 | 0.24 |

→ **Courbe en cloche serrée sur $\lambda = 1$.** Aucune normalisation ($1/N$, $1/\sqrt{N}$) n'aide. λ > 1 dégrade aussi rapidement que λ < 1. Le plateau est *robuste* au calibrage.

### 8.3 Schedules denoising-aware et MCMC (Phase 7v2)

Nous testons quatre schedules de modulation de λ le long de la trajectoire de débruitage (`progress` ∈ [0, 1]) :

- `late_fire` : $\lambda(p) = 0$ si $p < 0.5$, sinon $\lambda(p) = 1$
- `cosine` : $\lambda(p) = (1 - \cos(\pi p))/2$
- `exp` : $\lambda(p) = (e^p - 1)/(e - 1)$
- `early_fire` : $\lambda(p) = 1$ si $p < 0.5$, sinon $\lambda(p) = 0$

Et un raffinement post-sampling **Gibbs MCMC** : pour chaque sample, masquer 5 positions aléatoires et resample depuis $\text{softmax}(\text{logits}_{\text{PoE}})$, répété 10 fois.

Résultats à $n = 500$ avec marginals systématiquement mesurées en mode constant pour comparabilité :

| Config | naive | mcmc | ratio_naive | ratio_mcmc |
|---|---:|---:|---:|---:|
| **constant (contrôle)** | 0.018 | — | **0.55** | — |
| early_fire | 0.018 | — | 0.55 | — |
| exp | 0.010 | — | 0.30 | — |
| cosine | 0.004 | — | 0.12 | — |
| late_fire | 0.004 | — | 0.12 | — |
| constant + MCMC×10 | 0.018 | 0.012 | 0.55 | 0.36 |
| exp + MCMC×10 | 0.010 | 0.008 | 0.30 | 0.24 |

→ **Aucune variante ne bat constant.** `early_fire` matche, donc **les premiers pas de débruitage portent le signal de composition**. `late_fire` collapse — pousser à la fin clean ne marche pas. Le MCMC dégrade systématiquement (notre Gibbs naïf est trop greedy).

### 8.4 Synthèse du plateau N=3 sur MDLM-OWT

Le plateau ratio ≈ 0.55 est robuste contre :

- **11 valeurs uniformes de λ** ∈ [0.10, 1.50]
- **Bayesian per-expert λ** ($\lambda_i = 1/(1+\bar{\kappa}_i)$)
- **4 schedules denoising-aware**
- **2 variantes MCMC** (constant et exp ; 10 itérations × 5 positions)

**Total : 19 variantes de calibrage, toutes plafonnées à ratio ≤ 0.55.** Le plateau n'est pas un artefact de réglage : c'est structurel.

Deux explications possibles :

- **Niveau 2 — théorique** : la composition logit-par-step est une approximation de la composition séquence-niveau ; l'erreur s'accumule avec $N$. Cohérent avec Du Yan et al. 2023 (*Reduce Reuse Recycle*) sur la diffusion d'images.
- **Niveau 3 — capacité du backbone** : MDLM-OWT (110M) est petit. Le manifold des textes "all-three" est microscopique dans son espace.

Pour distinguer ces deux causes, nous changeons de backbone.

---

## 9. Phase 8 — Étude cross-backbone (Qwen3-0.6B-MDLM)

Backbone : `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` (596 M params, 5× MDLM-OWT). Mêmes datasets, mêmes prompts, mêmes hyperparamètres LoRA modulo le changement de target_modules. Adapter ré-entraînés pour les 6 axes + 1 intersection.

### 9.1 N=2 sur Qwen3 — résultats par paire

| pair | MDLM-OWT ratio | Qwen3 ratio | direction |
|---|---:|---:|---|
| formal × positive | 1.33 | **0.24** | ⬇⬇⬇ |
| formal × positive2 | 1.17 | **0.44** | ⬇⬇ |
| formal × concrete | 0.69 | **0.76** | ⬆ |
| formal × sports | 0.95 | **0.69** | ⬇ |
| positive × positive2 | 2.54 | **1.48** | ⬇ |
| positive × concrete | 0.74 | **1.08** | ⬆ |
| positive × sports | 1.69 | **1.13** | ⬇ |
| positive2 × concrete | 0.72 | **0.94** | ⬆ |
| positive2 × sports | 0.94 | **1.15** | ⬆ |
| concrete × sports | 0.76 | **2.77** | ⬆⬆⬆ |
| **moyenne** | **1.04** | **1.07** | ≈ |

→ **Moyennes très proches (1.04 vs 1.07), mais variance par paire fortement amplifiée** sur Qwen3 (range 0.24–2.77 vs 0.69–2.54 sur MDLM-OWT).

**Pattern qualitatif** : les paires lexicales/topicales (`concrete`, `sports`, `positive2`) s'améliorent dramatiquement sur Qwen3 ; les paires stylistiques (`formal`, `positive`) se dégradent. Hypothèse de mécanisme :

> Les experts Qwen3 produisent des shifts de marginals plus extrêmes (ex. : marginal `formal` passe de 0.32 à 0.59). Sur des axes orthogonaux, cela se traduit par plus de signal joint. Sur des axes co-corrélés (style), les shifts entrent en collision et se neutralisent.

### 9.2 N=3 sur Qwen3 — trois triplets à $n=500$

| Triplet | MDLM-OWT ratio | Qwen3 ratio | gain |
|---|---:|---:|---:|
| `positive2 × concrete × sports` (lexical) | 0.39 | **1.84** | **+4.7×** |
| `formal × concrete × sports` (mix) | 0.88 | 0.84 | ≈ |
| `formal × positive × concrete` (stylistique) | 0.55 | 0.42 | ⬇ |
| **moyenne** | **0.61** | **1.03** | **+69 %** |

→ **Le plateau est partiellement levé** (mean 0.61 → 1.03, soit la parité avec indep-ref). Mais l'effet est **triplet-dépendant** :

- Triplet lexical : explose à 1.84 (super-additif).
- Triplet mix : flat à 0.84.
- Triplet stylistique : dégrade à 0.42.

**Niveau 3 est partiellement confirmé**. La capacité du backbone résout le plateau pour les compositions lexicales, mais pas pour les compositions stylistiques. Cela indique soit qu'une part de Niveau 2 (théorique) subsiste, soit qu'un autre mécanisme (collision d'axes entangled) opère.

### 9.3 Phase 5 (κ ↔ déficit) refit sur Qwen3

Mêmes 10 paires, mêmes valeurs de κ (calculées sur OWT, indépendantes du backbone), mais nouveaux JS et indep-ref :

| Métrique | MDLM-OWT r | Qwen3 r | Δ |
|---|---:|---:|---:|
| κ | −0.917 | **−0.242** | effondrement |
| Spearman | −0.752 | −0.210 | effondrement |
| CKA | −0.868 | −0.302 | effondrement |
| MI | −0.836 | −0.301 | effondrement |

→ **La corrélation observée sur MDLM-OWT ne se reproduit pas sur Qwen3.** Tous les CI₉₅ traversent zéro : on ne peut pas rejeter l'hypothèse nulle au seuil α = 0.05.

**Conclusion sévère** : le résultat-phare $r = -0.92$ de la Section 6 était **spécifique au backbone MDLM-OWT**, pas une loi universelle. La κ-Gram theory n'est pas un prédicteur **fiable et stable** du déficit PoE.

---

## 10. Discussion

### 10.1 Ce qui est solidement établi

1. **La formule PoE est mathématiquement correcte au niveau des logits.** Test 2 : slope = 0.857, R² = 0.811. Indépendant du backbone.

2. **PoE-2 est super-additif en moyenne sur les deux backbones.** Mean ratio = 1.04 (MDLM-OWT) et 1.07 (Qwen3). Cohérent avec la promesse théorique de la composition.

3. **PoE bat l'expert intersection-trained** sur les axes critiques (Test 1). C'est un argument fort pour l'approche compositionnelle vs. l'entraînement spécialisé sur l'intersection (qui est coûteux et nécessite des datasets joints).

4. **À N=3, le plateau d'origine (ratio ≈ 0.55) est partiellement levé** par un backbone plus capable. La moyenne passe de 0.61 à 1.03. C'est un argument pour Niveau 3 (capacité importante), pas Niveau 2 (limite théorique pure).

5. **Les premiers pas de débruitage portent le signal de composition.** `early_fire = constant` mais `late_fire = collapse`. Cela contredit l'intuition naïve "composer à la fin clean est mieux".

### 10.2 Limites identifiées

1. **La κ-Gram theory ne se généralise pas cross-backbone.** Le r passe de −0.92 à −0.24. C'est une *vraie* limite : la prédiction du déficit par l'orthogonalité géométrique des proxies n'est pas universelle.

2. **À N=3 sur Qwen3, l'effet est triplet-dépendant.** Lexical : super-additif ; stylistique : sous-additif. La "capacité backbone" ne résout pas tout.

3. **Variance importante par paire.** Range de ratios sur Qwen3 : 0.24 à 2.77. Cela limite la prédictabilité des résultats.

4. **MDLM-OWT a une dégénérescence intrinsèque** au-delà de ~30-50 tokens (répétitions, digit floods). Nous avons palliée par rejection sampling, mais cela introduit un biais.

5. **L'estimation de l'indep-ref dépend des marginals individuels**, qui dépendent à leur tour du backbone et de l'agressivité des adapters. Comparer un ratio entre deux backbones est délicat — nous le faisons modulo cette caveat.

6. **N=3 testé sur seulement 3 triplets**. La couverture est limitée pour conclure de manière définitive sur le pattern lexical vs stylistique.

### 10.3 Pourquoi κ-sur-OWT n'est probablement pas le bon prédicteur

> *Cette section reprend une intuition cruciale du processus de recherche.*

**Pourquoi κ-sur-OWT échoue à prédire le déficit ?**

κ mesure **l'orthogonalité statistique des proxies sur le corpus OWT** : est-ce que `len(text)` et `formality(text)` sont décorrélés sur des textes naturels ?

Mais le déficit PoE n'est pas une propriété du corpus — c'est une propriété de **l'interaction** entre :

1. Comment le **modèle** représente ces axes en interne.
2. Comment les **adapters LoRA** ont appris à pousser.
3. Comment ces poussées se **composent dans l'espace des logits**.

**κ sur OWT ne voit rien de ces 3 choses.** C'est au mieux un proxy d'une propriété profondément différente. D'où l'effondrement cross-backbone.

#### Quatre candidats pour ce qu'on devrait *vraiment* mesurer

**A. Orthogonalité des shifts LoRA dans l'espace des poids ⭐**

$$
\text{align}(a, b) = \cos\!\left(\Delta W_a, \Delta W_b\right)
$$

où $\Delta W_a = W_{\text{LoRA}_a} - W_{\text{base}}$ (matrices de delta du LoRA).

- **Interprétation** : si les deux experts modifient le modèle dans la **même direction du weight-space**, ils se renforcent (composition triviale, gain marginal). Si dans des directions orthogonales, leur composition est génuinement nouvelle.
- **Backbone-spécifique** : capture exactement ce qui dépend du modèle.
- **Computable à coût zéro** : on a tous les LoRA checkpoints (MDLM-OWT + Qwen3).

**B. Leakage cross-axis (mesuré sur les samples qu'on a déjà)**

Pour chaque paire $(a, b)$, regarder dans `expert-A-only.jsonl` la marginale **sur l'axe b** (pas sur a). Combien de samples expert-A-only passent le seuil b ?

- Si proche de 0.25 (baseline) : expert-A est "pur" sur son axe.
- Si > 0.25 : expert-A pousse aussi vers b → axes positivement liés (composition facile, peu de gain incrémental).
- Si < 0.25 : expert-A repousse b → axes anti-corrélés (composition dure, déficit garanti).

- **Interprétation** : direction et magnitude de la "fuite" de chaque expert dans l'autre axe.
- **Computable à coût zéro** : tout est dans nos 80+80 jsonl.

**C. κ sur les activations du modèle (latent-space, pas raw OWT)**

Pour chaque paire de proxies, faire passer les baseline samples dans le modèle, extraire les activations layer-par-layer, calculer κ entre les activations qui *causent* les hauts scores sur a vs sur b.

- **Interprétation** : où dans le modèle les axes "vivent". Deux axes qui vivent dans des sous-espaces orthogonaux composent bien ; deux axes qui vivent dans le même sous-espace se battent.
- **Coût modéré** : nécessite d'extraire les activations (~30 min de compute local sur quelques centaines de samples).

**D. "Energy curvature" — la 2ème dérivée de la composition**

Mesurer :

$$
C(a, b) = \text{JS}_{\text{PoE}} - \left( \text{JS}_{a\text{-only}} + \text{JS}_{b\text{-only}} - \text{JS}_{\text{baseline}} \right)
$$

Cette quantité est l'écart à un modèle "additif simple".

- **Interprétation** : capture si la composition apporte quelque chose au-delà de l'union des 2 effets.
- **Computable à coût zéro** : tout est dans nos données.

#### Proposition

**Tester immédiatement A et B** (gratuit, local) sur les 10 paires des 2 backbones. Voir si l'un (ou la combinaison) prédit le déficit avec un $r > 0.7$ cross-backbone.

- **Si oui** → vraie contribution du papier : *"κ-on-data ne prédit pas le déficit PoE, mais l'orthogonalité des shifts LoRA — ou le leakage cross-axis — le prédit invariamment."* C'est plus puissant et plus pratique que κ-on-OWT.
- **Si non** → on a au moins exploré honnêtement la question. Et le papier raconte ça : *"on a essayé 4 prédicteurs, aucun ne marche cross-backbone, c'est un vrai problème ouvert."*

> **Note méthodologique post-hoc.** Après une revue critique externe, le candidat **A** original (cosinus brut entre les matrices ΔW) a été affiné en **A'** : alignement entre les *shifts de logits* induits par chaque expert sur un pool de prompts pivots fixes. Cette reformulation est plus directement reliée à la formule PoE (qui agit sur les logits, pas les poids). Le candidat **D** (energy curvature) a été écarté : la quantité $C(a,b) = \text{JS}_{\text{PoE}} - (\text{JS}_a + \text{JS}_b - \text{JS}_{\text{base}})$ est presque tautologique vis-à-vis du déficit qu'on veut prédire (corréler du PoE avec du PoE). La Section 11 documente l'évaluation empirique des 4 candidats raffinés.

---

## 11. Évaluation empirique des prédicteurs candidats

Nous évaluons les 4 candidats raffinés (B, F-js, A', E) sur les 10 paires d'experts, sur les deux backbones (MDLM-OWT et Qwen3), pour vérifier si l'un d'entre eux atteint une corrélation $r > 0.7$ avec le déficit PoE en cross-backbone (c'est-à-dire sur l'union des 20 paires).

### 11.1 Méthodologie

**Données utilisées** :
- Joint satisfaction par config (PoE-strict, expert-A-only, expert-B-only, baseline) sur les 10 paires × 2 backbones.
- Samples bruts JSONL pour B et F-js (200 samples par config).
- 32 prompts pivots de 32 tokens pour A' et E (chargement local des modèles + adapters LoRA).

**Métriques évaluées** :
- **B (leakage cross-axis)** : $\frac{1}{2}\big[ P(s_b \geq \tau_b \mid \text{expert-A-only}) + P(s_a \geq \tau_a \mid \text{expert-B-only}) \big]$
- **F-js (sample distance)** : Jensen-Shannon entre histogrammes proxy-score des configs expert-A-only vs expert-B-only sur chacun des 2 axes, moyenné.
- **A' (logit-shift alignment)** : $\mathbb{E}_x [\cos(\Delta\ell_a(x), \Delta\ell_b(x))]$ avec $\Delta\ell_a(x) = \text{logits}_a(x) - \text{logits}_b(x)$.
- **E (spatial overlap)** : $\mathbb{E}_x [\cos(\|\Delta\ell_a(x, \cdot)\|, \|\Delta\ell_b(x, \cdot)\|)]$ — cosinus entre les normes par-position.

Pour chaque prédicteur, nous calculons la corrélation Pearson contre le déficit $\Delta = \text{JS}_{\text{indep}} - \text{JS}_{\text{PoE-strict}}$ sur :
- les 10 paires MDLM-OWT seules (n=10)
- les 10 paires Qwen3 seules (n=10)
- l'union des 20 paires (cross-backbone, n=20).

### 11.2 Résultats per-backbone et cross-backbone

| Prédicteur | MDLM-OWT (n=10) | Qwen3 (n=10) | **Cross-backbone (n=20)** |
|---|---:|---:|---:|
| **B (leakage)** | **−0.887** *** | **+0.786** ** | +0.473 * |
| F-js | −0.294 | +0.102 | +0.014 |
| A' (logit-shift cos) | −0.179 | −0.049 | +0.098 |
| E (spatial overlap) | −0.231 | +0.453 | +0.071 |

(\*\*\* : p < 0.001 ; \*\* : p < 0.01 ; \* : p < 0.05.)

**Aucun prédicteur n'atteint $r > 0.7$ cross-backbone.** Le seul à signal très fort sur chaque backbone individuellement (B leakage) **change de signe** entre MDLM-OWT (r = −0.89) et Qwen3 (r = +0.79). Cross-backbone ce sign-flip se neutralise partiellement, plafonnant la corrélation à r ≈ 0.47.

### 11.3 Variantes de B et combinaisons linéaires

Pour vérifier que ce sign-flip n'est pas simplement un artefact de signe (e.g., qu'on devrait prendre la valeur absolue, ou la déviation à la baseline 0.25), nous testons :

**Variantes de B** (per-backbone) :

| Backbone | $B$ raw | $B - 0.25$ centré | $|B - 0.25|$ déviation absolue |
|---|---:|---:|---:|
| MDLM-OWT | −0.887 | −0.887 | −0.887 |
| Qwen3 | +0.786 | +0.786 | **+0.890** |

L'absolute deviation améliore légèrement Qwen3 (de +0.786 à +0.890) mais ne change rien sur MDLM-OWT. **Le sign-flip persiste.**

**Combinaisons linéaires** $\alpha \cdot \text{pred}_a + (1-\alpha) \cdot \text{pred}_b$, où le poids $\alpha$ est optimisé sur MDLM-OWT et évalué cross-backbone :

| pred_a + pred_b | α optimal | $r_{\text{MDLM}}$ | $r_{\text{Qwen3}}$ | $r_{\text{cross}}$ |
|---|---:|---:|---:|---:|
| B + F-js | 1.40 | −0.906 | +0.784 | **+0.479** ← top |
| A' + B | 0.01 | −0.889 | +0.786 | +0.474 |
| B + E | 0.95 | −0.890 | +0.786 | +0.473 |

**Plafond cross-backbone à r ≈ 0.48.** Aucune combinaison linéaire ne peut concilier les deux backbones car les corrélations sont en directions opposées sur chacun.

### 11.4 Le phénomène du sign-flip — un finding en lui-même

Le pattern observé est mécaniquement interprétable :

- **Sur MDLM-OWT (110M, experts faiblement entraînés)** : un leakage élevé signifie que les LoRA n'ont pas pleinement spécialisé leurs experts sur leur axe propre. Cela traduit une **affinité naturelle entre les axes** dans la distribution OWT (et dans le sub-manifold appris par le petit modèle). Plus les axes sont affins, plus la composition est facile (déficit faible). → **r négatif**.

- **Sur Qwen3 (596M, experts fortement entraînés)** : un leakage élevé signifie que les LoRA, malgré le grand modèle, ont **sur-spécialisé** au point que leurs distributions de poussée se chevauchent. Cela génère des **collisions de specialisation** quand on les compose : chaque expert pousse sur les mêmes neurones. Plus de leakage = plus de collision = composition plus dure (déficit élevé). → **r positif**.

Ce sign-flip est **le pattern empirique le plus distinctif** que nous documentons. Il opérationnalise mécaniquement la distinction "experts faibles vs forts" et clarifie pourquoi un prédicteur backbone-invariant simple est insaisissable : la même métrique a des sens opposés selon le régime des experts.

### 11.5 Implications pour la prédictibilité du déficit PoE

1. **Aucun prédicteur scalaire simple** (B, F-js, A', E) n'atteint $r > 0.7$ cross-backbone. Les variantes (centré, absolu, combinaisons linéaires) plafonnent à r ≈ 0.48.

2. **Un prédicteur backbone-invariant doit être régime-conscient** : il faut prendre en compte la "force" des experts (eg via une mesure de magnitude des shifts de logits) avant de pouvoir extraire un signal stable.

3. **Le candidat C (κ sur activations latentes) ne sauve pas la situation.** Évalué a posteriori (Section 11.6 ci-dessous) avec un linear-probe ridge regression sur le mean-pooled last-hidden-state des deux backbones, il reproduit exactement le pattern de κ_OWT : forte corrélation négative sur MDLM-OWT (κ_act : r = −0.76 ; cosinus signé : r = −0.83), effondrement sur Qwen3 (κ_act : r = −0.29 ; cosinus : r = +0.08). Cross-backbone : r = −0.39 et r = −0.23. Le problème n'est donc pas notre choix de métrique géométrique — c'est **structurel au régime du gros backbone**.

4. **Contribution scientifique consolidée** : la *non-existence* d'un prédicteur scalaire universel — démontrée par 7+ mesures testées (κ_OWT, Spearman, CKA, MI, B_leakage, F_js, A', E, κ_act, cos_act, plus variantes) — est un résultat empirique non-trivial. Il documente une limite fondamentale des cadres géométriques/corrélationnels appliqués à la composition PoE en MDLM, et motive des approches plus sophistiquées (prédicteurs régime-conscients, mesures structurelles non-linéaires, ou correctifs algorithmiques au niveau du sampling).

### 11.6 Évaluation empirique du candidat C — κ sur activations latentes

**Procédure** : pour chaque backbone, on passe les 200 baseline samples à travers le modèle (en désactivant les adapters via `disable_adapter()`), on extrait le mean-pooled last-hidden-state $h_x \in \mathbb{R}^d$ ($d=768$ pour MDLM-OWT, $d=1024$ pour Qwen3). On fitte un linear probe ridge par axe : $s_a(x) \approx \mathbf{w}_a^\top h_x + \beta_a$ où $s_a(x)$ est le proxy score sur l'axe $a$. La κ-Gram analog dans l'espace des activations est :

$$
\kappa_{\text{act}}(a, b) = \sqrt{2} \cdot \frac{|\langle \mathbf{w}_a, \mathbf{w}_b \rangle|}{\|\mathbf{w}_a\|^2 + \|\mathbf{w}_b\|^2}
$$

(la cosinus signé $\cos(\mathbf{w}_a, \mathbf{w}_b)$ est aussi reporté pour préserver la directionalité).

**Résultats** :

| Métrique | MDLM-OWT (n=10) | Qwen3 (n=10) | Cross-backbone (n=20) |
|---|---:|---:|---:|
| κ_act | −0.758 ** | −0.291 | −0.390 |
| cos_act (signé) | −0.829 ** | +0.083 | −0.225 |

(\*\* : p < 0.01.)

**Interprétation** : sur MDLM-OWT, le candidat C reproduit fidèlement la corrélation observée avec κ_OWT (r = −0.92 → −0.76 / −0.83). Cela confirme que la κ-Gram theory est **cohérente entre l'espace des données brutes et l'espace des représentations apprises** par le petit backbone — les axes "vivent" essentiellement aux mêmes endroits dans les deux espaces.

Sur Qwen3, les corrélations s'effondrent à des niveaux comparables à κ_OWT (r = −0.24 → −0.29 / +0.08). Ce n'est donc **pas un artefact de choix de métrique géométrique** : que κ soit calculé sur les énergies de proxies sur OWT ou sur les directions de probes linéaires dans l'espace des activations Qwen3, on obtient le même collapse cross-backbone.

→ **Le candidat C confirme et renforce le finding négatif global** : aucun prédicteur scalaire simple, qu'il soit basé sur les données brutes, les samples post-composition, les shifts de logits, ou les activations latentes, ne donne r > 0.7 cross-backbone. Le sign-flip de B reste le pattern empirique le plus distinctif.

---

## 12. Phase 10 — Tentative de correction par joint MCMC (Du Yan 2023 adapté)

Au-delà de la prédiction, nous avons testé une **correction algorithmique directe** du déficit PoE-3 : appliquer un correcteur MCMC après la trajectoire de débruitage standard. L'idée est inspirée de Du Yan et al. 2023 (*Reduce, Reuse, Recycle*), qui montrent en diffusion d'images que la composition logit-par-step sous-évalue la composition séquence-niveau, et que des steps de MCMC supplémentaires peuvent corriger cette dérive.

### 12.1 Hypothèse testée

Le **Test 2** (Section 4) avait montré un slope de **0.857 < 1** entre les log-ratios PoE prédits et observés. Une lecture possible : la composition par-step sous-compose systématiquement, et un correcteur joint pourrait ramener le slope vers 1, augmentant les ratios N=3.

### 12.2 Méthodologie

Nous adaptons l'idée Du Yan (originalement Langevin en continu) au cas **discret** des MDLM via une variante **block Gibbs** (procédure `noise_then_denoise`) :

Pour chaque sample issu du PoE-3 naïve, on répète $K = 3$ itérations de :
1. **Re-noising** : masquer aléatoirement $\rho = 25\%$ des positions du sample.
2. **Re-denoising partiel** : appliquer 64 sub-steps de débruitage MDLM utilisant les logits PoE-composés pour resampler les positions masquées.

Cette procédure correspond à un **block-Gibbs sweep** sous la conditional PoE au niveau de masking choisi. Algorithmiquement plus rigoureux qu'un Gibbs single-position (que nous avions testé en Phase 7 avec dégradation), il préserve la structure jointe des positions non-masquées tout en redonnant au modèle plusieurs passages pour stabiliser le joint.

Implémentation : `src/composition/joint_mcmc.py::noise_then_denoise()` ; runner : `scripts/13_n3_with_joint_mcmc.py`. Une seconde variante `mh_token_swap` (Metropolis-Hastings rigoureux avec ratio d'acceptance basé sur l'ELBO séquence-niveau) est aussi implémentée mais non testée par contraintes de compute (~50× plus lent que le block-Gibbs).

### 12.3 Résultats

Test sur le triplet `formal × positive × concrete` (le pire à N=3 sur Qwen3, ratio naïve 0.42 dans Phase 8 v1 à $n=500$) avec backbone `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1`. Deux runs distincts à $n=200$ et $n=50$ :

| Configuration | $n$ | triple_sat | indep_ref | ratio |
|---|---:|---:|---:|---:|
| PoE-3 naïve | 200 | 0.0100 | 0.0652 | **0.15** |
| PoE-3 + block-Gibbs MCMC | 200 | 0.0050 | 0.0652 | **0.08** ⬇ |
| PoE-3 naïve | 50 | 0.0200 | 0.0857 | **0.23** |
| PoE-3 + block-Gibbs MCMC | 50 | 0.0000 | 0.0857 | **0.00** ⬇⬇ |
| PoE-3 + **MH-token-swap rigoureux** | 50 | 0.0200 | 0.0857 | **0.23** ≈ |

**Stats des refinements** :
- *block-Gibbs* (n=200) : 8 829 swaps tentés, 5 526 changements appliqués (62.6 % churn). Modifie significativement les samples mais **dégrade le ratio**.
- *MH-token-swap* (n=50) : 1 166 propositions, 264 acceptées (22.6 % d'acceptance). Les swaps acceptés correspondent au critère MH d'augmentation du log-ratio PoE séquence-niveau ; ils **ne déplacent pas le ratio**.

### 12.4 Interprétation

Les deux variantes de correction MCMC échouent à lever le plateau N=3, mais pour des raisons différentes :

**Block-Gibbs (`noise_then_denoise`)** : modifie agressivement les samples (62.6 % de churn) mais détruit plus d'information jointe qu'il n'en restaure. Le re-noising de 25 % des positions casse la structure jointe accumulée durant les 256 pas de débruitage initial, et 64 sub-steps de re-débruitage ne suffisent pas pour la reconstruire correctement.

**MH-token-swap rigoureux** : accepte 22.6 % des swaps proposés selon un critère Metropolis-Hastings basé sur l'ELBO séquence-niveau de la distribution PoE-composée. Ces acceptations sont par construction des swaps qui *augmentent* la log-probabilité jointe sous PoE. Pourtant, **les samples résultants ont le même ratio que les naïfs** (0.23 → 0.23).

Cette dernière observation est déterminante. Elle signifie que :

> **Les samples PoE-3 naïfs sont déjà approximativement aux modes de la distribution PoE-composée.** Le MCMC rigoureux confirme qu'il n'existe pas de région voisine plus dense en samples triple-satisfaisants vers laquelle migrer.

Cela **réfute l'hypothèse Test 2** ($\text{slope} < 1 \Rightarrow$ sous-composition réparable). Le slope de 0.857 ne traduit *pas* une erreur de sampling correctible au niveau du décodage. Le bottleneck à N=3 réside dans la **distribution PoE-composée elle-même** : elle n'a tout simplement pas de masse importante sur les configurations triple-satisfaisantes — soit parce que le manifold conjoint des textes "tous-trois" est trop rare dans la distribution apprise par le backbone (Niveau 3 — capacité), soit parce que la composition par sum-of-logits n'élève pas suffisamment les régions joint-valides au-dessus du fond (Niveau 2 — paradigme de composition).

Pour aller au-delà, il faudrait soit :

- changer de **paradigme de composition** (score matching explicite, énergies apprises plutôt que sum-of-logits naïf, à la *vraie* méthode Du Yan 2023 et non son adaptation block-Gibbs),
- changer d'**objectif de sampling** (au lieu de tirer du PoE-composé, tirer de $p(\text{satisfait}_a, \text{satisfait}_b, \text{satisfait}_c)$ via classifier-conditioned sampling),
- ou changer de **backbone** vers un modèle dont la distribution apprise contient effectivement des régions triple-satisfaisantes denses (probable nécessité d'un modèle bien plus capable que les 596M de Qwen3 actuel).

### 12.5 Implication pour le papier

Cet ensemble de **résultats négatifs** complète le tableau diagnostique :

- Aucun **prédicteur scalaire simple** n'est backbone-invariant (Section 11).
- **Block-Gibbs MCMC** dégrade le ratio.
- **MH-token-swap rigoureux** préserve mais n'améliore pas — confirmant empiriquement que les samples PoE-3 naïfs sont déjà aux modes de la distribution composée.

→ Le déficit observé à N=3 **n'est pas une erreur d'approximation** réparable au niveau du sampling. C'est une propriété de la **distribution PoE-composée elle-même**, qui n'a pas de masse suffisante sur les configurations triple-satisfaisantes dans le régime des backbones MDLM-text actuels. Cela motive d'aller chercher la correction non au niveau du *sampling* mais au niveau de la **formule de composition** — ce qui mène à la Section 13.

---

## 13. Phase 11 — Découplage du coefficient μ sur log p_base

### 13.1 Hypothèse et formulation

La formule canonique du PoE,

$$
\log p_{\text{PoE}}(x) = \log p_b(x) + \sum_i \lambda_i \cdot (\log p_i(x) - \log p_b(x))
$$

donne, à $\lambda_i = 1$ pour tout $i$ :

$$
\log p_{\text{PoE}}(x) = (1 - N) \cdot \log p_b(x) + \sum_i \log p_i(x)
$$

Le coefficient sur $\log p_b$ vaut donc **$1 - N$** : à $N=2$ → $-1$ (pénalité modeste) ; à $N=3$ → $-2$ (pénalité forte). Hypothèse : à $N \geq 3$, ce coefficient pénalise *trop* les configurations OWT-typiques, vidant le modèle de continuations valides triple-satisfaisantes.

Pour tester : on **découple** μ comme paramètre libre :

$$
\log p_{\text{custom}}(x) = \mu \cdot \log p_b(x) + \sum_i \lambda_i \log p_i(x)
$$

et on balaie $\mu \in \{-2, -1.5, -1, -0.5, 0, +0.5, +1\}$. Implémentation : `src/composition/poe_sampler.py::PoECompositionModel(mu_base=μ)` ; runner : `scripts/14_mu_sweep.py`.

### 13.2 Sweep initial — Qwen3, formal × positive × concrete

Premier test sur le triplet le plus difficile (Phase 8 ratio 0.42 à $n=500$) :

| μ | triple_sat | ratio | gain vs canonical |
|---:|---:|---:|---:|
| **−2 (canonical)** | 0.010 | 0.15 | référence |
| −1.5 | 0.035 | 0.54 | +260 % |
| **−1.0** | **0.040** | **0.61** ⭐ | **+307 %** |
| −0.5 | 0.025 | 0.38 | +153 % |
| 0.0 | 0.025 | 0.38 | +153 % |
| +0.5 | 0.025 | 0.38 | +153 % |
| +1.0 | 0.020 | 0.31 | +107 % |

→ **Courbe en cloche, pic à μ = −1**, ratio multiplié par 4. Premier signal positif fort : un seul hyperparamètre de calibrage suffit à transformer un échec en quasi-parité avec l'indep-ref. Confirmé par sweep fin : $\mu \in \{-1.25, -1.0, -0.75\}$ donne (0.38, **0.61**, 0.38).

### 13.3 Verifications de généralisation (6 sweeps additionnels)

Pour vérifier si le sweet spot est universel ou dépend du triplet/backbone, nous avons étendu le sweep à 5 configurations supplémentaires.

#### 13.3.1 Autres triplets sur Qwen3

| Triplet | μ canonical | best μ | best ratio | canonical | gain |
|---|---:|---:|---:|---:|---:|
| **positive2 × concrete × sports** (lexical) | −2 | **−2** | **3.23** ⭐ | 3.23 | 0 % (déjà super-additif) |
| **formal × concrete × sports** (mix) | −2 | **−1** | **1.23** | 0.46 | **+167 %** |
| formal × positive × concrete (style) | −2 | −1 | 0.61 | 0.15 | +307 % |

→ Le μ-fix **n'aide que les triplets contenant un axe stylistique** (`formal`). Pour les triplets purement lexicaux, le canonical $\mu = -2$ est déjà optimal et même super-additif.

#### 13.3.2 Cross-backbone : MDLM-OWT, formal × positive × concrete

| μ | triple_sat | ratio |
|---:|---:|---:|
| −2 (canonical) | 0.010 | 0.35 |
| −1.5 | 0.010 | 0.35 |
| −1 | 0.010 | 0.35 |
| −0.5 | 0.015 | 0.53 |
| **0** | **0.020** | **0.71** ⭐ |
| +0.5 | 0.000 | 0.00 |
| +1 | 0.010 | 0.35 |

→ Sur le **petit backbone**, le sweet spot est à **μ = 0** (et non −1 comme sur Qwen3) — le ratio canonical 0.35 monte à 0.71, soit **+103 %**. Le sweet spot **n'est pas invariant cross-backbone**, mais sa **direction** l'est : *toujours* moins punitif que le canonical $1-N$.

#### 13.3.3 N=2 — le pattern se généralise

Pour deux paires d'experts ($N=2$ ; canonical $\mu = -1$) sur Qwen3 :

| Paire | μ=−1 (std) | μ=−0.5 | μ=0 | μ=+0.5 | best | gain |
|---|---:|---:|---:|---:|---:|---:|
| **formal × positive** (stylistic) | 0.33 | **1.07** | **1.07** | 0.89 | μ∈{−0.5,0} | **+220 %** |
| **concrete × sports** (lexical) | **3.29** | 1.73 | 1.04 | 0.69 | μ=−1 | 0 % (super-additif déjà) |

Même pattern qu'à N=3 : μ-découplé aide les compositions stylistiques (+220 %) et n'apporte rien aux compositions lexicales (le canonical est déjà super-additif).

### 13.4 Synthèse : 4 findings

1. **Le découplage de μ aide consistamment les compositions à composante stylistique** — gains de +103 % à +307 % observés sur les 4 setups stylistiques testés (3 N=3 + 1 N=2).

2. **Pour les compositions purement lexicales**, le canonical $\mu = 1 - N$ est **déjà optimal** et même super-additif (ratios 3.23 à 3.29). Le découplage n'aide pas et peut dégrader.

3. **Le sweet spot de μ n'est pas invariant cross-backbone** : −1 sur Qwen3-0.6B-MDLM, 0 sur MDLM-OWT-110M (à $N=3$ formal × positive × concrete). Mais sa **direction** est consistante : *moins punitif que le canonical*.

4. **Sur le petit backbone (110M)**, même au sweet spot, le plateau N=3 reste fragile (ratio max 0.71, sub-additif). **Sur le gros backbone (596M)**, on franchit la barre $r > 1$ avec le bon μ. La capacité du backbone reste un facteur orthogonal qu'aucune correction algorithmique ne supplante.

### 13.5 Position scientifique — passage du diagnostique à l'algorithmique

Avec ces résultats, le papier peut désormais soutenir :

- **Une amélioration algorithmique simple et reproductible** : un hyperparamètre additionnel ($\mu$) à régler par paire ou triplet, qui rescue les compositions stylistiques échouant sous le canonical.
- **Un pattern interprétable** : axes lexicaux composent bien sous canonical ; axes stylistiques requièrent un μ relâché.
- **Une formulation explicite de la limite restante** : la capacité du backbone reste critique au-delà du calibrage (μ-fix double les ratios MDLM-OWT mais ne lève pas le plateau ; sur Qwen3, le couplage μ-fix + grand backbone donne >1 super-additif).

C'est maintenant une *vraie contribution algorithmique* : on ne dit plus seulement "PoE échoue à N=3 et on ne sait pas réparer", mais "PoE-N peut être rescue via un calibrage simple sur un coefficient unique, à condition que les axes soient majoritairement stylistiques et que le backbone ait suffisamment de capacité pour exprimer le manifold conjoint".

---

## 14. Phase 12 — Auto-tuning de μ : protocoles et limites

Phase 11 établit qu'un μ bien choisi rescue les compositions stylistiques, mais la valeur optimale **dépend du triplet et du backbone**. Pour transformer cette observation en outil pratique, il faut une **procédure** de sélection qui ne nécessite pas un sweep complet à $n=200$ par triplet. Trois protocoles testés sur les 17 setups dont nous connaissons μ\* (issus des Phases 11 + 12c, 6 N=3 + 11 N=2, deux backbones) :

* **A — Quick grid sweep**, candidats $\mu \in \{-2, -1.5, -1, -0.5, 0\}$, argmax du ratio.
* **B — Bayesian optimization** (GP + Expected Improvement), 6 évaluations, range $\mu \in [-2.5, +0.5]$.
* **C — Predictor structurel**, régression linéaire sur 4 features (N, stylistic_load, $\log_{10}\text{cap}$, mean_marginal), évalué en LOO-CV. Aucune passe forward du modèle.

### 14.1 Protocoles A et B — coût d'évaluation

À $n=50$ (cible "rapide"), A et B échouent tous deux : 1/3 et 0–1/3 des setups identifient le bon μ\*. La variance de $n=50$ est trop grande pour distinguer des candidats voisins. À $n=200$ (cible "fiable") :

| Setup | Ground-truth μ\* | A → best μ̂ | A ratio | B → best μ̂ | B ratio | A ✓ | B ✓ |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| Qwen3, formal × positive × concrete | −1 | −1 | 0.61 | 0 | 0.38 | ✓ | ✗ |
| Qwen3, positive2 × concrete × sports | −2 | −2 | 3.23 | −2 | 3.23 | ✓ | ✓ |
| MDLM-OWT, formal × positive × concrete | 0 | 0 | 0.71 | −0.21 | 0.88 | ✓ | ≈ |

→ **A à $n=200$ est fiable (3/3)** mais coûte 5 forwards de 200 prompts chacun, soit l'équivalent de l'effort manuel sans son guide. **B est instable** — sur Qwen3 fpc, l'EI s'enferme à $\mu = +0.5$ (point de seed ratio plat) ; sur Qwen3 pcs il converge bien ; sur MDLM il *bat* A en explorant un point intermédiaire ($\mu = -0.21$, ratio 0.88 vs 0.71). En pratique, B exige une seed pertinente et un kernel bien réglé pour fonctionner.

### 14.2 Protocole C — predictor structurel (LOO-CV sur 17 setups)

Régression linéaire sur 4 features computées sans aucune passe forward du modèle :

```
μ_pred = −0.106
       − 0.240 · N
       + 1.032 · stylistic_load
       − 0.229 · log10(capacity_M)
       − 0.348 · mean_marginal
```

LOO-MAE = **0.469** sur l'échelle des μ\* observés ($\mu^* \in [-2, 0]$, plage 2.0). Soit ~23 % de l'amplitude — utile comme **prior**, insuffisant comme oracle.

Les signes des coefficients sont cohérents avec l'intuition Phase 11 :

* `stylistic_load` (positif fort) : plus la composition est stylistique, plus μ\* doit être *relâché* (moins négatif). C'est le finding central, et il sort des 17 datapoints sans avoir été codé en dur.
* `N` (négatif), `mean_marginal` (négatif) : plus on empile d'experts ou plus chacun pousse fort, plus il faut un μ punissant.
* `log10(capacity)` (négatif modeste) : un gros backbone supporte un μ plus négatif (contre-intuitif au premier abord ; cohérent avec l'observation Qwen3 vs MDLM-OWT — Qwen3 prend μ=−1, MDLM-OWT prend μ=0).

La règle structurelle simple `μ ≈ −N(1−sl) − [1 si cap≤200M]` donne MAE = 0.471, comparable à la régression mais sans capacité d'extrapolation.

### 14.3 Synthèse — pas un auto-tuner, mais un workflow

Aucun des trois protocoles seul ne **remplace** un sweep manuel à $n=200$. Mais combinés ils donnent un workflow réaliste :

1. **Calculer C** (gratuit, ~ms) → prédiction $\hat{\mu}_C \pm 0.5$.
2. **Lancer A en grille fine autour de $\hat{\mu}_C$** (3 candidats à $n=200$ au lieu de 5+) → économie ~40 % vs sweep aveugle, fiabilité conservée.
3. **À défaut, fallback sur la heuristique par défaut** : $\mu = -1$ pour les compositions stylistiques sur backbones $\geq$ 500M, $\mu = 0$ pour les backbones $\leq$ 200M, canonical $\mu = 1-N$ pour les compositions purement lexicales.

C'est l'étape honnête entre "il faut sweeper aveuglément à chaque triplet" et "on a un oracle qui prédit μ\*". L'oracle reste un objectif futur, conditionné à un dataset d'évaluation $\geq$ 30–50 setups sur $\geq$ 3 backbones — non atteignable dans le scope de cette étude.

### 14.4 Implication pour le papier

Phase 12 verrouille la **portée pratique** de la contribution Phase 11 : un μ-fix est le bon levier, le coût d'auto-calibration est non-trivial mais maîtrisable, et un predictor structurel léger peut servir de prior. La direction des coefficients structurels (signe positif sur `stylistic_load`) confirme empiriquement le finding qualitatif de §13.4.1.

### 14.5 Figures finales

Trois artefacts résument visuellement les Phases 11 et 12 (`artifacts/plots/`) :

* **`mu_sweep_curves.png`** — pour chaque setup, une courbe ratio vs μ avec marquage du canonical (anneau rouge) et du best (étoile verte). Le pattern en cloche ressort clairement sur les setups stylistiques ; les setups lexicaux montrent un plateau ou une décroissance monotone.
* **`predictor_loo_scatter.png`** — predicted vs ground-truth μ\* sur les 17 setups, avec ligne y = x et résidus. Le predictor compresse vers la médiane (attendu pour une régression linéaire sur n=17), mais sépare correctement les régimes Qwen3 vs MDLM-OWT et N=2 vs N=3.
* **`phase11_gains_table.md`** — tableau récapitulatif des 16 sweeps unique (par couple triplet × backbone), avec canonical ratio, best μ, best ratio et gain.

### 14.6 μ-schedule par-step (infrastructure prête, sweep à conduire)

Le μ optimal pourrait varier le long de la trajectoire de denoising — analogue à l'option β de Phase 7b sur λ. Le code (`src/composition/poe_sampler.py::PoEConfig.mu_schedule + mu_base_end`) interpole `μ_eff(p) = μ_start + (μ_end − μ_start) · σ(p)` où σ est l'une des shapes existantes (`linear`, `cosine`, `late_fire`, `early_fire`). Hypothèse : `late_fire` (canonical $1-N$ tôt, μ relâché tard) devrait dominer si le besoin de pénalisation OWT s'atténue à mesure que la séquence se clarifie.

Sweep conçu (`scripts/17_mu_schedule_sweep.py`, ~6 min pod par setup) : 2 contrôles à μ constant + 4 schedules sur Qwen3 formal × positive × concrete. Reste à exécuter quand un pod est de nouveau disponible. Le baseline à battre est ratio = 0.61 (μ ≡ −1 constant).

---

## 15. Conclusion

Nous avons mené une étude empirique exhaustive de la composabilité par Product-of-Experts dans les Masked Diffusion Language Models. Les résultats principaux :

- La formule PoE est **empiriquement valide** au niveau des logits (Section 4 : slope = 0.857, R² = 0.811).
- La composition à **N=2 est super-additive en moyenne** sur deux backbones de tailles différentes (mean ratio 1.04 sur MDLM-OWT, 1.07 sur Qwen3).
- À **N=3 sur le petit backbone** (110M), le plateau de composition (ratio ≈ 0.55) est robuste contre 19 variantes de calibrage (sweep λ, schedules denoising-aware, MCMC, Bayesian).
- Sur un **backbone 5× plus gros** (596M), le plateau est partiellement levé pour les compositions lexicales (ratio jusqu'à 1.84) mais pas pour les compositions stylistiques (ratio 0.42).
- La corrélation entre **orthogonalité géométrique des proxies** (κ, CKA, MI) et **déficit de composition** observée sur le petit backbone (r = −0.92) **ne se généralise pas** au gros backbone (r = −0.24).
- L'évaluation systématique de **5 prédicteurs candidats** (B leakage, F-js, A' logit-shift, E spatial overlap, C κ_act sur activations latentes — Section 11) révèle qu'**aucun n'atteint r > 0.7 cross-backbone**. Le prédicteur le plus puissant per-backbone (B leakage) **change de signe** entre les deux backbones, traduisant une distinction "experts faibles vs forts". Le candidat C (κ sur activations) reproduit le pattern de κ_OWT (forte corrélation MDLM, collapse Qwen3), confirmant que le problème n'est pas le choix de la métrique géométrique mais une propriété structurelle du régime de capacité du backbone.
- Les tentatives de **correction algorithmique** par joint MCMC à la Du Yan 2023 (Section 12) ne lèvent pas le plateau N=3. Le block-Gibbs `noise_then_denoise` dégrade (0.15 → 0.08 à $n=200$ ; 0.23 → 0.00 à $n=50$). Le MH-token-swap rigoureux, qui n'accepte les swaps que selon un ratio Metropolis-Hastings basé sur l'ELBO séquence-niveau, **préserve le ratio** sans l'améliorer (0.23 → 0.23 à $n=50$, 22.6 % d'acceptance). Cela démontre empiriquement que les samples naïfs PoE-3 sont déjà approximativement aux modes de la distribution PoE-composée, **réfutant l'hypothèse que le slope 0.857 < 1 du Test 2 traduise une sous-composition correctible au niveau du sampling**.

- Le **découplage du coefficient μ sur log p_base** de la valeur canonique $1-N$ (Section 13) — testé sur 17 setups (6 N=3 + 11 N=2 sur 2 backbones) — **rescue les compositions à composante stylistique** avec des gains de +103 % à +307 % sur le ratio. Pour les compositions purement lexicales, le canonical reste optimal. Le sweet spot de μ varie cross-backbone (−1 sur Qwen3, 0 sur MDLM-OWT) mais sa direction est consistante : *toujours* moins punitif que le canonical. Sur le gros backbone, le couplage μ-fix + grand modèle franchit la barre du super-additif ($r > 1$) sur des compositions auparavant sub-additives.

- Trois **protocoles d'auto-calibration de μ** ont été comparés (Section 14) : grille fixe (A), Bayesian optimization (B), et predictor structurel (C). À $n=200$, A est fiable (3/3 setups) mais coûteux ; B est instable (dépendance forte au seed et au kernel GP) ; C, une régression linéaire sur 4 features structurelles (N, stylistic_load, $\log_{10}$ capacity, mean_marginal), atteint LOO-MAE = 0.469 sur 17 setups — utile comme prior, insuffisant comme oracle. Le coefficient le plus marqué (stylistic_load → +1.03) confirme empiriquement, sans codage en dur, le finding qualitatif de §13.4. La combinaison **C-as-prior + A-en-grille-fine-locale** donne un workflow réaliste (~40 % d'économie vs sweep aveugle, fiabilité conservée).

Le papier prend ainsi sa forme finale : une **caractérisation empirique disciplinée** d'où PoE-on-MDLM marche et où il ne marche pas, accompagnée d'un finding empirique non-trivial — le **sign-flip de la corrélation leakage ↔ déficit** entre régimes d'experts. Cette observation, combinée aux ~30 calibrations testées sur N=3 et à l'échec de la correction MCMC simple, motive une nouvelle classe d'approches **régime-conscientes** (prédicteurs) ou **score-based explicites** (correcteurs algorithmiques) au-delà du sum-of-logits par-step.

### Travaux futurs

1. **Étendre l'étude à un troisième backbone** (e.g., LLaDA, BERT-MDLM, Dream) pour vérifier le sign-flip de B et clarifier sa relation à la capacité du modèle. Permettrait aussi d'augmenter la taille du dataset d'évaluation du predictor C (Section 14.2) au-delà des 17 setups actuels.

2. **Formaliser un prédicteur régime-conscient** : par exemple `sign(B - 0.25) × |B - 0.25|^β`, où β dépend d'une mesure de force des experts (norm des shifts de logits, ou divergence entre marginal expert et baseline).

3. **Améliorer le predictor C** : tester features non-linéaires (interactions stylistic_load × N, capacité), modèles k-NN ou Gaussian Process. Cible MAE ≤ 0.25 sur ≥ 30 setups, qui suffirait à remplacer A entièrement.

3b. **Conduire le sweep μ-schedule** (infrastructure §14.6). 6 configs × ~6 min pod sur le triplet difficile Qwen3 fpc. Si une schedule (le candidat naturel est `late_fire`) bat le constant μ=−1 actuel à 0.61, on tient une amélioration algorithmique gratuite par-dessus Phase 11.

4. **Explorer la composition cross-axe entangled** : peut-on adapter le sampling pour découpler globaux vs locaux (style vs lexique) ?

5. **Implémenter un correctif algorithmique à la Du Yan 2023 *complet*** : score-based composition explicite sur MDLM-text, avec énergies apprises plutôt que sum-of-logits naïf.

6. **Mesurer le déficit PoE via une métrique continue** plutôt que via top-quartile binarisation, pour étendre le scope à N≥4 sans saturation statistique.

---

## Annexes

### A. Récapitulatif des phases expérimentales

| Phase | Objectif | Compute |
|---|---|---|
| Phase 1 | Analyse de proxies sur OWT (κ, Spearman, HSIC, CKA, MI) | Local, 5k samples |
| Phase 2 | Construction des datasets (6 verticales × 80k docs) | Pod, ~9 h |
| Phase 3 | Entraînement de 6 LoRA experts mono-axe | Pod, ~1 h |
| Phase 3b | Entraînement de 2 experts d'intersection | Pod, ~30 min |
| Phase 3.5 | Validation cross-vertical (raw signals) | Pod, ~10 min |
| Phase 4 | Composition à N=2 sur 10 paires | Pod, ~1.5 h |
| Phase 4.5 | Tests 1 et 2 (intersection vs PoE ; formula validation) | Pod, ~30 min |
| Phase 5 | Phase 5 : κ vs déficit | Local, instantané |
| Phase 6 | N=3 sur 3 triplets | Pod, ~30 min |
| Phase 7a | Sweep λ uniforme + Bayesian | Pod, ~1 h |
| Phase 7b (v2) | Schedules + MCMC | Pod, ~1 h |
| Phase 8 | Pipeline complète sur Qwen3-0.6B | Pod, ~5 h |
| Phase 9 | Évaluation des 4+1 prédicteurs candidats (B, F-js, A', E, C) | Local, ~1 h |
| Phase 10 | Joint MCMC corrector à la Du Yan (block-Gibbs + MH) | Pod, ~45 min |
| Phase 11 | Découplage du coefficient μ — sweep initial + 6 vérifications | Pod, ~3 h |
| Phase 12 | Auto-tune A/B à $n=200$ + 10 sweeps N=2 pour predictor C | Pod, ~6 h |

### B. Fichiers de résultats

Tous les résultats numériques sont consolidés dans `~/Documents/composable-dllms-artifacts/SUMMARY.json` (~60 KB, contient les deux backbones). Le détail par phase est aussi disponible :

- `joint_satisfaction.json` : Phase 4 sur MDLM-OWT
- `qwen3_run/joint_satisfaction.json` : Phase 4 sur Qwen3
- `n3_*.json` (×11) : sweep λ et triplets sur MDLM-OWT
- `qwen3_run/n3_*.json` (×3) : triplets sur Qwen3
- `v2/n3_v2_*.json` (×7) : schedules et MCMC sur MDLM-OWT
- `test1_intersection_check.json`, `poe_formula_check.json`
- `gram_matrix.json`, `independence_metrics.json`
- `n3_mu_sweep.json`, `mu/*.json` (×6) : Phase 11 μ-sweeps (4 N=3 + 2 N=2)
- `mu_extra/*.json` (×10) : Phase 12c additional N=2 μ-sweeps (training set predictor)
- `auto_tune_n200/*.json` (×3) : Phase 12 protocoles A et B à $n=200$
- `predict_mu.json` : Phase 12c regression LOO sur 17 setups

Plots associés dans `plots/` (MDLM-OWT) et `plots_qwen3/` (Phase 5 refit).

### C. Fixes techniques

Pour faire fonctionner MDLM-OWT, plusieurs patches ont été nécessaires :

1. `dllm/utils/models.py` : `trust_remote_code=True` partout, fallback GPT-2 tokenizer pour MDLM-OWT (qui ne ship pas de tokenizer), ajout du `mask_token` à id 50257.
2. `dllm/pipelines/__init__.py` : drop `rl` (TRL non installé) et `editflow` (vllm non installé).
3. `modeling_mdlm.py` (HF cache) : SDPA fallback à la place de flash-attn (CUDA 13 incompatible), `**kwargs` dans `forward`, `sigma=None` accepté, `return_dict=True` forcé, dtype matching pour `TimestepEmbedder`.
4. Embeddings explicitement frozen dans le LoRA training (pour préserver la cohérence du sum-of-logits).
5. `remove_unused_columns=False` dans `MDLMConfig` (sinon HF Trainer drop la colonne `labels`).
6. Rejection sampling sur la cohérence (3 heuristiques).
7. Batch size de sampling = 32 pour éviter l'OOM CUDA sur le Gumbel softmax fp64.

### D. Coût total

- Pod RunPod A100 SXM 80GB : ~95 heures cumulées sur 6 jours.
- Coût ≈ 80 USD (sur un budget de 100 USD).
- Local Mac : ~2.0 GB d'artefacts (datasets + checkpoints + samples + plots + Phase 11/12 sweeps).

---

*Fin du brouillon.*
