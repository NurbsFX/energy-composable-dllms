# Next steps — Plan de match du papier

> Document de pilotage stratégique. Mis à jour au fil de l'eau.
> Dernière mise à jour : 2026-05-02.

## État actuel — étude empirique close

L'étude expérimentale est **terminée**. Le papier porte désormais une vraie
contribution algorithmique (μ-fix Phase 11) accompagnée d'une étude honnête de
sa portée pratique (auto-tune Phase 12). Voir PAPER_DRAFT.md §13–§14.

**Reste à faire** : finalisation rédactionnelle (figures, traduction
anglais ?), choix de venue, soumission.

## TL;DR — narratif final du papier

1. PoE-MDLM marche bien à N=2 (super-additif moyen sur 2 backbones).
2. À N=3, plateau robuste contre 19 calibrations sur le petit backbone ; gros backbone le lève partiellement.
3. La géométrie OWT (κ, CKA, MI) prédit le déficit sur le petit backbone (r=−0.92) mais collapse sur le gros (r=−0.24).
4. 5 prédicteurs candidats (B, F-js, A', E, C) testés cross-backbone — tous échouent à r > 0.7. Sign-flip de B entre régimes d'experts.
5. Joint MCMC (block-Gibbs + MH rigoureux) ne lève pas le plateau ; les samples PoE-3 naïfs sont déjà aux modes.
6. **Découplage du coefficient μ sur log p_base** rescue les compositions stylistiques : +103 % à +307 % sur le ratio (Phase 11, 17 setups).
7. **Auto-tuning de μ** (Phase 12) : grille à n=200 fiable mais coûteuse, BO instable, predictor structurel à MAE 0.469 (utile comme prior).
8. **μ-schedule par-step** (Phase 12d, 2026-05-03) : schedules ne battent pas constant ; la phase early du denoising fixe le résultat. Un seul scalaire global suffit.

---

## Pourquoi N=2-3 est le sweet spot

| Scope | Faisabilité | Intérêt scientifique | Compétitivité |
|---|---|---|---|
| N=2 seul | trivial à montrer | faible (déjà fait pour AR, images) | très peu compétitif |
| **N=2-3 avec prédicteur stable** | **réaliste avec ce qu'on a** | **élevé (gap réel non comblé)** | **publiable, distinguant** |
| N=3 avec correcteur algorithmique | possible (~$5-10 compute) | très élevé (vraie contribution algo) | conférence majeure si propre |
| N≥4 | difficile, fondamentalement limité | haut mais risqué | papier risqué |

**N=2-3 stable** est le scope où on a :
- Des données qui montrent qu'il y a un vrai problème (variance par paire, κ-collapse cross-backbone)
- Une méthode raisonnable pour le résoudre (prédicteurs alternatifs)
- Des applications pratiques immédiates (compositional text generation, guided sampling, DLLM-based agents)

---

## Les 2 piliers du papier final

### Pilier 1 — Prédicteur stable du déficit PoE (priorité 1, gratuit à tester)

C'est le point central manquant après Phase 8. On a 4 candidats identifiés (cf. PAPER_DRAFT.md §10.3) :

| Candidat | Quoi | Coût |
|---|---|---|
| **A. Orthogonalité des shifts LoRA** | `cos(ΔW_a, ΔW_b)` directement sur les matrices de poids | gratuit |
| **B. Leakage cross-axis** | marginal de l'expert A sur l'axe B (depuis les samples existants) | gratuit |
| **C. κ sur activations latentes** | κ dans le modèle plutôt que sur OWT brut | ~30 min compute local |
| **D. Energy curvature** | `JS_PoE − (JS_a + JS_b − JS_baseline)` (2e dérivée) | gratuit |

**Effort total** : ~1 heure de Python local sur le Mac (artefacts déjà téléchargés dans `~/Documents/composable-dllms-artifacts/`).

**Critère de succès** : trouver au moins une métrique avec **r > 0.7 cross-backbone** sur 10 paires (MDLM-OWT + Qwen3 = 20 points).

**Si c'est trouvé** → contribution centrale du papier :
> *"Le prédicteur du déficit PoE n'est pas l'orthogonalité des proxies sur les données brutes, mais X — qu'on valide cross-backbone."*

### Pilier 2 — Correcteur simple à N=3 (priorité 2, ~$5-10 si nécessaire)

Optionnel selon le résultat du Pilier 1 :

| Option | Ce que ça fait | Coût | Risque |
|---|---|---|---|
| Joint MCMC à la Du Yan (texte) | Réduit l'erreur per-step à N=3 | ~$5-10 pod | moyen |
| λ-schedule par-position (pas par-step) | Active la composition seulement où elle compte | ~$3-5 pod | faible |
| Score matching explicite (sortir de logit-sum) | Vraie composition energy-based | ~10h dev + ~$10 | élevé |

Pas obligatoire si le prédicteur seul est convaincant. Le correcteur transformerait le papier de "diagnostique" à "algorithmique" (publication conf majeure).

---

## Plan de match étape par étape

### Étape 1 — Tester les 4 prédicteurs sur artefacts existants (gratuit, ~1h)

**À faire** :
1. Coder un script `scripts/10_predictor_eval.py` qui calcule chacun des 4 prédicteurs (A, B, C, D) sur les 10 paires × 2 backbones.
2. Pour chaque prédicteur, calculer la corrélation Pearson r avec le déficit PoE observé, séparément sur MDLM-OWT et Qwen3, puis cross-backbone (20 points).
3. Reporter dans `~/Documents/composable-dllms-artifacts/predictor_eval.json`.

**Sources de données** :
- LoRA checkpoints : `~/Documents/composable-dllms-artifacts/checkpoints/` (MDLM-OWT) et `checkpoints_qwen3/` (Qwen3)
- Samples : `samples/` et `qwen3_run/samples/`
- Joint satisfaction : `joint_satisfaction.json` (MDLM) et `qwen3_run/joint_satisfaction.json` (Qwen3)
- Métriques d'indépendance pré-calculées : `gram_matrix.json`, `independence_metrics.json`

**Critère de décision** :
- Si l'un des 4 a r > 0.7 sur les 20 points cross-backbone → **prédicteur trouvé**, on passe à Étape 3.
- Sinon → Étape 2.

### Étape 2 — Décision selon Étape 1

**Cas A : prédicteur trouvé**
- Direct passage à Étape 3 (rédaction).
- Papier visé : NeurIPS/ICLR Workshop ou Findings (EMNLP/ACL).
- Coût restant : 0 USD pod.

**Cas B : aucun prédicteur ne fonctionne**
- Choix : (a) papier diagnostique honnête (workshop only), ou (b) ajouter un correcteur algorithmique.
- Si (b) : implémenter joint MCMC à la Du Yan sur N=3, lancer sur le pod (~$10).
- Coût restant : 0–10 USD pod.

### Étape 3 — Rédaction (à partir de PAPER_DRAFT.md)

PAPER_DRAFT.md a déjà une structure complète. À ajouter/modifier :

1. **Nouvelle section** "Cross-backbone predictor evaluation" :
   - Méthodologie (4 candidats)
   - Résultats par candidat (r sur MDLM, sur Qwen3, cross-backbone)
   - Identification du prédicteur stable (si trouvé)

2. **Reformulation de §10 (Discussion)** :
   - Cas A : "We identified [X] as the stable predictor of PoE deficit"
   - Cas B : "All 4 candidates fail cross-backbone — open problem"

3. **Conclusion mise à jour** selon Étape 1.

4. **Figures finales** :
   - κ-vs-deficit panel (déjà dans plots/)
   - Nouveau prédicteur panel (à générer)
   - Phase 5 fits side-by-side (MDLM vs Qwen3) pour montrer le collapse

5. **Traduction anglais** (en option, selon venue cible).

---

## Position scientifique attendue selon résultat

### Cas A — prédicteur trouvé

> *"We show empirically that the orthogonality of proxy energies on raw data does NOT predict PoE composition deficit cross-backbone. Instead, we identify [X] as a backbone-invariant predictor (r > 0.7 on 2 backbones, 10 pairs), enabling reliable selection of compositional axes for N=2-3 in DLLM."*

**Caractère** : contribution algorithmique-pratique.

**Venues possibles** :
- NeurIPS Workshop (ML for compositional generation, diffusion track)
- ICLR Workshop (efficient generative models)
- Findings of EMNLP/ACL
- Possiblement main conf si combiné avec un correcteur (Pilier 2)

### Cas B — aucun prédicteur ne fonctionne

> *"We exhaustively test 4 candidate predictors of PoE composition deficit on DLLM. None reproduce a strong cross-backbone correlation. This negative result documents a fundamental open problem in compositional generation: backbone-invariant predictors of PoE composability remain elusive, motivating future work on learned composition or score-based explicit methods."*

**Caractère** : finding diagnostique honnête.

**Venues possibles** :
- NeurIPS/ICLR Workshop diagnostique (ex : Workshop on Failure Modes, Workshop on Compositional Generation)
- ACL Findings
- arXiv-only si on veut juste documenter

---

## Recommandation immédiate

**Lancer Étape 1 maintenant** — ~1h de code Python local, sans pod, sans compute supplémentaire. Le résultat **change la trajectoire du papier** et c'est gratuit à obtenir.

**Action concrète** : créer `scripts/10_predictor_eval.py` qui implémente les 4 prédicteurs et les évalue cross-backbone. Charger LoRA weights, samples, joint_satisfaction depuis les artefacts locaux. Output dans `predictor_eval.json` + figures dans `plots/predictors/`.

---

## Suivi des budgets

- Pod RunPod : ~80 USD dépensés sur 100 USD initial (~95h cumulées sur 6 jours)
- Mac compute disponible : illimité pour tâches CPU-light
- Datasets locaux : 1.4 GB
- LoRA checkpoints locaux : 358 MB (MDLM) + 1.2 GB (Qwen3) = 1.55 GB
- Total artefacts locaux : ~3.2 GB (incluant Phase 11/12 sweeps)

Étude empirique close. Compute restant alloué à d'éventuelles relances ciblées (figure polish, replication on demand).

---

## Décisions actives à prendre

- [x] Lancer Étape 1 (4 prédicteurs cross-backbone) — fait, scripts/10 et /11
- [x] Tester le candidat C (κ activations latentes) — fait, scripts/12
- [x] Tester un correcteur algorithmique simple (block-Gibbs Du Yan-style) — fait, scripts/13
- [x] Tester `mh_token_swap` rigoureux — fait, n=50, ratio 0.23 → 0.23 (réfute l'hypothèse Test 2)
- [x] **Phase 11** — découplage du coefficient μ : sweep initial + 6 vérifications (4 N=3 + 2 N=2 sur 2 backbones)
- [x] **Phase 12** — auto-tuning de μ : protocoles A (grille n=200, 3/3), B (Bayesian opt, instable), C (predictor structurel, LOO-MAE 0.469 sur 17 setups)
- [x] **Phase 12d** — μ-schedule per-step : 6 configs sur Qwen3 fpc, schedules ne battent pas constant (early μ fixe le résultat)
- [x] **Décision finale Paper 1** : papier passe de "diagnostique" à "diagnostique + algorithmique avec étude de calibrage". Étude empirique close.

## Pendings pour Paper 1 — pod-gated

### A2 OK — conversion AR→MDLM est reproductible verbatim sur Qwen3-1.7B

Vérifié le 2026-05-03 : le script `external/dllm/dllm/pipelines/a2d/convert.py` est
**size-agnostic** :
- Le mapping `A2D_CONFIG_MAP["qwen3"]` couvre toutes les variantes Qwen3-* (model_type=`qwen3`)
- La config A2D est construite à partir de `src_config.to_dict()` — pas d'hyperparamètre size-dependent
- Le state_dict est chargé `strict=False` — la conversion préserve l'arch en ajoutant des composants spécifiques A2D (modules MDLM)

Le script de pretraining `external/dllm/examples/a2d/mdlm/pt.py` est aussi
size-agnostic :
- lr=1e-4, num_epochs=20, batch_size=16 — non size-dependent
- Modèle chargé via `dllm.utils.get_model(model_args)` — auto-arch
- Trainer = `MDLMTrainer` HF générique

**Seul ajustement attendu sur Qwen3-1.7B** : la taille mémoire. 1.7B vs 0.6B = ~3×
plus gros. Sur A100 80GB, prévoir `per_device_train_batch_size=8` ou
gradient_accumulation_steps=2 pour rester dans le budget mémoire.

Source : `Qwen/Qwen3-1.7B` est disponible sur HF Hub.

| Phase | Action | Coût pod | Pré-requis |
|---|---|---:|---|
| B0 | Conversion AR→MDLM Qwen3-1.7B | $30-50 | A2 OK ✓ |
| B1 | 6 LoRA experts sur Qwen3-1.7B-MDLM | $15-20 | B0 |
| B2 | PoE-2 sweep Qwen3-1.7B (10 paires) | $5-8 | B1 |
| B3 | PoE-3 Qwen3-1.7B (3 triplets) | $3-5 | B1 |
| B4 | μ-sweep Qwen3-1.7B sur fpc | $5-8 | B1 |
| C2 | μ-sweep complémentaire triplet lexical | $3-5 | éventuellement |
| **Total Phase B/C pod** | | **~$60-100** | |

**Décision pré-pod** : valider A1 (venue + deadline) et A2 (reproducibility conversion) localement avant tout pod re-up.

## Pendings pour Paper 2 (SEDD) — pod-gated

| Test | Discriminateur | Coût pod |
|---|---|---:|
| (a) μ-sweep sur triplet homogène `positive × concrete × sports` | Disambigue §10.6 : paradigm-level vs selectivity-driven | ~$5 |
| (b) μ-sweep sur autre style anchor `long × positive × concrete` | Robustness sur style non-formal | ~$5 |
| (optionnel) Re-train formal SEDD seul +5000 steps | Tests si formal-faible est undertraining | ~$1 |
| (optionnel) Prompted SEDD repeat | Cleane le caveat unconditional | ~$5 |

§10.8/(c) **DONE local** : ratio = 0.79 + 1.26·C, Pearson r = +0.74 sur 15 paires (commit `c3a870c`). Le finding load-bearing du Paper 2 est verrouillé sans pod ; (a)+(b) sont des tests de robustesse complémentaires.

## Pendings nécessitant intervention humaine

- **B6** : évaluation humaine légère. 30-50 samples × 4 conditions × 2-3 labellers, accord avec proxy energies (target > 70%). Préparation du form-template fait localement, mais le labeling lui-même ne peut être fait sans humains.
- [ ] **A1** Choisir venue cible (workshop vs Findings vs main conf) — voir tableau ci-dessous.

#### Options de venue pour Paper 1 (à confirmer par l'utilisateur)

> **Caveat** : ces dates sont des estimations basées sur les patterns récents (cycles ARR/EMNLP/ACL/NAACL). À vérifier sur les sites officiels avant de commiter sur l'une.

| Venue | Cycle typique | Fit pour Paper 1 | Notes |
|---|---|---|---|
| **NeurIPS Workshop (ML for compositional generation, diffusion track)** | Submission ~août, accept ~sept-oct | Excellent pour étude empirique compositionnelle, audience natural | Court (4-6 pages), pas archival, faible risque |
| **ICLR Workshop** | Submission ~fév-mars, accept ~mars-avr | Idem | Permet d'écrire à l'aise sans pression de la main conf |
| **EMNLP Findings 2026** | Submission via ARR (rolling) ; commit ~août 2026 | Très bon — audience NLP, valorise les findings empiriques disciplinés | Archival, full paper (long ou short) |
| **ACL Findings 2026** | Idem ARR, commit ~déc 2025 ou plus tard | Idem | Plus prestigieux que EMNLP Findings |
| **NAACL 2027** | Submission ARR ~oct 2026, commit ~jan 2027 | Idem | Bonne fenêtre si extension Qwen3-1.7B prête |
| **ACL 2027 main** | Submission ~jan-fév 2027 | Si Phase B (Qwen3-1.7B) confirme cross-scale | Plus exigeant : nécessite Paper 1 + extensions Phase B/C |

**Recommandation** : si Phase B (Qwen3-1.7B) ne sera pas faite, viser **Findings of EMNLP/ACL 2026** avec le scope actuel (2 backbones + Phase 11 μ-fix + Phase 12 auto-tune). Si Phase B est faite, viser **NAACL 2027 ou ACL 2027 main**.

→ **Action utilisateur** : décider parmi ces options et fixer la deadline. Toutes les autres décisions (Phase B oui/non, traduction anglaise, longueur du papier) en découlent.
- [ ] Décider du langage final (français → anglais ?)
- [ ] Identifier les 1-2 figures-clés à polir pour la soumission (μ-sweep curves, predictor LOO scatter)

## Synthèse Phases 9 → 12

| Phase | Méthode | Résultat |
|---|---|---|
| Phase 9 — 5 prédicteurs | B leakage, F-js, A' logit-shift, E spatial, C κ_act | r ≤ 0.48 cross-backbone (échec). **Sign-flip de B** entre régimes d'experts = finding distinctif. |
| Phase 10a | Block-Gibbs MCMC (`noise_then_denoise`) | dégrade le ratio (0.15 → 0.08) |
| Phase 10b | MH-token-swap rigoureux (sequence-level ELBO) | préserve sans améliorer (0.23 → 0.23, 22.6% accept) — **réfute Test 2 slope < 1** |
| **Phase 11** | **Découplage μ sur log p_base** | **+103 % à +307 % gain ratio sur compositions stylistiques** (4 N=3 + 2 N=2, 2 backbones) |
| Phase 12a | Auto-tune A (grid n=50) + B (BO n=50) | 1/3 et 0–1/3 succès — n=50 trop bruité |
| Phase 12b | Auto-tune A et B à n=200 | A : 3/3 (fiable mais coûteux). B : 1/3 strict (instable). |
| Phase 12c | Predictor C, training set 17 setups | LOO-MAE 0.469. Coef stylistic_load = +1.03 (confirme §13.4 sans codage en dur). |
| Phase 12d | μ-schedule per-step (6 configs sur Qwen3 fpc) | **Schedules ne battent pas constant** — μ early fixe le résultat (0.15 ou 0.61 selon μ_start, indépendant de μ_end). |

→ Le papier porte désormais **une contribution algorithmique** (μ-fix) **avec une étude de portée pratique** (3 protocoles d'auto-tuning, workflow réaliste C-as-prior + A-en-grille-fine).

## Pour la rédaction finale

PAPER_DRAFT.md contient désormais :
- Sections 1–9 : étude empirique multi-phase (Phases 1 à 8)
- Section 10 : discussion + 4 candidats prédicteurs alternatifs
- Section 11 : évaluation empirique des 5 prédicteurs (B, F-js, A', E, C)
- Section 12 : tentative correction MCMC (block-Gibbs + MH)
- Section 13 : Phase 11 — découplage du coefficient μ (rescue stylistic compositions)
- Section 14 : Phase 12 — auto-tuning de μ (protocoles A, B, C ; LOO-MAE 0.469)
- Section 15 : conclusion révisée + travaux futurs
- Annexes : récap phases (14 lignes), fichiers, fixes techniques, coût (~80 USD pod)
