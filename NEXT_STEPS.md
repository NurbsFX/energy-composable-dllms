# Next steps — Plan de match du papier

> Document de pilotage stratégique. Mis à jour au fil de l'eau.
> Dernière mise à jour : 2026-04-29.

## TL;DR

Le scope cible du papier est **N=2-3 avec un prédicteur stable du déficit PoE**. La prochaine étape critique est de **tester localement** quatre prédicteurs candidats sur les artefacts existants pour identifier celui qui généralise cross-backbone. Coût : ~1h de Python local, zéro compute pod.

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

- Pod RunPod restant : ~85 USD non dépensés (sur budget initial 100 USD)
- Pod temps cumulé : ~80h
- Mac compute disponible : illimité pour tâches CPU-light
- Datasets locaux : 1.4 GB
- LoRA checkpoints locaux : 358 MB (MDLM) + 1.2 GB (Qwen3) = 1.55 GB
- Total artefacts locaux : ~3 GB

Tout est en place pour Étape 1 sans coût supplémentaire.

---

## Décisions actives à prendre

- [ ] Lancer Étape 1 (4 prédicteurs cross-backbone)
- [ ] Selon résultat : décider entre rédaction directe ou ajout correcteur
- [ ] Choisir venue cible (workshop vs Findings vs main conf)
- [ ] Décider du langage final (français → anglais ?)
- [ ] Identifier les 1-2 figures-clés à polir pour la soumission
