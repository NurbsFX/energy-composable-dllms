"""Non-linear dependence metrics on the proxy-energy samples.

Complements the linear orthogonality index κ of :mod:`gram_matrix` with three
additional dependence measures, each filling a different gap:

* **Spearman rank correlation** — bounded in [-1, 1], detects *monotone*
  non-linear coupling (e.g. ``Y = exp(X)`` gets ρ_s ≈ 1 while Pearson is
  strictly < 1). Cheap and robust.
* **HSIC** with an RBF kernel and the median heuristic for the bandwidth,
  plus the normalised variant **CKA** = HSIC(K, L) / sqrt(HSIC(K, K) ·
  HSIC(L, L)) ∈ [0, 1] which detects *any* non-linear coupling and is
  comparable across pairs with different marginals.
* **Mutual information** via the Kraskov-Stögbauer-Grassberger (KSG) k-NN
  estimator from scikit-learn — information-theoretic, also fully
  non-linear, used as an independent cross-check.

Conceptually they form a hierarchy from "Pearson catches it" to "any
dependence catches it":

    κ        — linear only
    Spearman — linear + monotone non-linear
    CKA / HSIC / MI — linear + monotone non-linear + non-monotone non-linear

Memory note. We cache one ``(n, n)`` centred kernel matrix per energy:
  6 energies × 5000² × 8 bytes ≈ 1.2 GB.
For substantially larger ``n`` the implementation should switch to a
streaming or block-wise pairwise computation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _rbf_centered_kernel(x: np.ndarray) -> np.ndarray:
    """Double-centred RBF kernel matrix; bandwidth from the median heuristic."""
    n = x.shape[0]
    sq = (x[:, None] - x[None, :]) ** 2
    triu = sq[np.triu_indices(n, k=1)]
    sigma2 = max(float(np.median(triu)) / 2.0, 1e-12) if triu.size else 1.0
    K = np.exp(-sq / (2.0 * sigma2))
    return K - K.mean(0) - K.mean(1)[:, None] + K.mean()


def hsic(x: np.ndarray, y: np.ndarray) -> float:
    """Empirical HSIC. ≥ 0; vanishes iff X ⊥ Y in the RBF-kernel limit."""
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"length mismatch: {x.shape[0]} vs {y.shape[0]}")
    n = x.shape[0]
    Kc = _rbf_centered_kernel(x)
    Lc = _rbf_centered_kernel(y)
    return float((Kc * Lc).sum() / (n - 1) ** 2)


def mutual_info(x: np.ndarray, y: np.ndarray, *, n_neighbors: int = 3, seed: int = 0) -> float:
    """KSG mutual-information estimator (in nats). ≥ 0; vanishes iff X ⊥ Y."""
    from sklearn.feature_selection import mutual_info_regression

    return float(
        mutual_info_regression(x.reshape(-1, 1), y, n_neighbors=n_neighbors, random_state=seed)[0]
    )


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation. ∈ [-1, 1]; ±1 iff Y is a monotone function of X.

    Detects monotone non-linear dependence that Pearson misses (e.g.
    ``Y = exp(X)``), but is blind to non-monotone non-linear dependence
    (e.g. ``Y = X²`` with X centred).
    """
    from scipy.stats import spearmanr

    rho, _ = spearmanr(x, y)
    return float(rho)


@dataclass
class IndependenceResult:
    energy_names: list[str]
    pair_spearman: dict[tuple[str, str], float]  # ∈ [-1, 1]; monotone non-linear
    pair_hsic: dict[tuple[str, str], float]  # raw HSIC; scale-dependent
    pair_cka: dict[tuple[str, str], float]  # normalised HSIC ∈ [0, 1]
    pair_mi: dict[tuple[str, str], float]  # KSG mutual information, nats

    def to_json(self) -> dict:
        encode = lambda d: {f"{a}|{b}": float(v) for (a, b), v in d.items()}  # noqa: E731
        return {
            "energy_names": self.energy_names,
            "pair_spearman": encode(self.pair_spearman),
            "pair_hsic": encode(self.pair_hsic),
            "pair_cka": encode(self.pair_cka),
            "pair_mi": encode(self.pair_mi),
        }


def compute_independence(
    E_matrix: np.ndarray, energy_names: list[str], *, mi_seed: int = 0
) -> IndependenceResult:
    """Spearman, HSIC, CKA and MI for every unordered pair of columns in ``E_matrix``."""
    if E_matrix.ndim != 2:
        raise ValueError(f"E_matrix must be 2-D, got shape {E_matrix.shape}")
    n, k = E_matrix.shape
    if len(energy_names) != k:
        raise ValueError(f"len(energy_names)={len(energy_names)} ≠ E_matrix.shape[1]={k}")

    Kcs = [_rbf_centered_kernel(E_matrix[:, i]) for i in range(k)]
    hsic_self = [float((Kc * Kc).sum() / (n - 1) ** 2) for Kc in Kcs]

    pair_spearman: dict[tuple[str, str], float] = {}
    pair_hsic: dict[tuple[str, str], float] = {}
    pair_cka: dict[tuple[str, str], float] = {}
    pair_mi: dict[tuple[str, str], float] = {}
    for i in range(k):
        for j in range(i + 1, k):
            pair = (energy_names[i], energy_names[j])
            pair_spearman[pair] = spearman(E_matrix[:, i], E_matrix[:, j])
            h = float((Kcs[i] * Kcs[j]).sum() / (n - 1) ** 2)
            pair_hsic[pair] = h
            denom = np.sqrt(hsic_self[i] * hsic_self[j])
            pair_cka[pair] = float(h / denom) if denom > 0 else float("nan")
            pair_mi[pair] = mutual_info(E_matrix[:, i], E_matrix[:, j], seed=mi_seed)

    return IndependenceResult(
        energy_names=list(energy_names),
        pair_spearman=pair_spearman,
        pair_hsic=pair_hsic,
        pair_cka=pair_cka,
        pair_mi=pair_mi,
    )
