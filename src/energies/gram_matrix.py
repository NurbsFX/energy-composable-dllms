"""Empirical Gram matrix and orthogonality index κ on samples of energy values."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GramResult:
    energy_names: list[str]
    G: np.ndarray  # covariance matrix, shape (k, k)
    C: np.ndarray  # Pearson correlation matrix, shape (k, k)
    kappa_global: float
    pair_kappas: dict[tuple[str, str], float]

    def to_json(self) -> dict:
        return {
            "energy_names": self.energy_names,
            "G": self.G.tolist(),
            "C": self.C.tolist(),
            "kappa_global": float(self.kappa_global),
            "pair_kappas": {f"{a}|{b}": float(v) for (a, b), v in self.pair_kappas.items()},
        }


def kappa_from_gram(G: np.ndarray) -> float:
    """κ = ||G − diag(G)||_F / tr(G). Returns NaN if tr(G) ≤ 0."""
    tr = float(np.trace(G))
    if tr <= 0:
        return float("nan")
    off = G - np.diag(np.diag(G))
    return float(np.linalg.norm(off, ord="fro") / tr)


def compute_gram(E_matrix: np.ndarray, energy_names: list[str]) -> GramResult:
    if E_matrix.ndim != 2:
        raise ValueError(f"E_matrix must be 2-D, got shape {E_matrix.shape}")
    k = E_matrix.shape[1]
    if len(energy_names) != k:
        raise ValueError(f"len(energy_names)={len(energy_names)} ≠ E_matrix.shape[1]={k}")

    G = np.cov(E_matrix, rowvar=False)
    C = np.corrcoef(E_matrix, rowvar=False)

    pair_kappas: dict[tuple[str, str], float] = {}
    for i in range(k):
        for j in range(i + 1, k):
            sub = G[np.ix_([i, j], [i, j])]
            pair_kappas[(energy_names[i], energy_names[j])] = kappa_from_gram(sub)

    return GramResult(
        energy_names=list(energy_names),
        G=G,
        C=C,
        kappa_global=kappa_from_gram(G),
        pair_kappas=pair_kappas,
    )


def spectral_decomposition(G_pair: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.linalg.eigh(G_pair)
