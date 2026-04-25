"""Linear fit and plot of the per-pair PoE deficit against the orthogonality index κ.

For each expert pair (i, j) we collect:
* κ_ij from the proxy Gram matrix (pre-training).
* JS_ij^PoE: joint satisfaction of the PoE-strict configuration.
* JS_ij^indep = P(A_i) · P(A_j): the perfect-independence reference computed
  from the single-expert runs.

deficit_ij = JS_ij^indep − JS_ij^PoE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KappaDeficitPoint:
    pair: tuple[str, str]
    kappa: float
    js_poe: float
    js_indep: float

    @property
    def deficit(self) -> float:
        return self.js_indep - self.js_poe


@dataclass
class KappaDeficitFit:
    points: list[KappaDeficitPoint]
    pearson_r: float
    pearson_p: float
    slope: float
    intercept: float


def fit(points: list[KappaDeficitPoint]) -> KappaDeficitFit:
    from scipy.stats import linregress, pearsonr

    xs = np.array([p.kappa for p in points])
    ys = np.array([p.deficit for p in points])
    r, p = pearsonr(xs, ys)
    res = linregress(xs, ys)
    return KappaDeficitFit(
        points=points,
        pearson_r=float(r),
        pearson_p=float(p),
        slope=float(res.slope),
        intercept=float(res.intercept),
    )


def plot_kappa_vs_deficit(fit_result: KappaDeficitFit, out_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = np.array([p.kappa for p in fit_result.points])
    ys = np.array([p.deficit for p in fit_result.points])
    ax.scatter(xs, ys, s=60, color="#1f77b4", zorder=3)
    for p in fit_result.points:
        ax.annotate(
            f"{p.pair[0]}×{p.pair[1]}",
            (p.kappa, p.deficit),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    grid = np.linspace(xs.min(), xs.max(), 100)
    ax.plot(
        grid,
        fit_result.slope * grid + fit_result.intercept,
        linestyle="--",
        color="grey",
        label=rf"Pearson $r = {fit_result.pearson_r:.2f}$ (p = {fit_result.pearson_p:.3f})",
    )
    ax.set_xlabel(r"$\kappa_{ij}$ (proxy Gram, pre-training)")
    ax.set_ylabel(
        r"deficit $= \mathrm{JS}^{\mathrm{indep}}_{ij} - \mathrm{JS}^{\mathrm{PoE}}_{ij}$"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
