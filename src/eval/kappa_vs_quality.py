"""Linear fit of the per-pair PoE deficit against a chosen dependence metric.

For each expert pair (i, j) we collect its dependence metric value (κ, |ρ_s|,
CKA or MI) measured on ``p_base`` *before* training, and the joint-satisfaction
deficit measured *after* composition:

    deficit_ij = JS_ij^indep − JS_ij^PoE

with JS_ij^indep = P(A_i) · P(A_j) the perfect-independence reference computed
from the single-expert runs and JS_ij^PoE the joint satisfaction of the
PoE-strict configuration.

Because we expect to fit on N = 6–15 pairs only, every regression also reports
a 95% bootstrap CI on Pearson r and on the slope, plus a jackknife (drop one
pair at a time) that flags high-leverage points whose removal changes the
slope materially.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Metric name → attribute on MetricDeficitPoint that stores its value.
METRIC_ATTRS: dict[str, str] = {
    "kappa": "kappa",
    "spearman": "spearman_abs",
    "cka": "cka",
    "mi": "mi",
}

# Human-readable axis labels.
METRIC_LABELS: dict[str, str] = {
    "kappa": r"$\kappa_{ij}$ (Gram, linear)",
    "spearman": r"$|\rho_{s,ij}|$ (Spearman, monotone)",
    "cka": r"$\mathrm{CKA}_{ij}$ (kernel, full non-linear)",
    "mi": r"$\mathrm{MI}_{ij}$ (KSG, nats)",
}


@dataclass
class MetricDeficitPoint:
    """One pair of experts, with all four dependence metrics + the deficit."""

    pair: tuple[str, str]
    kappa: float
    spearman_abs: float  # |ρ_s|; we treat sign as informational only
    cka: float
    mi: float
    js_poe: float
    js_indep: float

    @property
    def deficit(self) -> float:
        return self.js_indep - self.js_poe


@dataclass
class MetricDeficitFit:
    points: list[MetricDeficitPoint]
    metric_name: str
    pearson_r: float
    pearson_p: float
    slope: float
    intercept: float
    pearson_r_ci95: tuple[float, float]
    slope_ci95: tuple[float, float]
    jackknife_pearson_r: list[float]  # one entry per leave-one-out
    jackknife_slopes: list[float]

    @property
    def jackknife_r_range(self) -> tuple[float, float]:
        if not self.jackknife_pearson_r:
            return (float("nan"), float("nan"))
        return (min(self.jackknife_pearson_r), max(self.jackknife_pearson_r))


def _xs_for_metric(points: list[MetricDeficitPoint], metric: str) -> np.ndarray:
    if metric not in METRIC_ATTRS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {sorted(METRIC_ATTRS)}")
    attr = METRIC_ATTRS[metric]
    return np.array([getattr(p, attr) for p in points], dtype=np.float64)


def fit(
    points: list[MetricDeficitPoint],
    *,
    metric: str = "kappa",
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> MetricDeficitFit:
    """Pearson + OLS slope/intercept, with bootstrap CIs and a jackknife."""
    from scipy.stats import linregress, pearsonr

    if len(points) < 3:
        raise ValueError(f"need ≥ 3 points to fit, got {len(points)}")

    xs = _xs_for_metric(points, metric)
    ys = np.array([p.deficit for p in points], dtype=np.float64)

    r, p = pearsonr(xs, ys)
    res = linregress(xs, ys)

    rng = np.random.default_rng(seed)
    n = len(points)
    boot_rs: list[float] = []
    boot_slopes: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        x_b, y_b = xs[idx], ys[idx]
        if np.var(x_b) > 0 and np.var(y_b) > 0:
            r_b, _ = pearsonr(x_b, y_b)
            slope_b = linregress(x_b, y_b).slope
            if np.isfinite(r_b) and np.isfinite(slope_b):
                boot_rs.append(float(r_b))
                boot_slopes.append(float(slope_b))

    jack_rs: list[float] = []
    jack_slopes: list[float] = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        x_j, y_j = xs[mask], ys[mask]
        if np.var(x_j) > 0 and np.var(y_j) > 0:
            r_j, _ = pearsonr(x_j, y_j)
            slope_j = linregress(x_j, y_j).slope
            if np.isfinite(r_j) and np.isfinite(slope_j):
                jack_rs.append(float(r_j))
                jack_slopes.append(float(slope_j))

    def _ci(values: list[float]) -> tuple[float, float]:
        if not values:
            return (float("nan"), float("nan"))
        return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))

    return MetricDeficitFit(
        points=points,
        metric_name=metric,
        pearson_r=float(r),
        pearson_p=float(p),
        slope=float(res.slope),
        intercept=float(res.intercept),
        pearson_r_ci95=_ci(boot_rs),
        slope_ci95=_ci(boot_slopes),
        jackknife_pearson_r=jack_rs,
        jackknife_slopes=jack_slopes,
    )


def plot_metric_vs_deficit(fit_result: MetricDeficitFit, out_path: str) -> None:
    """Single-metric scatter with regression line and bootstrap CI band."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = _xs_for_metric(fit_result.points, fit_result.metric_name)
    ys = np.array([p.deficit for p in fit_result.points])
    _draw_metric_panel(ax, fit_result, xs, ys)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_all_metrics(fits: dict[str, MetricDeficitFit], out_path: str) -> None:
    """Side-by-side κ / Spearman / CKA / MI panels for visual comparison."""
    import matplotlib.pyplot as plt

    metrics = [m for m in ("kappa", "spearman", "cka", "mi") if m in fits]
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics, strict=True):
        fr = fits[metric]
        xs = _xs_for_metric(fr.points, metric)
        ys = np.array([p.deficit for p in fr.points])
        _draw_metric_panel(ax, fr, xs, ys, show_y_label=(ax is axes[0]))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _draw_metric_panel(
    ax,
    fit_result: MetricDeficitFit,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    show_y_label: bool = True,
) -> None:
    ax.scatter(xs, ys, s=60, color="#1f77b4", zorder=3)
    for p, x, y in zip(fit_result.points, xs, ys, strict=True):
        ax.annotate(
            f"{p.pair[0]}×{p.pair[1]}",
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    grid = np.linspace(xs.min(), xs.max(), 100)
    ax.plot(
        grid,
        fit_result.slope * grid + fit_result.intercept,
        linestyle="--",
        color="grey",
    )
    r_lo, r_hi = fit_result.pearson_r_ci95
    ax.set_title(
        f"{fit_result.metric_name}: r = {fit_result.pearson_r:.2f} [{r_lo:.2f}, {r_hi:.2f}]"
    )
    ax.set_xlabel(METRIC_LABELS[fit_result.metric_name])
    if show_y_label:
        ax.set_ylabel(
            r"deficit $= \mathrm{JS}^{\mathrm{indep}}_{ij} - \mathrm{JS}^{\mathrm{PoE}}_{ij}$"
        )
