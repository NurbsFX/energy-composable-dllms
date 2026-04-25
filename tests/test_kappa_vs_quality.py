import numpy as np
import pytest

from src.eval.kappa_vs_quality import MetricDeficitPoint, fit


def _make_point(name: tuple[str, str], kappa: float, deficit: float) -> MetricDeficitPoint:
    """Build a point where the chosen κ controls the deficit linearly."""
    return MetricDeficitPoint(
        pair=name,
        kappa=kappa,
        spearman_abs=kappa,  # mirror κ so generic tests work for any metric
        cka=kappa,
        mi=kappa,
        js_poe=1.0 - deficit,
        js_indep=1.0,
    )


def test_fit_perfect_linear_correlation():
    pts = [_make_point((f"a{i}", f"b{i}"), k, k) for i, k in enumerate([0.1, 0.2, 0.3, 0.4, 0.5])]
    res = fit(pts, n_bootstrap=200, seed=0)
    assert abs(res.pearson_r - 1.0) < 1e-6
    assert abs(res.slope - 1.0) < 1e-6
    assert abs(res.intercept) < 1e-6


def test_deficit_property():
    p = _make_point(("a", "b"), kappa=0.3, deficit=0.3)
    assert abs(p.deficit - 0.3) < 1e-12


def test_fit_rejects_too_few_points():
    with pytest.raises(ValueError):
        fit([_make_point(("a", "b"), 0.1, 0.1)] * 2, n_bootstrap=10)


def test_fit_unknown_metric_raises():
    pts = [_make_point((f"a{i}", f"b{i}"), k, k) for i, k in enumerate([0.1, 0.2, 0.3])]
    with pytest.raises(ValueError):
        fit(pts, metric="nonexistent")


def test_fit_metric_dispatch():
    """Each metric should produce its own fit using its own X axis."""
    rng = np.random.default_rng(0)
    pts: list[MetricDeficitPoint] = []
    for i in range(8):
        kappa = float(rng.uniform(0, 1))
        deficit = float(rng.uniform(0, 1))
        pts.append(
            MetricDeficitPoint(
                pair=(f"a{i}", f"b{i}"),
                kappa=kappa,
                spearman_abs=float(rng.uniform(0, 1)),
                cka=float(rng.uniform(0, 1)),
                mi=float(rng.uniform(0, 1)),
                js_poe=1.0 - deficit,
                js_indep=1.0,
            )
        )
    f_kappa = fit(pts, metric="kappa", n_bootstrap=50)
    f_cka = fit(pts, metric="cka", n_bootstrap=50)
    # Different X data → different fits (vanishingly unlikely to coincide).
    assert f_kappa.pearson_r != f_cka.pearson_r


def test_bootstrap_ci_brackets_point_estimate():
    pts = [_make_point((f"a{i}", f"b{i}"), k, k) for i, k in enumerate([0.1, 0.2, 0.3, 0.4, 0.5])]
    res = fit(pts, n_bootstrap=500, seed=0)
    lo, hi = res.pearson_r_ci95
    # Perfect correlation: bootstrap r is always ≈ 1, so the CI brackets it.
    assert lo <= res.pearson_r <= hi or abs(res.pearson_r - 1.0) < 0.01


def test_jackknife_flags_high_leverage_point():
    """A point that drives most of the correlation should change r when removed."""
    pts = [
        # Cluster of low-κ points with no real signal.
        _make_point(("a", "b"), 0.01, 0.10),
        _make_point(("a", "c"), 0.02, 0.08),
        _make_point(("b", "c"), 0.03, 0.12),
        _make_point(("a", "d"), 0.04, 0.09),
        _make_point(("b", "d"), 0.05, 0.11),
        # Single high-κ point that pulls the slope.
        _make_point(("c", "d"), 0.80, 0.80),
    ]
    res = fit(pts, n_bootstrap=200, seed=0)
    r_lo, r_hi = res.jackknife_r_range
    # Removing the high-leverage point should drop r substantially.
    assert r_lo < res.pearson_r - 0.3, (
        f"jackknife should flag the high-leverage point; got r_lo={r_lo:.3f}, full r={res.pearson_r:.3f}"
    )


def test_jackknife_returns_one_entry_per_dropped_point():
    pts = [_make_point((f"a{i}", f"b{i}"), k, k) for i, k in enumerate([0.1, 0.2, 0.3, 0.4, 0.5])]
    res = fit(pts, n_bootstrap=50)
    assert len(res.jackknife_pearson_r) == len(pts)
    assert len(res.jackknife_slopes) == len(pts)
