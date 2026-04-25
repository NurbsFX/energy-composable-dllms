from src.eval.kappa_vs_quality import KappaDeficitPoint, fit


def test_fit_perfect_linear_correlation():
    pts = [
        KappaDeficitPoint(pair=("a", "b"), kappa=k, js_poe=1.0 - k, js_indep=1.0)
        for k in [0.1, 0.2, 0.3, 0.4, 0.5]
    ]
    res = fit(pts)
    assert abs(res.pearson_r - 1.0) < 1e-6
    assert abs(res.slope - 1.0) < 1e-6
    assert abs(res.intercept) < 1e-6


def test_deficit_property():
    p = KappaDeficitPoint(pair=("a", "b"), kappa=0.3, js_poe=0.4, js_indep=0.7)
    assert abs(p.deficit - 0.3) < 1e-12
