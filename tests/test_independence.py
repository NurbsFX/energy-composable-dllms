import numpy as np

from src.energies.independence import compute_independence, hsic, mutual_info


def test_hsic_independent_gaussians_is_small():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(400)
    y = rng.standard_normal(400)
    assert hsic(x, y) < 0.01


def test_hsic_self_is_strictly_positive():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(400)
    assert hsic(x, x) > 0.01


def test_hsic_detects_nonlinear_dependence_pearson_misses():
    """Y = X² has zero linear correlation but is fully determined by X."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(400)
    y = x**2
    # Linear correlation is near zero by symmetry of N(0,1).
    assert abs(np.corrcoef(x, y)[0, 1]) < 0.2
    # But HSIC must dominate the independent baseline by orders of magnitude.
    indep = hsic(rng.standard_normal(400), rng.standard_normal(400))
    assert hsic(x, y) > 10 * max(indep, 1e-3)


def test_hsic_length_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        hsic(np.zeros(10), np.zeros(11))


def test_mi_independent_gaussians_is_small():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(400)
    y = rng.standard_normal(400)
    assert mutual_info(x, y) < 0.1


def test_mi_self_dominates_independent_baseline():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(400)
    indep = mutual_info(x, rng.standard_normal(400))
    assert mutual_info(x, x) > 5 * max(indep, 0.05)


def test_compute_independence_returns_all_pairs_with_cka_in_unit_interval():
    rng = np.random.default_rng(0)
    E = rng.standard_normal((200, 3))
    res = compute_independence(E, ["a", "b", "c"])
    expected = {("a", "b"), ("a", "c"), ("b", "c")}
    assert set(res.pair_hsic.keys()) == expected
    assert set(res.pair_cka.keys()) == expected
    assert set(res.pair_mi.keys()) == expected
    for v in res.pair_cka.values():
        assert -0.05 <= v <= 1.0  # CKA must lie in [0, 1] up to rounding


def test_compute_independence_rejects_mismatched_names():
    import pytest

    with pytest.raises(ValueError):
        compute_independence(np.zeros((10, 3)), ["a", "b"])
