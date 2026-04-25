import numpy as np

from src.energies.gram_matrix import compute_gram, kappa_from_gram


def test_kappa_zero_for_diagonal():
    G = np.diag([1.0, 2.0, 3.0])
    assert kappa_from_gram(G) == 0.0


def test_kappa_nan_for_zero_trace():
    assert np.isnan(kappa_from_gram(np.zeros((2, 2))))


def test_compute_gram_independent_columns_have_small_kappa():
    rng = np.random.default_rng(0)
    E = rng.standard_normal((10_000, 3))
    res = compute_gram(E, ["a", "b", "c"])
    assert res.kappa_global < 0.1
    assert all(k < 0.1 for k in res.pair_kappas.values())


def test_compute_gram_correlated_columns_have_large_kappa():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(10_000)
    b = a + 0.1 * rng.standard_normal(10_000)
    E = np.column_stack([a, b])
    res = compute_gram(E, ["a", "b"])
    assert res.pair_kappas[("a", "b")] > 0.4


def test_compute_gram_rejects_mismatched_names():
    import pytest

    with pytest.raises(ValueError):
        compute_gram(np.zeros((10, 3)), ["a", "b"])
