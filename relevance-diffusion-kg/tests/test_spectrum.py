"""Тести спектра матриці переходів (api.spectrum) проти аналітичних спектрів (docs, розділи 1–4)."""

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

from reldiff import models
from reldiff.api import spectrum


def _sorted_desc(x):
    return np.sort(np.asarray(x))[::-1]


@pytest.mark.parametrize("n", [5, 8, 13])
def test_cycle(n):
    """Цикл: λ_m = cos(2πm/n)."""
    np.testing.assert_allclose(spectrum(nx.cycle_graph(n)),
                               _sorted_desc(np.cos(2 * np.pi * np.arange(n) / n)), atol=1e-12)


@pytest.mark.parametrize("n", [4, 7, 12])
def test_path(n):
    """Ланцюжок (складання циклу C_{2(n−1)}): λ_m = cos(πm/(n − 1)), m = 0..n − 1."""
    np.testing.assert_allclose(spectrum(nx.path_graph(n)),
                               _sorted_desc(np.cos(np.pi * np.arange(n) / (n - 1))), atol=1e-12)


def test_star_and_complete():
    """Зірка S_m: 1, 0 (m − 1 разів), −1; повний K_n: 1, −1/(n − 1) (n − 1 разів)."""
    np.testing.assert_allclose(spectrum(nx.star_graph(5)), [1, 0, 0, 0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(spectrum(nx.complete_graph(6)), [1] + [-0.2] * 5, atol=1e-12)


def test_summary_and_lazy():
    lam, s = spectrum(nx.cycle_graph(10), return_summary=True)
    assert s["real"] and s["lambda_min"] == pytest.approx(-1) and s["modulus_2"] == pytest.approx(1)
    assert s["lambda_2"] == pytest.approx(np.cos(2 * np.pi / 10))
    assert s["gap"] == pytest.approx(1 - np.cos(2 * np.pi / 10))
    np.testing.assert_allclose(spectrum(nx.cycle_graph(10), lazy=True), (1 + lam) / 2)


def test_weighted_matches_dense_general():
    """Зважений неорієнтований граф: симетричний спосіб = загальні власні значення P."""
    g = nx.connected_watts_strogatz_graph(40, 4, 0.3, seed=1)
    rng = np.random.default_rng(1)
    for u, v in g.edges():
        g[u][v]["weight"] = rng.uniform(0.1, 3)
    lam = spectrum(g)
    W, _ = models.weight_matrix(g)
    ref = np.linalg.eigvals(models.transition_matrix(W).toarray())
    np.testing.assert_allclose(lam, _sorted_desc(ref.real), atol=1e-10)
    assert np.abs(ref.imag).max() < 1e-10


def test_directed_cycle_is_complex():
    """Орієнтований цикл: власні значення — корені n-го степеня з одиниці (комплексні),
    а не косинуси, які дав би симетричний розрахунок."""
    n = 6
    lam = spectrum(nx.cycle_graph(n, create_using=nx.DiGraph))
    assert np.iscomplexobj(lam)
    np.testing.assert_allclose(np.abs(lam), 1, atol=1e-12)
    ref = np.exp(2j * np.pi * np.arange(n) / n)
    np.testing.assert_allclose(np.sort_complex(np.round(lam, 12)), np.sort_complex(np.round(ref, 12)))
    # як неорієнтований — дійсний спектр циклу
    np.testing.assert_allclose(spectrum(nx.cycle_graph(n, create_using=nx.DiGraph), as_undirected=True),
                               _sorted_desc(2 * np.cos(2 * np.pi * np.arange(n) / n) / 2), atol=1e-12)


def test_directed_with_dangling():
    """Орієнтований ланцюжок 0 → 1 → 2 з петлею на висячому вузлі 2: P трикутна, λ = {1, 0, 0}."""
    lam = spectrum(nx.DiGraph([(0, 1), (1, 2)]))
    np.testing.assert_allclose(np.sort(np.real(lam)), [0, 0, 1], atol=1e-7)
    with pytest.raises(ValueError):
        spectrum(nx.DiGraph([(0, 1)]), dangling="restart")


def test_symmetric_method_rejects_asymmetric():
    """Явна помилка, якщо симетричний спосіб викликано для несиметричної W."""
    W = sp.csr_matrix(np.array([[0, 1.0], [2.0, 0]]))
    with pytest.raises(ValueError):
        models.transition_spectrum(W, symmetric=True)


def test_isolated_node_uses_general():
    """Ізольований вузол (d = 0) у неорієнтованому графі: загальний розрахунок з петлею."""
    g = nx.path_graph(3); g.add_node(9)
    lam = spectrum(g)
    np.testing.assert_allclose(np.sort(np.real(lam)), [-1, 0, 1, 1], atol=1e-12)
