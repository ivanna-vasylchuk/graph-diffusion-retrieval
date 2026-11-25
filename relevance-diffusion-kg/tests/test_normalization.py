"""Нормування за степенем: π/d і π/√d.

Симетричне нормування: якщо замість P = D⁻¹W поширювати з S = D^(−1/2)·W·D^(−1/2)
(S = D^(1/2)·P·D^(−1/2)), то для неорієнтованого графа
  x = α·e_q·(I − (1 − α)S)⁻¹ = √d_q · π / √d   і   e_q·e^(−t(I − S)) = √d_q · u(t) / √d,
тобто ранжування за π_i/√d_i. Перевіряємо цю тотожність прямим розрахунком.
"""

import networkx as nx
import numpy as np
import pytest
import scipy.linalg as sla

from reldiff.api import rank


@pytest.fixture
def g():
    gr = nx.connected_watts_strogatz_graph(30, 4, 0.3, seed=3)
    rng = np.random.default_rng(3)
    for u, v in gr.edges():
        gr[u][v]["weight"] = rng.uniform(0.5, 3.0)
    return gr


def _sym(g):
    W = nx.to_numpy_array(g, nodelist=list(g.nodes()), weight="weight")
    d = W.sum(axis=1)
    S = W / np.sqrt(np.outer(d, d))
    return S, d


@pytest.mark.parametrize("alpha", [0.05, 0.15, 0.5])
def test_sqrt_degree_equals_symmetric_ppr(g, alpha):
    S, d = _sym(g)
    q = 4
    e = np.zeros(len(d)); e[q] = 1
    x = alpha * e @ np.linalg.inv(np.eye(len(d)) - (1 - alpha) * S)
    y = rank(g, q, "ppr", alpha=alpha, normalize="sqrt_degree").to_numpy()
    np.testing.assert_allclose(x, np.sqrt(d[q]) * y, atol=1e-12)


@pytest.mark.parametrize("t", [0.5, 3.0, 10.0])
def test_sqrt_degree_equals_symmetric_heat(g, t):
    S, d = _sym(g)
    q = 7
    e = np.zeros(len(d)); e[q] = 1
    x = e @ sla.expm(-t * (np.eye(len(d)) - S))
    y = rank(g, q, "heat", t=t, normalize="sqrt_degree").to_numpy()
    np.testing.assert_allclose(x, np.sqrt(d[q]) * y, atol=1e-12)


def test_degree_normalization_by_hand():
    """Ланцюжок x–y–z, α = 0.5: π = (7/12, 1/3, 1/12), d = (1, 2, 1) ⇒ π/d = (7/12, 1/6, 1/12),
    π/√d = (7/12, 1/(3√2), 1/12)."""
    g = nx.Graph([("x", "y"), ("y", "z")])
    np.testing.assert_allclose(rank(g, "x", alpha=0.5, normalize="degree").to_numpy(), [7 / 12, 1 / 6, 1 / 12])
    np.testing.assert_allclose(rank(g, "x", alpha=0.5, normalize="sqrt_degree").to_numpy(),
                               [7 / 12, 1 / (3 * np.sqrt(2)), 1 / 12])
    with pytest.raises(ValueError):
        rank(g, "x", normalize="щось")


def test_reversibility_of_degree_normalized_ppr(g):
    """Оборотність на неорієнтованому графі: π_q(i)/d_i = π_i(q)/d_q — нормована оцінка симетрична."""
    a, b = 2, 17
    assert rank(g, a, normalize="degree")[b] == pytest.approx(rank(g, b, normalize="degree")[a], rel=1e-10)
