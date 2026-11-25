"""Поєднання релевантностей від кількох вузлів запиту (reldiff.multi, rank(combine=...))."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from reldiff.api import rank
from reldiff.multi import combine_scores


@pytest.fixture
def g():
    return nx.connected_watts_strogatz_graph(40, 4, 0.3, seed=7)


@pytest.mark.parametrize("method,kw", [("ppr", dict(alpha=0.15)), ("heat", dict(t=3.0))])
def test_mean_equals_uniform_seed_vector(g, method, kw):
    """Лінійність: середнє окремих розподілів = один запуск з рівномірним s."""
    a = rank(g, [3, 17], method, combine="mean", **kw).to_numpy()
    b = rank(g, [3, 17], method, **kw).to_numpy()
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_path_by_hand():
    """Ланцюжок a–m–b, α = 0.5: від a π = (7/12, 1/3, 1/12), від b — дзеркально.
    Середнє: (1/3, 1/3, 1/3) — нічия; добуток: (7, 16, 7)/144 — міст m найвищий; мінімум: (1, 4, 1)/12."""
    g = nx.Graph([("a", "m"), ("m", "b")])
    mean = rank(g, ["a", "b"], alpha=0.5, combine="mean")
    prod = rank(g, ["a", "b"], alpha=0.5, combine="product")
    mn = rank(g, ["a", "b"], alpha=0.5, combine="min")
    np.testing.assert_allclose(mean.to_numpy(), [1 / 3] * 3)
    np.testing.assert_allclose(prod.to_numpy(), np.array([7, 16, 7]) / 144)
    np.testing.assert_allclose(mn.to_numpy(), np.array([1, 4, 1]) / 12)
    assert prod.idxmax() == "m" and mn.idxmax() == "m"


def test_normalized_product_identity(g):
    """Оборотність: (π_a(i)/d_i)·(π_b(i)/d_i) = π_i(a)·π_i(b)/(d_a·d_b)."""
    a, b = 3, 17
    lhs = rank(g, [a, b], alpha=0.2, normalize="degree", combine="product")
    d = dict(g.degree())
    rhs = pd.Series({i: rank(g, i, alpha=0.2)[a] * rank(g, i, alpha=0.2)[b] / (d[a] * d[b]) for i in g.nodes()})
    np.testing.assert_allclose(lhs.to_numpy(), rhs.reindex(lhs.index).to_numpy(), rtol=1e-9)


def test_combine_scores_zeros_and_errors():
    s1 = pd.Series([0.5, 0.0, 0.2], index=list("xyz"))
    s2 = pd.Series([0.1, 0.3, 0.4], index=list("xyz"))
    np.testing.assert_allclose(combine_scores([s1, s2], "product").to_numpy(), [0.05, 0.0, 0.08])
    np.testing.assert_allclose(combine_scores([s1, s2], "min").to_numpy(), [0.1, 0.0, 0.2])
    with pytest.raises(ValueError):
        combine_scores([s1], "max")
    with pytest.raises(ValueError):
        rank(nx.path_graph(3), [0, 2], "k_hop", combine="product")
