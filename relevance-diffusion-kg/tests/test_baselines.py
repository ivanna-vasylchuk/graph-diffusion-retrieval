"""Тести базових методів (reldiff.baselines) на прикладах із ручною відповіддю."""

import networkx as nx
import numpy as np
import pytest

from reldiff import baselines as B
from reldiff.api import rank

NAN = np.nan


def _eq(series, expected):
    np.testing.assert_allclose(series.to_numpy(), expected, equal_nan=True)


def test_shortest_path_on_path():
    """Ланцюжок 0–4, запит 0: відстані 0..4 → оцінки 1/(1 + d)."""
    _eq(B.shortest_path(nx.path_graph(5), 0), [1, 1 / 2, 1 / 3, 1 / 4, 1 / 5])


def test_unreachable_is_nan():
    g = nx.path_graph(3); g.add_node(9)
    _eq(B.shortest_path(g, 0), [1, 1 / 2, 1 / 3, NAN])


@pytest.mark.parametrize("agg,dist", [("min", [0, 1, 2, 1, 0]),
                                      ("mean", [2, 2, 2, 2, 2]),
                                      ("max", [4, 3, 2, 3, 4])])
def test_shortest_path_two_seeds(agg, dist):
    """Ланцюжок 0–4, запити {0, 4}: агрегування відстаней."""
    _eq(B.shortest_path(nx.path_graph(5), [0, 4], agg=agg), 1 / (1 + np.array(dist)))


def test_shortest_path_weighted():
    """Довжина ребра 1/w: a–b (w=1), b–c (w=4), a–c (w=0.5) ⇒ d(a,c) = min(2, 1 + 0.25) = 1.25."""
    g = nx.Graph()
    g.add_edge("a", "b", weight=1.0); g.add_edge("b", "c", weight=4.0); g.add_edge("a", "c", weight=0.5)
    s = B.shortest_path(g, "a", weighted=True)
    assert s["c"] == pytest.approx(1 / 2.25)
    assert B.shortest_path(g, "a")["c"] == pytest.approx(1 / 2)    # без ваг — 1 ребро


def test_k_hop_by_hand():
    """Ланцюжок 0–5, запит 0, радіус 2: оцінки 3, 2, 1, далі NaN (не відібрано)."""
    _eq(B.k_hop(nx.path_graph(6), 0, hops=2), [3, 2, 1, NAN, NAN, NAN])
    _eq(B.k_hop(nx.path_graph(4), 0, hops=0), [1, NAN, NAN, NAN])


def test_k_hop_two_seeds():
    """Запити {0, 5} на ланцюжку 0–5, радіус 1, agg=min: вузли 0, 1, 4, 5."""
    _eq(B.k_hop(nx.path_graph(6), [0, 5], hops=1), [2, 1, NAN, NAN, 1, 2])


def test_common_neighbors_by_hand():
    """Ребра q–a, q–b, x–a, x–b, y–a, z–x: CN(x) = 2, CN(y) = 1, CN(z) = 0, CN(a) = 0."""
    g = nx.Graph([("q", "a"), ("q", "b"), ("x", "a"), ("x", "b"), ("y", "a"), ("z", "x")])
    s = B.common_neighbors(g, "q")
    assert s["x"] == 2 and s["y"] == 1 and s["z"] == 0 and s["a"] == 0 and s["b"] == 0


def test_common_neighbors_two_seeds_sum():
    """Два запити — сума: зірка з центром 0, запити листки 1 і 2 ⇒ CN(3) = 1 + 1 = 2."""
    s = B.common_neighbors(nx.star_graph(4), [1, 2])
    assert s[3] == 2 and s[0] == 0


def test_directed_treated_as_undirected():
    g = nx.DiGraph([(1, 0), (2, 1)])
    s = B.shortest_path(g, 0)
    assert (s[0], s[1], s[2]) == pytest.approx((1, 1 / 2, 1 / 3))


def test_rank_dispatches_baselines():
    g = nx.path_graph(6)
    _eq(rank(g, 0, "k_hop", hops=2), B.k_hop(g, 0, hops=2).to_numpy())
    _eq(rank(g, [0, 5], "shortest_path", agg="max"), B.shortest_path(g, [0, 5], agg="max").to_numpy())
    _eq(rank(g, 0, "common_neighbors"), B.common_neighbors(g, 0).to_numpy())
    with pytest.raises(KeyError):
        rank(g, 99, "k_hop")
