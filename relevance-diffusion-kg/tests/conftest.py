"""Спільні тестові графи (усі випадкові — з явним seed)."""

import networkx as nx
import numpy as np
import pytest


def _connected_er(n, p, seed):
    """Зв'язний граф Ердеша–Реньї (перебираємо seed, доки граф не стане зв'язним)."""
    while True:
        g = nx.gnp_random_graph(n, p, seed=seed)
        if nx.is_connected(g):
            return g
        seed += 1000


@pytest.fixture
def er_graph():
    """Неорієнтований зв'язний недводольний ER-граф, n = 60."""
    g = _connected_er(60, 0.08, seed=1)
    assert not nx.is_bipartite(g)
    return g


@pytest.fixture
def weighted_graph():
    """Зважений неорієнтований граф (Ваттс–Строгац) з вагами з [0.1, 5]."""
    g = nx.connected_watts_strogatz_graph(50, 4, 0.2, seed=2)
    rng = np.random.default_rng(2)
    for u, v in g.edges():
        g[u][v]["weight"] = rng.uniform(0.1, 5.0)
    return g


@pytest.fixture
def directed_dangling():
    """Орієнтований зважений граф із висячими вузлами (без вихідних ребер)."""
    g = nx.gnp_random_graph(50, 0.08, seed=3, directed=True)
    for v in (5, 17, 33):
        g.remove_edges_from(list(g.out_edges(v)))
    rng = np.random.default_rng(3)
    for u, v in g.edges():
        g[u][v]["weight"] = rng.uniform(0.5, 2.0)
    assert sum(1 for v in g if g.out_degree(v) == 0) >= 3
    return g


@pytest.fixture
def all_graphs(er_graph, weighted_graph, directed_dangling):
    """Набір графів для загальних властивостей: (назва, граф, вузол запиту)."""
    return [("ER", er_graph, 0), ("WS зважений", weighted_graph, 7),
            ("орієнтований з висячими", directed_dangling, 1)]
