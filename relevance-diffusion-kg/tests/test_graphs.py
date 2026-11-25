"""Тести генераторів графів, ваг і шуму в ребрах (reldiff.graphs)."""

import networkx as nx
import numpy as np
import pytest

from reldiff import graphs as G


def _edge_set(g):
    return {frozenset(e) for e in g.edges()} if not g.is_directed() else set(g.edges())


# --- Прості і випадкові графи ------------------------------------------------

def test_simple_graphs():
    """Ланцюжок: n − 1 ребер; цикл: n; зірка: m ребер, центр 0; повний: n(n − 1)/2."""
    assert G.path(6).number_of_edges() == 5
    assert G.cycle(6).number_of_edges() == 6
    s = G.star(5)
    assert s.number_of_nodes() == 6 and s.degree(0) == 5
    assert G.complete(6).number_of_edges() == 15


@pytest.mark.parametrize("make", [
    lambda sd: G.erdos_renyi(80, 0.06, seed=sd),
    lambda sd: G.watts_strogatz(80, 4, 0.1, seed=sd),
    lambda sd: G.barabasi_albert(80, 2, seed=sd),
    lambda sd: G.sbm([30, 30], 0.2, 0.01, seed=sd),
])
def test_random_graphs_reproducible_and_connected(make):
    """Однаковий seed — однаковий граф; графи зв'язні."""
    g1, g2, g3 = make(5), make(5), make(6)
    assert _edge_set(g1) == _edge_set(g2)
    assert _edge_set(g1) != _edge_set(g3)
    assert nx.is_connected(g1)


def test_barabasi_albert_edges():
    """БА: m(n − m) ребер (стартова зірка з m+1 вузлів має m ребер)."""
    g = G.barabasi_albert(50, 3, seed=1)
    assert g.number_of_edges() == 3 * (50 - 3)


def test_sbm_communities():
    """SBM: атрибут спільноти, розміри блоків; при p_out = 0 немає ребер між спільнотами."""
    g = G.sbm([10, 15, 5], 0.5, 0.0, seed=1, connected=False)
    comm = G.communities(g)
    assert {c: len(v) for c, v in comm.items()} == {0: 10, 1: 15, 2: 5}
    assert all(g.nodes[u]["community"] == g.nodes[v]["community"] for u, v in g.edges())
    assert G.bridges(g) == set()


# --- Граф з явними мостами ---------------------------------------------------

def test_bridged_two_communities():
    """2 спільноти по 10 вузлів, 3 мости зі ступенем 2 в кожну: 23 вузли, мости 20–22."""
    g = G.bridged_communities(2, 10, 0.4, n_bridges=3, bridge_degree=2, seed=1)
    assert g.number_of_nodes() == 23
    assert G.bridges(g) == {20, 21, 22}
    for b in (20, 21, 22):
        assert g.degree(b) == 4
        assert g.nodes[b]["connects"] == (0, 1) and g.nodes[b]["community"] == -1
        assert sorted(g.nodes[v]["community"] for v in g.neighbors(b)) == [0, 0, 1, 1]
    # прямих ребер між спільнотами немає
    for u, v in g.edges():
        cu, cv = g.nodes[u]["community"], g.nodes[v]["community"]
        assert cu == -1 or cv == -1 or cu == cv
    # без мостів граф розпадається на 2 зв'язні спільноти
    h = g.copy(); h.remove_nodes_from(G.bridges(g))
    assert sorted(len(c) for c in nx.connected_components(h)) == [10, 10]
    assert nx.is_connected(g)


@pytest.mark.parametrize("pairs,n_pairs", [("all", 3), ("chain", 2)])
def test_bridged_pairs(pairs, n_pairs):
    """3 спільноти: усі пари — 3 пари мостів, ланцюжок — 2 (0–1, 1–2)."""
    g = G.bridged_communities(3, 8, 0.5, n_bridges=2, bridge_degree=1, seed=2, pairs=pairs)
    assert len(G.bridges(g)) == 2 * n_pairs
    connects = {g.nodes[b]["connects"] for b in G.bridges(g)}
    assert connects == ({(0, 1), (0, 2), (1, 2)} if pairs == "all" else {(0, 1), (1, 2)})


def test_bridged_reproducible():
    a = G.bridged_communities(2, 10, 0.4, 2, 2, seed=7)
    b = G.bridged_communities(2, 10, 0.4, 2, 2, seed=7)
    assert _edge_set(a) == _edge_set(b)


# --- Ваги -----------------------------------------------------------------

def test_random_weights():
    g = G.cycle(30)
    w1 = G.random_weights(g, 0.5, 2.0, seed=1)
    w2 = G.random_weights(g, 0.5, 2.0, seed=1)
    vals = [d["weight"] for *_, d in w1.edges(data=True)]
    assert all(0.5 <= x <= 2.0 for x in vals)
    assert vals == [d["weight"] for *_, d in w2.edges(data=True)]
    assert "weight" not in next(iter(g.edges(data=True)))[2]      # вихідний граф не змінено


def test_type_weights_by_hand():
    """Ваги за типами: актор → 3, рік → 0.5; невідомий тип — помилка."""
    g = nx.MultiGraph()
    g.add_edge("фільм", "актор", type="актор")
    g.add_edge("фільм", "1999", type="рік")
    h = G.type_weights(g, {"актор": 3.0, "рік": 0.5})
    assert sorted(d["weight"] for *_, d in h.edges(data=True)) == [0.5, 3.0]
    with pytest.raises(KeyError):
        G.type_weights(g, {"актор": 1.0})


def test_random_edge_types():
    g = G.random_edge_types(G.cycle(200), ["a", "b"], seed=3, probs=[0.75, 0.25])
    types = [d["type"] for *_, d in g.edges(data=True)]
    assert set(types) == {"a", "b"}
    assert 0.6 < types.count("a") / len(types) < 0.9


# --- Шум у ребрах -------------------------------------------------------------

def test_perturb_counts_by_hand():
    """Цикл з 20 ребер: remove 0.25 → 5 видалено, add 0.1 → 2 додано, разом 17."""
    g = G.random_weights(G.cycle(20), 1.0, 2.0, seed=0)
    h, rec = G.perturb_edges(g, remove_frac=0.25, add_frac=0.1, seed=1)
    assert len(rec["removed"]) == 5 and len(rec["added"]) == 2
    assert h.number_of_edges() == 17
    orig = _edge_set(g)
    assert {frozenset(e) for e in rec["removed"]} <= orig
    assert all(frozenset(e) not in orig for e in rec["added"])
    for u, v in rec["added"]:
        assert h[u][v]["noise"] is True and 1.0 <= h[u][v]["weight"] <= 2.0
    assert g.number_of_edges() == 20                            # вихідний граф не змінено


def test_perturb_reproducible():
    g = G.erdos_renyi(50, 0.1, seed=1)
    h1, r1 = G.perturb_edges(g, 0.2, 0.2, seed=9)
    h2, r2 = G.perturb_edges(g, 0.2, 0.2, seed=9)
    h3, _ = G.perturb_edges(g, 0.2, 0.2, seed=10)
    assert _edge_set(h1) == _edge_set(h2) and r1 == r2
    assert _edge_set(h1) != _edge_set(h3)


def test_perturb_zero_is_identity():
    g = G.erdos_renyi(30, 0.2, seed=1)
    h, rec = G.perturb_edges(g, 0.0, 0.0, seed=1)
    assert _edge_set(h) == _edge_set(g) and rec == {"removed": [], "added": []}


def test_perturb_directed():
    """Орієнтований граф: хибні дуги — упорядковані пари, яких не було."""
    g = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 0)])
    h, rec = G.perturb_edges(g, 0.5, 0.5, seed=2)
    assert len(rec["removed"]) == 2 and len(rec["added"]) == 2
    assert all(not g.has_edge(u, v) for u, v in rec["added"])
    assert h.number_of_edges() == 4


def test_perturb_too_many():
    with pytest.raises(ValueError):
        G.perturb_edges(G.complete(4), 0.0, 0.5, seed=0)


# --- Орієнтування і логнормальні ваги (п. 4) -------------------------------

@pytest.mark.parametrize("rho", [0.0, 1.0])
def test_orient_extremes(rho):
    """ρ = 0: кожне ребро — рівно одна дуга; ρ = 1: кожне ребро — дві дуги."""
    g = G.erdos_renyi(60, 0.1, seed=1)
    d = G.orient_with_reciprocity(g, rho, seed=2)
    assert d.is_directed() and set(d.nodes()) == set(g.nodes())
    for u, v in g.edges():
        n_arcs = int(d.has_edge(u, v)) + int(d.has_edge(v, u))
        assert n_arcs == (2 if rho == 1 else 1)
    assert d.number_of_edges() == g.number_of_edges() * (2 if rho == 1 else 1)


def test_orient_fraction_and_attrs():
    g = G.random_weights(G.erdos_renyi(400, 0.05, seed=1), 1.0, 2.0, seed=1)
    d = G.orient_with_reciprocity(g, 0.3, seed=5)
    recip = sum(1 for u, v in g.edges() if d.has_edge(u, v) and d.has_edge(v, u)) / g.number_of_edges()
    assert 0.27 < recip < 0.33
    u, v = next(iter(d.edges()))
    assert d[u][v]["weight"] == g[u][v]["weight"]
    assert set(G.orient_with_reciprocity(g, 0.3, seed=5).edges()) == set(d.edges())
    with pytest.raises(ValueError):
        G.orient_with_reciprocity(g, 1.5, seed=0)


def test_lognormal_weights():
    g = G.erdos_renyi(500, 0.05, seed=1)
    assert all(d["weight"] == 1.0 for *_, d in G.lognormal_weights(g, 0.0, seed=1).edges(data=True))
    w = np.array([d["weight"] for *_, d in G.lognormal_weights(g, 1.0, seed=1).edges(data=True)])
    assert np.all(w > 0) and abs(w.mean() - 1) < 0.1          # середнє ≈ 1
    assert abs(np.std(np.log(w)) - 1.0) < 0.05                # σ log-ваг ≈ σ
    w2 = [d["weight"] for *_, d in G.lognormal_weights(g, 1.0, seed=1).edges(data=True)]
    assert list(w) == w2
