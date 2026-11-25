"""Граничні випадки, матриця переходів і інтерфейс rank."""

import networkx as nx
import numpy as np
import pytest

from reldiff import models
from reldiff.api import rank


def test_alpha_one_gives_seed(er_graph):
    """α = 1: π* = s."""
    x = rank(er_graph, [3, 4], "ppr", alpha=1.0)
    assert x[3] == pytest.approx(0.5) and x[4] == pytest.approx(0.5)
    assert x.sum() == pytest.approx(1.0)


def test_t_zero_gives_seed(er_graph):
    """t = 0: u(0) = s."""
    for solver in ("expm_multiply", "series", "dense"):
        x = rank(er_graph, 7, "heat", t=0.0, solver=solver)
        assert x[7] == pytest.approx(1.0)


def test_bad_parameters(er_graph):
    with pytest.raises(ValueError):
        rank(er_graph, 0, "ppr", alpha=0.0)
    with pytest.raises(ValueError):
        rank(er_graph, 0, "heat", t=-1.0)
    with pytest.raises(ValueError):
        rank(er_graph, 0, "щось")
    with pytest.raises(KeyError):
        rank(er_graph, "немає", "ppr")


@pytest.mark.parametrize("dangling", ["restart", "self_loop"])
def test_transition_matrix_stochastic(directed_dangling, dangling):
    """P стохастична за рядками, у т.ч. для висячих вузлів, і невід'ємна."""
    W, nodes = models.weight_matrix(directed_dangling)
    s = models.seed_vector(len(nodes), [0, 2])
    P = models.transition_matrix(W, dangling=dangling, s=s)
    assert np.allclose(np.asarray(P.sum(axis=1)).ravel(), 1.0)
    assert P.min() >= 0
    dang = np.flatnonzero(models.degrees(W) == 0)
    for i in dang:
        row = P.getrow(i).toarray().ravel()
        if dangling == "restart":
            assert np.allclose(row, s)
        else:
            assert row[i] == 1.0


def test_restart_dangling_returns_mass_to_seed():
    """Ланцюжок 0 → 1 → 2 (2 висячий): маса з 2 повертається на s = e_0."""
    g = nx.DiGraph([(0, 1), (1, 2)])
    alpha = 0.2
    x = rank(g, 0, "ppr", alpha=alpha, dangling="restart").to_numpy()
    # з (1) при P̃: π0 = α + (1−α)π2, π1 = (1−α)π0, π2 = (1−α)π1
    b = 1 - alpha
    p0 = alpha / (1 - b ** 3)
    assert np.allclose(x, [p0, b * p0, b * b * p0])


def test_self_loop_dangling_accumulates():
    """Той самий ланцюжок зі стратегією петлі: π2 = (1−α)(π1 + π2)."""
    g = nx.DiGraph([(0, 1), (1, 2)])
    alpha = 0.2
    b = 1 - alpha
    x = rank(g, 0, "ppr", alpha=alpha, dangling="self_loop").to_numpy()
    p0 = alpha                      # у вузол 0 нічого не входить
    p1 = b * p0
    p2 = b * p1 / (1 - b)
    assert np.allclose(x, [p0, p1, p2])


def test_weights_and_type_weights():
    """Ваги ребер (атрибут і за типами) змінюють P; паралельні ребра сумуються."""
    g = nx.MultiGraph()
    g.add_edge("a", "b", type="актор")
    g.add_edge("a", "c", type="рік")
    g.add_edge("a", "c", type="рік")
    W, nodes = models.weight_matrix(g, type_weights={"актор": 3.0, "рік": 0.5})
    i = {v: k for k, v in enumerate(nodes)}
    assert W[i["a"], i["b"]] == 3.0 and W[i["a"], i["c"]] == 1.0
    assert (W != W.T).nnz == 0
    with pytest.raises(KeyError):
        models.weight_matrix(g, type_weights={"актор": 1.0})


def test_as_undirected():
    g = nx.DiGraph([(0, 1), (1, 2)])
    W, _ = models.weight_matrix(g, as_undirected=True)
    assert (W != W.T).nnz == 0 and W.sum() == 4


def test_degree_normalization_flattens_stationary():
    """π_i/d_i для стаціонарного розподілу d/Σd стає сталою — роль хаба зникає."""
    g = nx.star_graph(6)
    x = rank(g, 1, "heat", t=200.0, normalize="degree").to_numpy()
    assert np.allclose(x, x[0])


def test_rank_returns_series_indexed_by_nodes():
    g = nx.Graph([("x", "y"), ("y", "z")])
    x = rank(g, "x", "ppr", alpha=0.5)
    assert list(x.index) == ["x", "y", "z"]
    # при α = 0.5: π_x = 7/12, π_y = 1/3, π_z = 1/12 (з рівнянь (1))
    assert np.allclose(x.to_numpy(), [7 / 12, 1 / 3, 1 / 12])
