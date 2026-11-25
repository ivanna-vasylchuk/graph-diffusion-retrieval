"""Перевірка адекватності 2: збіжність без перезапуску до d/Σd."""

import networkx as nx
import numpy as np
import pytest

from reldiff import models
from reldiff.api import rank


def _stationary(g):
    W, _ = models.weight_matrix(g)
    return models.stationary_degree(W)


NON_BIPARTITE = [("непарний цикл C_11", nx.cycle_graph(11)),
                 ("повний K_6", nx.complete_graph(6)),
                 ("ланцюжок + трикутник", nx.lollipop_graph(3, 6))]
BIPARTITE = [("ланцюжок P_8", nx.path_graph(8)),
             ("парний цикл C_10", nx.cycle_graph(10)),
             ("зірка S_6", nx.star_graph(6))]


@pytest.mark.parametrize("name,g", NON_BIPARTITE)
def test_walk_converges_non_bipartite(name, g):
    """Недводольний зв'язний граф: π_k → d/Σd."""
    x = rank(g, 0, "walk", steps=3000).to_numpy()
    assert np.abs(x - _stationary(g)).sum() < 1e-8


def test_walk_converges_er(er_graph):
    """Випадковий ER-граф: π_k → d/Σd."""
    x = rank(er_graph, 0, "walk", steps=2000).to_numpy()
    assert np.abs(x - _stationary(er_graph)).sum() < 1e-8


def test_walk_converges_weighted(weighted_graph):
    """Зважений граф: π_k → d/Σd, де d — зважені степені."""
    x = rank(weighted_graph, 0, "walk", steps=5000).to_numpy()
    assert np.abs(x - _stationary(weighted_graph)).sum() < 1e-8


@pytest.mark.parametrize("name,g", BIPARTITE)
def test_walk_oscillates_bipartite(name, g):
    """Дводольний граф: звичайне блукання не збігається (коливання з періодом 2)."""
    traj = rank(g, 0, "walk", steps=2001, keep_trajectory=True, return_info=True)[1]["trajectory"]
    assert nx.is_bipartite(g)
    assert np.abs(traj[-1] - traj[-2]).sum() > 0.5          # сусідні кроки сильно різні
    assert np.abs(traj[-1] - traj[-3]).sum() < 1e-8         # а через крок — однакові


@pytest.mark.parametrize("name,g", BIPARTITE)
def test_lazy_walk_converges_bipartite(name, g):
    """Дводольний граф: ліниве блукання P' = (I + P)/2 збігається до d/Σd."""
    x = rank(g, 0, "walk", steps=5000, lazy=True).to_numpy()
    assert np.abs(x - _stationary(g)).sum() < 1e-8


@pytest.mark.parametrize("name,g", BIPARTITE + NON_BIPARTITE)
def test_heat_converges_any(name, g):
    """Теплове ядро збігається до d/Σd і на дводольних графах."""
    for solver in ("expm_multiply", "series", "dense"):
        x = rank(g, 0, "heat", t=300.0, solver=solver).to_numpy()
        assert np.abs(x - _stationary(g)).sum() < 1e-8, solver


def test_ppr_small_alpha_tends_to_stationary():
    """α → 0: π* → d/Σd (і на дводольному графі)."""
    g = nx.path_graph(7)
    errs = [np.abs(rank(g, 0, "ppr", alpha=a).to_numpy() - _stationary(g)).sum()
            for a in (1e-2, 1e-3, 1e-4)]
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-2
