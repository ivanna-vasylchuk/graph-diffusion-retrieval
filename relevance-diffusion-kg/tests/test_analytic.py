"""Перевірка адекватності 3: порівняння з аналітичними розв'язками (docs/analitychni_rozvyazky.md)."""

import networkx as nx
import numpy as np
import pytest

from reldiff import analytic as A
from reldiff.api import rank

TOL = 1e-12          # точні формули звіряємо з лінійною системою (solver="linear")
ALPHAS = [0.01, 0.15, 0.5, 0.9, 1.0]
TIMES = [0.0, 0.3, 1.0, 4.0, 15.0]


def _close(x, ref, tol=TOL):
    err = np.abs(np.asarray(x) - ref).max()
    assert err < tol, f"макс. похибка {err:.2e}"


# --- Повний граф ---------------------------------------------------------

@pytest.mark.parametrize("n", [3, 5, 12])
@pytest.mark.parametrize("alpha", ALPHAS)
def test_complete_ppr(n, alpha):
    _close(rank(nx.complete_graph(n), 1, "ppr", alpha=alpha, solver="linear"), A.complete_ppr(n, alpha, q=1))


@pytest.mark.parametrize("n", [3, 5, 12])
@pytest.mark.parametrize("t", TIMES)
def test_complete_heat(n, t):
    for solver in ("expm_multiply", "series", "dense"):
        _close(rank(nx.complete_graph(n), 1, "heat", t=t, solver=solver),
               A.complete_heat(n, t, q=1), tol=1e-11)


@pytest.mark.parametrize("k", [0, 1, 2, 7])
def test_complete_walk(k):
    _close(rank(nx.complete_graph(6), 0, "walk", steps=k), A.complete_walk(6, k))


# --- Цикл ----------------------------------------------------------------

@pytest.mark.parametrize("n", [3, 8, 15])
@pytest.mark.parametrize("alpha", ALPHAS)
def test_cycle_ppr(n, alpha):
    _close(rank(nx.cycle_graph(n), 2, "ppr", alpha=alpha, solver="linear"), A.cycle_ppr(n, alpha, q=2))


def test_cycle_ppr_geometric_decay():
    """На циклі π_j / π_{j+1} ≈ 1/r(α) далеко від протилежної точки (формула (10))."""
    n, alpha = 41, 0.15
    x = rank(nx.cycle_graph(n), 0, "ppr", alpha=alpha, solver="linear").to_numpy()
    ratios = x[1:6] / x[0:5]
    assert np.allclose(ratios, A.cycle_r(alpha), rtol=1e-6)


@pytest.mark.parametrize("n", [3, 8, 15])
@pytest.mark.parametrize("t", TIMES)
def test_cycle_heat(n, t):
    for solver in ("expm_multiply", "series", "dense"):
        _close(rank(nx.cycle_graph(n), 2, "heat", t=t, solver=solver),
               A.cycle_heat(n, t, q=2), tol=1e-11)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("k", [0, 1, 5, 20])
def test_cycle_walk(k, lazy):
    _close(rank(nx.cycle_graph(9), 0, "walk", steps=k, lazy=lazy),
           A.cycle_walk(9, k, lazy=lazy))


# --- Ланцюжок ------------------------------------------------------------

@pytest.mark.parametrize("q", [0, 3, 6])
@pytest.mark.parametrize("alpha", ALPHAS)
def test_path_ppr(q, alpha):
    _close(rank(nx.path_graph(7), q, "ppr", alpha=alpha, solver="linear"), A.path_ppr(7, alpha, q=q))


def test_path_ppr_formula_12():
    """Явна формула (12) для кінця ланцюжка."""
    n, alpha = 7, 0.2
    r = A.cycle_r(alpha)
    N = 2 * (n - 1)
    C = (1 - r) / ((1 + r) * (1 - r ** N))
    j = np.arange(1, n - 1)
    ref = np.r_[C * (1 + r ** N), 2 * C * (r ** j + r ** (N - j)), 2 * C * r ** (n - 1)]
    _close(rank(nx.path_graph(n), 0, "ppr", alpha=alpha, solver="linear"), ref)


@pytest.mark.parametrize("q", [0, 3])
@pytest.mark.parametrize("t", TIMES)
def test_path_heat(q, t):
    for solver in ("expm_multiply", "series", "dense"):
        _close(rank(nx.path_graph(7), q, "heat", t=t, solver=solver),
               A.path_heat(7, t, q=q), tol=1e-11)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("k", [0, 1, 4, 11])
def test_path_walk(k, lazy):
    _close(rank(nx.path_graph(6), 2, "walk", steps=k, lazy=lazy),
           A.path_walk(6, k, q=2, lazy=lazy))


# --- Зірка ---------------------------------------------------------------

@pytest.mark.parametrize("m", [2, 5, 20])
@pytest.mark.parametrize("alpha", ALPHAS)
def test_star_ppr(m, alpha):
    g = nx.star_graph(m)
    _close(rank(g, 0, "ppr", alpha=alpha, solver="linear"), A.star_ppr(m, alpha))
    _close(rank(g, 1, "ppr", alpha=alpha, solver="linear"), A.star_ppr(m, alpha, leaf=1))


@pytest.mark.parametrize("m", [2, 5, 20])
@pytest.mark.parametrize("t", TIMES)
def test_star_heat(m, t):
    g = nx.star_graph(m)
    for solver in ("expm_multiply", "series", "dense"):
        _close(rank(g, 0, "heat", t=t, solver=solver), A.star_heat(m, t), tol=1e-11)
        _close(rank(g, 1, "heat", t=t, solver=solver), A.star_heat(m, t, leaf=1), tol=1e-11)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("k", [0, 1, 2, 3, 10])
def test_star_walk(k, lazy):
    g = nx.star_graph(5)
    _close(rank(g, 0, "walk", steps=k, lazy=lazy), A.star_walk(5, k, lazy=lazy))
    _close(rank(g, 3, "walk", steps=k, lazy=lazy), A.star_walk(5, k, leaf=3, lazy=lazy))


def test_star_lazy_center_exact_after_one_step():
    """Зірка, ліниве блукання з центру: точно d/Σd уже після 1 кроку (розділ 4)."""
    m = 7
    x = rank(nx.star_graph(m), 0, "walk", steps=1, lazy=True).to_numpy()
    _close(x, np.r_[0.5, np.full(m, 1 / (2 * m))])
