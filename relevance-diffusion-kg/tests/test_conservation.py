"""Перевірка адекватності 1: збереження суми (= 1) і невід'ємність на кожному кроці."""

import numpy as np
import pytest

from reldiff.api import rank

ATOL = 1e-12


def _check_distribution(x, atol=ATOL):
    x = np.asarray(x)
    assert abs(x.sum() - 1.0) < atol, f"сума = {x.sum()!r}"
    assert x.min() > -atol, f"мінімум = {x.min()!r}"


@pytest.mark.parametrize("dangling", ["restart", "self_loop"])
@pytest.mark.parametrize("alpha", [0.05, 0.15, 0.5, 1.0])
def test_ppr_iterations_conserve_mass(all_graphs, alpha, dangling):
    """Кожна ітерація PPR π_k — розподіл ймовірностей."""
    for name, g, q in all_graphs:
        _, info = rank(g, q, "ppr", alpha=alpha, solver="iterative", dangling=dangling,
                       keep_trajectory=True, return_info=True)
        for pi_k in info["trajectory"]:
            _check_distribution(pi_k)


@pytest.mark.parametrize("dangling", ["restart", "self_loop"])
@pytest.mark.parametrize("solver", ["linear", "dense"])
def test_ppr_solvers_conserve_mass(all_graphs, solver, dangling):
    """Розв'язки лінійної системи та оберненої матриці — розподіли ймовірностей."""
    for name, g, q in all_graphs:
        _check_distribution(rank(g, q, "ppr", alpha=0.15, solver=solver, dangling=dangling))


@pytest.mark.parametrize("dangling", ["restart", "self_loop"])
@pytest.mark.parametrize("solver", ["expm_multiply", "series", "dense"])
def test_heat_conserves_mass(all_graphs, solver, dangling):
    """Теплове ядро u(t) — розподіл ймовірностей для всіх t на сітці."""
    for name, g, q in all_graphs:
        for t in (0.0, 0.1, 1.0, 5.0, 30.0, 100.0):
            _check_distribution(rank(g, q, "heat", t=t, solver=solver, dangling=dangling),
                                atol=1e-10)


@pytest.mark.parametrize("lazy", [False, True])
def test_walk_conserves_mass(all_graphs, lazy):
    """Блукання без перезапуску (звичайне і ліниве) зберігає суму на кожному кроці."""
    for name, g, q in all_graphs:
        _, info = rank(g, q, "walk", steps=200, lazy=lazy, keep_trajectory=True,
                       return_info=True)
        for pi_k in info["trajectory"]:
            _check_distribution(pi_k)


def test_multiple_seeds_conserve_mass(er_graph):
    """Кілька вузлів запиту з вагами: s і результат — розподіли ймовірностей."""
    x = rank(er_graph, [0, 5, 9], "ppr", seed_weights=[1, 2, 3])
    _check_distribution(x)
