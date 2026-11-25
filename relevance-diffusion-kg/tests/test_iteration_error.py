"""Перевірка адекватності 5: похибка ітерацій ‖π_k − π*‖₁ спадає не повільніше ніж (1 − α)^k."""

import numpy as np
import pytest

from reldiff.api import rank


@pytest.mark.parametrize("dangling", ["restart", "self_loop"])
@pytest.mark.parametrize("alpha", [0.05, 0.15, 0.3, 0.7])
def test_error_bound_18(all_graphs, alpha, dangling):
    """Оцінка (18): ‖π_k − π*‖₁ ≤ (1 − α)^k·‖s − π*‖₁ на кожному кроці."""
    for name, g, q in all_graphs:
        _, info = rank(g, q, "ppr", alpha=alpha, solver="iterative", dangling=dangling,
                       track_error=True, return_info=True)
        err = info["errors"]
        k = np.arange(err.size)
        bound = (1 - alpha) ** k * err[0]
        assert np.all(err <= bound + 1e-14), name
        assert err[0] <= 2.0


@pytest.mark.parametrize("alpha", [0.05, 0.15, 0.5])
def test_stopping_criterion_19(er_graph, alpha):
    """Критерій зупинки (19) гарантує ‖π_K − π*‖₁ < tol."""
    for tol in (1e-4, 1e-8, 1e-12):
        _, info = rank(er_graph, 0, "ppr", alpha=alpha, solver="iterative", tol=tol,
                       track_error=True, return_info=True)
        assert info["converged"]
        assert info["errors"][-1] < tol


def test_iterations_count_grows_as_alpha_decreases(er_graph):
    """Кількість ітерацій ~ ln(tol)/ln(1 − α): зростає при зменшенні α."""
    n_iter = [rank(er_graph, 0, "ppr", alpha=a, solver="iterative", tol=1e-10,
                   return_info=True)[1]["n_iter"] for a in (0.5, 0.15, 0.05)]
    assert n_iter[0] < n_iter[1] < n_iter[2]
    # теоретична верхня межа з (19): (1 − α)^K·2/α < tol
    for a, k in zip((0.5, 0.15, 0.05), n_iter):
        assert k <= np.ceil(np.log(1e-10 * a / 2) / np.log(1 - a)) + 1


def test_stagnation_on_unreachable_tol():
    """Недосяжний tol (нижче похибок округлення): ітерації зупиняються за застоєм,
    а не крутяться до max_iter, і похибка все одно на рівні машинної точності."""
    import networkx as nx
    g = nx.path_graph(40)
    _, info = rank(g, 0, "ppr", alpha=0.05, solver="iterative", tol=1e-17,
                   track_error=True, return_info=True)
    assert info["stagnated"] and not info["converged"]
    assert info["n_iter"] < 2000
    assert info["errors"][-1] < 1e-13
