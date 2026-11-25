"""Перевірка адекватності 4: збіг способів розрахунку всередині кожної моделі."""

import numpy as np
import pytest

from reldiff.api import rank

TOL = 1e-10


@pytest.mark.parametrize("dangling", ["restart", "self_loop"])
@pytest.mark.parametrize("alpha", [0.01, 0.15, 0.5, 0.95])
def test_ppr_three_solvers(all_graphs, alpha, dangling):
    """PPR: ітерація = лінійна система = щільна обернена матриця."""
    for name, g, q in all_graphs:
        x = {sv: rank(g, q, "ppr", alpha=alpha, solver=sv, dangling=dangling,
                      tol=1e-13).to_numpy()
             for sv in ("iterative", "linear", "dense")}
        assert np.abs(x["iterative"] - x["dense"]).sum() < TOL, name
        assert np.abs(x["linear"] - x["dense"]).sum() < TOL, name


@pytest.mark.parametrize("dangling", ["restart", "self_loop"])
@pytest.mark.parametrize("t", [0.0, 0.5, 3.0, 12.0, 50.0])
def test_heat_three_solvers(all_graphs, t, dangling):
    """Теплове ядро: обрізаний ряд = expm_multiply = щільна expm."""
    for name, g, q in all_graphs:
        x = {sv: rank(g, q, "heat", t=t, solver=sv, dangling=dangling).to_numpy()
             for sv in ("series", "expm_multiply", "dense")}
        assert np.abs(x["series"] - x["dense"]).sum() < TOL, name
        assert np.abs(x["expm_multiply"] - x["dense"]).sum() < TOL, name


def test_ppr_power_series(er_graph):
    """PPR як геометричний ряд α·Σ_k (1 − α)^k·s·P^k = π* (зв'язок з блуканням)."""
    alpha = 0.3
    ref = rank(er_graph, 0, "ppr", alpha=alpha, solver="linear").to_numpy()
    traj = rank(er_graph, 0, "walk", steps=150, keep_trajectory=True,
                return_info=True)[1]["trajectory"]
    k = np.arange(traj.shape[0])
    series = alpha * ((1 - alpha) ** k) @ traj
    assert np.abs(series - ref).sum() < 1e-12


def test_heat_power_series(er_graph):
    """Теплове ядро як пуассонівський ряд e^(−t)·Σ_k (t^k/k!)·s·P^k = u(t)."""
    from scipy.stats import poisson
    t = 4.0
    ref = rank(er_graph, 0, "heat", t=t).to_numpy()
    traj = rank(er_graph, 0, "walk", steps=80, keep_trajectory=True,
                return_info=True)[1]["trajectory"]
    weights = poisson.pmf(np.arange(traj.shape[0]), t)
    assert np.abs(weights @ traj - ref).sum() < 1e-12


@pytest.mark.parametrize("alpha", [0.05, 0.15, 0.5])
def test_default_ppr_solver_guarantee(all_graphs, alpha):
    """Замовчування PPR — ітерації з tol = 1e-10 (рішення після п. 7.4): ‖π − π*‖₁ < 1e-10."""
    for name, g, q in all_graphs:
        x = rank(g, q, "ppr", alpha=alpha).to_numpy()
        ref = rank(g, q, "ppr", alpha=alpha, solver="linear").to_numpy()
        assert np.abs(x - ref).sum() < 1e-10, name
