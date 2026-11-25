"""Аналітичні розв'язки на простих графах (для перевірки адекватності).

Виведення — у docs/analitychni_rozvyazky.md, номери формул (5)–(17) звідти.
Порядок вузлів збігається з генераторами networkx:
nx.complete_graph(n), nx.cycle_graph(n), nx.path_graph(n) — вузли 0, …, n − 1;
nx.star_graph(m) — центр 0, листки 1, …, m.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Повний граф K_n
# ---------------------------------------------------------------------------

def complete_ppr(n, alpha, q=0):
    """PPR на повному графі, формула (6)."""
    x = np.full(n, (1 - alpha) / (n - alpha))
    x[q] = (alpha * (n - 2) + 1) / (n - alpha)
    return x


def complete_heat(n, t, q=0):
    """Теплове ядро на повному графі, формула (7)."""
    e = np.exp(-t * n / (n - 1))
    x = np.full(n, (1 - e) / n)
    x[q] = 1 / n + (1 - 1 / n) * e
    return x


def complete_walk(n, k, q=0):
    """Блукання без перезапуску на повному графі після k кроків, формула (5)."""
    lam = (-1.0 / (n - 1)) ** k
    x = np.full(n, 1 / n - lam / n)
    x[q] = 1 / n + (1 - 1 / n) * lam
    return x


# ---------------------------------------------------------------------------
# Цикл C_n
# ---------------------------------------------------------------------------

def _cycle_spectral(n, q, gain):
    """Загальна формула (4) на циклі: x_j = (1/n)·Σ_m gain(cos θ_m)·cos(θ_m (j − q))."""
    theta = 2 * np.pi * np.arange(n) / n
    j = (np.arange(n) - q) % n
    g = gain(np.cos(theta))
    return (np.cos(np.outer(j, theta)) @ g) / n


def cycle_heat(n, t, q=0):
    """Теплове ядро на циклі, формула (8)."""
    return _cycle_spectral(n, q, lambda lam: np.exp(-t * (1 - lam)))


def cycle_walk(n, k, q=0, lazy=False):
    """Блукання (або ліниве) на циклі після k кроків, формула (9)."""
    if lazy:
        return _cycle_spectral(n, q, lambda lam: ((1 + lam) / 2) ** k)
    return _cycle_spectral(n, q, lambda lam: lam ** k)


def cycle_r(alpha):
    """Знаменник геометричного спадання PPR на циклі: r = (1 − √(α(2 − α)))/(1 − α)."""
    if alpha == 1:
        return 0.0
    return (1 - np.sqrt(alpha * (2 - alpha))) / (1 - alpha)


def cycle_ppr(n, alpha, q=0):
    """PPR на циклі у замкненому вигляді, формула (10)."""
    r = cycle_r(alpha)
    j = (np.arange(n) - q) % n
    C = (1 - r) / ((1 + r) * (1 - r ** n))
    return C * (r ** j + r ** (n - j))


# ---------------------------------------------------------------------------
# Ланцюжок P_n — складання циклу C_{2(n−1)}, формула (11)
# ---------------------------------------------------------------------------

def _fold(x_cycle, n):
    """Склеює вузли j і N − j циклу C_N (N = 2(n − 1)) у вузол j ланцюжка, формула (11)."""
    N = 2 * (n - 1)
    x = x_cycle[:n].copy()
    x[1:n - 1] += x_cycle[N - np.arange(1, n - 1)]
    return x


def path_ppr(n, alpha, q=0):
    """PPR на ланцюжку через складання циклу; для q = 0 — формула (12)."""
    return _fold(cycle_ppr(2 * (n - 1), alpha, q), n)


def path_heat(n, t, q=0):
    """Теплове ядро на ланцюжку через складання циклу, формула (11)."""
    return _fold(cycle_heat(2 * (n - 1), t, q), n)


def path_walk(n, k, q=0, lazy=False):
    """Блукання (або ліниве) на ланцюжку через складання циклу, формула (11)."""
    return _fold(cycle_walk(2 * (n - 1), k, q, lazy=lazy), n)


# ---------------------------------------------------------------------------
# Зірка S_m (центр 0, листки 1..m)
# ---------------------------------------------------------------------------

def star_ppr(m, alpha, leaf=None):
    """PPR на зірці: вузол запиту — центр (leaf=None, формула (14)) або листок leaf (формула (16))."""
    x = np.empty(m + 1)
    if leaf is None:
        x[0] = 1 / (2 - alpha)
        x[1:] = (1 - alpha) / ((2 - alpha) * m)
    else:
        c = (1 - alpha) / (2 - alpha)
        x[0] = c
        x[1:] = (1 - alpha) * c / m
        x[leaf] = alpha + (1 - alpha) * c / m
    return x


def star_heat(m, t, leaf=None):
    """Теплове ядро на зірці: центр (формула (13)) або листок leaf (формула (15))."""
    e1, e2 = np.exp(-t), np.exp(-2 * t)
    x = np.empty(m + 1)
    if leaf is None:
        x[0] = (1 + e2) / 2
        x[1:] = (1 - e2) / (2 * m)
    else:
        x[0] = (1 - e2) / 2
        x[1:] = (1 + e2) / (2 * m) - e1 / m
        x[leaf] = (1 + e2) / (2 * m) + (1 - 1 / m) * e1
    return x


def star_walk(m, k, leaf=None, lazy=False):
    """Блукання на зірці після k кроків (розклад за φ₊, φ₋, w з розділу 4)."""
    phi_p = np.r_[1.0, np.full(m, 1 / m)] / 2      # ½φ₊ = d/Σd
    phi_m = np.r_[1.0, np.full(m, -1 / m)] / 2     # ½φ₋
    lam_m, lam_0 = (0.0, 0.5) if lazy else (-1.0, 0.0)
    if leaf is None:
        return phi_p + lam_m ** k * phi_m
    w = np.r_[0.0, np.full(m, -1 / m)]
    w[leaf] += 1.0
    return phi_p - lam_m ** k * phi_m + lam_0 ** k * w   # формула (17) для lazy
