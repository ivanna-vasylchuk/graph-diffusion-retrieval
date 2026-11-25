"""Математичні моделі поширення релевантності.

- матриця ваг W і матриця переходів P = D⁻¹W з вибором стратегії для висячих вузлів
  (за замовчуванням для обох моделей — restart, маса на s; див. api.DEFAULT_DANGLING);
- особистий PageRank: ітерація (з історією похибки), лінійна система,
  щільна обернена матриця (перевірка на малих графах);
- теплове ядро u(t) = s·e^(−tL), L = I − P: обрізаний ряд, expm_multiply,
  щільна expm (перевірка на малих графах);
- блукання без перезапуску та ліниве блукання P' = (I + P)/2.

Розподіли — вектори-рядки (1-D масиви numpy), сума компонент = 1.
Добуток вектора-рядка на матрицю x·P обчислюємо як P.T @ x.
Номери формул (1)–(19) — з docs/analitychni_rozvyazky.md.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, spsolve

# Стратегії обробки висячих вузлів (d_i = 0)
DANGLING_RESTART = "restart"      # маса висячого вузла повертається на s
DANGLING_SELF_LOOP = "self_loop"  # маса залишається у висячому вузлі (петля)
DANGLING_STRATEGIES = (DANGLING_RESTART, DANGLING_SELF_LOOP)

# Обмеження на розмір для щільних перевірочних методів
MAX_DENSE_N = 3000


# ---------------------------------------------------------------------------
# Граф -> матриці
# ---------------------------------------------------------------------------

def weight_matrix(graph, weight="weight", type_weights=None, type_attr="type",
                  as_undirected=False, nodelist=None):
    """Матриця ваг W (CSR) і список вузлів у порядку рядків.

    Вага ребра: type_weights[тип ребра], якщо задано словник type_weights,
    інакше атрибут `weight` (за відсутності — 1). Паралельні ребра
    мультиграфа сумуються. Для неорієнтованого графа W симетрична.
    as_undirected=True симетризує орієнтований граф: W ← W + Wᵀ.
    """
    if nodelist is None:
        nodelist = list(graph.nodes())
    index = {v: i for i, v in enumerate(nodelist)}
    n = len(nodelist)
    directed = graph.is_directed()

    rows, cols, vals = [], [], []
    for u, v, data in graph.edges(data=True):
        if type_weights is not None:
            etype = data.get(type_attr)
            if etype not in type_weights:
                raise KeyError(f"Немає ваги для типу ребра {etype!r} ({u} → {v})")
            w = float(type_weights[etype])
        else:
            w = float(data.get(weight, 1.0))
        if w < 0:
            raise ValueError(f"Від'ємна вага ребра {u} → {v}: {w}")
        i, j = index[u], index[v]
        rows.append(i); cols.append(j); vals.append(w)
        if not directed and i != j:
            rows.append(j); cols.append(i); vals.append(w)

    W = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=float)
    W.sum_duplicates()
    if directed and as_undirected:
        W = (W + W.T).tocsr()
    return W, nodelist


def seed_vector(n, seed_idx, seed_weights=None):
    """Початковий розподіл s на вузлах запиту (індекси seed_idx).

    За замовчуванням — рівномірний на вузлах запиту; для одного вузла s = e_q.
    """
    seed_idx = np.atleast_1d(np.asarray(seed_idx, dtype=int))
    if seed_idx.size == 0:
        raise ValueError("Потрібен хоча б один вузол запиту")
    s = np.zeros(n)
    if seed_weights is None:
        np.add.at(s, seed_idx, 1.0)
    else:
        w = np.asarray(seed_weights, dtype=float)
        if w.shape != seed_idx.shape or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("seed_weights: невід'ємні, тієї ж довжини, що й вузли запиту")
        np.add.at(s, seed_idx, w)
    return s / s.sum()


def degrees(W):
    """Вихідні (зважені) степені d_i = Σ_j w_ij."""
    return np.asarray(W.sum(axis=1)).ravel()


def stationary_degree(W):
    """Стаціонарний розподіл блукання без перезапуску на неорієнтованому графі: d/Σd."""
    d = degrees(W)
    return d / d.sum()


def transition_matrix(W, dangling=DANGLING_RESTART, s=None):
    """Матриця переходів P = D⁻¹W (CSR, стохастична за рядками).

    Рядки висячих вузлів (d_i = 0) заповнюються за стратегією:
    - "restart": рядок = s (маса повертається на вузли запиту; потрібен s);
    - "self_loop": P_ii = 1 (маса залишається у вузлі).
    Повертає P, у якій кожен рядок має суму 1.
    """
    if dangling not in DANGLING_STRATEGIES:
        raise ValueError(f"Невідома стратегія для висячих вузлів: {dangling!r}")
    W = sp.csr_matrix(W, dtype=float)
    n = W.shape[0]
    d = degrees(W)
    is_dangling = d == 0
    inv_d = np.divide(1.0, d, out=np.zeros(n), where=~is_dangling)
    P = sp.diags(inv_d) @ W

    idx = np.flatnonzero(is_dangling)
    if idx.size:
        if dangling == DANGLING_RESTART:
            if s is None:
                raise ValueError("Для стратегії 'restart' потрібен розподіл s")
            s = np.asarray(s, dtype=float)
            support = np.flatnonzero(s)
            r = np.repeat(idx, support.size)
            c = np.tile(support, idx.size)
            v = np.tile(s[support], idx.size)
        else:
            r, c, v = idx, idx, np.ones(idx.size)
        P = P + sp.csr_matrix((v, (r, c)), shape=(n, n))
    return sp.csr_matrix(P)


def lazy_matrix(P):
    """Ліниве блукання P' = (I + P)/2: прибирає коливання на дводольних графах."""
    n = P.shape[0]
    return sp.csr_matrix((sp.identity(n, format="csr") + P) / 2.0)


def laplacian_rw(P):
    """Нормований лапласіан випадкового блукання L = I − P."""
    n = P.shape[0]
    return sp.csr_matrix(sp.identity(n, format="csr") - P)


def _check_dense(n):
    if n > MAX_DENSE_N:
        raise ValueError(f"Щільний метод лише для малих графів (n ≤ {MAX_DENSE_N}), а n = {n}")


def _check_alpha(alpha):
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"α має належати (0, 1], а задано α = {alpha}")


# ---------------------------------------------------------------------------
# Спектр матриці переходів
# ---------------------------------------------------------------------------

def transition_spectrum(W, symmetric, dangling=DANGLING_SELF_LOOP, s=None, lazy=False):
    """Власні значення матриці переходів P = D⁻¹W (щільний розрахунок, n ≤ MAX_DENSE_N).

    symmetric=True (неорієнтований граф, W = Wᵀ, усі d_i > 0): P подібна до симетричної
    D^(−1/2)·W·D^(−1/2), тому спектр дійсний; рахуємо eigvalsh і сортуємо за спаданням.
    symmetric=False (орієнтований граф або є вузли з d_i = 0): загальний розрахунок
    власних значень P (з обраною стратегією для висячих вузлів); значення можуть бути
    комплексними, сортуємо за спаданням модуля.
    lazy=True — спектр лінивого блукання (I + P)/2, тобто (1 + λ)/2.
    """
    W = sp.csr_matrix(W, dtype=float)
    n = W.shape[0]
    _check_dense(n)
    d = degrees(W)
    if symmetric:
        if abs(W - W.T).max() > 1e-12 * max(1.0, abs(W).max()):
            raise ValueError("symmetric=True, але матриця ваг несиметрична: граф орієнтований? "
                             "Використайте symmetric=False (загальний розрахунок).")
        if np.any(d == 0):
            raise ValueError("Є вузли з нульовим степенем: D^(−1/2) не визначена; "
                             "використайте symmetric=False.")
        r = 1.0 / np.sqrt(d)
        S = (sp.diags(r) @ W @ sp.diags(r)).toarray()
        lam = np.linalg.eigvalsh((S + S.T) / 2)[::-1]
    else:
        P = transition_matrix(W, dangling=dangling, s=s).toarray()
        lam = np.linalg.eigvals(P)
        lam = lam[np.lexsort((-lam.real, -np.abs(lam)))]
        if np.allclose(lam.imag, 0, atol=1e-12):
            lam = lam.real
    return (1 + lam) / 2 if lazy else lam


def spectral_summary(lam):
    """Ключові характеристики спектра P (λ відсортовані, як повертає transition_spectrum).

    modulus_2 — друге за модулем власне значення (визначає швидкість збіжності блукання
    і, з множником (1 − α), ітерацій PPR); для дійсного спектра також λ₂ (друге за
    величиною), λ_min, спектральна щілина 1 − λ₂ (швидкість теплового ядра і лінивого блукання).
    """
    lam = np.asarray(lam)
    mods = np.sort(np.abs(lam))[::-1]
    out = {"modulus_2": float(mods[1]) if lam.size > 1 else 0.0, "real": bool(np.isrealobj(lam))}
    if out["real"]:
        out.update(lambda_2=float(lam[1]), lambda_min=float(lam[-1]), gap=float(1 - lam[1]),
                   lazy_rate=float((1 + lam[1]) / 2))
    return out


# ---------------------------------------------------------------------------
# Особистий PageRank
# ---------------------------------------------------------------------------

def ppr_iterative(P, s, alpha, tol=1e-12, max_iter=100_000, pi_ref=None,
                  keep_trajectory=False, patience=50):
    """PPR ітерацією π_{k+1} = α·s + (1 − α)·π_k·P, π_0 = s.

    Зупинка за апостеріорною оцінкою (19): ‖π_{k+1} − π_k‖₁·(1 − α)/α < tol,
    що гарантує ‖π* − π_{k+1}‖₁ < tol.
    Якщо tol менший за досяжну в арифметиці з плаваючою комою точність, нев'язка
    перестає спадати через похибки округлення; тоді зупиняємося, коли вона не
    оновлює мінімум `patience` кроків поспіль (info["stagnated"] = True, converged = False).
    Повертає (π, info): info["residuals"] — ‖π_{k+1} − π_k‖₁ на кожному кроці;
    info["errors"] — ‖π_k − π*‖₁ для k = 0, 1, … (якщо задано pi_ref = π*);
    info["trajectory"] — масив π_k (якщо keep_trajectory=True).
    """
    _check_alpha(alpha)
    PT = sp.csr_matrix(P.T)
    s = np.asarray(s, dtype=float)
    pi = s.copy()
    residuals, errors, traj = [], [], []
    if pi_ref is not None:
        errors.append(np.abs(pi - pi_ref).sum())
    if keep_trajectory:
        traj.append(pi.copy())

    factor = (1.0 - alpha) / alpha
    converged = stagnated = False
    best, since_best = np.inf, 0
    for _ in range(max_iter):
        new = alpha * s + (1.0 - alpha) * (PT @ pi)
        res = np.abs(new - pi).sum()
        pi = new
        residuals.append(res)
        if pi_ref is not None:
            errors.append(np.abs(pi - pi_ref).sum())
        if keep_trajectory:
            traj.append(pi.copy())
        if res * factor < tol:
            converged = True
            break
        if res < best:
            best, since_best = res, 0
        else:
            since_best += 1
            if since_best >= patience:
                stagnated = True
                break

    info = {"n_iter": len(residuals), "converged": converged, "stagnated": stagnated,
            "residuals": np.array(residuals)}
    if pi_ref is not None:
        info["errors"] = np.array(errors)
    if keep_trajectory:
        info["trajectory"] = np.array(traj)
    return pi, info


def ppr_linear(P, s, alpha):
    """PPR розв'язанням лінійної системи (I − (1 − α)P)ᵀ·πᵀ = α·sᵀ (розріджено, spsolve)."""
    _check_alpha(alpha)
    n = P.shape[0]
    A = sp.csc_matrix(sp.identity(n) - (1.0 - alpha) * P.T)
    return spsolve(A, alpha * np.asarray(s, dtype=float))


def ppr_dense_inverse(P, s, alpha):
    """PPR через щільну обернену матрицю: π* = α·s·(I − (1 − α)P)⁻¹ (лише для перевірки)."""
    _check_alpha(alpha)
    n = P.shape[0]
    _check_dense(n)
    Pd = P.toarray() if sp.issparse(P) else np.asarray(P)
    M = np.linalg.inv(np.eye(n) - (1.0 - alpha) * Pd)
    return alpha * np.asarray(s, dtype=float) @ M


# ---------------------------------------------------------------------------
# Теплове ядро
# ---------------------------------------------------------------------------

def heat_series(P, s, t, tol=1e-15, max_step=10.0, max_terms=10_000):
    """Теплове ядро обрізаним рядом u(t) = e^(−t)·Σ_k (t^k/k!)·s·P^k.

    Щоб уникнути переповнення і втрати точності при великих t, інтервал [0, t]
    ділимо на m = ⌈t / max_step⌉ рівних кроків τ = t/m і застосовуємо ряд m разів
    (напівгрупова властивість e^(−tL) = (e^(−τL))^m). У кожному кроці ряд
    обриваємо, коли k ≥ τ і ‖доданок‖₁ < tol.
    Повертає (u, info): info["n_terms"] — загальна кількість доданків.
    """
    if t < 0:
        raise ValueError(f"t має бути ≥ 0, а задано t = {t}")
    PT = sp.csr_matrix(P.T)
    u = np.asarray(s, dtype=float).copy()
    if t == 0:
        return u, {"n_terms": 0, "n_steps": 0}
    m = max(1, math.ceil(t / max_step))
    tau = t / m
    total_terms = 0
    for _ in range(m):
        term = math.exp(-tau) * u
        acc = term.copy()
        k = 0
        while k < max_terms:
            k += 1
            term = (PT @ term) * (tau / k)
            acc += term
            if k >= tau and np.abs(term).sum() < tol:
                break
        total_terms += k
        u = acc
    return u, {"n_terms": total_terms, "n_steps": m}


def heat_expm_multiply(P, s, t):
    """Теплове ядро через scipy.sparse.linalg.expm_multiply: u(t)ᵀ = e^(−t·Lᵀ)·sᵀ."""
    if t < 0:
        raise ValueError(f"t має бути ≥ 0, а задано t = {t}")
    L = laplacian_rw(P)
    return expm_multiply(-t * sp.csr_matrix(L.T), np.asarray(s, dtype=float))


def heat_dense_expm(P, s, t):
    """Теплове ядро через щільну scipy.linalg.expm: u(t) = s·e^(−tL) (лише для перевірки)."""
    if t < 0:
        raise ValueError(f"t має бути ≥ 0, а задано t = {t}")
    n = P.shape[0]
    _check_dense(n)
    L = laplacian_rw(P).toarray()
    return np.asarray(s, dtype=float) @ sla.expm(-t * L)


# ---------------------------------------------------------------------------
# Блукання без перезапуску
# ---------------------------------------------------------------------------

def random_walk(P, s, steps, lazy=False):
    """Блукання без перезапуску π_{k+1} = π_k·P (або ліниве, P' = (I + P)/2).

    Повертає траєкторію — масив розміру (steps + 1, n), рядок k = π_k.
    """
    if lazy:
        P = lazy_matrix(P)
    PT = sp.csr_matrix(P.T)
    pi = np.asarray(s, dtype=float).copy()
    traj = np.empty((steps + 1, pi.size))
    traj[0] = pi
    for k in range(1, steps + 1):
        pi = PT @ pi
        traj[k] = pi
    return traj


def is_bipartite(graph):
    """Чи є (неорієнтований) граф дводольним — тоді блукання без перезапуску коливається."""
    g = graph.to_undirected(as_view=True) if graph.is_directed() else graph
    return nx.is_bipartite(g)
