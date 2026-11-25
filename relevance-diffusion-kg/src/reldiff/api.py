"""Головний інтерфейс модуля.

rank(graph, seeds, method="ppr", **params) -> оцінка релевантності для кожного вузла.
Експерименти звертаються до моделей лише через цю функцію.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import baselines, models, multi

DIFFUSION_METHODS = ("ppr", "heat", "walk")
BASELINE_METHODS = ("shortest_path", "common_neighbors", "k_hop")
METHODS = DIFFUSION_METHODS + BASELINE_METHODS
PPR_SOLVERS = ("linear", "iterative", "dense")
HEAT_SOLVERS = ("expm_multiply", "series", "dense")
# Замовчування для висячих вузлів: restart для обох моделей (рішення після п. 4 ТЗ: петля
# підсилює висячий вузол у 1/α раза в PPR, а в тепловому ядрі робить його поглинаючим —
# на ER з 10% висячих при t = 40 у тупиках опиняється 0.97 маси). Для блукання без
# перезапуску (перевірки збіжності) лишається петля.
DEFAULT_DANGLING = {"ppr": models.DANGLING_RESTART,
                    "heat": models.DANGLING_RESTART,
                    "walk": models.DANGLING_SELF_LOOP}


def rank(graph, seeds, method="ppr", *, alpha=0.15, t=5.0, steps=50, lazy=False,
         solver=None, dangling=None, weight="weight", type_weights=None,
         type_attr="type", as_undirected=False, seed_weights=None,
         normalize=None, tol=1e-10, max_iter=100_000, track_error=False,
         keep_trajectory=False, hops=2, agg="min", weighted=False,
         combine=None, return_info=False):
    """Оцінки релевантності вузлів графа відносно вузлів запиту.

    graph   — networkx-граф (Graph, DiGraph, MultiGraph, MultiDiGraph);
    seeds   — вузол запиту або список вузлів (s рівномірний на них, якщо не задано seed_weights);
    method  — "ppr" (особистий PageRank, параметр alpha ∈ (0, 1]),
              "heat" (теплове ядро, параметр t ≥ 0),
              "walk" (блукання без перезапуску, `steps` кроків; lazy=True — ліниве);
              базові методи (reldiff.baselines): "shortest_path" (1/(1 + відстань)),
              "common_neighbors", "k_hop" (радіус `hops`); для кількох вузлів запиту
              відстань агрегується за `agg` ("min", "mean", "max"); weighted=True —
              довжина ребра 1/w для найкоротшого шляху; невідібрані вузли мають NaN;
    solver  — для "ppr": "iterative" (за замовч., гарантує ‖π − π*‖₁ < tol = 1e-10), "linear", "dense";
              для "heat": "expm_multiply" (за замовч.), "series", "dense";
    dangling — стратегія для висячих вузлів: "restart" (за замовч. для PPR і теплового ядра)
              або "self_loop" (за замовч. для блукання без перезапуску);
    weight / type_weights / type_attr / as_undirected — побудова матриці ваг W;
    normalize — None, "degree" (оцінка π_i / d_i — прибирає перевагу хабів) або
              "sqrt_degree" (π_i / √d_i — те саме ранжування, що дає PPR / теплове ядро
              з симетричною матрицею D^(−1/2)·W·D^(−1/2), див. tests/test_normalization.py);
              для вузлів з d_i = 0 оцінка 0;
    track_error — для ітерацій PPR: обчислити π* лінійною системою і зберегти ‖π_k − π*‖₁;
    keep_trajectory — зберегти π_k для ітерацій PPR і блукання;
    combine — None (s рівномірний на вузлах запиту) або "mean" / "product" / "min":
              рахувати розподіл окремо для кожного вузла запиту (з тим самим нормуванням)
              і поєднати їх (reldiff.multi.combine_scores);
    return_info — повернути (оцінки, info) замість лише оцінок.

    Повертає pd.Series: індекс — вузли графа, значення — оцінки.
    """
    if method not in METHODS:
        raise ValueError(f"Невідомий метод {method!r}; доступні: {METHODS}")
    if combine is not None:
        if method in BASELINE_METHODS:
            raise ValueError("combine підтримується лише для дифузійних методів")
        seed_list = list(seeds) if isinstance(seeds, (list, tuple, set, np.ndarray)) else [seeds]
        kw = dict(alpha=alpha, t=t, steps=steps, lazy=lazy, solver=solver, dangling=dangling,
                  weight=weight, type_weights=type_weights, type_attr=type_attr,
                  as_undirected=as_undirected, normalize=normalize, tol=tol, max_iter=max_iter)
        parts = [rank(graph, q, method, **kw) for q in seed_list]
        result = multi.combine_scores(parts, combine)
        info = {"method": method, "combine": combine, "n_seeds": len(seed_list)}
        return (result, info) if return_info else result
    if method in BASELINE_METHODS:
        if method == "shortest_path":
            result = baselines.shortest_path(graph, seeds, agg=agg, weighted=weighted, weight=weight)
        elif method == "common_neighbors":
            result = baselines.common_neighbors(graph, seeds)
        else:
            result = baselines.k_hop(graph, seeds, hops=hops, agg=agg)
        info = {"method": method, "n": len(result)}
        return (result, info) if return_info else result
    if dangling is None:
        dangling = DEFAULT_DANGLING[method]

    W, nodes = models.weight_matrix(graph, weight=weight, type_weights=type_weights,
                                    type_attr=type_attr, as_undirected=as_undirected)
    index = {v: i for i, v in enumerate(nodes)}
    seed_list = list(seeds) if isinstance(seeds, (list, tuple, set, np.ndarray)) else [seeds]
    missing = [q for q in seed_list if q not in index]
    if missing:
        raise KeyError(f"Вузлів запиту немає в графі: {missing}")
    s = models.seed_vector(len(nodes), [index[q] for q in seed_list], seed_weights)
    P = models.transition_matrix(W, dangling=dangling, s=s)

    info = {"method": method, "dangling": dangling, "n": len(nodes)}

    if method == "ppr":
        # замовчування — ітерації (рішення після п. 7.4: spsolve масштабується як ~n^2.9,
        # при n = 16000 — 359 с проти 0.011 с ітерацій з похибкою ~1e-11)
        solver = solver or "iterative"
        if solver == "linear":
            scores = models.ppr_linear(P, s, alpha)
        elif solver == "iterative":
            pi_ref = models.ppr_linear(P, s, alpha) if track_error else None
            scores, it_info = models.ppr_iterative(P, s, alpha, tol=tol, max_iter=max_iter,
                                                   pi_ref=pi_ref,
                                                   keep_trajectory=keep_trajectory)
            info.update(it_info)
        elif solver == "dense":
            scores = models.ppr_dense_inverse(P, s, alpha)
        else:
            raise ValueError(f"Невідомий спосіб для PPR {solver!r}; доступні: {PPR_SOLVERS}")
        info["alpha"] = alpha
    elif method == "heat":
        solver = solver or "expm_multiply"
        if solver == "expm_multiply":
            scores = models.heat_expm_multiply(P, s, t)
        elif solver == "series":
            scores, h_info = models.heat_series(P, s, t)
            info.update(h_info)
        elif solver == "dense":
            scores = models.heat_dense_expm(P, s, t)
        else:
            raise ValueError(f"Невідомий спосіб для теплового ядра {solver!r}; доступні: {HEAT_SOLVERS}")
        info["t"] = t
    else:  # walk
        traj = models.random_walk(P, s, steps, lazy=lazy)
        scores = traj[-1]
        if keep_trajectory:
            info["trajectory"] = traj
        info.update(steps=steps, lazy=lazy)
    info["solver"] = solver

    if normalize in ("degree", "sqrt_degree"):
        d = models.degrees(W)
        den = d if normalize == "degree" else np.sqrt(d)
        scores = np.divide(scores, den, out=np.zeros_like(scores), where=d > 0)
    elif normalize is not None:
        raise ValueError(f"Невідоме нормування {normalize!r}; доступні: None, 'degree', 'sqrt_degree'")

    result = pd.Series(np.asarray(scores, dtype=float), index=pd.Index(nodes, name="вузол", tupleize_cols=False),
                       name=f"{method}")
    info["degrees"] = pd.Series(models.degrees(W), index=result.index, name="d")
    return (result, info) if return_info else result


def spectrum(graph, *, weight="weight", type_weights=None, type_attr="type",
             as_undirected=False, dangling="self_loop", lazy=False, return_summary=False):
    """Власні значення матриці переходів P графа (щільно, для графів до кількох тисяч вузлів).

    Неорієнтований граф (або as_undirected=True) без ізольованих вузлів — через симетричну
    D^(−1/2)·W·D^(−1/2): дійсний спектр за спаданням. Орієнтований граф або граф з
    ізольованими вузлами — загальний розрахунок власних значень P (стратегія dangling
    для висячих вузлів; для "restart" спектр залежить від s, тому тут не підтримується),
    значення можуть бути комплексними, сортування за спаданням модуля.
    return_summary=True — повернути також словник: modulus_2, λ₂, λ_min, щілина 1 − λ₂.
    """
    if dangling == models.DANGLING_RESTART:
        raise ValueError("Спектр для стратегії 'restart' залежить від вузлів запиту; "
                         "використайте dangling='self_loop'.")
    W, _ = models.weight_matrix(graph, weight=weight, type_weights=type_weights,
                                type_attr=type_attr, as_undirected=as_undirected)
    symmetric = (not graph.is_directed() or as_undirected) and bool(np.all(models.degrees(W) > 0))
    lam = models.transition_spectrum(W, symmetric=symmetric, dangling=dangling, lazy=lazy)
    return (lam, models.spectral_summary(lam)) if return_summary else lam
