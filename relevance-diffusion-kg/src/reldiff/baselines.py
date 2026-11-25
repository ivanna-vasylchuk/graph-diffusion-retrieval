"""Прості способи оцінювання близькості до вузлів запиту.

Довжина найкоротшого шляху, кількість спільних сусідів,
вибір вузлів у межах k кроків з упорядкуванням за відстанню.

Інтерфейс як у reldiff.api.rank: f(graph, seeds, ...) -> pd.Series оцінок
(індекс — вузли графа, більше = релевантніше). Вузли, яких метод не відбирає
(недосяжні; поза радіусом k), мають оцінку NaN.
Орієнтованість ребер ігнорується (графи розглядаються як неорієнтовані).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

AGGREGATIONS = ("min", "mean", "max")


def _seed_list(graph, seeds):
    seeds = list(seeds) if isinstance(seeds, (list, tuple, set, np.ndarray)) else [seeds]
    missing = [q for q in seeds if q not in graph]
    if missing:
        raise KeyError(f"Вузлів запиту немає в графі: {missing}")
    return seeds


def _undirected(graph):
    return graph.to_undirected(as_view=True) if graph.is_directed() else graph


def _series(graph, values, name):
    return pd.Series(values, index=pd.Index(list(graph.nodes()), name="вузол",
                                            tupleize_cols=False), name=name, dtype=float)


def distances(graph, seeds, agg="min", weighted=False, weight="weight"):
    """Відстань від кожного вузла до вузлів запиту (np.inf — недосяжний).

    weighted=False — кількість ребер; weighted=True — довжина ребра 1/w
    (більша вага = сильніший зв'язок = коротше ребро).
    Для кількох вузлів запиту: agg = "min" (до найближчого), "mean", "max".
    """
    if agg not in AGGREGATIONS:
        raise ValueError(f"agg має бути одним із {AGGREGATIONS}")
    g = _undirected(graph)
    nodes = list(graph.nodes())
    D = np.full((len(seeds), len(nodes)), np.inf)
    for r, q in enumerate(seeds):
        if weighted:
            length = lambda u, v, d: 1.0 / d.get(weight, 1.0) if d.get(weight, 1.0) > 0 else None
            dist = nx.single_source_dijkstra_path_length(g, q, weight=length)
        else:
            dist = nx.single_source_shortest_path_length(g, q)
        for c, v in enumerate(nodes):
            if v in dist:
                D[r, c] = dist[v]
    return {"min": D.min, "mean": D.mean, "max": D.max}[agg](axis=0)


def shortest_path(graph, seeds, agg="min", weighted=False, weight="weight"):
    """Оцінка за найкоротшим шляхом: 1/(1 + dist); недосяжні вузли — NaN."""
    seeds = _seed_list(graph, seeds)
    d = distances(graph, seeds, agg=agg, weighted=weighted, weight=weight)
    with np.errstate(divide="ignore"):
        scores = np.where(np.isfinite(d), 1.0 / (1.0 + d), np.nan)
    return _series(graph, scores, "shortest_path")


def common_neighbors(graph, seeds):
    """Кількість спільних сусідів з вузлами запиту: Σ_q |N(v) ∩ N(q)|."""
    seeds = _seed_list(graph, seeds)
    g = _undirected(graph)
    seed_nbrs = [set(g.neighbors(q)) - {q} for q in seeds]
    scores = [sum(len((set(g.neighbors(v)) - {v}) & nq) for nq in seed_nbrs)
              for v in graph.nodes()]
    return _series(graph, scores, "common_neighbors")


def k_hop(graph, seeds, hops=2, agg="min"):
    """Вузли в межах hops кроків від вузлів запиту, упорядковані за відстанню.

    Оцінка hops + 1 − dist для dist ≤ hops (ближчі — вище), інакше NaN (не відібрано).
    """
    if hops < 0:
        raise ValueError("hops має бути ≥ 0")
    seeds = _seed_list(graph, seeds)
    d = distances(graph, seeds, agg=agg)
    scores = np.where(d <= hops, hops + 1.0 - d, np.nan)
    return _series(graph, scores, "k_hop")
