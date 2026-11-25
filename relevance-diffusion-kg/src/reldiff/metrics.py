"""Метрики якості ранжування.

precision@k, nDCG@k, MRR, мінімальна кількість верхніх вузлів,
що охоплює всі релевантні.

Вхід — pd.Series оцінок (як повертає reldiff.api.rank; більше = релевантніше)
і множина релевантних вузлів. Релевантність бінарна.
- Вузли запиту за замовчуванням виключаються з ранжування і з множини релевантних
  (exclude_seeds=True).
- Вузли з оцінкою NaN вважаються невідібраними: їх немає в ранжуванні.
- Нічиї (однакові оцінки) розбиваються за параметром ties:
  "random" — випадково з фіксованим seed (за замовчуванням),
  "pessimistic" — релевантні вузли в кінці групи нічиїх,
  "optimistic" — релевантні вузли на початку групи.
MRR — середнє reciprocal_rank за набором запитів.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIES = ("random", "pessimistic", "optimistic")


def _prepare(scores, relevant, seeds, exclude_seeds):
    scores = pd.Series(scores, dtype=float)
    relevant = set(relevant)
    unknown = relevant - set(scores.index)
    if unknown:
        raise KeyError(f"Релевантних вузлів немає серед оцінених: {sorted(map(str, unknown))[:5]}")
    if seeds is not None and exclude_seeds:
        seeds = set(seeds) if isinstance(seeds, (list, tuple, set, np.ndarray)) else {seeds}
        scores = scores.drop(index=[q for q in seeds if q in scores.index])
        relevant = relevant - seeds
    if not relevant:
        raise ValueError("Множина релевантних вузлів порожня")
    return scores, relevant


def ranking(scores, relevant=None, seeds=None, exclude_seeds=True, ties="random", seed=0):
    """Упорядкований список відібраних вузлів (за спаданням оцінки, без NaN)."""
    if ties not in TIES:
        raise ValueError(f"ties має бути одним із {TIES}")
    if relevant is None:
        scores = pd.Series(scores, dtype=float)
        if seeds is not None and exclude_seeds:
            seeds = set(seeds) if isinstance(seeds, (list, tuple, set, np.ndarray)) else {seeds}
            scores = scores.drop(index=[q for q in seeds if q in scores.index])
        relevant = set()
    else:
        scores, relevant = _prepare(scores, relevant, seeds, exclude_seeds)
    scores = scores.dropna()
    nodes = list(scores.index)
    vals = scores.to_numpy()
    if ties == "random":
        tie_key = np.random.default_rng(seed).random(len(nodes))
    else:
        is_rel = np.array([v in relevant for v in nodes], dtype=float)
        tie_key = is_rel if ties == "pessimistic" else -is_rel
    order = np.lexsort((tie_key, -vals))        # головний ключ — останній
    return [nodes[i] for i in order]


def _ranked(scores, relevant, seeds, exclude_seeds, ties, seed):
    scores, relevant = _prepare(scores, relevant, seeds, exclude_seeds)
    order = ranking(scores, relevant, None, False, ties, seed)
    hits = np.array([v in relevant for v in order], dtype=bool)
    return hits, relevant


def precision_at_k(scores, relevant, k, seeds=None, exclude_seeds=True, ties="random", seed=0):
    """precision@k = |топ-k ∩ релевантні| / k (якщо відібрано < k вузлів, знаменник усе одно k)."""
    hits, _ = _ranked(scores, relevant, seeds, exclude_seeds, ties, seed)
    return hits[:k].sum() / k


def ndcg_at_k(scores, relevant, k, seeds=None, exclude_seeds=True, ties="random", seed=0):
    """nDCG@k з бінарною релевантністю: DCG = Σ_{i≤k} rel_i / log₂(i + 1),
    IDCG — те саме для ідеального ранжування (min(k, |R|) релевантних на початку)."""
    hits, relevant = _ranked(scores, relevant, seeds, exclude_seeds, ties, seed)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits[:k] * disc[:min(k, hits.size)]).sum()
    idcg = disc[:min(k, len(relevant))].sum()
    return dcg / idcg


def reciprocal_rank(scores, relevant, seeds=None, exclude_seeds=True, ties="random", seed=0):
    """1 / (позиція першого релевантного вузла); 0, якщо жоден не відібраний. MRR — середнє за запитами."""
    hits, _ = _ranked(scores, relevant, seeds, exclude_seeds, ties, seed)
    pos = np.flatnonzero(hits)
    return 1.0 / (pos[0] + 1) if pos.size else 0.0


def coverage_size(scores, relevant, seeds=None, exclude_seeds=True, ties="random", seed=0):
    """Мінімальна кількість верхніх вузлів ранжування, що охоплює всі релевантні
    (= позиція останнього релевантного); np.inf, якщо якийсь релевантний не відібраний."""
    hits, relevant = _ranked(scores, relevant, seeds, exclude_seeds, ties, seed)
    if hits.sum() < len(relevant):
        return np.inf
    return int(np.flatnonzero(hits)[-1] + 1)


def evaluate(scores, relevant, ks=(5, 10, 20), seeds=None, exclude_seeds=True,
             ties="random", seed=0):
    """Усі метрики для одного запиту: словник {"P@k", "nDCG@k", "RR", "coverage"}."""
    kw = dict(seeds=seeds, exclude_seeds=exclude_seeds, ties=ties, seed=seed)
    out = {}
    for k in ks:
        out[f"P@{k}"] = precision_at_k(scores, relevant, k, **kw)
        out[f"nDCG@{k}"] = ndcg_at_k(scores, relevant, k, **kw)
    out["RR"] = reciprocal_rank(scores, relevant, **kw)
    out["coverage"] = coverage_size(scores, relevant, **kw)
    return out
