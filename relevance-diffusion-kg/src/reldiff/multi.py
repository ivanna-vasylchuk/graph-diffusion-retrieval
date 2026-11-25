"""Поєднання релевантностей від кількох вузлів запиту.

Адитивне поєднання (середнє окремих розподілів π^(q)) — через лінійність дорівнює
одному запуску з s, рівномірним на вузлах запиту (для неорієнтованого графа або без
висячих вузлів; зі стратегією restart рядки висячих вузлів залежать від s).
Мультиплікативне — добуток (через суму логарифмів, нуль лишається нулем) або мінімум:
нагороджує вузли, до яких маса доходить від усіх вузлів запиту («мости»).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COMBINE = ("mean", "product", "min")


def combine_scores(scores_list, how="mean"):
    """Поєднує оцінки від окремих вузлів запиту (список pd.Series з однаковим індексом).

    how = "mean" (адитивне), "product" (добуток; рахуємо як exp(Σ ln x), x ≤ 0 → 0),
    "min" (мінімум). Повертає pd.Series з тим самим індексом.
    """
    if how not in COMBINE:
        raise ValueError(f"Невідомий спосіб поєднання {how!r}; доступні: {COMBINE}")
    if not scores_list:
        raise ValueError("Потрібна хоча б одна множина оцінок")
    index = scores_list[0].index
    X = np.vstack([s.reindex(index).to_numpy(dtype=float) for s in scores_list])
    if how == "mean":
        out = X.mean(axis=0)
    elif how == "min":
        out = X.min(axis=0)
    else:
        pos = np.all(X > 0, axis=0)
        out = np.zeros(X.shape[1])
        out[pos] = np.exp(np.log(X[:, pos]).sum(axis=0))
    return pd.Series(out, index=index, name=f"{scores_list[0].name}_{how}")
