"""Тести метрик (reldiff.metrics) на прикладах із ручною відповіддю."""

import numpy as np
import pandas as pd
import pytest

from reldiff import metrics as M

# Оцінки: запит q найвищий, далі a > b > c > d > e; релевантні {b, d}
SCORES = pd.Series({"q": 1.0, "a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6, "e": 0.5})
REL = {"b", "d"}
L2, L3, L5 = np.log2(2), np.log2(3), np.log2(5)


def test_ranking_excludes_seed():
    assert M.ranking(SCORES, seeds="q") == ["a", "b", "c", "d", "e"]
    assert M.ranking(SCORES, seeds="q", exclude_seeds=False)[0] == "q"


@pytest.mark.parametrize("k,expected", [(1, 0.0), (2, 0.5), (4, 0.5), (10, 0.2)])
def test_precision_by_hand(k, expected):
    """Ранжування a, b, c, d, e: P@1 = 0, P@2 = 1/2, P@4 = 2/4, P@10 = 2/10."""
    assert M.precision_at_k(SCORES, REL, k, seeds="q") == pytest.approx(expected)


def test_ndcg_by_hand():
    """Релевантні на позиціях 2 і 4: DCG@4 = 1/log₂3 + 1/log₂5; IDCG@4 = 1 + 1/log₂3."""
    assert M.ndcg_at_k(SCORES, REL, 2, seeds="q") == pytest.approx((1 / L3) / (1 / L2 + 1 / L3))
    assert M.ndcg_at_k(SCORES, REL, 4, seeds="q") == pytest.approx((1 / L3 + 1 / L5) / (1 + 1 / L3))
    # ідеальне ранжування дає 1
    ideal = pd.Series({"b": 2.0, "d": 1.5, "a": 1.0})
    assert M.ndcg_at_k(ideal, REL, 3) == pytest.approx(1.0)


def test_rr_and_coverage_by_hand():
    """Перший релевантний на позиції 2 → RR = 1/2; останній на позиції 4 → охоплення 4."""
    assert M.reciprocal_rank(SCORES, REL, seeds="q") == pytest.approx(0.5)
    assert M.coverage_size(SCORES, REL, seeds="q") == 4
    # без виключення запиту все зсувається на 1
    assert M.reciprocal_rank(SCORES, REL, seeds="q", exclude_seeds=False) == pytest.approx(1 / 3)
    assert M.coverage_size(SCORES, REL, seeds="q", exclude_seeds=False) == 5


def test_seed_removed_from_relevant():
    """Якщо запит потрапив до релевантних, при виключенні він прибирається і звідти."""
    assert M.coverage_size(SCORES, {"q", "b"}, seeds="q") == 2
    with pytest.raises(ValueError):
        M.precision_at_k(SCORES, {"q"}, 3, seeds="q")


def test_nan_not_selected():
    """NaN — невідібрані: релевантний c не відібраний ⇒ RR = 0, охоплення = ∞."""
    s = pd.Series({"a": 2.0, "b": 1.0, "c": np.nan})
    assert M.reciprocal_rank(s, {"c"}) == 0.0
    assert M.coverage_size(s, {"c"}) == np.inf
    assert M.precision_at_k(s, {"c", "a"}, 2) == pytest.approx(0.5)


def test_ties():
    """4 вузли з однаковою оцінкою, релевантний x: песимістично RR = 1/4, оптимістично 1,
    випадково — у середньому (1 + 1/2 + 1/3 + 1/4)/4."""
    s = pd.Series({"w": 1.0, "x": 1.0, "y": 1.0, "z": 1.0})
    assert M.reciprocal_rank(s, {"x"}, ties="pessimistic") == pytest.approx(0.25)
    assert M.reciprocal_rank(s, {"x"}, ties="optimistic") == pytest.approx(1.0)
    assert M.reciprocal_rank(s, {"x"}, seed=3) == M.reciprocal_rank(s, {"x"}, seed=3)
    mean_rr = np.mean([M.reciprocal_rank(s, {"x"}, seed=i) for i in range(4000)])
    assert mean_rr == pytest.approx((1 + 1 / 2 + 1 / 3 + 1 / 4) / 4, abs=0.02)


def test_ties_do_not_follow_index_order():
    """Нічиї не розбиваються за порядком індексу (інакше SBM-нумерація давала б зсув)."""
    s = pd.Series(1.0, index=range(100))
    rel = set(range(10))                       # «релевантні» — перші за індексом
    p = np.mean([M.precision_at_k(s, rel, 10, seed=i) for i in range(200)])
    assert p == pytest.approx(0.1, abs=0.03)


def test_errors():
    with pytest.raises(KeyError):
        M.precision_at_k(SCORES, {"немає"}, 3)
    with pytest.raises(ValueError):
        M.ranking(SCORES, ties="як-небудь")


def test_evaluate_keys():
    out = M.evaluate(SCORES, REL, ks=(2, 4), seeds="q")
    assert out == pytest.approx({"P@2": 0.5, "nDCG@2": (1 / L3) / (1 + 1 / L3), "P@4": 0.5,
                                 "nDCG@4": (1 / L3 + 1 / L5) / (1 + 1 / L3), "RR": 0.5,
                                 "coverage": 4})
