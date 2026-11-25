"""Тести reldiff.metaqa на маленькому штучному KB з ручними відповідями.

Штучний KB (фільми M1–M5):
  M1: режисер D1, актори A1, A2, жанр Drama, рік 2000
  M2: режисер D1, актори A2, A3, жанр Comedy, рік 2001
  M3: режисер D2, актор A3, жанр Drama, рік 2000
  M4: режисер D2, актор A1, сценарист A1, жанр Drama
  M5: режисер D3, актор A4, жанр Horror        (люди M5 більше ніде не знімались)
"""

import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from reldiff import metaqa as MQ

KB = """M1|directed_by|D1
M1|starred_actors|A1
M1|starred_actors|A2
M1|has_genre|Drama
M1|release_year|2000
M2|directed_by|D1
M2|starred_actors|A2
M2|starred_actors|A3
M2|has_genre|Comedy
M2|release_year|2001
M3|directed_by|D2
M3|starred_actors|A3
M3|has_genre|Drama
M3|release_year|2000
M4|directed_by|D2
M4|starred_actors|A1
M4|written_by|A1
M4|has_genre|Drama
M5|directed_by|D3
M5|starred_actors|A4
M5|has_genre|Horror
M1|has_genre|Drama
"""

QA = {  # hop -> [(питання, відповідь MetaQA, qtype)]
    1: [("what did [A1] act in", "M1|M4", "actor_to_movie"),
        ("[A1] appears in which movies", "M1|M4", "actor_to_movie"),          # дублікат (A1, qtype)
        ("who directed [M2]", "D1", "movie_to_director")],
    2: [("who acted with [A1]", "A2|A3", "actor_to_movie_to_actor")],          # відповідь MetaQA неточна
    3: [("genres of films sharing directors with [M1]", "Comedy", "movie_to_director_to_movie_to_genre"),
        ("genres of films sharing directors with [M5]", "Horror", "movie_to_director_to_movie_to_genre")],
}


@pytest.fixture
def root(tmp_path):
    (tmp_path / "kb.txt").write_text(KB, encoding="utf-8")
    for h in MQ.HOPS:
        (tmp_path / f"{h}-hop" / "vanilla").mkdir(parents=True)
        for sp in MQ.SPLITS:
            rows = QA[h] if sp == "train" else []
            (tmp_path / f"{h}-hop" / "vanilla" / f"qa_{sp}.txt").write_text(
                "".join(f"{q}\t{a}\n" for q, a, _ in rows), encoding="utf-8")
            (tmp_path / f"{h}-hop" / f"qa_{sp}_qtype.txt").write_text(
                "".join(f"{t}\n" for *_, t in rows), encoding="utf-8")
    return tmp_path


@pytest.fixture
def kb(root):
    return MQ.load_kb(root / "kb.txt")[0]


def test_load_kb_dedup(root):
    triples, n_dup = MQ.load_kb(root / "kb.txt")
    assert n_dup == 1 and len(triples) == 21


def test_load_kb_malformed(tmp_path):
    p = tmp_path / "kb.txt"; p.write_text("a|b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        MQ.load_kb(p)


def test_chain_answers_by_hand(kb):
    ix = MQ.KBIndex(kb)
    assert MQ.chain_answers(ix, "A1", "actor_to_movie") == {"M1", "M4"}
    assert MQ.chain_answers(ix, "M4", "movie_to_writer") == {"A1"}
    # A1 → {M1, M4} → актори {A1, A2} − A1 = {A2}  (A3 не знімався з A1)
    assert MQ.chain_answers(ix, "A1", "actor_to_movie_to_actor") == {"A2"}
    # M1 → D1 → {M1, M2} − M1 = {M2} → Comedy (без виключення на проміжному кроці був би й Drama)
    assert MQ.chain_answers(ix, "M1", "movie_to_director_to_movie_to_genre") == {"Comedy"}
    assert MQ.chain_answers(ix, "M5", "movie_to_director_to_movie_to_genre") == set()


def test_snowball_follows_people(kb):
    """2 фільми: друга вибірка завжди пов'язана спільною людиною з першою."""
    for seed in range(20):
        m = MQ.sample_movies_snowball(kb, 2, seed)
        people = [{o for s, r, o in kb if s == x and r in MQ.PERSON_RELATIONS} for x in m]
        assert len(m) == 2 and (people[0] & people[1] or "M5" in m)


def test_snowball_reproducible_and_sizes(kb):
    assert MQ.sample_movies_snowball(kb, 3, 7) == MQ.sample_movies_snowball(kb, 3, 7)
    assert len(MQ.sample_movies_snowball(kb, 5, 0)) == 5
    with pytest.raises(ValueError):
        MQ.sample_movies_snowball(kb, 6, 0)


def test_subgraph_and_graph(kb):
    sub = MQ.subgraph_triples(kb, {"M1", "M2"})
    assert len(sub) == 10 and all(s in {"M1", "M2"} for s, _, _ in sub)
    g = MQ.to_graph(sub)
    assert isinstance(g, nx.MultiGraph) and g.number_of_edges() == 10
    assert g.nodes["M1"]["kinds"] == ("movie",) and g.nodes["D1"]["kinds"] == ("director",)
    assert {d["type"] for *_, d in g.edges(data=True)} == {
        "directed_by", "starred_actors", "has_genre", "release_year"}
    dg = MQ.to_graph(sub, directed=True)
    assert dg.has_edge("M1", "D1") and not dg.has_edge("D1", "M1")


def test_node_kinds_multi_role(kb):
    """A1 — актор і сценарист: два типи."""
    assert MQ.node_kinds(kb)["A1"] == {"actor", "writer"}


def test_load_questions_dedup(root):
    q = MQ.load_questions(root)
    assert q.attrs["n_before_dedup"] == 6 and len(q) == 5
    assert set(q["topic"]) == {"A1", "M2", "M1", "M5"}


def test_build_questions_criteria(root, kb):
    """Підграф {M1, M2}: основний критерій — перерахунок на підграфі; строгий — прапорець."""
    sub = MQ.subgraph_triples(kb, {"M1", "M2"})
    q = MQ.build_questions(MQ.load_questions(root), MQ.KBIndex(sub), MQ.KBIndex(kb))
    by = {(r.topic, r.qtype): r for r in q.itertuples()}
    # A1 у підграфі: на підграфі {M1}, на повному KB {M1, M4} — придатне, але не повне
    r = by[("A1", "actor_to_movie")]
    assert r.answers == ["M1"] and not r.complete_wrt_full_kb and not r.matches_original
    # M2 → D1: повне і збігається з MetaQA
    r = by[("M2", "movie_to_director")]
    assert r.answers == ["D1"] and r.complete_wrt_full_kb and r.matches_original
    # A1 → актори: на підграфі {A2}, повний KB {A2}, MetaQA {A2, A3} — повне, але не як MetaQA
    r = by[("A1", "actor_to_movie_to_actor")]
    assert r.answers == ["A2"] and r.complete_wrt_full_kb and not r.matches_original
    # M1, 3-hop: M1 → D1 → M2 → Comedy
    assert by[("M1", "movie_to_director_to_movie_to_genre")].answers == ["Comedy"]
    # M5 не в підграфі — питання немає
    assert ("M5", "movie_to_director_to_movie_to_genre") not in by
    assert len(q) == 4


def test_pair_queries_by_hand(kb):
    sub = MQ.subgraph_triples(kb, {"M1", "M2", "M3"})
    p = MQ.pair_queries(sub, MQ.KBIndex(kb))
    by = {(r.kind, r.seed_1, r.seed_2): r for r in p.itertuples()}
    assert by[("actor_actor", "A1", "A2")].answers == ["M1"]
    assert by[("actor_actor", "A2", "A3")].answers == ["M2"]
    assert by[("actor_director", "A2", "D1")].answers == ["M1", "M2"]
    assert by[("actor_director", "A3", "D2")].answers == ["M3"]
    # A1 + D2 спільний фільм лише M4, якого немає в підграфі — такого запиту немає
    assert ("actor_director", "A1", "D2") not in by
    assert all(r.complete_wrt_full_kb for r in p.itertuples())
    # якщо M4 у KB, а в підграфі лише M1: A1 + D1 → M1 (повне); з M2 поза підграфом A2 + D1 неповне
    p2 = MQ.pair_queries(MQ.subgraph_triples(kb, {"M1"}), MQ.KBIndex(kb))
    by2 = {(r.kind, r.seed_1, r.seed_2): r for r in p2.itertuples()}
    assert by2[("actor_director", "A1", "D1")].complete_wrt_full_kb
    assert not by2[("actor_director", "A2", "D1")].complete_wrt_full_kb


def test_prepare_and_load_roundtrip(root):
    res = MQ.prepare(root, n_movies=3, seed=0)
    d = MQ.processed_dir(root, 3, 0)
    assert {p.name for p in d.iterdir()} == {"subgraph_edges.csv", "questions.csv",
                                             "pair_queries.csv", "stats.json"}
    g, q, p = MQ.load_processed(root, 3, 0)
    assert g.number_of_edges() == len(res["triples"])
    assert q["answers"].tolist() == res["questions"]["answers"].tolist()
    assert p["answers"].tolist() == res["pairs"]["answers"].tolist()
    stats = json.loads((d / "stats.json").read_text(encoding="utf-8"))
    assert stats["фільмів"] == 3 and stats["KB"]["дублікатів"] == 1
    dg, _, _ = MQ.load_processed(root, 3, 0, directed=True)
    assert dg.is_directed()


def test_rank_works_on_metaqa_graph(kb):
    """Граф MetaQA підходить для rank, у т.ч. з вагами за типами ребер."""
    from reldiff.api import rank
    g = MQ.to_graph(kb)
    w = {r: 1.0 for r in MQ.RELATION_KIND}
    x = rank(g, "A1", "ppr", type_weights=w)
    assert x.sum() == pytest.approx(1.0) and x.idxmax() in {"A1", "M1", "M4"}


# --- Двосторонній граф і твердження про відношення ваг «вперед/назад» (п. 4) ----

def test_bidirected_graph(kb):
    g = MQ.to_graph(kb, bidirected=True)
    assert g.is_directed() and g.number_of_edges() == 2 * len(kb)
    types = {d["type"] for *_, d in g.edges(data=True)}
    assert "directed_by" in types and "directed_by" + MQ.REVERSE_SUFFIX in types
    assert any(d["type"] == "starred_actors⁻¹" for _, _, d in g.edges("A1", data=True))


def _pure_kb(kb):
    # без вузлів, у яких є дуги обох напрямів (фільм, що є й об'єктом) — у штучному KB таких немає
    subjects = {s for s, _, _ in kb}
    assert not ({o for _, _, o in kb} & subjects)
    return kb


@pytest.mark.parametrize("ratio", [0.1, 1.0, 7.0])
def test_direction_ratio_invariance(kb, ratio):
    """Твердження: якщо жоден вузол не має дуг обох напрямів, P (отже π і u(t)) не залежить
    від спільного відношення ваг «вперед» / «назад» — множник скорочується в кожному рядку."""
    from reldiff.api import rank
    g = MQ.to_graph(_pure_kb(kb), bidirected=True)
    rels = set(MQ.RELATION_KIND)
    tw = {r: ratio for r in rels} | {r + MQ.REVERSE_SUFFIX: 1.0 for r in rels}
    und = MQ.to_graph(kb)
    for q in ("A1", "M1", "Drama"):
        x = rank(g, q, "ppr", type_weights=tw).sort_index()
        y = rank(und, q, "ppr").sort_index()
        np.testing.assert_allclose(x.to_numpy(), y.to_numpy(), atol=1e-12)
        xh = rank(g, q, "heat", t=3.0, type_weights=tw).sort_index()
        yh = rank(und, q, "heat", t=3.0).sort_index()
        np.testing.assert_allclose(xh.to_numpy(), yh.to_numpy(), atol=1e-12)


def test_direction_ratio_matters_for_mixed_node():
    """Якщо вузол має дуги обох напрямів (фільм «X», що є й роком іншого фільму), відношення діє."""
    from reldiff.api import rank
    kb = [("M1", "release_year", "X"), ("M1", "starred_actors", "A"), ("X", "starred_actors", "B"),
          ("M2", "starred_actors", "B")]
    g = MQ.to_graph(kb, bidirected=True)
    rels = set(MQ.RELATION_KIND)
    x = [rank(g, "M1", type_weights={r: f for r in rels} | {r + MQ.REVERSE_SUFFIX: 1.0 for r in rels})
         for f in (0.2, 5.0)]
    assert np.abs(x[0].sort_index().to_numpy() - x[1].sort_index().to_numpy()).max() > 1e-3


# --- Шум зі збереженням схеми KG (п. 7) -----------------------------------------

def test_perturb_triples_counts_and_schema(kb):
    new, rec = MQ.perturb_triples(kb, remove_frac=0.2, add_frac=0.3, seed=1)
    m = len(kb)
    assert len(rec["removed"]) == round(0.2 * m) and len(rec["added"]) == round(0.3 * m)
    assert len(new) == m - len(rec["removed"]) + len(rec["added"])
    movies = {s for s, _, _ in kb}
    objects_of = {}
    for _, r, o in kb:
        objects_of.setdefault(r, set()).add(o)
    for s, r, o in rec["added"]:
        assert s in movies and o in objects_of[r] and (s, r, o) not in set(kb)
    assert set(rec["removed"]) <= set(kb)


def test_perturb_triples_reproducible_and_zero(kb):
    a = MQ.perturb_triples(kb, 0.1, 0.1, seed=3)
    b = MQ.perturb_triples(kb, 0.1, 0.1, seed=3)
    assert a == b
    same, rec = MQ.perturb_triples(kb, 0.0, 0.0, seed=3)
    assert sorted(same) == sorted(kb) and rec == {"removed": [], "added": []}
