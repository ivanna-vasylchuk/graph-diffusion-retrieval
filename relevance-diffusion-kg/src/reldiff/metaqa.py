"""Робота з графом знань MetaQA.

- завантаження kb.txt (трійки subject|relation|object, без дублікатів);
- підграф «сніговою кулею»: випадковий фільм → його люди (актори, режисери,
  сценаристи) → їхні інші фільми → …, доки не набереться n_movies фільмів;
  у підграф входять усі трійки вибраних фільмів (параметри n_movies, seed);
- відповіді на питання — ланцюжок зв'язків із qtype з виключенням сутності
  запиту на кожному кроці (правило відтворює відповіді MetaQA для 2/3-hop);
- відбір питань: основний критерій — сутність запиту є в підграфі, а відповідь,
  перерахована на підграфі, непорожня; строгий (прапорець complete_wrt_full_kb) —
  перерахунок на підграфі збігається з перерахунком на повному KB;
- запити з двома вузлами (два актори / актор + режисер → спільні фільми);
- збереження/завантаження оброблених даних і статистика.
Вузли ідентифікуються за назвою, як у MetaQA (однойменні сутності злиті).
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# Слова з qtype → відношення KB (у KB суб'єкт завжди — фільм)
QTYPE_RELATION = {
    "actor": "starred_actors", "director": "directed_by", "writer": "written_by",
    "genre": "has_genre", "year": "release_year", "language": "in_language",
    "tags": "has_tags", "tag": "has_tags",
    "imdbrating": "has_imdb_rating", "imdbvotes": "has_imdb_votes",
}
# Тип об'єкта за відношенням (для статистики «частка вузлів кожного типу»)
RELATION_KIND = {
    "starred_actors": "actor", "directed_by": "director", "written_by": "writer",
    "has_genre": "genre", "release_year": "year", "in_language": "language",
    "has_tags": "tag", "has_imdb_rating": "imdb_rating", "has_imdb_votes": "imdb_votes",
}
PERSON_RELATIONS = ("starred_actors", "directed_by", "written_by")
HOPS = (1, 2, 3)
SPLITS = ("train", "dev", "test")
_TOPIC_RE = re.compile(r"\[(.+?)\]")


# ---------------------------------------------------------------------------
# База знань
# ---------------------------------------------------------------------------

def load_kb(path):
    """Трійки (s, r, o) з kb.txt: без дублікатів, відсортовані. Повертає (трійки, кількість дублікатів)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    triples = []
    for i, line in enumerate(lines, 1):
        parts = line.split("|")
        if len(parts) != 3:
            raise ValueError(f"Рядок {i} не у форматі subject|relation|object: {line!r}")
        triples.append(tuple(parts))
    unique = sorted(set(triples))
    return unique, len(triples) - len(unique)


class KBIndex:
    """Індекс трійок для обходу: fwd[r][s] = {o}, bwd[r][o] = {s}."""

    def __init__(self, triples):
        self.fwd = defaultdict(lambda: defaultdict(set))
        self.bwd = defaultdict(lambda: defaultdict(set))
        self.nodes = set()
        for s, r, o in triples:
            self.fwd[r][s].add(o)
            self.bwd[r][o].add(s)
            self.nodes.update((s, o))

    def step(self, frontier, src, dst):
        """Один крок ланцюжка src → dst: від фільму вперед по відношенню dst, до фільму — назад по src."""
        out = set()
        for x in frontier:
            if src == "movie":
                out |= self.fwd[QTYPE_RELATION[dst]].get(x, set())
            else:
                out |= self.bwd[QTYPE_RELATION[src]].get(x, set())
        return out


def chain_answers(index, topic, qtype):
    """Відповідь на питання типу qtype (напр. movie_to_director_to_movie_to_genre):
    проходимо ланцюжок, виключаючи сутність запиту на кожному кроці."""
    parts = qtype.split("_to_")
    frontier = {topic}
    for src, dst in zip(parts, parts[1:]):
        frontier = index.step(frontier, src, dst) - {topic}
    return frontier


# ---------------------------------------------------------------------------
# Підграф
# ---------------------------------------------------------------------------

def sample_movies_snowball(triples, n_movies, seed):
    """Набір фільмів «сніговою кулею» через людей (актори, режисери, сценаристи).

    Пошук у ширину від випадкового фільму: фільм → його люди → їхні інші фільми.
    Порядок обходу людей і фільмів випадковий (seed). Якщо черга вичерпалась,
    новий старт з випадкового невибраного фільму.
    """
    people_of = defaultdict(list)
    movies_of = defaultdict(set)
    for s, r, o in triples:
        if r in PERSON_RELATIONS:
            people_of[s].append(o)
            movies_of[o].add(s)
    movies = sorted({s for s, _, _ in triples})
    if n_movies > len(movies):
        raise ValueError(f"У KB лише {len(movies)} фільмів, а запитано {n_movies}")

    rng = np.random.default_rng(seed)
    chosen = set()
    while len(chosen) < n_movies:
        start = movies[int(rng.integers(len(movies)))]
        if start in chosen:
            continue
        chosen.add(start)
        queue = deque([start])
        while queue and len(chosen) < n_movies:
            m = queue.popleft()
            people = sorted(set(people_of[m]))
            rng.shuffle(people)
            for p in people:
                others = sorted(movies_of[p] - chosen)
                rng.shuffle(others)
                for m2 in others:
                    if len(chosen) >= n_movies:
                        break
                    chosen.add(m2)
                    queue.append(m2)
    return chosen


def subgraph_triples(triples, movies):
    """Усі трійки, суб'єкт яких — вибраний фільм."""
    movies = set(movies)
    return [t for t in triples if t[0] in movies]


def node_kinds(triples):
    """Типи кожного вузла: "movie" для суб'єктів, тип об'єкта — за відношенням.
    Вузол може мати кілька типів (людина в кількох ролях, злиті однойменні сутності)."""
    kinds = defaultdict(set)
    for s, r, o in triples:
        kinds[s].add("movie")
        kinds[o].add(RELATION_KIND.get(r, r))
    return kinds


REVERSE_SUFFIX = "⁻¹"


def to_graph(triples, directed=False, bidirected=False):
    """networkx-мультиграф: ребро s — o (або дуга s → o) з атрибутом type = відношення.
    bidirected=True — орієнтований мультиграф з дугами в обидва боки: s → o типу r
    («вперед», фільм → атрибут) і o → s типу r⁻¹ («назад»); тоді ваги «вперед» і «назад»
    задаються окремо через type_weights.
    Атрибут вузла "kinds" — кортеж типів вузла."""
    g = nx.MultiDiGraph() if (directed or bidirected) else nx.MultiGraph()
    kinds = node_kinds(triples)
    for v in sorted(kinds):
        g.add_node(v, kinds=tuple(sorted(kinds[v])))
    for s, r, o in triples:
        g.add_edge(s, o, type=r)
        if bidirected:
            g.add_edge(o, s, type=r + REVERSE_SUFFIX)
    return g


def perturb_triples(triples, remove_frac=0.0, add_frac=0.0, seed=0):
    """Шум у KB зі збереженням схеми (модель помилки вилучення трійок).

    Видаляє round(remove_frac·m) випадкових трійок і додає round(add_frac·m) хибних:
    для кожної береться відношення r випадкової справжньої трійки, випадковий фільм
    (суб'єкт KB) і випадковий об'єкт, що трапляється в KB з відношенням r (допустимий тип);
    хибна трійка не збігається з жодною справжньою. Повертає (нові трійки, {"removed", "added"}).
    Вузли, що втратили всі трійки, з трійок зникають — щоб зберегти їх у графі,
    додайте вузли чистого графа окремо.
    """
    if not (0 <= remove_frac <= 1 and add_frac >= 0):
        raise ValueError("remove_frac ∈ [0, 1], add_frac ≥ 0")
    rng = np.random.default_rng(seed)
    triples = list(triples)
    m = len(triples)
    present = set(triples)
    movies = sorted({s for s, _, _ in triples})
    objects_of = defaultdict(set)
    for _, r, o in triples:
        objects_of[r].add(o)
    objects_of = {r: sorted(v) for r, v in objects_of.items()}

    n_remove = int(round(remove_frac * m))
    idx = set(rng.choice(m, size=n_remove, replace=False).tolist()) if n_remove else set()
    removed = [triples[i] for i in sorted(idx)]
    kept = [t for i, t in enumerate(triples) if i not in idx]

    n_add = int(round(add_frac * m))
    added, chosen = [], set()
    tries = 0
    while len(added) < n_add:
        tries += 1
        if tries > 100 * (n_add + 1):
            raise RuntimeError("Не вдалося згенерувати достатньо хибних трійок")
        r = triples[int(rng.integers(m))][1]
        t = (movies[int(rng.integers(len(movies)))], r, objects_of[r][int(rng.integers(len(objects_of[r])))])
        if t in present or t in chosen:
            continue
        chosen.add(t)
        added.append(t)
    return kept + added, {"removed": removed, "added": added}


# ---------------------------------------------------------------------------
# Питання
# ---------------------------------------------------------------------------

def load_questions(root, hops=HOPS, splits=SPLITS):
    """Питання з vanilla і qtype-файлів. Дублікати (сутність, qtype) залишаються один раз
    (перше входження в порядку hop, split, рядок). Повертає DataFrame."""
    root = Path(root)
    rows = []
    for h in hops:
        for sp in splits:
            qs = (root / f"{h}-hop" / "vanilla" / f"qa_{sp}.txt").read_text(encoding="utf-8").splitlines()
            ts = (root / f"{h}-hop" / f"qa_{sp}_qtype.txt").read_text(encoding="utf-8").split()
            if len(qs) != len(ts):
                raise ValueError(f"{h}-hop/{sp}: {len(qs)} питань, але {len(ts)} типів")
            for i, (line, qtype) in enumerate(zip(qs, ts)):
                text, ans = line.split("\t")
                m = _TOPIC_RE.search(text)
                if m is None:
                    raise ValueError(f"{h}-hop/{sp}, рядок {i + 1}: немає сутності в [..]")
                rows.append(dict(hop=h, split=sp, line=i, qtype=qtype, topic=m.group(1),
                                 question=text, original_answers=sorted(set(ans.split("|")))))
    df = pd.DataFrame(rows)
    n_all = len(df)
    df = df.drop_duplicates(subset=["topic", "qtype"], keep="first").reset_index(drop=True)
    df.attrs["n_before_dedup"] = n_all
    return df


def build_questions(questions, sub_index, full_index):
    """Питання, придатні за основним критерієм: сутність запиту в підграфі і відповідь,
    перерахована на підграфі, непорожня. Додає поля answers, n_answers,
    complete_wrt_full_kb (= перерахунок на повному KB), matches_original (= відповідь MetaQA)."""
    rows = []
    for q in questions.itertuples(index=False):
        if q.topic not in sub_index.nodes:
            continue
        ans = chain_answers(sub_index, q.topic, q.qtype)
        if not ans:
            continue
        full = chain_answers(full_index, q.topic, q.qtype)
        rows.append(dict(hop=q.hop, split=q.split, line=q.line, qtype=q.qtype, topic=q.topic,
                         question=q.question, answers=sorted(ans), n_answers=len(ans),
                         complete_wrt_full_kb=(ans == full),
                         matches_original=(ans == set(q.original_answers))))
    cols = ["hop", "split", "line", "qtype", "topic", "question", "answers", "n_answers",
            "complete_wrt_full_kb", "matches_original"]
    return pd.DataFrame(rows, columns=cols)


def pair_queries(sub_triples, full_index):
    """Запити з двома вузлами: пари акторів зі спільним фільмом (actor_actor) і пари
    актор + режисер зі спільним фільмом (actor_director) у підграфі.
    Відповідь — спільні фільми в підграфі (без самих вузлів запиту);
    complete_wrt_full_kb — чи збігається вона зі спільними фільмами в повному KB."""
    cast, directors = defaultdict(set), defaultdict(set)
    for s, r, o in sub_triples:
        if r == "starred_actors":
            cast[s].add(o)
        elif r == "directed_by":
            directors[s].add(o)
    pairs = defaultdict(set)
    for m in sorted(set(cast) | set(directors)):
        actors = sorted(cast[m])
        for i, a in enumerate(actors):
            for b in actors[i + 1:]:
                pairs[("actor_actor", a, b)].add(m)
            for d in sorted(directors[m]):
                if d != a:
                    pairs[("actor_director", a, d)].add(m)

    def full_common(kind, a, b):
        ma = full_index.bwd["starred_actors"].get(a, set())
        rel_b = "starred_actors" if kind == "actor_actor" else "directed_by"
        mb = full_index.bwd[rel_b].get(b, set())
        return (ma & mb) - {a, b}

    rows = []
    for (kind, a, b), movies in sorted(pairs.items()):
        ans = movies - {a, b}
        if not ans:
            continue
        rows.append(dict(kind=kind, seed_1=a, seed_2=b, answers=sorted(ans), n_answers=len(ans),
                         complete_wrt_full_kb=(ans == full_common(kind, a, b))))
    return pd.DataFrame(rows, columns=["kind", "seed_1", "seed_2", "answers", "n_answers",
                                       "complete_wrt_full_kb"])


# ---------------------------------------------------------------------------
# Статистика
# ---------------------------------------------------------------------------

def statistics(sub_triples, questions_df, pairs_df, top=10):
    """Статистика підграфа і запитів (словник, придатний для JSON)."""
    g = to_graph(sub_triples)
    deg = dict(g.degree())
    d = np.array(list(deg.values()))
    kinds = node_kinds(sub_triples)
    kind_counts = Counter(k for ks in kinds.values() for k in ks)
    n = g.number_of_nodes()
    multi = {v: ks for v, ks in kinds.items() if len(ks) > 1}
    movie_and_other = sorted(v for v, ks in multi.items() if "movie" in ks)
    years = defaultdict(set)
    for s, r, o in sub_triples:
        if r == "release_year":
            years[s].add(o)
    merged_movies = sorted(m for m, ys in years.items() if len(ys) > 1)
    comps = sorted((len(c) for c in nx.connected_components(g)), reverse=True)

    def q_counts(df, strict):
        sub = df[df["complete_wrt_full_kb"]] if strict else df
        by_hop = {int(h): int(c) for h, c in sub.groupby("hop").size().items()}
        by_qtype = {f"{int(h)}-hop/{t}": int(c) for (h, t), c in sub.groupby(["hop", "qtype"]).size().items()}
        return {"усього": int(len(sub)), "за_hop": by_hop, "за_qtype": by_qtype}

    def p_counts(df, strict):
        sub = df[df["complete_wrt_full_kb"]] if strict else df
        return {k: int(c) for k, c in sub.groupby("kind").size().items()}

    return {
        "вузлів": n,
        "ребер (трійок)": g.number_of_edges(),
        "пар вузлів, з'єднаних хоча б одним ребром": nx.Graph(g).number_of_edges(),
        "фільмів": kind_counts.get("movie", 0),
        "компонент зв'язності": len(comps),
        "розмір найбільшої компоненти": comps[0] if comps else 0,
        "степінь": {"мін": int(d.min()), "медіана": float(np.median(d)), "середній": float(d.mean()),
                    "макс": int(d.max()), "частка степеня 1": float(np.mean(d == 1)),
                    "квантилі 90/99%": [float(np.quantile(d, 0.9)), float(np.quantile(d, 0.99))]},
        "топ хабів": [{"вузол": v, "степінь": int(k), "типи": sorted(kinds[v])}
                      for v, k in sorted(deg.items(), key=lambda x: (-x[1], x[0]))[:top]],
        "вузлів за типом (вузол з кількома типами враховано в кожному)": dict(sorted(kind_counts.items())),
        "частка вузлів за типом": {k: round(c / n, 4) for k, c in sorted(kind_counts.items())},
        "вузлів з кількома типами": len(multi),
        "злиті назви 'фільм + інший тип'": movie_and_other,
        "фільмів з кількома роками випуску (ймовірно злиті однойменні)": len(merged_movies),
        "приклади злитих фільмів": merged_movies[:10],
        "питання, основний критерій": q_counts(questions_df, strict=False),
        "питання, строгий критерій (complete_wrt_full_kb)": q_counts(questions_df, strict=True),
        "частка питань, що збігаються з відповіддю MetaQA": float(questions_df["matches_original"].mean())
        if len(questions_df) else float("nan"),
        "запити з двома вузлами, основний критерій": p_counts(pairs_df, strict=False),
        "запити з двома вузлами, строгий критерій": p_counts(pairs_df, strict=True),
    }


# ---------------------------------------------------------------------------
# Повний конвеєр, збереження, завантаження
# ---------------------------------------------------------------------------

def processed_dir(root, n_movies, seed):
    """Папка з обробленими даними для підграфа (n_movies, seed)."""
    return Path(root) / "processed" / f"movies{n_movies}_seed{seed}"


def prepare(root, n_movies=500, seed=0, save=True):
    """Повний конвеєр: KB → підграф → питання → запити з двома вузлами → статистика.

    Зберігає в data/metaqa/processed/movies{n}_seed{s}/:
    subgraph_edges.csv (s, r, o — напрям і тип ребра), questions.csv, pair_queries.csv, stats.json.
    """
    root = Path(root)
    triples, n_dup = load_kb(root / "kb.txt")
    full_index = KBIndex(triples)
    movies = sample_movies_snowball(triples, n_movies, seed)
    sub = subgraph_triples(triples, movies)
    sub_index = KBIndex(sub)

    q_all = load_questions(root)
    questions = build_questions(q_all, sub_index, full_index)
    pairs = pair_queries(sub, full_index)
    stats = statistics(sub, questions, pairs)
    stats = {"параметри": {"n_movies": n_movies, "seed": seed},
             "KB": {"трійок (унікальних)": len(triples), "дублікатів": n_dup,
                    "сутностей": len(full_index.nodes)},
             "питань у пулі до/після вилучення дублікатів (сутність, qtype)":
                 [int(q_all.attrs["n_before_dedup"]), int(len(q_all))],
             **stats}

    if save:
        out = processed_dir(root, n_movies, seed)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sub, columns=["s", "r", "o"]).to_csv(out / "subgraph_edges.csv", index=False)
        _to_csv(questions, out / "questions.csv")
        _to_csv(pairs, out / "pair_queries.csv")
        (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"triples": sub, "questions": questions, "pairs": pairs, "stats": stats}


def _to_csv(df, path):
    df = df.copy()
    df["answers"] = df["answers"].map(lambda a: json.dumps(a, ensure_ascii=False))
    df.to_csv(path, index=False)


def load_processed(root, n_movies=500, seed=0, directed=False, bidirected=False):
    """Завантажує збережений підграф і запити: (граф, questions, pairs).
    directed=True — орієнтований мультиграф s → o; bidirected=True — дуги в обидва боки
    з типами r і r⁻¹ (для асиметричних ваг, п. 4)."""
    d = processed_dir(root, n_movies, seed)
    edges = pd.read_csv(d / "subgraph_edges.csv", dtype=str, keep_default_na=False)
    g = to_graph(list(edges.itertuples(index=False, name=None)), directed=directed, bidirected=bidirected)
    read = lambda name: pd.read_csv(d / name, keep_default_na=False,
                                    converters={"answers": json.loads, "topic": str,
                                                "seed_1": str, "seed_2": str})
    return g, read("questions.csv"), read("pair_queries.csv")
