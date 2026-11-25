"""Генерація графів для експериментів.

Прості графи (ланцюжок, цикл, зірка, повний граф), випадкові графи
(Ердеш–Реньї, Ваттс–Строгац, Барабаші–Альберт, стохастична блокова модель
з відомими спільнотами), граф зі спільнотами, з'єднаними лише через явні
мостові вузли, призначення ваг ребрам, шум у ребрах (видалення справжніх
і додавання хибних). Усі випадкові генератори приймають явний seed.

Атрибути вузлів: "community" — номер спільноти (−1 для мостових вузлів),
"is_bridge" — чи є вузол мостовим. Атрибути ребер: "weight", "type", "noise".
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np

MAX_TRIES = 1000


# ---------------------------------------------------------------------------
# Прості графи
# ---------------------------------------------------------------------------

def path(n):
    """Ланцюжок 0 — 1 — … — (n − 1)."""
    return nx.path_graph(n)


def cycle(n):
    """Цикл на вузлах 0, …, n − 1."""
    return nx.cycle_graph(n)


def star(m):
    """Зірка: центр 0, листки 1, …, m."""
    return nx.star_graph(m)


def complete(n):
    """Повний граф K_n."""
    return nx.complete_graph(n)


# ---------------------------------------------------------------------------
# Випадкові графи
# ---------------------------------------------------------------------------

def _resample_connected(make, seed):
    """Генерує граф make(seed'), доки граф не стане зв'язним.

    Перша спроба — з seed; наступні — з похідних seed' від генератора,
    ініціалізованого seed (а не seed + 1, …: інакше різні seed давали б однакові графи).
    """
    rng = np.random.default_rng(seed)
    sd = seed
    for _ in range(MAX_TRIES):
        g = make(sd)
        if nx.is_connected(g):
            g.graph["seed_used"] = sd
            return g
        sd = int(rng.integers(2**31))
    raise RuntimeError(f"Не вдалося отримати зв'язний граф за {MAX_TRIES} спроб")


def erdos_renyi(n, p, seed, connected=True):
    """Граф Ердеша–Реньї G(n, p). connected=True — перебір seed до зв'язного графа."""
    make = lambda sd: nx.gnp_random_graph(n, p, seed=sd)
    return _resample_connected(make, seed) if connected else make(seed)


def watts_strogatz(n, k, p, seed):
    """Граф Ваттса–Строгаца (кільце з k сусідами, перепідключення з ймовірністю p), зв'язний."""
    return nx.connected_watts_strogatz_graph(n, k, p, seed=seed)


def barabasi_albert(n, m, seed):
    """Граф Барабаші–Альберт (переважне приєднання, m ребер на новий вузол) — граф з хабами."""
    return nx.barabasi_albert_graph(n, m, seed=seed)


def sbm(sizes, p_in, p_out, seed, connected=True):
    """Стохастична блокова модель: спільноти розмірів sizes, ймовірність ребра p_in
    всередині спільноти та p_out між спільнотами. Атрибут вузла "community"."""
    k = len(sizes)
    probs = [[p_in if a == b else p_out for b in range(k)] for a in range(k)]

    def make(sd):
        g = nx.stochastic_block_model(sizes, probs, seed=sd)
        g = nx.Graph(g)                                  # прибираємо службові атрибути графа
        for v, data in g.nodes(data=True):
            data["community"] = data.pop("block")
            data["is_bridge"] = False
        return g

    return _resample_connected(make, seed) if connected else make(seed)


def bridged_communities(n_communities, community_size, p_in, n_bridges, bridge_degree,
                        seed, pairs="all"):
    """Спільноти, з'єднані ЛИШЕ через задані мостові вузли (еталон для п. 5 ТЗ).

    Кожна спільнота — зв'язний граф Ердеша–Реньї G(community_size, p_in); прямих
    ребер між спільнотами немає. Для кожної пари спільнот (pairs="all" — усі пари,
    "chain" — сусідні 0–1, 1–2, …) додаються n_bridges мостових вузлів, кожен
    з'єднаний з bridge_degree випадковими вузлами в кожній зі своїх двох спільнот.
    Атрибути: "community" (−1 для мостів), "is_bridge", для мостів — "connects" = (a, b).
    Будь-який шлях між спільнотами проходить через мостовий вузол.
    """
    if pairs == "all":
        pair_list = list(itertools.combinations(range(n_communities), 2))
    elif pairs == "chain":
        pair_list = [(a, a + 1) for a in range(n_communities - 1)]
    else:
        raise ValueError(f"pairs має бути 'all' або 'chain', а задано {pairs!r}")
    if bridge_degree > community_size:
        raise ValueError("bridge_degree не може перевищувати розмір спільноти")

    rng = np.random.default_rng(seed)
    g = nx.Graph()
    members = []
    for c in range(n_communities):
        sub_seed = int(rng.integers(2**31))
        block = erdos_renyi(community_size, p_in, seed=sub_seed, connected=True)
        offset = c * community_size
        nodes = [offset + v for v in range(community_size)]
        members.append(nodes)
        g.add_nodes_from(nodes, community=c, is_bridge=False)
        g.add_edges_from((offset + u, offset + v) for u, v in block.edges())

    next_id = n_communities * community_size
    for a, b in pair_list:
        for _ in range(n_bridges):
            g.add_node(next_id, community=-1, is_bridge=True, connects=(a, b))
            for c in (a, b):
                ends = rng.choice(members[c], size=bridge_degree, replace=False)
                g.add_edges_from((next_id, int(v)) for v in ends)
            next_id += 1
    return g


def communities(g):
    """Словник {номер спільноти: множина вузлів} за атрибутом "community" (без мостів)."""
    out = {}
    for v, c in g.nodes(data="community"):
        if c is not None and c >= 0:
            out.setdefault(c, set()).add(v)
    return out


def bridges(g):
    """Множина мостових вузлів (атрибут "is_bridge")."""
    return {v for v, b in g.nodes(data="is_bridge") if b}


# ---------------------------------------------------------------------------
# Ваги ребер
# ---------------------------------------------------------------------------

def random_weights(g, low, high, seed, attr="weight"):
    """Копія графа з вагами ребер, рівномірно розподіленими на [low, high]."""
    g = g.copy()
    rng = np.random.default_rng(seed)
    for *_, data in _edges_with_data(g):
        data[attr] = float(rng.uniform(low, high))
    return g


def random_edge_types(g, types, seed, probs=None, attr="type"):
    """Копія графа, у якій кожному ребру випадково призначено тип із types (з ймовірностями probs)."""
    g = g.copy()
    rng = np.random.default_rng(seed)
    types = list(types)
    for *_, data in _edges_with_data(g):
        data[attr] = types[int(rng.choice(len(types), p=probs))]
    return g


def type_weights(g, mapping, type_attr="type", attr="weight"):
    """Копія графа з вагою ребра, заданою його типом: weight = mapping[type]."""
    g = g.copy()
    for u, v, *_, data in _edges_with_data(g):
        t = data.get(type_attr)
        if t not in mapping:
            raise KeyError(f"Немає ваги для типу ребра {t!r} ({u} — {v})")
        data[attr] = float(mapping[t])
    return g


def lognormal_weights(g, sigma, seed, attr="weight"):
    """Копія графа з логнормальними вагами w = exp(σZ − σ²/2), Z ~ N(0, 1): середнє 1,
    розкид задає σ (σ = 0 — усі ваги 1)."""
    g = g.copy()
    rng = np.random.default_rng(seed)
    for *_, data in _edges_with_data(g):
        data[attr] = float(np.exp(sigma * rng.standard_normal() - sigma ** 2 / 2))
    return g


def orient_with_reciprocity(g, rho, seed):
    """Орієнтований граф з неорієнтованого: кожне ребро з імовірністю rho стає двостороннім
    (дуги u → v і v → u), інакше — однією дугою випадкового напряму. Атрибути ребер копіюються."""
    if not 0 <= rho <= 1:
        raise ValueError("rho ∈ [0, 1]")
    rng = np.random.default_rng(seed)
    d = nx.DiGraph()
    d.add_nodes_from(g.nodes(data=True))
    for u, v, data in g.edges(data=True):
        if rng.random() < rho:
            d.add_edge(u, v, **data); d.add_edge(v, u, **data)
        elif rng.random() < 0.5:
            d.add_edge(u, v, **data)
        else:
            d.add_edge(v, u, **data)
    return d


def _edges_with_data(g):
    """Ребра з даними (для мультиграфа — з ключами)."""
    return g.edges(keys=True, data=True) if g.is_multigraph() else g.edges(data=True)


# ---------------------------------------------------------------------------
# Шум у ребрах
# ---------------------------------------------------------------------------

def perturb_edges(g, remove_frac=0.0, add_frac=0.0, seed=0):
    """Шум у ребрах: видалення частки справжніх і додавання частки хибних ребер.

    Кількості рахуються від початкової кількості ребер m:
    видаляється round(remove_frac·m) випадкових ребер, додається round(add_frac·m)
    випадкових пар вузлів, не з'єднаних у вихідному графі (для орієнтованого — дуг).
    Хибне ребро копіює атрибути (вагу, тип) випадкового справжнього ребра і має noise=True,
    щоб за виглядом не відрізнятися від справжніх.
    Повертає (новий граф, {"removed": [...], "added": [...]}).
    """
    if not (0 <= remove_frac <= 1 and add_frac >= 0):
        raise ValueError("remove_frac ∈ [0, 1], add_frac ≥ 0")
    rng = np.random.default_rng(seed)
    h = g.copy()
    edges = list(_edges_with_data(g))
    m = len(edges)

    n_remove = int(round(remove_frac * m))
    idx = rng.choice(m, size=n_remove, replace=False) if n_remove else []
    removed = [edges[i][:-1] for i in idx]
    h.remove_edges_from(removed)

    n_add = int(round(add_frac * m))
    nodes = list(g.nodes())
    n = len(nodes)
    n_pairs = n * (n - 1) if g.is_directed() else n * (n - 1) // 2
    if n_add > n_pairs - g.number_of_edges():
        raise ValueError("Забагато хибних ребер: бракує вільних пар вузлів")
    added, chosen = [], set()
    while len(added) < n_add:
        i, j = rng.integers(n, size=2)
        if i == j:
            continue
        u, v = nodes[i], nodes[j]
        key = (u, v) if g.is_directed() else frozenset((u, v))
        if g.has_edge(u, v) or key in chosen:
            continue
        chosen.add(key)
        template = edges[int(rng.integers(m))][-1] if m else {}
        attrs = {k: val for k, val in template.items() if k != "noise"}
        attrs["noise"] = True
        h.add_edge(u, v, **attrs)
        added.append((u, v))
    return h, {"removed": removed, "added": added}
