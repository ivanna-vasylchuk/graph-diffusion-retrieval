"""Генерує experiments/04_vagy_i_napryam.ipynb."""
import sys
from pathlib import Path
import nbformat as nbf

out = sys.argv[1]
CONCL = Path(__file__).resolve().parent / "conclusions" / "concl04.md"
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Завдання 4. Ваги, напрям ребер і висячі вузли

**Пункт 4 ТЗ.** Дослідити вплив ваг і напряму ребер на розподіл релевантності, моделюючи різну «силу» зв'язків, зокрема задаючи різні ваги для різних типів зв'язків; врахувати вузли без вихідних ребер.

Параметри: PPR з α = 0.15, теплове ядро з t = 5.67. Вимірюємо **зміну розподілу**:
- перетин топ-10 і τ_b на топ-50 з базовим варіантом (незваженим або неорієнтованим);
- склад топу за типами вузлів;
- потік імовірності за типами ребер;
- локальність.

Якість відповідей (P@k, nDCG) — у п. 6.
""")

code(r"""
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

from reldiff.api import rank
from reldiff import graphs as G
from reldiff import metaqa as MQ

FIG = Path("../figures"); RES = Path("../results")
ALPHA, T = 0.15, (1 - 0.15) / 0.15
RESOLVE = 1e-10
pd.set_option("display.float_format", "{:.3f}".format); pd.set_option("display.width", 250)

def prep(s, exclude):
    x = s.drop(index=exclude).to_numpy(dtype=float)
    m = np.abs(x).max()
    x = x / m if m > 0 else x
    x[x < RESOLVE] = 0.0
    return np.round(x, 12)

def topk(x, k):
    pos = np.flatnonzero(x > 0)
    if pos.size <= k: return set(pos)
    thr = np.sort(x[pos])[::-1][k - 1]
    return set(np.flatnonzero(x >= thr))

def overlap(a, b, k=10):
    A, B = topk(a, k), topk(b, k)
    return len(A & B) / min(len(A), len(B)) if A and B else np.nan

def tau50(a, b):
    U = [i for i in sorted(topk(a, 50) | topk(b, 50)) if a[i] > 0 and b[i] > 0]
    if len(U) < 3: return np.nan
    t = kendalltau(a[U], b[U]).statistic
    return 1.0 if np.isnan(t) else t

def compare(s_new, s_base, q):
    # порядок вузлів може різнитися між графами — вирівнюємо за індексом базового варіанта
    s_new = s_new.reindex(s_base.index)
    a, b = prep(s_new, q), prep(s_base, q)
    return dict(перетин_топ10=overlap(a, b), τ_b_топ50=tau50(a, b), L1=np.abs(s_new - s_base).sum())

def locality(s, q, g):
    dd = nx.single_source_shortest_path_length(g.to_undirected(as_view=True) if g.is_directed() else g, q)
    dist = np.array([dd.get(v, np.inf) for v in s.index], dtype=float)
    x = s.to_numpy(dtype=float); p = x[x > 0]; fin = np.isfinite(dist)
    return dict(маса_r_до_2=x[dist <= 2].sum(), сер_відстань=(x[fin] * dist[fin]).sum() / max(x[fin].sum(), 1e-300),
                eff_вузлів=np.exp(-(p * np.log(p)).sum()))

def random_seeds(g, n=20, seed=0):
    r = np.random.default_rng(seed); nodes = list(g.nodes())
    return [nodes[i] for i in r.choice(len(nodes), n, replace=False)]

ER = G.erdos_renyi(1000, 0.01, seed=1)
BA = G.barabasi_albert(1000, 5, seed=5)
SYN = {"ER": ER, "BA": BA}
SEEDS = {k: random_seeds(g) for k, g in SYN.items()}

g_mq, q_mq, _ = MQ.load_processed("../data/metaqa", 500, 0)
r = np.random.default_rng(0)
movies = sorted({t for t in q_mq.topic if "movie" in g_mq.nodes[t]["kinds"]})
people = sorted({t for t in q_mq.topic if {"actor", "director", "writer"} & set(g_mq.nodes[t]["kinds"])
                 and "movie" not in g_mq.nodes[t]["kinds"]})
SEEDS_MQ = list(r.choice(movies, 10, replace=False)) + list(r.choice(people, 10, replace=False))
print(f"Синтетика: ER {ER.number_of_nodes()} вузлів / {ER.number_of_edges()} ребер, BA {BA.number_of_nodes()} / {BA.number_of_edges()}; "
      f"MetaQA {g_mq.number_of_nodes()} / {g_mq.number_of_edges()}")
""")

# ======================================================================== 4.1
md(r"""
## Завдання 4.1. Напрям ребер

### Синтетика: орієнтовані ER і BA з часткою взаємних ребер ρ

Кожне ребро з імовірністю ρ стає двостороннім, інакше отримує одну дугу випадкового напряму (`orient_with_reciprocity`). Порівнюємо з неорієнтованим графом на тих самих 20 запитах. Стратегія для висячих вузлів — restart для обох моделей (замовчування після рішення за результатами 4.2).
""")

code(r"""
RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
rows = []
for gname, g in SYN.items():
    for rho in RHOS:
        d = G.orient_with_reciprocity(g, rho, seed=10)
        dang = np.mean([d.out_degree(v) == 0 for v in d])
        for q in SEEDS[gname]:
            reach = len(nx.descendants(d, q)) + 1
            for model, kw in (("PPR", dict(method="ppr", alpha=ALPHA)), ("теплове ядро", dict(method="heat", t=T))):
                s_dir = rank(d, q, **kw); s_und = rank(g, q, **kw)
                rows.append(dict(граф=gname, ρ=rho, модель=model, частка_висячих=dang, досяжних=reach,
                                 **compare(s_dir, s_und, q), **locality(s_dir, q, d)))
dir_syn = pd.DataFrame(rows).groupby(["граф", "ρ", "модель"], sort=False).mean().reset_index()
dir_syn.to_csv(RES / "04_napryam_syntetyka.csv", index=False)
display(dir_syn)

fig, axs = plt.subplots(1, 4, figsize=(20, 4.5))
for i, gname in enumerate(SYN):
    for model, ls in (("PPR", "-"), ("теплове ядро", "--")):
        sub = dir_syn[(dir_syn["граф"] == gname) & (dir_syn["модель"] == model)]
        axs[0].plot(sub["ρ"], sub["перетин_топ10"], ls, marker="o", color=f"C{i}", label=f"{gname}, {model}")
        axs[1].plot(sub["ρ"], sub["τ_b_топ50"], ls, marker="o", color=f"C{i}")
        axs[2].plot(sub["ρ"], sub["маса_r_до_2"], ls, marker="o", color=f"C{i}")
    sub = dir_syn[(dir_syn["граф"] == gname) & (dir_syn["модель"] == "PPR")]
    axs[3].plot(sub["ρ"], sub["досяжних"], marker="o", color=f"C{i}", label=f"{gname}: досяжних вузлів")
for ax, lab in zip(axs, ("перетин топ-10 з неорієнтованим", "τ_b з неорієнтованим, топ-50",
                         "маса в радіусі r ≤ 2", "досяжних від запиту вузлів (з 1000)")):
    ax.set_xlabel("частка взаємних ребер ρ"); ax.set_ylabel(lab); ax.grid(alpha=0.3)
axs[0].legend(fontsize=8); axs[3].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "04_napryam_syntetyka.png", dpi=200); plt.show()
""")

md(r"""
### MetaQA (а): чисто орієнтований граф вироджений

У KB усі дуги ведуть фільм → атрибут. Тому кожен вузол, що не є фільмом, — висячий. Від людини чи тегу маса нікуди не йде: π = s за будь-якої стратегії. Від фільму вона за один крок потрапляє в атрибути і далі не йде.
""")

code(r"""
g_dir, _, _ = MQ.load_processed("../data/metaqa", 500, 0, directed=True)
n_dang = sum(1 for v in g_dir if g_dir.out_degree(v) == 0)
print(f"Орієнтований MetaQA: висячих вузлів {n_dang} з {g_dir.number_of_nodes()} "
      f"({n_dang / g_dir.number_of_nodes():.3f}); фільмів {sum('movie' in k for _, k in g_dir.nodes(data='kinds'))}")
rows = []
for q in SEEDS_MQ:
    kind = "фільм" if "movie" in g_dir.nodes[q]["kinds"] else "людина"
    for model, kw in (("PPR", dict(method="ppr", alpha=ALPHA)), ("теплове ядро", dict(method="heat", t=T))):
        for strat in ("restart", "self_loop"):
            s = rank(g_dir, q, dangling=strat, **kw)
            nb = set(g_dir.successors(q))
            rows.append(dict(запит=kind, модель=model, стратегія=strat, маса_у_запиті=s[q],
                             маса_на_сусідах=s[list(nb)].sum() if nb else 0.0,
                             ненульових_вузлів=int((s > 1e-15).sum())))
degen = pd.DataFrame(rows).groupby(["запит", "модель", "стратегія"], sort=False).mean().reset_index()
display(degen)
""")

md(r"""
### MetaQA (б): асиметричні ваги «вперед/назад»

Граф із дугами в обидва боки: s → o має тип r («вперед»), o → s — тип r⁻¹ («назад»).

**Твердження.** Нехай вага кожної дуги дорівнює w_ij = λ_{c(ij)}·ω_ij, де c(ij) ∈ {вперед, назад} — її напрям, а ω_ij — будь-яка вага, що не залежить від напряму. Якщо в графі немає вузла, з якого виходять дуги обох напрямів, то матриця переходів P = D⁻¹W, а отже π* і u(t), не залежать від λ_вперед і λ_назад.

**Доведення.** Для вузла i з дугами лише одного напряму c маємо d_i = Σ_j λ_c·ω_ij = λ_c·Σ_j ω_ij. Тому P_ij = λ_c·ω_ij/(λ_c·Σ_k ω_ik) = ω_ij/Σ_k ω_ik: λ_c скорочується. Так у кожному рядку, тож P від (λ_вперед, λ_назад) не залежить. π* і u(t) є функціями лише P і s. ∎

**Наслідки для MetaQA:**
- з фільму виходять лише дуги «вперед», з атрибутів і людей — лише дуги «назад»;
- люди в кількох ролях мають лише дуги «назад» (різних типів), тож на спільне відношення не реагують, але реагують на ваги «назад» **за типами**;
- твердження порушує лише вузол, що є одночасно фільмом і об'єктом. У підграфі це «2012» (фільм і рік).

Нижче — чисельна перевірка з «2012» і без нього.
""")

code(r"""
edges = pd.read_csv("../data/metaqa/processed/movies500_seed0/subgraph_edges.csv", dtype=str, keep_default_na=False)
trip = list(edges.itertuples(index=False, name=None))
trip_no2012 = [t for t in trip if "2012" not in (t[0], t[2])]
RELS = sorted(MQ.RELATION_KIND)
BACK = [r + MQ.REVERSE_SUFFIX for r in RELS]
VARIANTS = {"з «2012»": trip, "без «2012»": trip_no2012}
mixed = {name: [v for v, k in MQ.node_kinds(tr).items() if "movie" in k and len(k) > 1] for name, tr in VARIANTS.items()}
print("Вузли з дугами обох напрямів:", mixed)

rows = []
for vname, tr in VARIANTS.items():
    g_bi = MQ.to_graph(tr, bidirected=True); g_un = MQ.to_graph(tr)
    seeds = [q for q in SEEDS_MQ if q in g_un]
    for ratio in (0.1, 0.5, 1.0, 2.0, 10.0):
        tw = {r: ratio for r in RELS} | {b: 1.0 for b in BACK}
        for q in seeds:
            for model, kw in (("PPR", dict(method="ppr", alpha=ALPHA)), ("теплове ядро", dict(method="heat", t=T))):
                rows.append(dict(варіант=vname, w_f_до_w_b=ratio, модель=model,
                                 **compare(rank(g_bi, q, type_weights=tw, **kw), rank(g_un, q, **kw), q)))
prop = pd.DataFrame(rows)
prop_s = prop.groupby(["варіант", "w_f_до_w_b", "модель"], sort=False).agg(
    L1_макс=("L1", "max"), L1_сер=("L1", "mean"), перетин_топ10_мін=("перетин_топ10", "min"), τ_b_мін=("τ_b_топ50", "min")).reset_index()
prop_s.to_csv(RES / "04_metaqa_napryam_tverdzhennya.csv", index=False)
pd.set_option("display.float_format", "{:.3g}".format)
display(prop_s)

# Ваги «назад» за типами: starred_actors⁻¹ = c, решта «назад» = 1, «вперед» = 1 — діють через людей у кількох ролях
g_bi = MQ.to_graph(trip, bidirected=True); g_un = MQ.to_graph(trip)
multi_role = [v for v, k in g_un.nodes(data="kinds") if len({"actor", "director", "writer"} & set(k)) > 1]
print(f"Людей у кількох ролях: {len(multi_role)}")
rows = []
for c in (0.1, 0.5, 2.0, 10.0):
    tw = {r: 1.0 for r in RELS} | {b: 1.0 for b in BACK} | {"starred_actors" + MQ.REVERSE_SUFFIX: c}
    for q in SEEDS_MQ:
        for model, kw in (("PPR", dict(method="ppr", alpha=ALPHA)), ("теплове ядро", dict(method="heat", t=T))):
            rows.append(dict(вага_starred_назад=c, модель=model, **compare(rank(g_bi, q, type_weights=tw, **kw), rank(g_un, q, **kw), q)))
back_t = pd.DataFrame(rows).groupby(["вага_starred_назад", "модель"], sort=False).mean().reset_index()
back_t.to_csv(RES / "04_metaqa_vagy_nazad_za_typamy.csv", index=False)
pd.set_option("display.float_format", "{:.3f}".format)
display(back_t)

fig, axs = plt.subplots(1, 2, figsize=(14, 4.6))
for vname, ls in (("з «2012»", "-"), ("без «2012»", "--")):
    for model, c in (("PPR", "C0"), ("теплове ядро", "C1")):
        sub = prop_s[(prop_s["варіант"] == vname) & (prop_s["модель"] == model)]
        axs[0].semilogy(sub["w_f_до_w_b"], np.maximum(sub["L1_макс"], 1e-17), ls, marker="o", color=c, label=f"{model}, {vname}")
axs[0].set_xscale("log"); axs[0].axhline(1e-15, color="gray", lw=0.8, ls=":")
axs[0].text(0.11, 2e-15, "межа машинної точності", fontsize=8, color="gray")
axs[0].set_xlabel("відношення ваг «вперед» / «назад» w_f/w_b"); axs[0].set_ylabel("макс. ‖π − π_неорієнт‖₁ за 20 запитами")
axs[0].set_title("Твердження 4.1: від w_f/w_b залежить лише через вузол «2012»")
for model, c in (("PPR", "C0"), ("теплове ядро", "C1")):
    sub = back_t[back_t["модель"] == model]
    axs[1].plot(sub["вага_starred_назад"], sub["L1"], "o-", color=c, label=f"{model}: ‖Δπ‖₁")
    axs[1].plot(sub["вага_starred_назад"], 1 - sub["перетин_топ10"], "s--", color=c, label=f"{model}: 1 − перетин топ-10")
axs[1].set_xscale("log"); axs[1].set_xlabel("вага дуг starred_actors⁻¹ (інші = 1)"); axs[1].set_ylabel("зміна розподілу")
axs[1].set_title("Ваги «назад» за типами (через людей у кількох ролях)")
for ax in axs: ax.grid(alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "04_metaqa_napryam.png", dpi=200); plt.show()
""")

# ======================================================================== 4.2
md(r"""
## Завдання 4.2. Висячі вузли: restart проти self_loop

Беремо орієнтовані ER і BA (ρ = 0.5) і видаляємо вихідні дуги в частки f ∈ {0.01, 0.05, 0.1, 0.2} вузлів. Порівнюємо стратегії для обох моделей. Запити, що самі стали висячими, виключаємо, бо від них маса нікуди не йде; їхню кількість показано.

**Теорія (PPR).** Для вузла i, що не є запитом, з вхідним потоком F_i = Σ_{j≠i} π_j·P_ji:
- звичайний вузол: π_i = (1 − α)·F_i;
- висячий вузол із петлею (self_loop): π_i = (1 − α)(F_i + π_i), тобто π_i = (1 − α)·F_i/α;
- висячий вузол зі стратегією restart: π_i = (1 − α)·F_i (його маса йде на s).

Отже, петля підсилює висячий вузол **у 1/α ≈ 6.67 раза** відносно звичайного вузла з тим самим вхідним потоком. Перевіряємо відношення π_i/((1 − α)·F_i).

**Теплове ядро з петлею:** висячий вузол поглинає масу, тож вона там лише накопичується з часом.
""")

code(r"""
def make_dangling(g, f, seed):
    d = G.orient_with_reciprocity(g, 0.5, seed=20)
    r = np.random.default_rng(seed); nodes = list(d.nodes())
    kill = [nodes[i] for i in r.choice(len(nodes), int(round(f * len(nodes))), replace=False)]
    d.remove_edges_from([e for v in kill for e in list(d.out_edges(v))])
    return d

def inflow_ratio(d, s, q, alpha):
    # π_i / ((1 − α)·F_i), F_i = Σ_{j≠i} π_j w_ji / d_j (вхідний потік від інших вузлів)
    outw = {v: sum(dd.get("weight", 1.0) for _, _, dd in d.out_edges(v, data=True)) for v in d}
    F = {v: 0.0 for v in d}
    for j, i, dd in d.edges(data=True):
        if i != j and outw[j] > 0:
            F[i] += s[j] * dd.get("weight", 1.0) / outw[j]
    return {i: s[i] / ((1 - alpha) * F[i]) for i in d if i != q and F[i] > 1e-12}

FS = [0.01, 0.05, 0.1, 0.2]
rows, amp, skipped = [], [], []
for gname, g in SYN.items():
    for f in FS:
        d = make_dangling(g, f, seed=30)
        dang = np.array([d.out_degree(v) == 0 for v in d.nodes()])
        frac = dang.mean()
        # запит, що сам став висячим, вироджений (маса нікуди не йде) — виключаємо і рахуємо
        skipped.append(dict(граф=gname, f=f, виключено_запитів=sum(d.out_degree(q) == 0 for q in SEEDS[gname])))
        for q in SEEDS[gname]:
            if d.out_degree(q) == 0:
                continue
            res = {}
            for model, kw in (("PPR", dict(method="ppr", alpha=ALPHA)), ("теплове ядро", dict(method="heat", t=T))):
                for strat in ("restart", "self_loop"):
                    s = rank(d, q, dangling=strat, **kw)
                    res[(model, strat)] = s
                    x = prep(s, q); dd = np.delete(dang, list(d.nodes()).index(q))
                    T10 = sorted(topk(x, 10))
                    rows.append(dict(граф=gname, f=f, частка_висячих=frac, модель=model, стратегія=strat,
                                     маса_на_висячих=s[dang].sum() - (s[q] if d.out_degree(q) == 0 else 0),
                                     висячих_у_топ10=dd[T10].mean()))
                    if model == "PPR" and f == 0.1:
                        rat = inflow_ratio(d, s, q, ALPHA)
                        for i, v in rat.items():
                            amp.append(dict(граф=gname, стратегія=strat, висячий=d.out_degree(i) == 0, відношення=v))
            for model in ("PPR", "теплове ядро"):
                a, b = prep(res[(model, "restart")], q), prep(res[(model, "self_loop")], q)
                rows.append(dict(граф=gname, f=f, частка_висячих=frac, модель=model, стратегія="перетин restart/self_loop",
                                 перетин_топ10=overlap(a, b)))
dang_t = pd.DataFrame(rows).groupby(["граф", "f", "модель", "стратегія"], sort=False).mean().reset_index()
dang_t.to_csv(RES / "04_visyachi_vuzly.csv", index=False)
print("Запити, що самі стали висячими і виключені:", pd.DataFrame(skipped).to_dict("records"))
display(dang_t)
amp_t = pd.DataFrame(amp).groupby(["граф", "стратегія", "висячий"])["відношення"].agg(["median", "min", "max", "count"]).reset_index()
amp_t.to_csv(RES / "04_pidsylennya_1_alpha.csv", index=False)
pd.set_option("display.float_format", "{:.6f}".format)
print(f"Відношення π_i/((1 − α)F_i), f = 0.1 (теорія: 1 для звичайних і висячих з restart, 1/α = {1 / ALPHA:.6f} для висячих з self_loop):")
display(amp_t)
pd.set_option("display.float_format", "{:.3f}".format)
""")

code(r"""
fig, axs = plt.subplots(1, 3, figsize=(19, 4.8))
for i, gname in enumerate(SYN):
    for model, mk in (("PPR", "o"), ("теплове ядро", "s")):
        for strat, ls in (("restart", "--"), ("self_loop", "-")):
            sub = dang_t[(dang_t["граф"] == gname) & (dang_t["модель"] == model) & (dang_t["стратегія"] == strat)]
            axs[0].plot(sub["частка_висячих"], sub["висячих_у_топ10"], ls, marker=mk, color=f"C{2 * i + (model != 'PPR')}",
                        label=f"{gname}, {model}, {strat}")
            axs[1].plot(sub["частка_висячих"], sub["маса_на_висячих"], ls, marker=mk, color=f"C{2 * i + (model != 'PPR')}")
xx = np.linspace(0, 0.21, 10)
axs[0].plot(xx, xx, "k:", lw=1, label="y = x (як серед усіх вузлів)")
axs[0].set_xlabel("частка висячих вузлів у графі"); axs[0].set_ylabel("частка висячих у топ-10")
axs[0].set_title("Чи витягує стратегія тупикові вузли в топ"); axs[0].legend(fontsize=6)
axs[1].set_xlabel("частка висячих вузлів у графі"); axs[1].set_ylabel("сумарна маса на висячих вузлах")
axs[1].set_title("Маса на висячих вузлах")

d = make_dangling(ER, 0.1, seed=30); dang = np.array([d.out_degree(v) == 0 for v in d.nodes()])
ts = np.linspace(0, 40, 41)
for strat, ls in (("restart", "--"), ("self_loop", "-")):
    m = np.mean([[rank(d, q, "heat", t=t, dangling=strat)[dang].sum() for t in ts]
                 for q in [q for q in SEEDS["ER"] if d.out_degree(q) > 0][:10]], axis=0)
    axs[2].plot(ts, m, ls, lw=1.6, label=f"теплове ядро, {strat}")
axs[2].set_xlabel("час дифузії t"); axs[2].set_ylabel("маса на висячих вузлах")
axs[2].set_title("ER, f = 0.1: петля робить висячі вузли поглинаючими")
for ax in axs: ax.grid(alpha=0.3)
axs[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "04_visyachi_vuzly.png", dpi=200); plt.show()
""")

# ======================================================================== 4.3
md(r"""
## Завдання 4.3. Ваги

### Синтетика: логнормальні ваги з різним розкидом σ

Ваги ребер w = exp(σZ − σ²/2) мають середнє 1. Порівнюємо з незваженим графом.
""")

code(r"""
SIG = [0.0, 0.25, 0.5, 1.0, 2.0]
rows = []
for gname, g in SYN.items():
    for sig in SIG:
        gw = G.lognormal_weights(g, sig, seed=40)
        for q in SEEDS[gname]:
            for model, kw in (("PPR", dict(method="ppr", alpha=ALPHA)), ("теплове ядро", dict(method="heat", t=T))):
                rows.append(dict(граф=gname, σ=sig, модель=model, **compare(rank(gw, q, **kw), rank(g, q, **kw), q),
                                 **locality(rank(gw, q, **kw), q, gw)))
w_syn = pd.DataFrame(rows).groupby(["граф", "σ", "модель"], sort=False).mean().reset_index()
w_syn.to_csv(RES / "04_vagy_syntetyka.csv", index=False)
display(w_syn)

fig, axs = plt.subplots(1, 3, figsize=(17, 4.5))
for i, gname in enumerate(SYN):
    for model, ls in (("PPR", "-"), ("теплове ядро", "--")):
        sub = w_syn[(w_syn["граф"] == gname) & (w_syn["модель"] == model)]
        axs[0].plot(sub["σ"], sub["перетин_топ10"], ls, marker="o", color=f"C{i}", label=f"{gname}, {model}")
        axs[1].plot(sub["σ"], sub["τ_b_топ50"], ls, marker="o", color=f"C{i}")
        axs[2].plot(sub["σ"], sub["eff_вузлів"], ls, marker="o", color=f"C{i}")
for ax, lab in zip(axs, ("перетин топ-10 з незваженим", "τ_b з незваженим, топ-50", "exp(H), вузлів")):
    ax.set_xlabel("розкид логнормальних ваг σ"); ax.set_ylabel(lab); ax.grid(alpha=0.3)
axs[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "04_vagy_syntetyka.png", dpi=200); plt.show()
""")

md(r"""
### MetaQA: сценарії ваг (неорієнтований граф)

1. **Рівні ваги** — базовий сценарій.
2. **«Змістовні зв'язки важчі»:** актор, режисер, сценарист мають вагу r, решта (жанр, рік, мова, теги, imdb) — вагу 1, r ∈ {0.25, 0.5, 1, 2, 4, 8}.
3. **«Послаблені зв'язки до хабів» (графовий аналог IDF):** вага ребра фільм–об'єкт дорівнює d_об'єкта^(−β), β ∈ {0.5, 1}, де d — степінь об'єкта (атрибута або людини) у підграфі. Рідкісний атрибут вважаємо інформативнішим. Ця вага діє з боку фільму: з самого об'єкта всі його ребра однаково послаблені і після нормування рівні.

**Вимірюємо:**
- склад топ-10 за типами;
- потік імовірності за типами ребер: Σ π_i·P_ij по ребрах кожного типу, обидва напрями, частка від загального;
- перетин топ-10 і τ_b з рівними вагами;
- локальність;
- частку хабів у топ-10;
- медіанний ранг Comedy.
""")

code(r"""
PERSON = {"starred_actors", "directed_by", "written_by"}
deg_un = dict(g_mq.degree())
PRIOR = ["movie", "actor", "director", "writer", "genre", "year", "language", "tag", "imdb_rating", "imdb_votes"]
UA = {"movie": "фільм", "actor": "людина", "director": "людина", "writer": "людина", "genre": "жанр", "year": "рік",
      "language": "мова", "tag": "тег", "imdb_rating": "imdb", "imdb_votes": "imdb"}
kind_of = {v: UA[min(k, key=PRIOR.index)] for v, k in g_mq.nodes(data="kinds")}
hub_thr = np.quantile(list(deg_un.values()), 0.99)

def idf_graph(beta):
    h = nx.MultiGraph(); h.add_nodes_from(g_mq.nodes(data=True))
    for s_, r_, o_ in trip:
        h.add_edge(s_, o_, type=r_, weight=deg_un[o_] ** (-beta))
    return h

SCEN = [("рівні", g_mq, {})]
for rr in (0.25, 0.5, 2.0, 4.0, 8.0):
    SCEN.append((f"змістовні r={rr}", g_mq, dict(type_weights={r_: (rr if r_ in PERSON else 1.0) for r_ in RELS})))
for beta in (0.5, 1.0):
    SCEN.append((f"IDF β={beta}", idf_graph(beta), {}))

def flux_by_type(g, s, wdeg, weighted_attr=None, tw=None):
    fl = {}
    for u, v, dd in g.edges(data=True):
        w = tw[dd["type"]] if tw else dd.get("weight", 1.0)
        fl[dd["type"]] = fl.get(dd["type"], 0.0) + s[u] * w / wdeg[u] + s[v] * w / wdeg[v]
    tot = sum(fl.values())
    return {k: v / tot for k, v in fl.items()}

rows, flux_rows, type_rows = [], [], []
base = {}
for sname, gs, extra in SCEN:
    for q in SEEDS_MQ:
        for model, kw in (("PPR", dict(method="ppr", alpha=ALPHA)), ("теплове ядро", dict(method="heat", t=T))):
            s, info = rank(gs, q, return_info=True, **kw, **extra)
            if sname == "рівні":
                base[(q, model)] = s
            cmp = compare(s, base[(q, model)], q)
            x = prep(s, q); nodes_ex = [v for v in s.index if v != q]
            T10 = sorted(topk(x, 10))
            order = np.argsort(-x, kind="stable"); pos = {nodes_ex[i]: k + 1 for k, i in enumerate(order)}
            rows.append(dict(сценарій=sname, модель=model, **cmp, **locality(s, q, gs),
                             хаби_в_топ10=np.mean([deg_un[nodes_ex[i]] >= hub_thr for i in T10]),
                             ранг_Comedy=pos.get("Comedy", np.nan)))
            for kname, c in pd.Series([kind_of[nodes_ex[i]] for i in T10]).value_counts(normalize=True).items():
                type_rows.append(dict(сценарій=sname, модель=model, тип=kname, частка=c / len(SEEDS_MQ)))
            for k_, v_ in flux_by_type(gs, s, info["degrees"], tw=extra.get("type_weights")).items():
                flux_rows.append(dict(сценарій=sname, модель=model, тип_ребра=k_, потік=v_ / len(SEEDS_MQ)))
agg = dict(перетин_топ10=("перетин_топ10", "mean"), τ_b_топ50=("τ_b_топ50", "mean"), L1=("L1", "mean"),
           маса_r_до_2=("маса_r_до_2", "mean"), сер_відстань=("сер_відстань", "mean"), eff_вузлів=("eff_вузлів", "mean"),
           хаби_в_топ10=("хаби_в_топ10", "mean"), медіанний_ранг_Comedy=("ранг_Comedy", "median"))
scen_t = pd.DataFrame(rows).groupby(["сценарій", "модель"], sort=False).agg(**agg).reset_index()
scen_t.to_csv(RES / "04_metaqa_stsenarii.csv", index=False)
display(scen_t)
types_t = pd.DataFrame(type_rows).groupby(["сценарій", "модель", "тип"], sort=False)["частка"].sum().unstack(fill_value=0)
types_t.to_csv(RES / "04_metaqa_typy_top10.csv")
display(types_t)
flux_t = pd.DataFrame(flux_rows).groupby(["сценарій", "модель", "тип_ребра"], sort=False)["потік"].sum().unstack(fill_value=0)
flux_t.to_csv(RES / "04_metaqa_potik_za_typamy.csv")
display(flux_t)
""")

code(r"""
fig, axs = plt.subplots(1, 3, figsize=(23, 5.2))
order = [s for s, _, _ in SCEN]
sub = types_t.xs("PPR", level="модель").reindex(order)
bottom = np.zeros(len(sub))
for c, col in enumerate(sub.columns):
    axs[0].bar(range(len(sub)), sub[col], bottom=bottom, color=f"C{c}", label=col, edgecolor="white", linewidth=1)
    bottom += sub[col].to_numpy()
axs[0].set_xticks(range(len(sub))); axs[0].set_xticklabels(order, rotation=35, ha="right", fontsize=8)
axs[0].set_ylabel("частка вузлів у топ-10 (PPR)"); axs[0].set_title("Склад топ-10 за типами")
axs[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
fsub = flux_t.xs("PPR", level="модель").reindex(order)
bottom = np.zeros(len(fsub))
for c, col in enumerate(fsub.columns):
    axs[1].bar(range(len(fsub)), fsub[col], bottom=bottom, color=f"C{c}", label=col, edgecolor="white", linewidth=1)
    bottom += fsub[col].to_numpy()
axs[1].set_xticks(range(len(fsub))); axs[1].set_xticklabels(order, rotation=35, ha="right", fontsize=8)
axs[1].set_ylabel("частка потоку ймовірності (PPR)"); axs[1].set_title("Потік за типами ребер")
axs[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
for model, ls in (("PPR", "-"), ("теплове ядро", "--")):
    s2 = scen_t[scen_t["модель"] == model].set_index("сценарій").reindex(order)
    axs[2].plot(range(len(order)), s2["перетин_топ10"], ls, marker="o", label=f"{model}: перетин топ-10 з «рівні»")
    axs[2].plot(range(len(order)), s2["хаби_в_топ10"], ls, marker="s", label=f"{model}: частка хабів у топ-10")
axs[2].set_xticks(range(len(order))); axs[2].set_xticklabels(order, rotation=35, ha="right", fontsize=8)
axs[2].set_title("Зміна ранжування і роль хабів"); axs[2].legend(fontsize=7)
for ax in axs: ax.grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig(FIG / "04_metaqa_stsenarii_vag.png", dpi=200); plt.show()
""")

md(CONCL.read_text(encoding="utf-8") if CONCL.exists() else "## Висновки\n\n_(заповнюється за результатами виконання)_")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, out)
print("ok", out)
