"""Поглинання маси висячими вузлами в тепловому ядрі (п. 4.2) — числа рисунка 04_visyachi_vuzly.png (права панель).

Відтворює конфігурацію ноутбука 04: ER (n = 1000, p = 0.01, seed = 1), орієнтування з часткою
взаємних ребер ρ = 0.5 (seed = 20), видалення вихідних дуг у 10% вузлів (seed = 30); запити —
перші 10 невисячих серед 20 випадкових вузлів (seed = 0), як на рисунку. Сітка t — як на рисунку
(0…40, крок 1) плюс t = 5.67. Додатково — середнє при t = 5.67 за всіма невисячими запитами з 20
(вибірка таблиці 04_visyachi_vuzly.csv). Лише синтетика, ≈ 30 с.

Запуск з кореня relevance-diffusion-kg: python tools/poglynannya_t.py
Результат: results/04_poglynannya_t.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from reldiff import graphs as G
from reldiff.api import rank

ROOT = Path(__file__).resolve().parents[1]
T_FIG = (1 - 0.15) / 0.15

ER = G.erdos_renyi(1000, 0.01, seed=1)
d = G.orient_with_reciprocity(ER, 0.5, seed=20)
r = np.random.default_rng(30)
nodes = list(d.nodes())
kill = [nodes[i] for i in r.choice(len(nodes), int(round(0.1 * len(nodes))), replace=False)]
d.remove_edges_from([e for v in kill for e in list(d.out_edges(v))])
dang = np.array([d.out_degree(v) == 0 for v in d.nodes()])

r0 = np.random.default_rng(0)
seeds20 = [nodes[i] for i in r0.choice(len(nodes), 20, replace=False)]
seeds_ok = [q for q in seeds20 if d.out_degree(q) > 0]
seeds10 = seeds_ok[:10]

rows = []
ts = sorted(set(np.linspace(0, 40, 41).tolist()) | {round(T_FIG, 4)})
for strat in ("restart", "self_loop"):
    for t in ts:
        m10 = np.mean([rank(d, q, "heat", t=t, dangling=strat)[dang].sum() for q in seeds10])
        rows.append(dict(вибірка=f"{len(seeds10)} запитів (рисунок)", стратегія=strat, t=t, маса_на_висячих=m10))
    m_all = np.mean([rank(d, q, "heat", t=T_FIG, dangling=strat)[dang].sum() for q in seeds_ok])
    rows.append(dict(вибірка=f"{len(seeds_ok)} запитів (таблиця 04_visyachi_vuzly.csv)", стратегія=strat,
                     t=round(T_FIG, 4), маса_на_висячих=m_all))

out = pd.DataFrame(rows)
out.to_csv(ROOT / "results" / "04_poglynannya_t.csv", index=False)
print(f"Висячих вузлів: {dang.sum()} з {len(nodes)}; невисячих запитів з 20: {len(seeds_ok)}")
key = out[out["t"].isin([round(T_FIG, 4), 20.0, 40.0])]
print(key.pivot_table(index=["вибірка", "стратегія"], columns="t", values="маса_на_висячих").round(3).to_string())
