"""Бутстреп-інтервали (95%, 2000 повторень, seed 0) над збереженими результатами окремих запитів.

Без перерахунку моделей — лише арифметика над results/*.csv:
- п. 6: макро-nDCG@10 за hop — обраний варіант (PPR, α = 0.5, сирі) проти найкоротшого шляху (random-нічиї).
  Джерело: results/06_metaqa_test_zapyty.csv (окремі test-питання). Бутстреп стратифікований за qtype
  (питання переобираються всередині свого qtype), різниця — парна (ті самі переобрані питання).
- п. 7.2: MRR рекомендованого варіанта і оракула (k-hop ∩, k = 1, random) при f = 0.3 і 0.5.
  Джерело: results/07_pary_po_parakh.csv (окремі пари × 3 seed шуму). Кластерний бутстреп за парами
  (пара переобирається разом з усіма seed); два варіанти — ізольовані пари виключено / враховано як 0.

Інтервалів НЕМАЄ (збережено лише агреговані таблиці, перерахунку не робимо):
- п. 6, таблиця однакового бюджету з k-hop (results/06_metaqa_byudzhet_khop.csv — лише середні);
- п. 6, вибір варіанта 0.371 проти 0.372 (results/06_metaqa_verkhnya_mezha.csv — лише середні);
- п. 5, рекомендований варіант проти суми відстаней за r (results/05_mosty_syntetyka.csv — лише середні).

Запуск з кореня relevance-diffusion-kg: python tools/bootstrap_ci.py
Результат: results/ci_p6_makro_ndcg.csv, results/ci_p72_pary.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
B, SEED, LEVEL = 2000, 0, 0.95
LO, HI = 100 * (1 - LEVEL) / 2, 100 * (1 + LEVEL) / 2


def ci(x):
    return float(np.percentile(x, LO)), float(np.percentile(x, HI))


# ---------------------------------------------------------------- п. 6
q = pd.read_csv(RES / "06_metaqa_test_zapyty.csv")
q = q[(q["фільтр"] == "ні") & (q["нічиї"] == "random")]
dif = q[(q["метод"] == "PPR, α=0.5") & (q["варіант"] == "сирі")].set_index(["topic", "qtype"])
sp = q[(q["метод"] == "найкор. шлях") & q["варіант"].isna()].set_index(["topic", "qtype"])
both = dif[["hop", "nDCG@10"]].rename(columns={"nDCG@10": "дифузія"}).join(sp[["nDCG@10"]].rename(columns={"nDCG@10": "шлях"}),
                                                                           how="inner").reset_index()
assert len(both) == len(dif) == len(sp), (len(both), len(dif), len(sp))

rng = np.random.default_rng(SEED)
groups = {qt: g[["дифузія", "шлях"]].to_numpy() for qt, g in both.groupby("qtype")}
hop_of = both.groupby("qtype")["hop"].first().to_dict()
boot = {h: {"дифузія": [], "шлях": []} for h in (1, 2, 3, "усі")}
point = {}
for h in (1, 2, 3, "усі"):
    qts = [qt for qt in groups if h == "усі" or hop_of[qt] == h]
    point[h] = np.mean([groups[qt].mean(axis=0) for qt in qts], axis=0)
for _ in range(B):
    means = {qt: g[rng.integers(len(g), size=len(g))].mean(axis=0) for qt, g in groups.items()}
    for h in (1, 2, 3, "усі"):
        m = np.mean([means[qt] for qt in means if h == "усі" or hop_of[qt] == h], axis=0)
        boot[h]["дифузія"].append(m[0]); boot[h]["шлях"].append(m[1])
rows = []
for h in (1, 2, 3, "усі"):
    d_, s_ = np.array(boot[h]["дифузія"]), np.array(boot[h]["шлях"])
    diff = d_ - s_
    lo_d, hi_d = ci(d_); lo_s, hi_s = ci(s_); lo_r, hi_r = ci(diff)
    rows.append(dict(hop=h, питань=int(sum(len(groups[qt]) for qt in groups if h == "усі" or hop_of[qt] == h)),
                     дифузія=point[h][0], дифузія_ДІ_низ=lo_d, дифузія_ДІ_верх=hi_d,
                     найкор_шлях=point[h][1], шлях_ДІ_низ=lo_s, шлях_ДІ_верх=hi_s,
                     різниця=point[h][0] - point[h][1], різниця_ДІ_низ=lo_r, різниця_ДІ_верх=hi_r,
                     значуща=bool(lo_r > 0 or hi_r < 0)))
p6 = pd.DataFrame(rows)
p6.to_csv(RES / "ci_p6_makro_ndcg.csv", index=False)

# ---------------------------------------------------------------- п. 7.2
pr = pd.read_csv(RES / "07_pary_po_parakh.csv")
pr["пара"] = pr.groupby(["режим", "f", "seed"]).cumcount()
rng = np.random.default_rng(SEED)
rows = []
for (mode, f, kind), g in pr[pr["f"].isin([0.3, 0.5])].groupby(["режим", "f", "kind"]):
    for variant in ("ізольовані виключено", "ізольовані = 0"):
        gg = g if variant == "ізольовані = 0" else g[~g["ізольована"]]
        ids = np.sort(gg["пара"].unique())
        by_pair = {i: gg[gg["пара"] == i][["MRR_дифузії", "MRR_оракула_random"]].to_numpy() for i in ids}
        bd, bo = [], []
        for _ in range(B):
            smp = np.concatenate([by_pair[i] for i in rng.choice(ids, size=len(ids))])
            bd.append(smp[:, 0].mean()); bo.append(smp[:, 1].mean())
        bd, bo = np.array(bd), np.array(bo)
        lo_r, hi_r = ci(bd - bo)
        rows.append(dict(режим=mode, f=f, kind=kind, варіант=variant, пар=len(ids), рядків=len(gg),
                         MRR_дифузії=gg["MRR_дифузії"].mean(), дифузія_ДІ_низ=ci(bd)[0], дифузія_ДІ_верх=ci(bd)[1],
                         MRR_оракула=gg["MRR_оракула_random"].mean(), оракул_ДІ_низ=ci(bo)[0], оракул_ДІ_верх=ci(bo)[1],
                         різниця=gg["MRR_дифузії"].mean() - gg["MRR_оракула_random"].mean(),
                         різниця_ДІ_низ=lo_r, різниця_ДІ_верх=hi_r, значуща=bool(lo_r > 0 or hi_r < 0)))
p72 = pd.DataFrame(rows)
p72.to_csv(RES / "ci_p72_pary.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.float_format", "{:.3f}".format)
print(p6.to_string(index=False)); print(); print(p72.to_string(index=False))
