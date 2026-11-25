# graph-diffusion-retrieval

Графова дифузія для пошуку в графах знань: як поширювати релевантність від вузлів запиту графом і ранжувати сутності за близькістю до запиту.

Репозиторій об'єднує дві роботи:
- **курсову** — математичне ядро ранжування вузлів графа знань (папка `relevance-diffusion-kg/`);
- **дипломну** (згодом) — інтелектуальну систему пошуку і взаємодії з документами на основі RAG з елементом GraphRAG, яка використовуватиме ядро курсової на етапі ранжування вузлів.

## relevance-diffusion-kg/ — курсова робота

**Тема:** Моделювання поширення релевантності в графі знань як дифузійного процесу з перезапуском.

Дисципліна «Математичне моделювання», КПІ ім. Ігоря Сікорського, ФПМ, кафедра прикладної математики.

**Що всередині:**
- дві моделі — особистий PageRank (випадкове блукання з перезапуском, параметр α) і теплове ядро нормованого лапласіана (час t);
- модуль `src/reldiff` з єдиним інтерфейсом `rank(graph, seeds, method, **params)`;
- базові методи для порівняння: найкоротший шлях, спільні сусіди, k-hop;
- метрики P@k, nDCG@k, MRR і розмір вибірки для повного охоплення відповідей;
- дослідження на синтетичних графах і на підграфі MetaQA.

| Папка | Вміст |
|---|---|
| `src/reldiff/` | модуль ядра (див. `CLAUDE.md`, розділ «Структура проєкту») |
| `tests/` | тести `pytest`: перевірка адекватності моделей і модулів |
| `experiments/` | ноутбуки `00`–`07` (перевірка адекватності і пункти 1–7 ТЗ) і `08_demo.ipynb` (демонстрація для захисту) |
| `figures/`, `results/` | рисунки і таблиці, на яких побудовано висновки |
| `docs/` | аналітичні розв'язки (`analitychni_rozvyazky.md`), джерела даних (`dzherela.md`), матеріали для звіту (`dlya_zvitu.md`) |
| `tools/` | генератори ноутбуків і `docs/dlya_zvitu.md` (див. `tools/README.md`) |
| `materials/` | технічне завдання і загальні вимоги до курсової |
| `data/metaqa/` | дані MetaQA (у репозиторій не входять, див. нижче) |

## Встановлення і тести

Потрібен Python 3.10+.

```bash
cd relevance-diffusion-kg
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
```

Тести використовують лише синтетичні графи і штучну базу знань, тож дані MetaQA для них не потрібні.

Перебудувати ноутбук або матеріали для звіту — див. `relevance-diffusion-kg/tools/README.md`.

## Дані MetaQA

Ноутбуки `experiments/` використовують підграф бази знань **MetaQA** (MoviE Text Audio QA).

- **Джерело:** https://github.com/yuyuz/MetaQA. Дані завантажуються вручну з папки Google Drive, посилання на яку є в README репозиторію. Потрібні `kb.txt`, `LICENSE.txt`, а для 1-hop, 2-hop і 3-hop — папка `vanilla/` і файли `qa_*_qtype.txt`. Розкласти їх у `relevance-diffusion-kg/data/metaqa/` за схемою з `relevance-diffusion-kg/docs/dzherela.md`.
- **Підграф і запити** будує `reldiff.metaqa.prepare(root, n_movies=500, seed=0)`; результат зберігається в `data/metaqa/processed/movies500_seed0/`.
- **Ліцензія:** Creative Commons Attribution 3.0 Unported (CC BY 3.0). Копія ліцензії додається до даних.
- **Цитування:**

  Yuyu Zhang, Hanjun Dai, Zornitsa Kozareva, Alexander J. Smola, Le Song. *Variational Reasoning for Question Answering with Knowledge Graph*. AAAI 2018. arXiv:1709.04071.

  ```bibtex
  @inproceedings{zhang2017variational,
    title={Variational Reasoning for Question Answering with Knowledge Graph},
    author={Zhang, Yuyu and Dai, Hanjun and Kozareva, Zornitsa and Smola, Alexander J and Song, Le},
    booktitle={AAAI},
    year={2018}
  }
  ```

Дані MetaQA у репозиторій не входять (`.gitignore`).
