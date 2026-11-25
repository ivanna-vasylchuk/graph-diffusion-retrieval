# Джерела даних

## MetaQA (MoviE Text Audio QA)

- **Репозиторій:** https://github.com/yuyuz/MetaQA
- **Завантаження:** публічна папка Google Drive, посилання в README репозиторію:
  https://drive.google.com/drive/folders/0B-36Uca2AvwhTWVFSUZqRXVtbUE?resourcekey=0-kdv6ho5KcpEXdI2aUdLn_g
- **Що використовуємо в курсовій:**
  - `kb.txt` — граф знань про фільми, кожен рядок — трійка `subject|relation|object`;
  - `{1,2,3}-hop/vanilla/qa_{train,dev,test}.txt` — питання і відповіді: питання та відповіді розділені табуляцією, кілька відповідей розділені `|`, сутність питання виділена `[...]`;
  - `{1,2,3}-hop/qa_{train,dev,test}_qtype.txt` — типи питань (порядок рядків той самий, що у vanilla).
- **Не використовуємо:** аудіоверсію питань (`audio/`) і перефразовані питання (`ntm/`).
- **Завантаження:** вручну через браузер з папки Google Drive за посиланням вище (2026-11-24). Автоматичне завантаження (gdown) не використовується. Розкладено як `data/metaqa/kb.txt`, `data/metaqa/LICENSE.txt`, `data/metaqa/{1,2,3}-hop/vanilla/qa_*.txt`, `data/metaqa/{1,2,3}-hop/qa_*_qtype.txt`.
- **Перевірка цілісності:** `kb.txt` — 134 741 рядок, усі у форматі `s|r|o`; 43 234 сутності — збігається з опублікованим. Серед трійок 133 582 унікальні (1 159 повторів). Кількість питань у всіх 9 файлах збігається з таблицею нижче, кількість рядків у qtype-файлах дорівнює кількості питань. Типів питань у train: 13 (1-hop), 21 (2-hop), 15 (3-hop).
- **Опубліковані обсяги** (README і стаття): база знань — 134 741 трійка, 43 234 сутності. Кількість питань:

  |       |  1-hop |   2-hop |   3-hop |
  |-------|-------:|--------:|--------:|
  | Train | 96 106 | 118 980 | 114 196 |
  | Dev   |  9 992 |  14 872 |  14 274 |
  | Test  |  9 947 |  14 872 |  14 274 |

- **Походження:** база знань і 1-hop питання взято з WikiMovies (Facebook MovieQA, гілка `wiki_entities`). 2-hop і 3-hop питання автори згенерували за шаблонами: 21 тип для 2-hop і 15 типів для 3-hop, по 10 текстових шаблонів на тип.

## Ліцензія

**Creative Commons Attribution 3.0 Unported (CC BY 3.0)** — за файлом `data/metaqa/LICENSE.txt` (заголовок «CC License. THE METAQA DATASET … Creative Commons Legal Code. Attribution 3.0 Unported»). Ліцензія дозволяє використання й поширення за умови зазначення авторства (див. «Як цитувати»). У файлі ліцензії також є відсилання до README даних MovieQA (WikiMovies, https://research.fb.com/downloads/babi/), на яких частково побудовано MetaQA.

## Як цитувати

Yuyu Zhang, Hanjun Dai, Zornitsa Kozareva, Alexander J. Smola, Le Song. *Variational Reasoning for Question Answering with Knowledge Graph*. AAAI 2018. arXiv:1709.04071.

```bibtex
@inproceedings{zhang2017variational,
  title={Variational Reasoning for Question Answering with Knowledge Graph},
  author={Zhang, Yuyu and Dai, Hanjun and Kozareva, Zornitsa and Smola, Alexander J and Song, Le},
  booktitle={AAAI},
  year={2018}
}
```
