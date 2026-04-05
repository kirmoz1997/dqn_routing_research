# Experiment Log

Журнал экспериментов проекта Multi-Agent Set Routing with DQN.

---

## Версия датасета

### v1 (323 записи) — использована в экспериментах ниже

- **Файл:** `data/tasks_set.jsonl`
- **SHA-256:** `d6e8f4d0f50b950f4e3c168ddf82cb49ab5181ce02c7bcaece0034e64db6a718`
- **Записей:** 323
- **Сплиты:** train=227, val=47, test=49 (70/15/15, стратификация по |R|)
- **Seed сплита:** 42
- **Язык текстов:** русский
- **|R| диапазон:** 2..9

### v2 (1054 записи) — обновление 2026-03-01

Датасет расширен до **1054 записей**.

- **Сплиты:** train=736, val=158, test=159 (70/15/15, стратификация по |R|)
- **Seed сплита:** 42

Результаты baseline-ов (секция 1) — на v1. Секция 2 (baseline snapshot) и DDQN итерация 4 (секция 6) — на v2.

### Adaptive full (сплиты `data/splits_adaptive/`)

Датасет с полями `adaptive.trajectory` для режима `AdaptiveRoutingEnv`. Полный сгенерированный пул — `data/tasks_set_adaptive_full.jsonl`; стратифицированные сплиты строятся `tools/split_adaptive_dataset.py`.

- **Файл (полный пул):** `data/tasks_set_adaptive_full.jsonl`
- **SHA-256:** `4fa22c63b15e1d2c71933c87ddd205c4d605a59de4791db7a0290bc7df04df15`
- **Сплиты:** train=609, val=131, test=131 (70/15/15, стратификация по |R|)
- **Seed сплита:** 42

---

## Adaptive dataset — Baseline Random (2026-04-05)

**Скрипт:** `python -m multiagent_dqn_routing.experiments.run_random_set`

**Протокол:** тот же стохастический reward, что в `configs/baseline_protocol.json` (поле `reward`), передаётся через `--reward_config_json` (`p_bad = 0.35`).

**Параметры запуска:** `seed=42`, `max_steps=9` (по умолчанию), `--dataset_path data/tasks_set_adaptive_full.jsonl` только для meta/SHA в JSON-snapshot.

**Артефакты (локально, каталог в `.gitignore`):** `artifacts/baseline_adaptive/random_{train,val,test}.json`

### Train / val / test (overall)

| Сплит | n | mean_f1 | mean_jaccard | exact_match | success_rate | avg_steps | avg_over | avg_under | mean_episode_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 609 | 0.5119 | 0.3758 | 0.0082 | 0.2562 | 5.4680 | 2.5813 | 1.9146 | 0.0977 |
| val | 131 | 0.5155 | 0.3842 | 0.0000 | 0.2824 | 5.4427 | 2.5420 | 1.8779 | 0.1641 |
| test | 131 | 0.5120 | 0.3799 | 0.0076 | 0.2977 | 5.4427 | 2.5573 | 1.8931 | 0.1565 |

### Test (n=131) — по bucket-ам |R|

| Bucket | n | mean_f1 | mean_jaccard | success_rate | avg_over | avg_under |
|---|---:|---:|---:|---:|---:|---:|
| A | 42 | 0.3585 | 0.2479 | 0.4762 | 4.0952 | 0.9048 |
| B | 62 | 0.5485 | 0.4018 | 0.2419 | 2.3871 | 1.9677 |
| C | 27 | 0.6668 | 0.5348 | 0.1481 | 0.5556 | 3.2593 |

Границы bucket-ов совпадают с `evaluator_set.py`: A — размер требуемого множества 2–3; B — 4–6; C — 7–9.

**Воспроизведение (все три сплита):**

```bash
REWARD='{"alpha":1.0,"beta":0.5,"gamma":1.0,"p_good":0.85,"p_bad":0.35}'
for SPLIT in train val test; do
  python -m multiagent_dqn_routing.experiments.run_random_set \
    --split_path "data/splits_adaptive/${SPLIT}.jsonl" \
    --dataset_path data/tasks_set_adaptive_full.jsonl \
    --seed 42 \
    --reward_config_json "$REWARD" \
    --json_out "artifacts/baseline_adaptive/random_${SPLIT}.json"
done
```

---

## Adaptive dataset — Baseline Rule-based (2026-04-05)

**Скрипт:** `python -m multiagent_dqn_routing.experiments.run_rule_set`

**Протокол:** тот же стохастический reward, что в `configs/baseline_protocol.json` (поле `reward`), через `--reward_config_json` (`p_bad = 0.35`).

**Логика роутера:** эвристика по ключевым маркерам в тексте запроса (`TRIGGERS` в `run_rule_set.py`); при недоборе до `min_len=2` — дозаполнение случайными уникальными агентами (фиксированный `seed+81`).

**Параметры запуска:** `seed=42`, `max_steps=9`, `--dataset_path data/tasks_set_adaptive_full.jsonl` для meta в JSON-snapshot.

**Артефакты (локально, `.gitignore`):** `artifacts/baseline_adaptive/rule_{train,val,test}.json`

### Train / val / test (overall)

| Сплит | n | mean_f1 | mean_jaccard | exact_match | success_rate | avg_steps | avg_over | avg_under | mean_episode_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 609 | 0.5137 | 0.3719 | 0.0312 | 0.0361 | 2.3218 | 0.4778 | 2.9573 | −1.4745 |
| val | 131 | 0.5473 | 0.4059 | 0.0458 | 0.0534 | 2.3053 | 0.3817 | 2.8550 | −1.2634 |
| test | 131 | 0.5284 | 0.3881 | 0.0382 | 0.0382 | 2.3664 | 0.4427 | 2.8550 | −1.2099 |

### Test (n=131) — по bucket-ам

| Bucket | n | mean_f1 | mean_jaccard | success_rate | avg_over | avg_under |
|---|---:|---:|---:|---:|---:|---:|
| A | 42 | 0.5032 | 0.3869 | 0.0952 | 0.9048 | 1.3571 |
| B | 62 | 0.5597 | 0.4123 | 0.0161 | 0.3065 | 2.8226 |
| C | 27 | 0.4955 | 0.3344 | 0.0000 | 0.0370 | 5.2593 |

Границы bucket-ов: A — 2–3; B — 4–6; C — 7–9 требуемых агентов (как в `evaluator_set.py`).

**Воспроизведение (все три сплита):**

```bash
REWARD='{"alpha":1.0,"beta":0.5,"gamma":1.0,"p_good":0.85,"p_bad":0.35}'
for SPLIT in train val test; do
  python -m multiagent_dqn_routing.experiments.run_rule_set \
    --split_path "data/splits_adaptive/${SPLIT}.jsonl" \
    --dataset_path data/tasks_set_adaptive_full.jsonl \
    --seed 42 \
    --reward_config_json "$REWARD" \
    --json_out "artifacts/baseline_adaptive/rule_${SPLIT}.json"
done
```

---

## 1. Baseline Snapshot v1 (2026-02-11)

**Протокол:** `tools/baseline_snapshot.py --config configs/baseline_protocol.json`

**Reward-параметры (baseline snapshot):**

| Параметр | Значение |
|---|---|
| alpha | 1.0 |
| beta | 0.5 |
| gamma | 1.0 |
| p_good | 0.85 |
| p_bad | 0.35 |

### Overall (test, n=49)

| Метод | mean_f1 | mean_jaccard | exact_match | success_rate | precision | recall | avg_steps | avg_over | avg_under | reward |
|---|---|---|---|---|---|---|---|---|---|---|
| Random | 0.517 | 0.384 | 0.000 | 0.286 | 0.521 | 0.611 | 5.71 | 2.71 | 1.86 | 0.10 |
| Rule-based | 0.483 | 0.342 | 0.020 | 0.020 | 0.765 | 0.368 | 2.20 | 0.47 | 3.12 | −1.62 |
| Supervised (TF-IDF+LogReg) | **0.880** | **0.802** | **0.286** | **0.898** | 0.826 | **0.967** | 5.69 | 0.96 | **0.12** | **3.80** |
| LLM-Router | *skipped* | — | — | — | — | — | — | — | — | — |

### Bucket A (|R| ∈ {2, 3})

| Метод | mean_f1 | mean_jaccard | avg_over | avg_under |
|---|---|---|---|---|
| Random | 0.297 | 0.190 | 4.14 | 1.29 |
| Rule-based | 0.493 | 0.375 | 0.86 | 1.43 |
| Supervised | **0.820** | **0.715** | 0.93 | **0.14** |

### Bucket B (|R| ∈ {4, 5, 6})

| Метод | mean_f1 | mean_jaccard | avg_over | avg_under |
|---|---|---|---|---|
| Random | 0.560 | 0.410 | 2.54 | 1.96 |
| Rule-based | 0.464 | 0.319 | 0.42 | 3.38 |
| Supervised | **0.884** | **0.803** | 1.12 | **0.15** |

### Bucket C (|R| ∈ {7, 8, 9})

| Метод | mean_f1 | mean_jaccard | avg_over | avg_under |
|---|---|---|---|---|
| Random | 0.732 | 0.608 | 1.00 | 2.44 |
| Rule-based | 0.521 | 0.356 | 0.00 | 5.00 |
| Supervised | **0.965** | **0.934** | 0.56 | **0.00** |

---

## 2. Baseline Snapshot v2 (2026-03-01)

**Протокол:** `tools/baseline_snapshot.py --config configs/baseline_protocol.json`

**Датасет:** v2 (test n=159)

**Reward-параметры (baseline snapshot):**

| Параметр | Значение |
|---|---|
| alpha | 1.0 |
| beta | 0.5 |
| gamma | 1.0 |
| p_good | 0.85 |
| p_bad | 0.35 |

### Overall (test, n=159)

| Метод | mean_f1 | mean_jaccard | exact_match | success_rate | precision | recall | avg_steps | avg_over | avg_under | reward |
|---|---|---|---|---|---|---|---|---|---|---|
| Random | 0.533 | 0.402 | 0.031 | 0.251 | 0.580 | 0.593 | 5.38 | 2.21 | 2.14 | 0.20 |
| Rule-based | 0.523 | 0.379 | 0.025 | 0.025 | 0.819 | 0.406 | 2.44 | 0.38 | 3.25 | −1.58 |
| Supervised (TF-IDF+LogReg) | **0.876** | **0.797** | **0.252** | **0.836** | **0.836** | **0.952** | **6.16** | **1.04** | **0.189** | **4.03** |
| LLM-Router | 0.854 | 0.773 | 0.333 | 0.352 | 0.928 | 0.804 | 4.46 | 0.26 | 1.11 | 2.43 |

### Bucket A (|R| ∈ {2, 3}, n=42)

| Метод | mean_f1 | mean_jaccard | avg_over | avg_under |
|---|---|---|---|---|
| Random | 0.359 | 0.245 | 3.52 | 1.07 |
| Rule-based | 0.503 | 0.383 | 0.90 | 1.36 |
| Supervised | **0.824** | **0.730** | **0.95** | **0.14** |
| LLM-Router | 0.863 | 0.810 | 0.29 | 0.38 |

### Bucket B (|R| ∈ {4, 5, 6}, n=66)

| Метод | mean_f1 | mean_jaccard | avg_over | avg_under |
|---|---|---|---|---|
| Random | 0.527 | 0.377 | 2.71 | 2.03 |
| Rule-based | 0.540 | 0.391 | 0.32 | 3.00 |
| Supervised | **0.854** | **0.756** | **1.33** | **0.30** |
| LLM-Router | 0.850 | 0.767 | 0.36 | 0.95 |

### Bucket C (|R| ∈ {7, 8, 9}, n=51)

| Метод | mean_f1 | mean_jaccard | avg_over | avg_under |
|---|---|---|---|---|
| Random | 0.682 | 0.563 | 0.49 | 3.18 |
| Rule-based | 0.517 | 0.359 | 0.04 | 5.14 |
| Supervised | **0.948** | **0.905** | **0.75** | **0.08** |
| LLM-Router | 0.850 | 0.751 | 0.12 | 1.92 |

---

## 3. DDQN — Итерация 1 (конфиг `ddqn_set_default.json`)

**Дата:** ~2026-02-12

**Изменения относительно baseline:** первый запуск DQN, reward-параметры совпадают с дефолтными.

| Параметр reward | Значение |
|---|---|
| alpha | 1.0 |
| beta | 0.5 |
| gamma | 1.0 |
| step_cost | 0.0 (нет) |
| p_good | 0.85 |
| p_bad | 0.30 |

| Параметр обучения | Значение |
|---|---|
| total_steps | 30 000 |
| hidden_sizes | [256, 256] |
| lr | 0.001 |
| discount | 0.99 |
| buffer_size | 20 000 |
| batch_size | 128 |
| epsilon | 1.0 → 0.05 |
| target_update_every | 1 000 |
| use_action_mask | false |

**Результат:** артефакт не сохранён (перезаписан итерацией 3). Наблюдалась проблема: агент выбирает почти всех агентов (avg_steps ≈ 9), не учится нажимать STOP. Ожидаемый штраф за лишнего агента `E[penalty] = p_bad × beta = 0.3 × 0.5 = 0.15` слишком мал по сравнению с терминальным штрафом `gamma = 1.0` за пропуск.

---

## 4. DDQN — Итерация 2 (конфиг `ddqn_set_beta1_step005.json`)

**Дата:** ~2026-02-12

**Изменения относительно итерации 1:**
- `beta`: 0.5 → **1.0** (удвоен штраф за лишнего агента)
- `step_cost`: 0.0 → **0.05** (добавлен штраф за каждый шаг)

| Параметр reward | Значение | Δ |
|---|---|---|
| alpha | 1.0 | — |
| beta | **1.0** | ×2 |
| gamma | 1.0 | — |
| step_cost | **0.05** | новый |
| p_good | 0.85 | — |
| p_bad | 0.30 | — |

**Результат:** артефакт не сохранён (перезаписан итерацией 3). Ожидаемый штраф за лишнего агента вырос до `0.3 × 1.0 + 0.05 = 0.35`, но агент по-прежнему предпочитал набирать больше агентов.

---

## 5. DDQN — Итерация 3 (конфиг `ddqn_set_beta1_step005_gamma2_nomask.json`)

**Дата:** 2026-02-13

**Изменения относительно итерации 2:**
- `gamma`: 1.0 → **2.0** (удвоен терминальный штраф за пропуск)

| Параметр reward | Значение | Δ |
|---|---|---|
| alpha | 1.0 | — |
| beta | 1.0 | — |
| gamma | **2.0** | ×2 |
| step_cost | 0.05 | — |
| p_good | 0.85 | — |
| p_bad | 0.30 | — |

Параметры обучения — без изменений (те же, что в итерации 1).

### Overall (test, n=49)

| Метрика | DDQN (iter 3) | Supervised |
|---|---|---|
| mean_f1 | 0.732 | **0.880** |
| mean_jaccard | 0.600 | **0.802** |
| exact_match | 0.020 | **0.286** |
| success_rate | 0.653 | **0.898** |
| precision | 0.638 | **0.826** |
| recall | **0.918** | 0.967 |
| avg_steps | 8.43 | 5.69 |
| avg_over | 2.43 | **0.96** |
| avg_under | 0.41 | **0.12** |
| reward | 2.22 | **3.80** |

### Bucket A (|R| ∈ {2, 3}, n=14)

| Метрика | DDQN | Supervised |
|---|---|---|
| mean_f1 | 0.565 | **0.820** |
| mean_jaccard | 0.405 | **0.715** |
| avg_over | 3.36 | **0.93** |
| avg_under | 0.21 | 0.14 |

### Bucket B (|R| ∈ {4, 5, 6}, n=26)

| Метрика | DDQN | Supervised |
|---|---|---|
| mean_f1 | 0.773 | **0.884** |
| mean_jaccard | 0.642 | **0.803** |
| avg_over | 2.50 | **1.12** |
| avg_under | 0.27 | **0.15** |

### Bucket C (|R| ∈ {7, 8, 9}, n=9)

| Метрика | DDQN | Supervised |
|---|---|---|
| mean_f1 | 0.873 | **0.965** |
| mean_jaccard | 0.781 | **0.934** |
| avg_over | 0.78 | 0.56 |
| avg_under | 1.11 | **0.00** |

### Общий вывод

DDQN выбирает в среднем 8.43 агента из 9 — не научился вовремя нажимать STOP. Увеличение `gamma` до 2.0 усилило перестраховку: агент ещё больше боится пропустить нужного агента и набирает почти всех. Recall высокий (0.918), но Precision низкий (0.638) из-за массового over-selection.

Supervised LogReg решает задачу за один шаг (multi-hot prediction) и не страдает от проблемы последовательного выбора, поэтому стабильно лидирует.

---

## 6. DDQN — Итерация 4 (конфиг `ddqn_set_beta1_step005_gamma2_actionmask.json`) ← текущий

**Дата:** 2026-03-01

**Датасет:** v2 (test n=159)

**Изменения относительно итерации 3:**
- `use_action_mask`: false → **true** (запрет повторного выбора агента)

| Параметр reward | Значение | Δ |
|---|---|---|
| alpha | 1.0 | — |
| beta | 1.0 | — |
| gamma | 2.0 | — |
| step_cost | 0.05 | — |
| p_good | 0.85 | — |
| p_bad | 0.30 | — |

| Параметр обучения | Значение |
|---|---|
| total_steps | 30 000 |
| hidden_sizes | [256, 256] |
| lr | 0.001 |
| discount | 0.99 |
| buffer_size | 100 000 |
| batch_size | 128 |
| epsilon | 1.0 → 0.05 |
| target_update_every | 500 |
| use_action_mask | **true** |

### Overall (test, n=159)

| Метрика | DDQN (iter 4) | Supervised |
|---|---|---|
| mean_f1 | 0.721 | **0.876** |
| mean_jaccard | 0.600 | **0.797** |
| exact_match | 0.101 | **0.252** |
| success_rate | 0.981 | 0.836 |
| precision | 0.601 | **0.836** |
| recall | **0.995** | 0.952 |
| avg_steps | 8.70 | **6.16** |
| avg_over | 3.42 | **1.04** |
| avg_under | **0.019** | 0.189 |
| reward | 3.47 | 4.03 |

### Bucket A (|R| ∈ {2, 3}, n=42)

| Метрика | DDQN | Supervised |
|---|---|---|
| mean_f1 | 0.461 | **0.824** |
| mean_jaccard | 0.302 | **0.730** |
| avg_over | 5.79 | **0.95** |
| avg_under | 0.024 | 0.14 |

### Bucket B (|R| ∈ {4, 5, 6}, n=66)

| Метрика | DDQN | Supervised |
|---|---|---|
| mean_f1 | 0.720 | **0.854** |
| mean_jaccard | 0.569 | **0.756** |
| avg_over | 3.76 | **1.33** |
| avg_under | 0.030 | 0.30 |

### Bucket C (|R| ∈ {7, 8, 9}, n=51)

| Метрика | DDQN | Supervised |
|---|---|---|
| mean_f1 | 0.938 | **0.948** |
| mean_jaccard | 0.887 | **0.905** |
| avg_over | 1.02 | **0.75** |
| avg_under | **0.00** | 0.08 |

### Общий вывод

С action masking DDQN не может выбирать одного агента дважды — это устраняет логические ошибки, но не решает проблему позднего STOP. avg_steps = 8.70 (почти все 9 агентов), success_rate высокий (0.981), recall почти идеальный (0.995), но precision низкий (0.601) из‑за over-selection. В bucket C DDQN достигает уровня Supervised (mean_f1 0.938 vs 0.939). Supervised по-прежнему лидирует по F1 и exact_match в целом.

*Примечание: reward-параметры DDQN (beta=1.0, gamma=2.0, step_cost=0.05) отличаются от baseline snapshot (beta=0.5, gamma=1.0), поэтому reward не сопоставим напрямую.*

---

## 7. DDQN — Итерация 5 (конфиг `ddqn_jaccard_step005.json`) ← planned

**Дата:** 2026-03-28

**Датасет:** v2 (test n=159)

**Изменения относительно итерации 4:**
- Полностью заменена reward-функция: вместо стохастической пошаговой (p_good/p_bad + gamma·missing) — детерминированный терминальный Jaccard-reward
- step_cost = 0.05 (фиксированный штраф за шаг, без стохастики)
- total_steps увеличен до 150 000 (sparse reward требует больше опыта)
- Остальные параметры: те же, что в итерации 4

**Мотивация изменения:**
В итерациях 1–4 агент выбирает ~8.7 из 9 агентов (avg_steps=8.70).
Причина: конфликт между страхом пропустить агента (gamma=2.0) и штрафом за лишнего (beta=1.0). Стохастика p_good/p_bad создаёт высокодисперсный сигнал, не позволяющий Q-сети уверенно оценить действие STOP. Терминальный Jaccard даёт чистый, однозначный сигнал, напрямую соответствующий метрике оценки.

**Параметры reward:**

| Параметр | Значение |
|---|---|
| reward_mode | jaccard |
| step_cost | 0.05 |
| p_good | — (не используется) |
| p_bad | — (не используется) |
| alpha | — (не используется) |
| beta | — (не используется) |
| gamma | — (не используется) |

**Результат:** [TO BE FILLED after experiment]

---

## 8. DDQN — Итерация 6 (конфиг `ddqn_jaccard_curriculum.json`) ← planned

**Дата:** 2026-03-29

**Датасет:** v2 (test n=159)

**Изменения относительно итерации 5:**
- Добавлен Curriculum Learning: обучение в 3 фазах по сложности |R|
- Фаза 1 (0–50k шагов):   только примеры с |R| ∈ {2,3} — агент учится останавливаться
- Фаза 2 (50k–100k шагов): примеры с |R| ∈ {2..6} — переход к средней сложности
- Фаза 3 (100k–150k шагов): все примеры — полная задача
- reward_mode и step_cost без изменений (jaccard, 0.05)

**Мотивация:**
Ablation sweep по step_cost (0.01/0.05/0.10/0.20, по 50k шагов каждый)
показал, что все варианты дают avg_steps ≥ 8.4 — агент застревает
в одном локальном оптимуме независимо от штрафа за шаг. Причина:
равномерный сэмплинг из всех примеров создаёт конфликтующие сигналы.
Curriculum Learning устраняет конфликт, обучая навык раннего STOP
сначала на простых примерах (|R|=2,3), где ранняя остановка всегда верна.

**Ожидаемый результат:**
- avg_steps ≤ 6.0 (против 8.7+ в итерациях 1–5)
- precision ≥ 0.70 (против 0.60 в итерации 5)
- mean_f1 ≥ 0.78

**Результат:** [TO BE FILLED after experiment]

---

## 9. DDQN — Итерация 7 (конфиг `ddqn_adaptive_jaccard.json`)

**Дата:** 2026-04-05

**Датасет:** Adaptive v1 (871 записей), сплиты `data/splits_adaptive/`

**Изменения относительно статического DQN:**
- Введён `AdaptiveRoutingEnv`
- Состояние заменено с `[text_vec | selected_mask | step_feature]` на `[text_vec | selected_mask | context_vec]`
- `context_vec` строится из конкатенации outputs уже выбранных агентов из `adaptive.trajectory`
- Reward фиксирован как Jaccard + `step_cost=0.05`

**Мотивация:**
Это первая постановка, где RL действительно имеет доступ к информации, возникающей только после действия. В отличие от Supervised multi-hot модели, policy может учитывать промежуточные outputs и менять следующий выбор агента.

### Validation (best checkpoint)

| Метрика | Значение |
|---|---:|
| n_items | 131 |
| mean_episode_reward | 0.1360 |
| success_rate | 0.7252 |
| exact_match_rate | 0.0611 |
| mean_jaccard | 0.5334 |
| mean_precision | 0.5576 |
| mean_recall | 0.9310 |
| mean_f1 | **0.6682** |
| avg_steps | 7.9466 |
| avg_overselection | 3.5038 |
| avg_underselection | 0.3359 |

### Test (greedy)

| Метрика | Значение |
|---|---:|
| n_items | 131 |
| mean_episode_reward | 0.1291 |
| success_rate | 0.6031 |
| exact_match_rate | 0.0534 |
| mean_jaccard | 0.5169 |
| mean_precision | 0.5514 |
| mean_recall | 0.8923 |
| mean_f1 | **0.6517** |
| avg_steps | 7.7557 |
| avg_overselection | 3.4809 |
| avg_underselection | 0.5038 |

**Динамика обучения:**
- Резкий рост до `val_mean_f1 = 0.6682` уже к шагу 14k
- После ~36k начинается плато около `val_mean_f1 = 0.6653`
- Агент остаётся recall-biased: покрытие высокое, но precision низкий из-за over-selection

**Научный вывод:**
Adaptive formulation сама по себе полезна: test `mean_f1 = 0.6517` оказался выше adaptive Random (`0.5120`) и adaptive Rule-based (`0.5284`). Однако policy всё ещё не научилась устойчивому раннему STOP; промежуточный context помогает недостаточно, потому что словарь encoder почти полностью обучен на query texts и слабо покрывает язык agent outputs.

---

## 10. DDQN — Итерация 8 (конфиг `ddqn_adaptive_jaccard_v2.json`)

**Дата:** 2026-04-05

**Датасет:** Adaptive v1 (871 записей), сплиты `data/splits_adaptive/`

**Изменения относительно итерации 7:**
- Encoder обучается на расширенном корпусе: `texts + all adaptive.trajectory[*].output`
- Добавлен helper `_build_adaptive_corpus()` в `train_ddqn_set.py`
- На train-сплите encoder видит `3533` документов (`609` query texts + `2924` agent outputs)

**Мотивация:**
В итерации 7 `context_vec` был частично вырожден: многие токены из outputs агентов (`"DataFrame"`, `"EBITDA"`, `"SQL-запрос"`) не встречались при обучении TF-IDF и уходили в OOV. Гипотеза: если обучить encoder на совместном корпусе запросов и outputs, `context_vec` станет информативным и policy начнёт раньше останавливаться.

### Validation (best checkpoint)

| Метрика | Значение |
|---|---:|
| n_items | 131 |
| mean_episode_reward | 0.0810 |
| success_rate | 1.0000 |
| exact_match_rate | 0.0534 |
| mean_jaccard | 0.5310 |
| mean_precision | 0.5310 |
| mean_recall | 1.0000 |
| mean_f1 | **0.6653** |
| avg_steps | 9.0000 |
| avg_overselection | 4.2214 |
| avg_underselection | 0.0000 |

### Test (greedy)

| Метрика | Значение |
|---|---:|
| n_items | 131 |
| mean_episode_reward | 0.0810 |
| success_rate | 1.0000 |
| exact_match_rate | 0.0534 |
| mean_jaccard | 0.5310 |
| mean_precision | 0.5310 |
| mean_recall | 1.0000 |
| mean_f1 | **0.6653** |
| avg_steps | 9.0000 |
| avg_overselection | 4.2214 |
| avg_underselection | 0.0000 |

**Динамика обучения:**
- До ~20k шагов модель догоняет уровень итерации 7
- С 22k шагов и до конца обучения метрики полностью замирают
- Policy коллапсирует в детерминированную стратегию: выбрать всех 9 агентов

**Научный вывод:**
Гипотеза про OOV оказалась недостаточной. Расширение корпуса действительно убрало словарный разрыв между query space и output space, но не улучшило policy-качество: test `mean_f1` вырос лишь с `0.6517` до `0.6653`, а цена роста — полная деградация в `avg_steps = 9.0` и `recall = 1.0`. Значит, основное ограничение adaptive DQN сейчас лежит не только в encoder, но и в самой оптимизационной динамике Jaccard-reward: policy по-прежнему предпочитает recall-maximizing стратегию "взять всех".

---

## Следующие шаги

- [x] Увеличить датасет (аугментация через LLM)
- [x] Попробовать `use_action_mask = true` (запретить повторный выбор)
- [x] AdaptiveRoutingEnv (итерация 7)
- [x] Расширенный encoder corpus (итерация 8)
- [ ] Глобальный Jaccard reward (итерация 5) ← IN PROGRESS
- [ ] Curriculum Learning (итерация 6) ← IN PROGRESS
- [ ] Устранить collapse adaptive policy в стратегию "выбрать всех"
- [ ] Заменить TF-IDF на dense embeddings (sentence-transformers)
- [ ] Рассмотреть one-shot multi-hot prediction вместо MDP
- [ ] Запустить LLM-Router baseline с реальным API-ключом
