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

**Дата:** 2026-02-12

**Датасет:** v1, `data/tasks_set.jsonl` (323 записи; test n=49)

**Научная гипотеза:**
Мы проверяем гипотезу, что Double DQN может обучиться выбирать оптимальный набор агентов из 9, используя TF-IDF представление текста и бинарную маску выбранных агентов как состояние.

**Изменения относительно предыдущей итерации:**
- Первый RL-запуск вместо one-shot baseline-ов.
- Сохранена исходная стохастическая reward-схема без `step_cost`.
- Повторные выборы ещё не маскируются.

**Мотивация:**
Это была отправная точка исследования: сначала нужно было проверить, способен ли DDQN вообще выучить политику последовательного выбора в базовой постановке. Мы сознательно стартовали с минимального числа новых эвристик, чтобы увидеть естественную динамику агента.

**Параметры:**

| Параметр | Значение |
|---|---|
| reward_mode | stochastic |
| alpha / beta / gamma | 1.0 / 0.5 / 1.0 |
| step_cost | 0.0 |
| p_good / p_bad | 0.85 / 0.30 |
| total_steps | 30 000 |
| lr / discount | 0.001 / 0.99 |
| buffer_size / batch_size | 20 000 / 128 |
| epsilon | 1.0 → 0.05 |
| target_update_every | 1 000 |
| use_action_mask | false |

**Результат:** артефакт не сохранён отдельно, но по журналу обучения агент не научился нажимать `STOP`: `avg_steps ≈ 9`. Политика быстро ушла в стратегию выбора почти всех агентов.

**Вывод:** Агент не научился нажимать STOP: `avg_steps ≈ 9`. `E[penalty] = p_bad × beta = 0.3 × 0.5 = 0.15` слишком мал по сравнению с терминальным штрафом `gamma = 1.0` за пропуск агента. Агент предпочитает перестраховаться и выбирать всех.

---

## 4. DDQN — Итерация 2 (конфиг `ddqn_set_beta1_step005.json`)

**Дата:** 2026-02-12

**Датасет:** v1, `data/tasks_set.jsonl` (323 записи; test n=49)

**Научная гипотеза:**
Мы проверяем гипотезу, что увеличение beta (штраф за лишнего агента) и добавление step_cost создадут достаточное давление для раннего STOP.

**Изменения относительно предыдущей итерации:**
- `beta`: `0.5 → 1.0`.
- `step_cost`: `0.0 → 0.05`.
- Остальная архитектура и schedule сохранены без изменений.

**Мотивация:**
После итерации 1 стало ясно, что штраф за лишнего агента слишком слаб по сравнению со страхом недобора. Поэтому следующий шаг был простым reward shaping: усилить наказание за over-selection и сделать каждый дополнительный выбор ощутимо дорогим.

**Параметры:**

| Параметр | Значение |
|---|---|
| reward_mode | stochastic |
| alpha / beta / gamma | 1.0 / 1.0 / 1.0 |
| step_cost | 0.05 |
| p_good / p_bad | 0.85 / 0.30 |
| total_steps | 30 000 |
| lr / discount | 0.001 / 0.99 |
| buffer_size / batch_size | 20 000 / 128 |
| epsilon | 1.0 → 0.05 |
| target_update_every | 1 000 |
| use_action_mask | false |

**Результат:** отдельный артефакт не сохранён; по логу агент по-прежнему выбирал слишком много агентов и не демонстрировал устойчивого раннего `STOP`.

**Вывод:** Ожидаемый штраф за лишнего агента вырос до `0.35`, но агент по-прежнему предпочитает выбирать больше агентов. Дисбаланс между страхом пропустить (`gamma`) и штрафом за лишнего (`beta × p_bad`) сохраняется.

---

## 5. DDQN — Итерация 3 (конфиг `ddqn_set_beta1_step005_gamma2_nomask.json`)

**Дата:** 2026-02-13

**Датасет:** v1, `data/tasks_set.jsonl` (323 записи; test n=49)

**Научная гипотеза:**
Мы проверяем гипотезу, что удвоение терминального штрафа gamma (1.0 → 2.0) усилит сигнал о пропущенных агентах и заставит агента более тщательно выбирать набор.

**Изменения относительно предыдущей итерации:**
- `gamma`: `1.0 → 2.0`.
- Reward остаётся стохастическим.
- Маскирование действий всё ещё выключено.

**Мотивация:**
Итерация 2 показала, что плоское увеличение штрафа за лишний шаг не меняет поведение радикально. Следующей естественной гипотезой было усилить терминальный сигнал недобора и проверить, начнёт ли агент балансировать precision и recall более разумно.

**Параметры:**

| Параметр | Значение |
|---|---|
| reward_mode | stochastic |
| alpha / beta / gamma | 1.0 / 1.0 / 2.0 |
| step_cost | 0.05 |
| p_good / p_bad | 0.85 / 0.30 |
| total_steps | 30 000 |
| lr / discount | 0.001 / 0.99 |
| buffer_size / batch_size | 20 000 / 128 |
| epsilon | 1.0 → 0.05 |
| target_update_every | 1 000 |
| use_action_mask | false |

**Результат:** `mean_f1 = 0.732`, `mean_jaccard = 0.600`, `precision = 0.638`, `recall = 0.918`, `avg_steps = 8.43` на test `n=49`.

**Вывод:** Парадоксальный эффект: увеличение gamma усилило страх пропустить агента, что привело к ещё большему over-selection (`avg_steps = 8.43`). Recall вырос (`0.918`), но Precision упал (`0.638`). Это классический precision-recall trade-off при неправильно настроенной reward.

---

## 6. DDQN — Итерация 4 (конфиг `ddqn_set_beta1_step005_gamma2_actionmask.json`)

**Дата:** 2026-03-01

**Датасет:** v2, `data/tasks_set.jsonl` (1054 записи; test n=159)

**Научная гипотеза:**
Мы проверяем гипотезу, что запрет повторного выбора агента (action masking) устранит логические ошибки и улучшит качество выбора набора.

**Изменения относительно предыдущей итерации:**
- `use_action_mask`: `false → true`.
- Буфер увеличен до `100 000`.
- `target_update_every`: `1000 → 500`.

**Мотивация:**
После итерации 3 было важно убрать хотя бы технические артефакты поведения, не связанные с научной гипотезой о `STOP`. Маскирование действий должно было убрать дубли и показать, сколько проблемы связано именно с логикой среды, а сколько с формой reward.

**Параметры:**

| Параметр | Значение |
|---|---|
| reward_mode | stochastic |
| alpha / beta / gamma | 1.0 / 1.0 / 2.0 |
| step_cost | 0.05 |
| p_good / p_bad | 0.85 / 0.30 |
| total_steps | 30 000 |
| lr / discount | 0.001 / 0.99 |
| buffer_size / batch_size | 100 000 / 128 |
| epsilon | 1.0 → 0.05 |
| target_update_every | 500 |
| use_action_mask | true |

**Результат:** `mean_f1 = 0.721`, `mean_jaccard = 0.600`, `precision = 0.601`, `recall = 0.995`, `avg_steps = 8.70`, `exact_match = 0.101` на test `n=159`.

**Вывод:** Action masking устранил повторные выборы, но не решил STOP-проблему. `avg_steps = 8.70` — агент по-прежнему выбирает почти всех. В bucket C (`|R| = 7..9`) DDQN достигает уровня Supervised (`F1 = 0.938`), но Supervised стабильно лидирует по F1 и exact_match в целом.

---

## 7. DDQN — Итерация 5 (конфиг `ddqn_jaccard_step005.json`)

**Дата:** 2026-03-28

**Датасет:** v2, `data/tasks_set.jsonl` (1054 записи; test n=159)

**Научная гипотеза:**
Мы проверяем гипотезу, что замена стохастической пошаговой reward-функции на детерминированный терминальный Jaccard-reward устранит высокодисперсный обучающий сигнал и позволит Q-сети уверенно оценить действие STOP.

**Изменения относительно предыдущей итерации:**
- Reward полностью заменён на `reward_mode = "jaccard"`.
- Запущен ablation sweep по `step_cost ∈ {0.01, 0.05, 0.10, 0.20}`.
- Для sweep использованы короткие прогоны по `50 000` шагов на конфиг.

**Мотивация:**
Итерации 1–4 показали, что проблема выглядит не как нехватка одного коэффициента, а как следствие шумной и внутренне противоречивой reward-схемы. Поэтому следующим шагом было убрать стохастику и перейти к сигналу, который напрямую совпадает с целевой set-метрикой.

**Параметры:**

| Параметр | Значение |
|---|---|
| reward_mode | jaccard |
| step_cost | sweep: 0.01 / 0.05 / 0.10 / 0.20 |
| total_steps | 50 000 на конфиг |
| hidden_sizes | [256, 256] |
| lr / discount | 0.001 / 0.99 |
| buffer_size / batch_size | 100 000 / 128 |
| use_action_mask | true |

**Результат:** ablation sweep по `step_cost ∈ {0.01, 0.05, 0.10, 0.20}` показал, что все варианты дают `avg_steps ≥ 8.4`. Отдельные test-артефакты по каждому sweep-конфигу не были сохранены, но качественный итог одинаков: агент застревает в том же локальном оптимуме независимо от величины flat `step_cost`.

**Вывод:** Flat `step_cost` не является решением: форма штрафа важнее его абсолютного значения. Маргинальный прирост Jaccard от добавления агента (`~0.06`) всегда превышает flat `step_cost` (`0.05`). Это структурный локальный оптимум, а не параметрическая проблема.

---

## 8. DDQN — Итерация 6 (конфиг `ddqn_jaccard_curriculum.json`)

**Дата:** 2026-03-29

**Датасет:** v2, `data/tasks_set.jsonl` (1054 записи; test n=159)

**Научная гипотеза:**
Мы проверяем гипотезу, что обучение от простых примеров к сложным (`|R| = 2..3 → 4..6 → все`) позволит агенту сначала выучить навык раннего STOP на задачах где это очевидно выгодно, а затем перенести этот навык на сложные примеры.

**Изменения относительно предыдущей итерации:**
- Добавлен curriculum в 3 фазы по сложности `|R|`.
- Reward остаётся `jaccard` с `step_cost = 0.05`.
- Полный прогон увеличен до `150 000` шагов.

**Мотивация:**
После провала flat Jaccard стало понятно, что агент может усреднять конфликтующие сигналы от простых и сложных примеров. Curriculum learning должен был временно изолировать задачи, где ранний `STOP` явно выгоден, и затем перенести этот навык на полный датасет.

**Параметры:**

| Параметр | Значение |
|---|---|
| reward_mode | jaccard |
| step_cost | 0.05 |
| curriculum | 0–50k: `|R|≤3`; 50k–100k: `|R|≤6`; 100k–150k: all |
| total_steps | 150 000 |
| hidden_sizes | [256, 256] |
| lr / discount | 0.001 / 0.99 |
| buffer_size / batch_size | 100 000 / 128 |
| use_action_mask | true |

**Результат:** smoke test показал `avg_steps = 6.4` — лучший промежуточный результат до итерации 9. Однако полный прогон `150k` вернул политику к `avg_steps = 9.0`; отдельный точный test `mean_f1` для full-run не был сохранён как самостоятельный артефакт.

**Вывод:** Curriculum дал временный эффект, но не устойчивый. Catastrophic forgetting в фазе 3 вернул стратегию «выбрать всех». Проблема фундаментальна: нужно менять форму reward, а не только порядок обучения.

---

## 9. DDQN — Итерация 7 (конфиг `ddqn_adaptive_jaccard.json`)

**Дата:** 2026-04-05

**Датасет:** adaptive full, `data/tasks_set_adaptive_full.jsonl` (871 записей; test n=131)

**Научная гипотеза:**
Мы проверяем гипотезу, что расширение состояния контекстным вектором из промежуточных результатов агентов (AdaptiveRoutingEnv) даст агенту информацию недоступную Supervised-классификатору и позволит принимать лучшие решения о продолжении выбора.

**Изменения относительно предыдущей итерации:**
- Введён `AdaptiveRoutingEnv`.
- Состояние заменено на `[text_vec | selected_mask | context_vec]`.
- `context_vec` строится из `adaptive.trajectory[*].output`.
- Reward остаётся `jaccard` + `step_cost = 0.05`.

**Мотивация:**
Это первая действительно sequential постановка, где RL получает наблюдение, возникающее только после действия. Если эта гипотеза верна, adaptive-среда должна дать DDQN преимущество перед one-shot supervised-router не за счёт одной лишь reward-функции, а за счёт richer state.

**Параметры:**

| Параметр | Значение |
|---|---|
| env_mode | adaptive |
| reward_mode | jaccard |
| step_cost | 0.05 |
| total_steps | 150 000 |
| hidden_sizes | [256, 256] |
| lr / discount | 0.001 / 0.99 |
| buffer_size / batch_size | 100 000 / 128 |
| use_action_mask | true |

**Результат:** best val `mean_f1 = 0.6682`, test `mean_f1 = 0.6517`, test `mean_jaccard = 0.5169`, `precision = 0.5514`, `recall = 0.8923`, `avg_steps = 7.7557`.

**Вывод:** Плато с шага `~36k` на уровне `F1 ≈ 0.665`. `context_vec` оказался практически нулевым по полезной информации: TF-IDF encoder обучен в основном на текстах запросов и не знает словарь outputs агентов. Агент обучается игнорировать `context_vec` как неинформативный сигнал.

---

## 10. DDQN — Итерация 8 (конфиг `ddqn_adaptive_jaccard_v2.json`)

**Дата:** 2026-04-05

**Датасет:** adaptive full, `data/tasks_set_adaptive_full.jsonl` (871 записей; test n=131)

**Научная гипотеза:**
Мы проверяем гипотезу, что расширение обучающего корпуса TF-IDF encoder на outputs агентов (`texts + trajectory outputs`) устранит OOV-проблему и сделает `context_vec` информативным.

**Изменения относительно предыдущей итерации:**
- Encoder обучается на совместном корпусе `texts + adaptive.trajectory[*].output`.
- Добавлен helper `_build_adaptive_corpus()`.
- Корпус train-энкодера вырос с `609` до `3533` документов.

**Мотивация:**
Итерация 7 показала, что adaptive-state теоретически интересен, но представление контекста слишком бедное. Поэтому следующая гипотеза была узко технической: сначала устранить OOV-разрыв между языком запросов и языком промежуточных agent outputs, а уже потом судить о полезности `context_vec`.

**Параметры:**

| Параметр | Значение |
|---|---|
| env_mode | adaptive |
| reward_mode | jaccard |
| step_cost | 0.05 |
| encoder_corpus | `609` query texts + `2924` outputs |
| total_steps | 150 000 |
| hidden_sizes | [256, 256] |
| lr / discount | 0.001 / 0.99 |
| buffer_size / batch_size | 100 000 / 128 |
| use_action_mask | true |

**Результат:** best val `mean_f1 = 0.6653`, test `mean_f1 = 0.6653`, `mean_jaccard = 0.5310`, `precision = 0.5310`, `recall = 1.0000`, `avg_steps = 9.0000`.

**Вывод:** Гипотеза частично подтверждена технически (encoder corpus вырос с `609` до `3533` документов), но не дала улучшения качества. Агент коллапсировал в стратегию «выбрать всех» (`avg_steps = 9.0`). Основное ограничение — не качество encoder, а структурный локальный оптимум flat Jaccard reward.

---

## 11. DDQN — Итерация 9 probe (конфиг `ddqn_log_lambda010.json`)

**Дата:** 2026-04-06

**Датасет:** v2 static, `data/splits/` (1054 записи; test n=159)

**Научная гипотеза:**
Мы проверяем гипотезу, что нарастающий логарифмический штраф (адаптация из Puppeteer, Dang et al. NeurIPS 2025) сломает структурный локальный оптимум flat reward: первые шаги дёшевы (агент не боится начинать), поздние шаги дороги (добавление лишних агентов становится убыточным).

**Изменения относительно предыдущей итерации:**
- Введён `reward_mode = "jaccard_log"`.
- `lambda_eff = 0.10`.
- `discount`: `0.99 → 0.95`, `lr`: `1e-3 → 1e-4`, `target_update_every`: `500 → 200`.
- Probe ограничен `50 000` шагов.

**Мотивация:**
Итерации 5–8 показали, что flat penalties не ломают select-all коллапс. Поэтому следующая гипотеза уже меняла не уровень штрафа, а его геометрию во времени: сделать ранние шаги дешёвыми, а поздние ощутимо дорогими.

**Параметры:**

| Параметр | Значение |
|---|---|
| reward_mode | jaccard_log |
| lambda_eff | 0.10 |
| total_steps | 50 000 |
| epsilon_decay_steps | 50 000 |
| lr / discount | 0.0001 / 0.95 |
| buffer_size / batch_size | 100 000 / 128 |
| target_update_every | 200 |
| use_action_mask | true |

**Результат:** `val_mean_f1 = 0.856`, `avg_steps = 4.44`, `precision = 0.940`. STOP-проблема решена впервые за 8 итераций; модель продолжала расти до шага `50k` без признаков плато.

**Вывод:** Логарифмический штраф сломал select-all коллапс. Однако `epsilon_decay_steps = total_steps = 50k` означало, что в probe-прогоне epsilon быстро упал до `0.05`; позже это оказалось критическим фактором успеха.

---

## 12. DDQN — Итерация 9 full v1 (конфиг `ddqn_log_lambda010_full.json`)

**Дата:** 2026-04-06

**Датасет:** v2 static, `data/splits/` (1054 записи; test n=159)

**Научная гипотеза:**
Полный прогон probe-конфигурации на 150k шагов должен дать дальнейшее улучшение F1 за счёт более длительного обучения.

**Изменения относительно предыдущей итерации:**
- Тот же `reward_mode = "jaccard_log"` и `lambda_eff = 0.10`.
- `total_steps`: `50 000 → 150 000`.
- Ошибочно оставлен `epsilon_decay_steps = total_steps = 150 000`.

**Мотивация:**
После сильного probe-результата естественным было ожидать, что более длинное обучение улучшит политику ещё сильнее. Этот запуск проверял простую гипотезу «больше шагов = лучше», не меняя остальную конфигурацию.

**Параметры:**

| Параметр | Значение |
|---|---|
| reward_mode | jaccard_log |
| lambda_eff | 0.10 |
| total_steps | 150 000 |
| epsilon_decay_steps | 150 000 |
| lr / discount | 0.0001 / 0.95 |
| buffer_size / batch_size | 100 000 / 128 |
| target_update_every | 200 |
| use_action_mask | true |

**Результат:** плато с шага `~62k` на уровне `F1 ≈ 0.710`, `avg_steps = 9.0`. Select-all коллапс вернулся.

**Вывод:** Диагноз: `epsilon_decay_steps = total_steps = 150k`. Epsilon падал слишком медленно, replay buffer заполнился случайными длинными эпизодами, и Q-сеть обучилась на «мусорных» траекториях. Решение: фиксировать `epsilon_decay_steps = 50000` независимо от `total_steps`.

---

## 13. DDQN — Итерация 9 full v2 (конфиг `ddqn_log_lambda010_full_v2.json`)

**Дата:** 2026-04-07

**Датасет:** v2 static, `data/splits/` (1054 записи; test n=159)

**Научная гипотеза:**
Мы проверяем гипотезу, что фиксация `epsilon_decay_steps = 50000` (независимо от `total_steps = 150000`) позволит агенту сначала исследовать с правильным давлением, а затем 100k шагов обучаться на качественных эпизодах.

**Изменения относительно предыдущей итерации:**
- Сохранены `reward_mode = "jaccard_log"` и `lambda_eff = 0.10`.
- `epsilon_decay_steps`: `150 000 → 50 000`.
- Остальные гиперпараметры не менялись.

**Мотивация:**
Провал full v1 показал, что проблема находится не в самой идее log-reward, а в exploration schedule. Поэтому следующий эксперимент был точечной диагностической правкой: оставить reward неизменным и изменить только скорость decay epsilon.

**Параметры:**

| Параметр | Значение |
|---|---|
| reward_mode | jaccard_log |
| lambda_eff | 0.10 |
| total_steps | 150 000 |
| epsilon_decay_steps | 50 000 |
| lr / discount | 0.0001 / 0.95 |
| buffer_size / batch_size | 100 000 / 128 |
| target_update_every | 200 |
| use_action_mask | true |

**Результат:** `val_mean_f1 = 0.897`, `test_mean_f1 = 0.857`, `avg_steps = 4.50`, `precision = 0.941`, `cost_ratio = 0.847`. Конфиг превзошёл Supervised по precision и exact_match, но уступил лучшему варианту по F1.

**Вывод:** Epsilon schedule критичен для log-reward. При быстром decay агент быстро начинает exploit правильной политики. Разрыв `val/test = 0.897 / 0.857` указывает на небольшое переобучение под val-распределение.

---

## 14. DDQN — Итерация 9 ablation λ=0.05 (конфиг `ddqn_log_lambda005_full.json`)

**Дата:** 2026-04-07

**Датасет:** v2 static, `data/splits/` (1054 записи; test n=159)

**Научная гипотеза:**
Мы проверяем гипотезу, что меньший `λ = 0.05` (менее агрессивный штраф за шаг) улучшит recall (агент будет делать больше шагов и покрывать больше нужных агентов), сохраняя при этом улучшенный precision log-reward.

**Изменения относительно предыдущей итерации:**
- `lambda_eff`: `0.10 → 0.05`.
- Сохранён фиксированный `epsilon_decay_steps = 50000`.
- Остальной log-reward стек не менялся.

**Мотивация:**
После full v2 стало понятно, что log-reward работает, но `λ = 0.10` может быть слишком агрессивным и рано останавливать политику. Следующий шаг был ablation по силе лог-штрафа, чтобы найти лучший balance между precision и recall.

**Параметры:**

| Параметр | Значение |
|---|---|
| reward_mode | jaccard_log |
| lambda_eff | 0.05 |
| total_steps | 150 000 |
| epsilon_decay_steps | 50 000 |
| lr / discount | 0.0001 / 0.95 |
| buffer_size / batch_size | 100 000 / 128 |
| target_update_every | 200 |
| use_action_mask | true |

**Результат:** best val `mean_f1 = 0.9015`, test `mean_f1 = 0.8877`, `mean_jaccard = 0.8236`, `precision = 0.9267`, `recall = 0.8652`, `avg_steps = 4.9434`, `cost_ratio = 0.9313`, `exact_match = 0.4403`.

**Вывод:** `λ = 0.05` — лучшая конфигурация исследования. Умеренный логарифмический штраф создаёт правильный баланс: `precision = 0.927` (лучше Supervised `0.836`) при `recall = 0.865` (ниже Supervised `0.952`). DDQN впервые превзошёл Supervised по test F1 (`0.888 vs 0.876`).

---

## 15. DDQN — Итерация 10 (конфиг `ddqn_adaptive_log_lambda005.json`)

**Дата:** 2026-04-07

**Датасет:** adaptive full, `data/tasks_set_adaptive_full.jsonl` (871 записей; test n=131)

**Научная гипотеза:**
Мы проверяем гипотезу, что комбинация log-reward (решившего STOP-проблему на static) с AdaptiveRoutingEnv (дающей доступ к промежуточным результатам агентов) даст синергетический эффект: агент сможет использовать контекст для более точного решения о продолжении выбора.

**Изменения относительно предыдущей итерации:**
- Объединены `env_mode = "adaptive"` и `reward_mode = "jaccard_log"`.
- Использован лучший static-лог-штраф `lambda_eff = 0.05`.
- Для adaptive-режима добавлены отдельные train/val/test пути и `epsilon_decay_steps = 50000`.

**Мотивация:**
После прорыва на static-датасете логично было проверить, переносится ли этот механизм в richer adaptive-state. Это был главный тест гипотезы о синергии между правильной формой reward и доступом к промежуточному контексту агентов.

**Параметры:**

| Параметр | Значение |
|---|---|
| env_mode | adaptive |
| reward_mode | jaccard_log |
| lambda_eff | 0.05 |
| total_steps | 150 000 |
| epsilon_decay_steps | 50 000 |
| lr / discount | 0.0001 / 0.95 |
| buffer_size / batch_size | 100 000 / 128 |
| target_update_every | 200 |
| use_action_mask | true |

**Результат:** `val_mean_f1 = 0.668`, test `mean_f1 = 0.668`, `mean_jaccard = 0.532`, `precision = 0.568`, `recall = 0.938`, `avg_steps = 8.05`, `cost_ratio = 1.68`. В шагах `56k–66k` наблюдался catastrophic forgetting: F1 кратковременно падал до `~0.16`.

**Вывод:** Синергии не возникло. Log-reward не помог в adaptive-среде: select-all коллапс вернулся почти полностью. Расширенное state space adaptive env (`4009` против `2009` измерений) требует более мощной Q-сети или лучшего state encoder; TF-IDF `context_vec` остаётся слишком слабым.

---

## Следующие шаги

- [x] Увеличить датасет (аугментация через LLM)
- [x] Попробовать `use_action_mask = true` (запретить повторный выбор)
- [x] AdaptiveRoutingEnv (итерация 7)
- [x] Расширенный encoder corpus (итерация 8)
- [x] Probe 50k для log-reward `lambda_eff = 0.10`
- [x] Full 150k для log-reward `lambda_eff = 0.10`
- [x] Исправить `epsilon_decay_steps` для full-run (`50000` вместо `total_steps`)
- [x] Full 150k ablation log-reward `lambda_eff = 0.05` (`ddqn_log_lambda005_full.json`)
- [x] Проверить adaptive + log-reward (`ddqn_adaptive_log_lambda005.json`)
- [ ] Ablation sweep `lambda_eff in {0.03, 0.15}` при необходимости
- [ ] Dense embeddings для adaptive env
- [ ] Более мощная сеть / иной RL-алгоритм для adaptive env (например, PPO)
- [ ] Сравнить с one-shot adaptive policy
- [ ] Запустить LLM-Router baseline с реальным API-ключом
