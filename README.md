# Multi-Agent Set Routing with DQN

Исследование задачи **маршрутизации пользовательского запроса** через сеть из 9 специализированных агентов. Для каждого запроса требуется выбрать **оптимальный набор агентов** (от 2 до 9), минимизируя недобор и перебор.

Подробности — в [Research Plan](Research_Plan_MultiAgent_Set_Routing_v1.0.0.md). Полный журнал запусков — в [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Сводная таблица результатов

### Статический датасет v2 (test n=159)

| Метод | F1 | Jaccard | Precision | Recall | avg_steps | cost_ratio |
|---|---|---|---|---|---|---|
| Random | 0.533 | 0.402 | 0.580 | 0.593 | 5.38 | — |
| Rule-based | 0.523 | 0.379 | 0.819 | 0.406 | 2.44 | — |
| LLM-Router | 0.854 | 0.773 | 0.928 | 0.804 | 4.46 | — |
| TF-IDF + LogReg | 0.876 | 0.797 | 0.836 | 0.952 | 6.16 | 1.15 |
| DDQN flat reward (iter 4) | 0.721 | 0.600 | 0.601 | 0.995 | 8.70 | 1.62 |
| **DDQN log λ=0.05 (iter 9)** | **0.888** | **0.824** | **0.927** | **0.865** | **4.94** | **0.931** |

### Адаптивный датасет (test n=131)

| Метод | F1 | Jaccard | Precision | Recall | avg_steps |
|---|---|---|---|---|---|
| Random | 0.512 | 0.380 | — | — | 5.44 |
| Rule-based | 0.528 | 0.388 | — | — | 2.37 |
| TF-IDF + LogReg | 0.885 | 0.813 | 0.879 | 0.920 | 5.13 |
| DDQN adaptive flat (iter 7) | 0.652 | 0.517 | 0.551 | 0.892 | 7.76 |
| DDQN adaptive log λ=0.05 (iter 10) | 0.668 | 0.532 | 0.568 | 0.938 | 8.05 |

## Структура проекта

```text
multiagent_dqn_routing/
├── configs/                     # Все конфиги baseline/DDQN/smoke; полный список ниже
├── data/                        # Основной и adaptive датасеты + stratified splits
├── artifacts/                   # Зафиксированные baseline/DDQN артефакты
├── tools/                       # Скрипты подготовки данных и snapshot-протокол
├── src/multiagent_dqn_routing/
│   ├── data/                    # Загрузка JSONL-датасетов
│   ├── envs/                    # Static и adaptive routing environments
│   ├── eval/                    # Set-metrics, bucket evaluation, cost_ratio
│   ├── experiments/             # Baselines, DDQN training, snapshot utils
│   ├── rl/                      # Replay buffer, Q-network, state encoder, DDQN agent
│   ├── sim/                     # Reward models: stochastic / jaccard / jaccard_log
│   └── agents.py                # Описание 9 специализированных агентов
├── EXPERIMENT_LOG.md
├── Research_Plan_MultiAgent_Set_Routing_v1.0.0.md
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Быстрый старт

### Установка

> **Важно:** работайте в изолированном виртуальном окружении, чтобы зависимости проекта не конфликтовали с системными пакетами.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\\Scripts\\activate      # Windows

pip install -e '.[dev]'

# Для lock-окружения:
pip install -r requirements.txt
pip install -e .
```

Зависимости объявлены в `pyproject.toml`:
- **core** — `numpy`, `scipy`, `scikit-learn`, `joblib`, `torch`
- **dev** — `matplotlib`, `pandas`, `tqdm`

> **Примечание:** `artifacts/` и `cache/` генерируются скриптами. В репозитории сохранены только те артефакты, которые нужны для отчётности и воспроизводимости.

### Подготовка данных

```bash
# Конвертация TSV → JSONL (если редактировали черновик)
python tools/tsv_to_jsonl.py

# Валидация и статистика
python tools/dataset_stats_set.py

# Стратифицированный split основного датасета
python tools/split_jsonl_set.py

# Стратифицированный split adaptive-датасета
python tools/split_adaptive_dataset.py
```

Для генерации `data/tasks_set_adaptive_full.jsonl` не храните секреты в коде:

```bash
cp .env.example .env
source .env
python tools/generate_adaptive_dataset.py
```

## Воспроизведение лучшего результата (DDQN λ=0.05)

```bash
# 1. Установка
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

# 2. Подготовка данных (если splits ещё не созданы)
python tools/split_jsonl_set.py

# 3. Запуск лучшей конфигурации
python -m multiagent_dqn_routing.experiments.train_ddqn_set \
    --config configs/ddqn_log_lambda005_full.json

# 4. Ожидаемый результат на test:
#   mean_f1 ≈ 0.888, precision ≈ 0.927, avg_steps ≈ 4.94
```

### Запуск baseline-ов

```bash
# Random baseline
python -m multiagent_dqn_routing.experiments.run_random_set

# Rule-based baseline
python -m multiagent_dqn_routing.experiments.run_rule_set

# Supervised baseline
python -m multiagent_dqn_routing.experiments.run_supervised_set

# LLM-Router (требует API-ключ)
export LLM_API_KEY=sk-...
python -m multiagent_dqn_routing.experiments.run_llm_set
```

Adaptive baseline-прогоны используют `data/splits_adaptive/*.jsonl` и reward из `configs/baseline_protocol.json`; агрегированные результаты зафиксированы в [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

### Обучение Double DQN

```bash
# Лучшая конфигурация на static v2
python -m multiagent_dqn_routing.experiments.train_ddqn_set \
    --config configs/ddqn_log_lambda005_full.json

# Исторические stochastic reward эксперименты (итерации 1–4)
python -m multiagent_dqn_routing.experiments.train_ddqn_set \
    --config configs/ddqn_set_beta1_step005_gamma2_actionmask.json

# Jaccard reward / curriculum
python -m multiagent_dqn_routing.experiments.train_ddqn_set \
    --config configs/ddqn_jaccard_step005.json
python -m multiagent_dqn_routing.experiments.train_ddqn_set \
    --config configs/ddqn_jaccard_curriculum.json

# Adaptive env
python -m multiagent_dqn_routing.experiments.train_ddqn_set \
    --config configs/ddqn_adaptive_jaccard.json
python -m multiagent_dqn_routing.experiments.train_ddqn_set \
    --config configs/ddqn_adaptive_jaccard_v2.json
python -m multiagent_dqn_routing.experiments.train_ddqn_set \
    --config configs/ddqn_adaptive_log_lambda005.json

# Smoke-check любого конфига
python -m multiagent_dqn_routing.experiments.train_ddqn_set \
    --config configs/ddqn_log_lambda005_full.json --smoke_test
```

Артефакты DDQN сохраняются в `artifacts/ddqn/` (`model.pt`, `encoder.joblib`, `metrics_val_best.json`, `metrics_test.json`, `config_used.json`). Исторические smoke-конфиги сохранены в `configs/` для точной воспроизводимости, но в обычной работе достаточно флага `--smoke_test`.

### Baseline snapshot (official)

Для воспроизводимого сравнения baseline-ов перед этапом DQN используйте:

```bash
python tools/baseline_snapshot.py --config configs/baseline_protocol.json
```

Скрипт создаёт:
- `artifacts/baselines_summary.json`
- `artifacts/baselines_summary.md`
- `artifacts/baseline_<name>.json`

### Reward config used in baselines

Базовые set-routing скрипты (`run_random_set.py`, `run_rule_set.py`, `run_supervised_set.py`, `run_llm_set.py`) по умолчанию используют:

- `alpha = 1.0`
- `beta = 0.5`
- `gamma = 1.0`
- `p_good = 0.85`
- `p_bad = 0.30`

Для official snapshot reward берётся из `configs/baseline_protocol.json`, где сейчас зафиксировано `p_bad = 0.35`.

### Reproducibility for supervised artifact

Supervised baseline сохраняет артефакт в `artifacts/supervised_tfidf_ovr_logreg.joblib`.

- В `requirements.txt` зафиксирована рабочая версия: `scikit-learn==1.6.1`.
- В `.joblib` сохраняются `sklearn_version`, `seed`, `split_paths`, `reward_params`, `created_at_utc`.
- При загрузке артефакта выполняется проверка совместимости по `major.minor` версии sklearn.

Рекомендуемый порядок после смены окружения:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
python -m multiagent_dqn_routing.experiments.run_supervised_set --strict_artifact_load_check
```

## Конфиги экспериментов

| Конфиг | Описание | Итерация |
|---|---|---|
| `baseline_protocol.json` | Официальный baseline snapshot | — |
| `ddqn_set_default.json` | Базовый DDQN со стохастическим reward | 1 |
| `ddqn_set_beta1_step005.json` | `beta=1.0`, `step_cost=0.05` | 2 |
| `ddqn_set_beta1_step005_gamma2_nomask.json` | `gamma=2.0`, без action masking | 3 |
| `ddqn_set_beta1_step005_gamma2_actionmask.json` | + action masking | 4 |
| `ddqn_jaccard_step001.json` | Ablation flat Jaccard, `step_cost=0.01` | 5 |
| `ddqn_jaccard_step005.json` | Flat Jaccard, `step_cost=0.05` | 5 |
| `ddqn_jaccard_step010.json` | Ablation flat Jaccard, `step_cost=0.10` | 5 |
| `ddqn_jaccard_step020.json` | Ablation flat Jaccard, `step_cost=0.20` | 5 |
| `ddqn_jaccard_curriculum.json` | Curriculum learning поверх Jaccard reward | 6 |
| `ddqn_jaccard_curriculum_smoke.json` | Короткий smoke-конфиг для curriculum | 6 |
| `ddqn_adaptive_jaccard.json` | AdaptiveRoutingEnv + Jaccard reward | 7 |
| `ddqn_adaptive_jaccard_smoke.json` | Короткий smoke-конфиг для adaptive env | 7 |
| `ddqn_adaptive_jaccard_v2.json` | Adaptive env + расширенный TF-IDF корпус | 8 |
| `ddqn_adaptive_jaccard_v2_smoke.json` | Smoke-конфиг для iteration 8 | 8 |
| `ddqn_log_lambda010_smoke.json` | Smoke-конфиг для log-reward | 9 |
| `ddqn_log_lambda010.json` | Log-reward `λ=0.10` probe, 50k | 9 |
| `ddqn_log_lambda010_full.json` | Log-reward `λ=0.10` full v1, plateau | 9 |
| `ddqn_log_lambda010_full_v2.json` | Log-reward `λ=0.10` + `epsilon_decay_steps=50000` | 9 |
| `ddqn_log_lambda005.json` | Ablation `λ=0.05`, короткий probe | 9 |
| `ddqn_log_lambda005_full.json` | **Лучшая конфигурация: log-reward `λ=0.05`** | **9** |
| `ddqn_log_lambda003.json` | Ablation `λ=0.03` (не запускался) | — |
| `ddqn_log_lambda015.json` | Ablation `λ=0.15` (не запускался) | — |
| `ddqn_adaptive_log_lambda005.json` | Adaptive + log-reward `λ=0.05` | 10 |

## Ключевые научные выводы

1. **STOP-проблема структурна, а не параметрична.** Flat `step_cost` создаёт локальный оптимум «выбрать всех»: маргинальный прирост Jaccard (`~0.06`) систематически превышает любой фиксированный штраф за шаг.
2. **Логарифмический штраф ломает локальный оптимум.** DDQN с `λ=0.05` превзошёл TF-IDF+LogReg по test F1 (`0.888 vs 0.876`), Jaccard (`0.824 vs 0.797`) и exact_match (`0.440 vs 0.252`), выбирая при этом меньше агентов.
3. **Epsilon schedule критичен для sparse reward.** `epsilon_decay_steps = 50000` оказался обязательным условием успеха; при `epsilon_decay_steps = total_steps` воспроизводится plateau и возврат к select-all коллапсу.
4. **Adaptive-постановка требует более плотного state representation.** TF-IDF `context_vec` недостаточен для использования промежуточных outputs агентов; следующие кандидаты — dense embeddings и/или PPO.

## Агенты

| ID | Название | Специализация |
|----|----------|--------------|
| 0 | Code Agent (Python) | Пишет и исправляет Python-код |
| 1 | SQL Agent | SQL-запросы (SELECT/JOIN/GROUP BY) |
| 2 | Data Analysis Agent (Pandas) | Анализ данных с Pandas |
| 3 | Math Formula Solver | Математические формулы и вычисления |
| 4 | Structured Extraction Agent (JSON) | Извлечение данных в JSON |
| 5 | Summarization & Formatting Agent | Суммаризация и форматирование |
| 6 | Requirements / ТЗ Agent | ТЗ и требования (FR/NFR) |
| 7 | Rewrite / Style Constraints Agent | Рерайт и стилистика |
| 8 | Finance / Numeric Computation Agent | Финансовые расчёты |

## Метрики

Все эксперименты выводят единый набор метрик (overall + по 3 bucket-ам):

**Основные:**
- `mean_episode_reward` — средняя суммарная награда за эпизод; в baseline/static-экспериментах это reward из `RewardSetModel`, в adaptive/Jaccard-режиме — детерминированный reward текущей среды
- `success_rate` — доля задач с полным покрытием (missing = 0)
- `exact_match_rate` — доля задач с точным совпадением наборов
- `mean_jaccard` — среднее Jaccard similarity
- `mean_precision`, `mean_recall`, `mean_f1` — классические set-метрики

**Bucket-ы по `|R|`:**
- **A** (|R| ∈ {2,3}) — малые наборы
- **B** (|R| ∈ {4,5,6}) — средние наборы
- **C** (|R| ∈ {7,8,9}) — большие наборы

## Формат датасета

Каждая запись в `data/tasks_set.jsonl`:

```json
{
  "id": "ex_0001",
  "required_agents": [0, 2, 4],
  "eval_hint": "код + JSON извлечение",
  "text": "Напиши скрипт, который парсит логи...",
  "notes": "Code для парсинга + Structured Extraction для JSON."
}
```

- `required_agents` — отсортированный список уникальных int (0..8), длина 2..9
- `text` — текст запроса на русском языке
- `eval_hint` — подсказка для интерпретации результата
- `notes` — пояснение к разметке (не используется при обучении)

### Adaptive-формат

Для adaptive-экспериментов используется расширенный JSONL `data/tasks_set_adaptive_full.jsonl`.
Он сохраняет те же поля, что и базовый датасет, и добавляет `adaptive.trajectory`:

```json
{
  "id": "gen_9_0001",
  "required_agents": [1, 2, 5],
  "text": "Подготовь SQL-выгрузку и краткую сводку по cohort retention...",
  "adaptive": {
    "trajectory": [
      {
        "agent_id": 1,
        "agent_name": "SQL Agent",
        "output": "Сформирован SQL-запрос ...",
        "remaining_gap": "Нужен анализ retention и финальная сводка",
        "is_last": false
      }
    ]
  }
}
```

- `adaptive.trajectory` — упорядоченная цепочка промежуточных результатов выбранных агентов
- `output` — текст, который попадает в `context_vec` adaptive-среды
- `remaining_gap` — какая часть задачи ещё не закрыта после данного шага
- `is_last` — индикатор последнего шага в сгенерированной траектории
