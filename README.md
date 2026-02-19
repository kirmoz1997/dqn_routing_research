# Multi-Agent Set Routing with DQN

Исследование задачи **маршрутизации пользовательского запроса** через сеть из 9 специализированных агентов. Для каждого запроса требуется выбрать **оптимальный набор агентов** (от 2 до 9), минимизируя недобор и перебор.

Подробности — в [Research Plan](Research_Plan_MultiAgent_Set_Routing_v1.0.0.md).

## Структура проекта

```
multiagent_dqn_routing/
├── data/
│   ├── tasks_set.jsonl          # Основной датасет (323 записи, JSONL)
│   ├── tasks_set_draft.tsv      # Черновик для ручного редактирования
│   └── splits/
│       ├── train.jsonl           # Стратифицированный train-сплит
│       ├── val.jsonl             # Стратифицированный val-сплит
│       └── test.jsonl            # Стратифицированный test-сплит
├── src/multiagent_dqn_routing/
│   ├── agents.py                # Описание 9 агентов (id, name, description)
│   ├── data/
│   │   └── dataset.py           # Загрузка JSONL-датасетов
│   ├── envs/
│   │   └── set_routing_env.py   # MDP-среда для set routing (без gymnasium)
│   ├── eval/
│   │   └── evaluator_set.py     # Evaluator для set routing (метрики + buckets)
│   ├── experiments/
│   │   ├── train_ddqn_set.py    # Double DQN: обучение + eval + артефакты
│   │   ├── run_random_set.py    # Baseline 1: Random set routing
│   │   ├── run_rule_set.py      # Baseline 2: Rule-based set routing
│   │   ├── run_supervised_set.py # Baseline 3: TF-IDF + OVR LogReg
│   │   ├── run_llm_set.py       # Baseline 4: LLM-Router (API + prompt)
│   │   └── snapshot_utils.py    # Общие утилиты snapshot-протокола
│   ├── rl/
│   │   ├── replay_buffer.py     # Uniform replay buffer
│   │   ├── q_network.py         # MLP Q-network (PyTorch)
│   │   ├── state_encoder.py     # TF-IDF state encoder
│   │   └── ddqn_agent.py        # Double DQN agent
│   └── sim/
│       └── reward_set.py        # Reward model для set routing (alpha/beta/gamma)
├── configs/
│   ├── baseline_protocol.json   # Конфиг official baseline snapshot
│   └── ddqn_set_default.json    # Конфиг обучения Double DQN
├── tools/
│   ├── tsv_to_jsonl.py          # TSV → JSONL конвертер
│   ├── dataset_stats_set.py     # Статистика и валидация датасета
│   ├── fix_dataset.py           # Каноникализация (удаление difficulty, сортировка)
│   ├── split_jsonl_set.py       # Стратифицированный split на train/val/test
│   ├── baseline_snapshot.py     # Запуск и агрегация baseline snapshot
│   └── README.md                # Документация по утилитам
├── Research_Plan_MultiAgent_Set_Routing_v1.0.0.md
├── pyproject.toml
├── requirements.txt
└── README.md                    # ← этот файл
```

## Быстрый старт

### Установка

> **Важно:** работайте в изолированном виртуальном окружении, чтобы
> зависимости проекта не конфликтовали с системными/conda-пакетами.

```bash
# 1. Создайте и активируйте venv (один раз)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 2. Установите проект в editable-режиме со всеми зависимостями
pip install -e '.[dev]'

# ── Альтернатива: точный lock ──
# Если нужна 100% воспроизводимая среда (CI, ревью курсовой):
pip install -r requirements.txt
pip install -e .
```

Зависимости объявлены в `pyproject.toml`:
- **core** — `numpy`, `scipy`, `scikit-learn`, `joblib`, `torch`
- **dev** (extras) — `matplotlib`, `pandas`, `tqdm`

> **Примечание по Git:** Директории `artifacts/` (обученные модели) и `cache/` (ответы LLM-роутера) генерируются скриптами и исключены из системы контроля версий через `.gitignore`. Для воспроизведения запустите соответствующие скрипты обучения/baseline.

### Подготовка данных

```bash
# Конвертация TSV → JSONL (если редактировали черновик)
python tools/tsv_to_jsonl.py

# Валидация и статистика
python tools/dataset_stats_set.py

# Стратифицированный split
python tools/split_jsonl_set.py
```

### Запуск baseline-ов

```bash
# Random baseline (на test-сплите)
python -m multiagent_dqn_routing.experiments.run_random_set

# Rule-based baseline (на test-сплите)
python -m multiagent_dqn_routing.experiments.run_rule_set

# Supervised baseline (train/val/test сплиты, threshold sweep)
python -m multiagent_dqn_routing.experiments.run_supervised_set

# LLM-Router (требует API-ключ)
export LLM_API_KEY=sk-...
python -m multiagent_dqn_routing.experiments.run_llm_set
# или с указанием модели / base_url:
python -m multiagent_dqn_routing.experiments.run_llm_set --model gpt-4o-mini --base_url https://api.openai.com/v1
```

### Обучение Double DQN

```bash
# Полный запуск (с конфигом)
python -m multiagent_dqn_routing.experiments.train_ddqn_set \
    --config configs/ddqn_set_default.json

# Быстрая проверка (smoke test, ~2000 шагов)
python -m multiagent_dqn_routing.experiments.train_ddqn_set \
    --config configs/ddqn_set_default.json --smoke_test

# Артефакты сохраняются в artifacts/ddqn/:
#   model.pt, encoder.joblib, metrics_val_best.json,
#   metrics_test.json, config_used.json
```

### Reward config used in baselines

Базовые set-routing скрипты (`run_random_set.py`, `run_rule_set.py`,
`run_supervised_set.py`, `run_llm_set.py`) по умолчанию используют:

- `alpha = 1.0`
- `beta = 0.5`
- `gamma = 1.0`
- `p_good = 0.85`
- `p_bad = 0.30` (согласовано с `reward_set.py`)

Для **official snapshot** reward берётся из `configs/baseline_protocol.json`
(поле `reward`) и передаётся baseline-скриптам через `--reward_config_json`.
Сейчас в конфиге snapshot установлено `p_bad = 0.35`.

### Reproducibility for supervised artifact

Supervised baseline сохраняет артефакт в `artifacts/supervised_tfidf_ovr_logreg.joblib`.

- В `requirements.txt` зафиксирована рабочая версия: `scikit-learn==1.6.1`.
- Внутрь `.joblib` сохраняются метаданные: `sklearn_version`, `seed`, `split_paths`,
  `reward_params`, `created_at_utc`.
- При загрузке артефакта выполняется проверка совместимости по `major.minor` версии sklearn:
  - по умолчанию — **warning**;
  - при запуске с `--strict_artifact_load_check` — **fail-fast**.

Важно: `.joblib` может быть несовместим между версиями sklearn.  
Если меняли окружение (Python/sklearn) — пересоберите артефакт.

Рекомендуемый порядок после смены окружения:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
python -m multiagent_dqn_routing.experiments.run_supervised_set --strict_artifact_load_check
```

### Baseline snapshot (official)

Для воспроизводимого сравнения всех baseline-ов перед этапом DQN используйте протокол:

```bash
python tools/baseline_snapshot.py --config configs/baseline_protocol.json
```

Скрипт создаёт артефакты:

- `artifacts/baselines_summary.json` — машиночитаемый итоговый snapshot
- `artifacts/baselines_summary.md` — человекочитаемый отчёт
- `artifacts/baseline_<name>.json` — промежуточные результаты отдельных baseline-ов

Отключить LLM baseline можно в `configs/baseline_protocol.json`:

- установить `"llm": { "include": false, ... }`, или
- оставить `include=true`, но не задавать переменную окружения `api_key_env` (тогда LLM будет помечен как skipped).

Важно: в snapshot все baseline-ы принудительно получают одинаковые параметры
из конфига (`seed`, `max_steps`, `reward`, `test_split_path`), поэтому сравнение
между ними воспроизводимо.

Официальным snapshot перед DQN считается запуск этого протокола без ручного копипаста результатов.

### LLM-Router: детали

- **Промпт**: recall-biased (приоритет на полноту), версия фиксирована константой `PROMPT_VERSION`.
- **Кэш**: JSONL-файл `cache/llm_router_cache.jsonl`. Каждая запись содержит `id`, `pred`, `raw`, `model`, `prompt_version`.
  - При загрузке кэша фильтруются только записи, совпадающие по `model` и `prompt_version` — при смене модели или версии промпта старый кэш автоматически игнорируется.
- **Fallback**: при невалидном ответе LLM — keyword fallback (гарантирует ≥ 2 агента).
- **Логи**: в начале печатаются `PROMPT_VERSION`, cache hits / misses; в конце — API stats (calls, hits, fallbacks).

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
- `mean_episode_reward` — средняя суммарная награда
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
