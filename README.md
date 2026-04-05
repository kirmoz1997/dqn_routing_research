# Multi-Agent Set Routing with DQN

Исследование задачи **маршрутизации пользовательского запроса** через сеть из 9 специализированных агентов. Для каждого запроса требуется выбрать **оптимальный набор агентов** (от 2 до 9), минимизируя недобор и перебор.

Подробности — в [Research Plan](Research_Plan_MultiAgent_Set_Routing_v1.0.0.md). Результаты экспериментов — в [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Структура проекта

```
multiagent_dqn_routing/
├── data/
│   ├── tasks_set.jsonl          # Основной датасет (1054 записи, JSONL)
│   ├── tasks_set_draft.tsv      # Черновик для ручного редактирования
│   ├── splits/
│   │   ├── train.jsonl           # Стратифицированный train-сплит
│   │   ├── val.jsonl             # Стратифицированный val-сплит
│   │   └── test.jsonl            # Стратифицированный test-сплит
│   ├── splits_adaptive/          # train/val/test для adaptive (стратификация по |R|)
│   └── tasks_set_adaptive_full.jsonl  # полный adaptive-пул (после генерации)
├── src/multiagent_dqn_routing/
│   ├── agents.py                # Описание 9 агентов (id, name, description)
│   ├── data/
│   │   └── dataset.py           # Загрузка JSONL-датасетов
│   ├── envs/
│   │   ├── set_routing_env.py   # MDP-среда для static set routing
│   │   └── adaptive_routing_env.py  # Adaptive env: [text_vec | selected_mask | context_vec]
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
│       └── reward_set.py        # Reward models: RewardSetModel (stochastic) + RewardSetJaccard (terminal Jaccard)
├── configs/
│   ├── baseline_protocol.json   # Конфиг official baseline snapshot
│   ├── ddqn_set_default.json    # Конфиг обучения Double DQN (базовый)
│   ├── ddqn_set_beta1_step005.json
│   ├── ddqn_set_beta1_step005_gamma2_nomask.json
│   ├── ddqn_set_beta1_step005_gamma2_actionmask.json  # Stochastic reward: beta=1, gamma=2, action mask
│   ├── ddqn_adaptive_jaccard.json  # Adaptive env + context from trajectory outputs
│   ├── ddqn_adaptive_jaccard_smoke.json
│   ├── ddqn_adaptive_jaccard_v2.json  # Adaptive env + extended TF-IDF corpus
│   ├── ddqn_adaptive_jaccard_v2_smoke.json
│   ├── ddqn_jaccard_step005.json   # Jaccard reward: step_cost=0.05, 150k steps
│   ├── ddqn_jaccard_step001.json   # Ablation: step_cost=0.01
│   ├── ddqn_jaccard_step010.json   # Ablation: step_cost=0.10
│   └── ddqn_jaccard_step020.json   # Ablation: step_cost=0.20
├── tools/
│   ├── tsv_to_jsonl.py          # TSV → JSONL конвертер
│   ├── dataset_stats_set.py     # Статистика и валидация датасета
│   ├── fix_dataset.py           # Каноникализация (удаление difficulty, сортировка)
│   ├── split_jsonl_set.py       # Стратифицированный split на train/val/test
│   ├── generate_adaptive_dataset.py  # Генерация adaptive.trajectory через LLM
│   ├── split_adaptive_dataset.py     # Стратифицированный split adaptive-датасета
│   ├── baseline_snapshot.py     # Запуск и агрегация baseline snapshot
│   └── README.md                # Документация по утилитам
├── Research_Plan_MultiAgent_Set_Routing_v1.0.0.md
├── EXPERIMENT_LOG.md            # Журнал экспериментов (baseline, DDQN итерации)
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

# Adaptive dataset split (отдельно, не затрагивает data/splits/)
python tools/split_adaptive_dataset.py
```

Для генерации `data/tasks_set_adaptive_full.jsonl` не храните секреты в коде:

```bash
# один раз:
cp .env.example .env

# затем заполните .env и загрузите переменные в shell
source .env

python tools/generate_adaptive_dataset.py
```

Лучший практический подход здесь:
- секрет (`API key`) хранить только в env,
- несекретные параметры (`base_url`, `model`) задавать через env или CLI-флаги,
- не коммитить реальные ключи и не передавать их в командной строке.

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

**Random baseline на adaptive датасете** (`data/splits_adaptive/*.jsonl`, reward как в `configs/baseline_protocol.json`):

```bash
REWARD='{"alpha":1.0,"beta":0.5,"gamma":1.0,"p_good":0.85,"p_bad":0.35}'
python -m multiagent_dqn_routing.experiments.run_random_set \
  --split_path data/splits_adaptive/test.jsonl \
  --dataset_path data/tasks_set_adaptive_full.jsonl \
  --seed 42 \
  --reward_config_json "$REWARD"
```

Итоги полного прогона (train/val/test) и метрики — в [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) (секции *Adaptive dataset — Baseline Random* и *Baseline Rule-based*).

**Rule-based baseline на adaptive датасете** (те же reward и сплиты):

```bash
REWARD='{"alpha":1.0,"beta":0.5,"gamma":1.0,"p_good":0.85,"p_bad":0.35}'
python -m multiagent_dqn_routing.experiments.run_rule_set \
  --split_path data/splits_adaptive/test.jsonl \
  --dataset_path data/tasks_set_adaptive_full.jsonl \
  --seed 42 \
  --reward_config_json "$REWARD"
```

Полные adaptive-прогоны для `Random` и `Rule-based`, а также их сравнение по bucket-ам, зафиксированы в [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

### Обучение Double DQN

```bash
# Stochastic reward (итерации 1–4): action masking + beta=1, gamma=2
python -m multiagent_dqn_routing.experiments.train_ddqn_set \
    --config configs/ddqn_set_beta1_step005_gamma2_actionmask.json

# Jaccard reward (итерация 5+): терминальный Jaccard + step_cost
python -m multiagent_dqn_routing.experiments.train_ddqn_set \
    --config configs/ddqn_jaccard_step005.json

# Adaptive env: использует data/splits_adaptive/*.jsonl и sequential context
python -m multiagent_dqn_routing.experiments.train_ddqn_set \
    --config configs/ddqn_adaptive_jaccard.json

# Adaptive env + расширенный TF-IDF корпус
python -m multiagent_dqn_routing.experiments.train_ddqn_set \
    --config configs/ddqn_adaptive_jaccard_v2.json

# Базовый конфиг (без action mask)
python -m multiagent_dqn_routing.experiments.train_ddqn_set \
    --config configs/ddqn_set_default.json

# Быстрая проверка (smoke test, ~2000 шагов)
python -m multiagent_dqn_routing.experiments.train_ddqn_set \
    --config configs/ddqn_set_beta1_step005_gamma2_actionmask.json --smoke_test
python -m multiagent_dqn_routing.experiments.train_ddqn_set \
    --config configs/ddqn_jaccard_step005.json --smoke_test
python -m multiagent_dqn_routing.experiments.train_ddqn_set \
    --config configs/ddqn_adaptive_jaccard_v2_smoke.json --smoke_test

# Артефакты сохраняются в artifacts/ddqn/:
#   model.pt, encoder.joblib, metrics_val_best.json,
#   metrics_test.json, config_used.json
```

**Режимы reward:**
- **Stochastic** (`reward_mode: "stochastic"`, по умолчанию) — пошаговый стохастический reward (alpha/beta/p_good/p_bad) + терминальный штраф gamma. Конфиги: `ddqn_set_*.json`.
- **Jaccard** (`reward_mode: "jaccard"`) — фиксированный step_cost за каждый шаг + терминальный Jaccard `|S∩R|/|S∪R|`. Конфиги: `ddqn_jaccard_*.json`. Подробнее — в [Research Plan, §6.3](Research_Plan_MultiAgent_Set_Routing_v1.0.0.md).

**Режимы окружения (`env_mode`):**
- **`static`** — исходный `SetRoutingEnv`: состояние строится как `[text_vec | selected_mask | step_feature]`.
- **`adaptive`** — `AdaptiveRoutingEnv`: состояние строится как `[text_vec | selected_mask | context_vec]`, где `context_vec` — TF-IDF по конкатенации выходов уже выбранных агентов из `adaptive.trajectory`. Для этого режима train/val/test по умолчанию читаются из `data/splits_adaptive/`, а reward принудительно переключается на Jaccard-схему из Research Plan §6.3.
- В конфиге `ddqn_adaptive_jaccard_v2.json` encoder обучается не только на `text`, но и на всех `adaptive.trajectory[*].output` из train-сплита. Это расширяет словарь для `context_vec`, но по текущему эксперименту не устраняет коллапс в стратегию “выбрать всех”.

Результаты прогонов — в [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

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
