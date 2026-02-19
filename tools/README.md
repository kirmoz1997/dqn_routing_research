# Tools

Вспомогательные скрипты для работы с датасетом. Запускаются из **корня проекта**.

---

## tsv_to_jsonl.py

Конвертирует TSV-черновик в JSONL-датасет.

| Вход | Выход |
|------|-------|
| `data/tasks_set_draft.tsv` | `data/tasks_set.jsonl` |

Если входной файл не найден, он пропускается с предупреждением.

### Запуск

```bash
python tools/tsv_to_jsonl.py
```

### tasks_set_draft.tsv → tasks_set.jsonl

**Заголовок:**

```
id	required_agents	difficulty	eval_hint	text	notes
```

**Преобразования:**

- `required_agents` хранится как строка `"0,2,4"` и преобразуется в отсортированный список int `[0, 2, 4]`;
- `difficulty` — столбец TSV присутствует для совместимости, но **не включается** в выходной JSONL.

**Валидация:**

- все `id` уникальны и не пустые;
- `required_agents`: от 2 до 9 элементов, все уникальные, каждый в диапазоне 0..8;
- число колонок в каждой строке равно 6.

### Общее поведение

- пустые строки пропускаются;
- JSON пишется в UTF-8 без ASCII-экранирования (`ensure_ascii=False`);
- при ошибке выводится сообщение с номером строки и программа останавливается.

---

## dataset_stats_set.py

Валидация и статистика для `data/tasks_set.jsonl`.

### Запуск

```bash
python tools/dataset_stats_set.py [путь_к_файлу]
```

По умолчанию: `data/tasks_set.jsonl`.

### Что проверяет

- обязательные поля: `id`, `text`, `required_agents`;
- `required_agents`: список int, уникальный, диапазон 0..8, длина 2..9;
- уникальность `id` по всему файлу.

### Что выводит

- общее количество записей;
- распределение по `len(required_agents)` (2..9);
- частота каждого агента (0..8);
- топ-10 сигнатур `required_agents`;
- количество текстов < 20 символов.

При наличии ошибок выводит до 20 и завершается с `exit(1)`.

---

## fix_dataset.py

Однократный скрипт для каноникализации `data/tasks_set.jsonl`.

### Запуск

```bash
python tools/fix_dataset.py
```

### Что делает

1. Удаляет поле `difficulty` из каждой записи.
2. Приводит `required_agents` к каноническому виду (sorted, unique, int 0..8, len 2..9).
3. Валидация перед сохранением (если не проходит — файл не перезаписывается).
4. Создаёт бэкап: `data/tasks_set.jsonl.bak`.
5. Печатает отчёт: сколько записей обработано, у скольких удалён `difficulty`, у скольких изменён `required_agents`.

---

## split_jsonl_set.py

Разбиение `data/tasks_set.jsonl` на train / val / test сплиты.

### Запуск

```bash
python tools/split_jsonl_set.py [опции]
```

### Опции

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--in_path` | `data/tasks_set.jsonl` | Путь к исходному JSONL |
| `--out_dir` | `data/splits` | Директория для сплитов |
| `--seed` | `42` | Seed для перемешивания |
| `--train` | `0.70` | Доля train |
| `--val` | `0.15` | Доля val |
| `--test` | `0.15` | Доля test |
| `--stratify_by_set_size` | `1` | `1` = стратификация по `|R|`, `0` = простой split |

### Стратификация (по умолчанию)

- Группирует записи по `k = len(required_agents)` (k от 2 до 9).
- Внутри каждой группы — shuffle с seed, затем разбиение по долям.
- Гарантии: при `n >= 3` в группе — val >= 1, test >= 1; при `n == 2` — train=1, test=1; при `n == 1` — train=1 + предупреждение.
- Финальный shuffle каждого сплита для случайного порядка.

### Вывод

- Итоговые размеры train / val / test.
- Таблица counts по `k` (2..9) для каждого сплита.
- Предупреждения о покрытии агентов в train.

### Результат

```
data/splits/
  train.jsonl
  val.jsonl
  test.jsonl
```

---

## baseline_snapshot.py

Официальный протокол сравнения baseline-ов в одном запуске.

### Запуск

```bash
python tools/baseline_snapshot.py --config configs/baseline_protocol.json
```

### Что делает

- читает `configs/baseline_protocol.json`;
- запускает baseline-скрипты в порядке `order`;
- передаёт единые `seed`, `max_steps`, `reward`, `split_path`;
- пропускает LLM baseline, если `include=false` или не задан `api_key_env`;
- собирает результаты в:
  - `artifacts/baselines_summary.json`
  - `artifacts/baselines_summary.md`
