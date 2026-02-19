#!/usr/bin/env python3
"""Статистика датасета tasks_set.jsonl.

Проверяет корректность каждой записи и выводит сводную статистику:
  - общее количество записей
  - распределение по длине required_agents (2..9)
  - частота каждого агента (0..8)
  - топ-10 сигнатур required_agents
  - количество пустых / слишком коротких text (< 20 символов)

Использует только stdlib: json, collections, argparse, sys.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

# ── helpers ──────────────────────────────────────────────────────────

MIN_TEXT_LEN = 20
AGENT_RANGE = range(0, 9)          # 0..8 включительно
SET_SIZE_MIN = 2
SET_SIZE_MAX = 9


def _validate_record(record: dict, lineno: int) -> list[str]:
    """Возвращает список ошибок для одной записи."""
    errors: list[str] = []

    # --- обязательные поля ---
    for field in ("id", "text", "required_agents"):
        if field not in record:
            errors.append(f"строка {lineno}: отсутствует поле '{field}'")

    if "required_agents" in record:
        ra = record["required_agents"]

        # тип — список
        if not isinstance(ra, list):
            errors.append(
                f"строка {lineno}: required_agents должен быть списком, "
                f"получен {type(ra).__name__}"
            )
        else:
            # все элементы — int
            if not all(isinstance(x, int) for x in ra):
                errors.append(
                    f"строка {lineno}: required_agents содержит не-int элементы"
                )

            # длина 2..9
            if not (SET_SIZE_MIN <= len(ra) <= SET_SIZE_MAX):
                errors.append(
                    f"строка {lineno}: len(required_agents)={len(ra)}, "
                    f"ожидается {SET_SIZE_MIN}..{SET_SIZE_MAX}"
                )

            # уникальность
            if len(ra) != len(set(ra)):
                errors.append(
                    f"строка {lineno}: required_agents содержит дубликаты"
                )

            # диапазон 0..8
            out_of_range = [x for x in ra if x not in AGENT_RANGE]
            if out_of_range:
                errors.append(
                    f"строка {lineno}: required_agents вне диапазона 0..8: "
                    f"{out_of_range}"
                )

    return errors


# ── main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Статистика и валидация tasks_set.jsonl",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="data/tasks_set.jsonl",
        help="Путь к JSONL-файлу (по умолчанию data/tasks_set.jsonl)",
    )
    args = parser.parse_args()

    errors: list[str] = []
    seen_ids: set[str] = set()

    records: list[dict] = []

    # ── чтение и валидация ───────────────────────────────────────────
    try:
        with open(args.path, encoding="utf-8") as fh:
            for lineno, raw_line in enumerate(fh, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                # парсинг JSON
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    errors.append(f"строка {lineno}: невалидный JSON — {exc}")
                    continue

                # валидация полей
                errors.extend(_validate_record(record, lineno))

                # уникальность id
                rec_id = record.get("id")
                if rec_id is not None:
                    if rec_id in seen_ids:
                        errors.append(
                            f"строка {lineno}: дублирующийся id '{rec_id}'"
                        )
                    seen_ids.add(rec_id)

                records.append(record)

    except FileNotFoundError:
        print(f"Файл не найден: {args.path}", file=sys.stderr)
        sys.exit(1)

    # ── вывод ошибок (до 20) и выход ────────────────────────────────
    if errors:
        print("=== ОШИБКИ ВАЛИДАЦИИ ===\n")
        for err in errors[:20]:
            print(f"  • {err}")
        if len(errors) > 20:
            print(f"\n  … и ещё {len(errors) - 20} ошибок")
        print(f"\nВсего ошибок: {len(errors)}")
        sys.exit(1)

    # ── статистика ───────────────────────────────────────────────────
    n_items = len(records)

    # распределение по длине required_agents
    len_counter: Counter[int] = Counter()
    # частота каждого агента
    agent_counter: Counter[int] = Counter()
    # сигнатуры
    sig_counter: Counter[str] = Counter()
    # короткие тексты
    short_text_count = 0

    for rec in records:
        ra = rec.get("required_agents", [])
        sorted_ra = sorted(ra)

        len_counter[len(sorted_ra)] += 1

        for a in sorted_ra:
            agent_counter[a] += 1

        sig = ",".join(str(x) for x in sorted_ra)
        sig_counter[sig] += 1

        text = rec.get("text", "")
        if len(text) < MIN_TEXT_LEN:
            short_text_count += 1

    # ── вывод ────────────────────────────────────────────────────────
    print(f"Файл: {args.path}")
    print(f"n_items: {n_items}\n")

    print("── Распределение по len(required_agents) ──")
    for size in range(SET_SIZE_MIN, SET_SIZE_MAX + 1):
        count = len_counter.get(size, 0)
        bar = "█" * count
        print(f"  {size}: {count:>4}  {bar}")
    print()

    print("── Частота агентов (0..8) ──")
    for agent_id in AGENT_RANGE:
        count = agent_counter.get(agent_id, 0)
        bar = "█" * count
        print(f"  agent {agent_id}: {count:>4}  {bar}")
    print()

    print("── Топ-10 сигнатур required_agents ──")
    for rank, (sig, count) in enumerate(sig_counter.most_common(10), start=1):
        print(f"  {rank:>2}. [{sig}]  × {count}")
    print()

    print(f"Текстов < {MIN_TEXT_LEN} символов: {short_text_count}")


if __name__ == "__main__":
    main()
