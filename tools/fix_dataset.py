#!/usr/bin/env python3
"""Исправление датасета tasks_set.jsonl:
1) Удалить поле difficulty из каждой записи.
2) Привести required_agents к каноническому виду (sorted, unique, int, 0..8, len 2..9).
3) Валидация перед сохранением.
4) Бэкап исходного файла.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

SRC = Path("data/tasks_set.jsonl")
BAK = Path("data/tasks_set.jsonl.bak")

AGENT_MIN = 0
AGENT_MAX = 8
SET_SIZE_MIN = 2
SET_SIZE_MAX = 9


def main() -> None:
    if not SRC.exists():
        print(f"Файл не найден: {SRC}", file=sys.stderr)
        sys.exit(1)

    records: list[dict] = []
    difficulty_removed = 0
    ra_changed = 0

    # ── Чтение и трансформация ────────────────────────────────────────
    with open(SRC, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue

            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"ОШИБКА: строка {lineno}: невалидный JSON — {exc}", file=sys.stderr)
                sys.exit(1)

            # 1) Удалить difficulty
            if "difficulty" in rec:
                del rec["difficulty"]
                difficulty_removed += 1

            # 2) Канонический required_agents
            ra = rec.get("required_agents", [])
            canonical = sorted(set(int(x) for x in ra))
            if ra != canonical:
                ra_changed += 1
            rec["required_agents"] = canonical

            records.append(rec)

    # ── Валидация ─────────────────────────────────────────────────────
    errors: list[str] = []
    seen_ids: set[str] = set()

    for idx, rec in enumerate(records, start=1):
        # id
        rid = rec.get("id")
        if not rid or not isinstance(rid, str) or not rid.strip():
            errors.append(f"запись {idx}: id пустой или отсутствует")
        elif rid in seen_ids:
            errors.append(f"запись {idx}: дублирующийся id '{rid}'")
        else:
            seen_ids.add(rid)

        # text
        text = rec.get("text")
        if not text or not isinstance(text, str) or not text.strip():
            errors.append(f"запись {idx}: text пустой или отсутствует")

        # required_agents
        ra = rec.get("required_agents")
        if not isinstance(ra, list):
            errors.append(f"запись {idx}: required_agents не список")
        else:
            if not all(isinstance(x, int) for x in ra):
                errors.append(f"запись {idx}: required_agents содержит не-int")
            if not (SET_SIZE_MIN <= len(ra) <= SET_SIZE_MAX):
                errors.append(f"запись {idx}: len(required_agents)={len(ra)}, нужно {SET_SIZE_MIN}..{SET_SIZE_MAX}")
            if len(ra) != len(set(ra)):
                errors.append(f"запись {idx}: required_agents содержит дубликаты после каноникализации")
            out = [x for x in ra if x < AGENT_MIN or x > AGENT_MAX]
            if out:
                errors.append(f"запись {idx}: required_agents вне 0..8: {out}")

    if errors:
        print("ВАЛИДАЦИЯ НЕ ПРОЙДЕНА — файл НЕ перезаписан!\n", file=sys.stderr)
        for e in errors[:30]:
            print(f"  • {e}", file=sys.stderr)
        if len(errors) > 30:
            print(f"  … и ещё {len(errors) - 30}", file=sys.stderr)
        sys.exit(1)

    # ── Бэкап ─────────────────────────────────────────────────────────
    shutil.copy2(SRC, BAK)
    print(f"Бэкап сохранён: {BAK}")

    # ── Запись ────────────────────────────────────────────────────────
    with open(SRC, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── Отчёт ─────────────────────────────────────────────────────────
    print(f"\n=== Отчёт ===")
    print(f"Записей всего:                         {len(records)}")
    print(f"Записей с удалённым difficulty:         {difficulty_removed}")
    print(f"Записей с изменённым required_agents:   {ra_changed}")
    print(f"\nПервая строка результата:")
    first_line = json.dumps(records[0], ensure_ascii=False)
    print(f"  {first_line!r}")


if __name__ == "__main__":
    main()
