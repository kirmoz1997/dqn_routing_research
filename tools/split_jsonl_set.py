#!/usr/bin/env python3
"""Разбиение tasks_set.jsonl на train / val / test.

Читает JSONL, перемешивает детерминированно (seed), режет по долям
и записывает три файла в выходную директорию.
После разбиения проверяет покрытие агентов 0..8 в train-сплите.

Поддерживает стратифицированный split по |R| = len(required_agents)
(флаг --stratify_by_set_size, по умолчанию включён).

Использует только stdlib: json, argparse, os, random, sys, collections.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

AGENT_RANGE = range(0, 9)  # 0..8
MIN_AGENT_OCCURRENCES_TRAIN = 3
SET_SIZE_MIN = 2
SET_SIZE_MAX = 9


# ── I/O helpers ──────────────────────────────────────────────────────

def _read_jsonl(path: str) -> list[dict]:
    """Читает JSONL-файл и возвращает список объектов."""
    items: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                items.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                print(
                    f"Ошибка JSON в строке {lineno}: {exc}",
                    file=sys.stderr,
                )
                sys.exit(1)
    return items


def _write_jsonl(items: list[dict], path: str) -> None:
    """Записывает список объектов в JSONL (UTF-8, без ASCII-экранирования)."""
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


# ── coverage check ───────────────────────────────────────────────────

def _check_agent_coverage(train_items: list[dict]) -> None:
    """Предупреждает, если какой-то агент встречается < 3 раз в train."""
    counter: Counter[int] = Counter()
    for rec in train_items:
        for a in rec.get("required_agents", []):
            counter[a] += 1

    warnings: list[str] = []
    for agent_id in AGENT_RANGE:
        count = counter.get(agent_id, 0)
        if count < MIN_AGENT_OCCURRENCES_TRAIN:
            warnings.append(
                f"  ⚠ agent {agent_id}: {count} вхождений в train "
                f"(минимум {MIN_AGENT_OCCURRENCES_TRAIN})"
            )

    if warnings:
        print("\n── Предупреждения по покрытию агентов в train ──")
        for w in warnings:
            print(w)
        print()


# ── split helpers ────────────────────────────────────────────────────

def _simple_split(
    items: list[dict],
    rng: random.Random,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Простой shuffle + split с гарантией непустых val/test (если n >= 3)."""
    rng.shuffle(items)
    n = len(items)

    n_val = max(1, round(n * val_ratio))
    n_test = max(1, round(n * (1.0 - train_ratio - val_ratio)))
    n_train = n - n_val - n_test

    if n_train < 1:
        print(
            f"Невозможно выделить хотя бы 1 запись в train при n={n}, "
            f"val={n_val}, test={n_test}.",
            file=sys.stderr,
        )
        sys.exit(1)

    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def _split_group(
    group: list[dict],
    rng: random.Random,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Разбивает одну группу (по k=|R|) на train/val/test.

    Гарантии:
    - n >= 3: val >= 1, test >= 1, train >= 1
    - n == 2: train = 1, test = 1, val = 0
    - n == 1: train = 1, val = 0, test = 0 (с предупреждением)
    """
    rng.shuffle(group)
    n = len(group)

    if n == 1:
        return group[:], [], []
    if n == 2:
        return group[:1], [], group[1:]

    # n >= 3 — стандартное разбиение по долям с гарантиями
    n_val = max(1, round(n * val_ratio))
    n_test = max(1, round(n * (1.0 - train_ratio - val_ratio)))
    n_train = n - n_val - n_test

    # подстраховка: если train «съели» — отдаём из val
    if n_train < 1:
        n_train = 1
        # пересчитаем val/test из остатка
        rest = n - 1
        n_test = max(1, round(rest * 0.5))
        n_val = rest - n_test

    return group[:n_train], group[n_train:n_train + n_val], group[n_train + n_val:]


def _stratified_split(
    items: list[dict],
    rng: random.Random,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Стратифицированный split по k = len(required_agents)."""
    # группировка по k
    groups: dict[int, list[dict]] = defaultdict(list)
    for rec in items:
        k = len(rec.get("required_agents", []))
        groups[k].append(rec)

    train_all: list[dict] = []
    val_all: list[dict] = []
    test_all: list[dict] = []

    for k in range(SET_SIZE_MIN, SET_SIZE_MAX + 1):
        group = groups.get(k, [])
        if not group:
            continue

        if len(group) == 1:
            print(
                f"  ⚠ k={k}: только 1 запись — невозможно стратифицировать, "
                f"отправляется в train",
                file=sys.stderr,
            )

        g_train, g_val, g_test = _split_group(group, rng, train_ratio, val_ratio)
        train_all.extend(g_train)
        val_all.extend(g_val)
        test_all.extend(g_test)

    # финальный shuffle каждого сплита для случайного порядка
    rng.shuffle(train_all)
    rng.shuffle(val_all)
    rng.shuffle(test_all)

    return train_all, val_all, test_all


# ── report ───────────────────────────────────────────────────────────

def _print_stratification_report(
    train_items: list[dict],
    val_items: list[dict],
    test_items: list[dict],
) -> None:
    """Печатает таблицу counts по k (2..9) для каждого сплита."""
    def _count_by_k(items: list[dict]) -> Counter[int]:
        return Counter(len(r.get("required_agents", [])) for r in items)

    train_k = _count_by_k(train_items)
    val_k = _count_by_k(val_items)
    test_k = _count_by_k(test_items)

    print("\n── Распределение по k = |R| в каждом сплите ──")
    print(f"  {'k':>3}  {'train':>6}  {'val':>6}  {'test':>6}  {'total':>6}")
    print(f"  {'---':>3}  {'------':>6}  {'------':>6}  {'------':>6}  {'------':>6}")
    for k in range(SET_SIZE_MIN, SET_SIZE_MAX + 1):
        tr = train_k.get(k, 0)
        va = val_k.get(k, 0)
        te = test_k.get(k, 0)
        total = tr + va + te
        if total > 0:
            print(f"  {k:>3}  {tr:>6}  {va:>6}  {te:>6}  {total:>6}")
    print()


# ── main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Разбиение tasks_set.jsonl на train/val/test сплиты",
    )
    parser.add_argument(
        "--in_path",
        default="data/tasks_set.jsonl",
        help="Путь к исходному JSONL (по умолчанию data/tasks_set.jsonl)",
    )
    parser.add_argument(
        "--out_dir",
        default="data/splits",
        help="Директория для сплитов (по умолчанию data/splits)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed для перемешивания",
    )
    parser.add_argument(
        "--train", type=float, default=0.70, dest="train_ratio",
        help="Доля train (по умолчанию 0.70)",
    )
    parser.add_argument(
        "--val", type=float, default=0.15, dest="val_ratio",
        help="Доля val (по умолчанию 0.15)",
    )
    parser.add_argument(
        "--test", type=float, default=0.15, dest="test_ratio",
        help="Доля test (по умолчанию 0.15)",
    )
    parser.add_argument(
        "--stratify_by_set_size", type=int, default=1, choices=[0, 1],
        help="1 = стратификация по |R| (по умолчанию), 0 = простой split",
    )
    args = parser.parse_args()

    # ── проверка долей ───────────────────────────────────────────────
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(
            f"Сумма долей train+val+test = {total_ratio:.4f}, ожидается 1.0",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── чтение ───────────────────────────────────────────────────────
    items = _read_jsonl(args.in_path)
    n = len(items)
    if n < 3:
        print(
            f"Недостаточно записей для разбиения: {n} (минимум 3, "
            f"чтобы каждый сплит получил хотя бы 1 запись).",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── split ─────────────────────────────────────────────────────────
    rng = random.Random(args.seed)

    stratify = bool(args.stratify_by_set_size)

    if stratify:
        train_items, val_items, test_items = _stratified_split(
            items, rng, args.train_ratio, args.val_ratio,
        )
    else:
        train_items, val_items, test_items = _simple_split(
            items, rng, args.train_ratio, args.val_ratio,
        )

    assert len(train_items) + len(val_items) + len(test_items) == n

    # ── запись ────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    splits = {
        "train.jsonl": train_items,
        "val.jsonl": val_items,
        "test.jsonl": test_items,
    }

    mode_label = "стратифицированный" if stratify else "простой"
    print(f"Источник : {args.in_path}  ({n} записей)")
    print(f"Seed     : {args.seed}")
    print(f"Доли     : train={args.train_ratio}  val={args.val_ratio}  "
          f"test={args.test_ratio}")
    print(f"Режим    : {mode_label}")
    print(f"Выход    : {args.out_dir}/\n")

    for filename, split_items in splits.items():
        out_path = os.path.join(args.out_dir, filename)
        _write_jsonl(split_items, out_path)
        print(f"  {filename:<14} {len(split_items):>5} записей  → {out_path}")

    print(f"\n  {'ИТОГО':<14} {n:>5} записей")

    # ── таблица по k ─────────────────────────────────────────────────
    _print_stratification_report(train_items, val_items, test_items)

    # ── проверка покрытия агентов в train ─────────────────────────────
    _check_agent_coverage(train_items)


if __name__ == "__main__":
    main()
