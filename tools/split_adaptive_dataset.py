#!/usr/bin/env python3
"""Split the adaptive dataset into stratified train/val/test JSONL files.

The split mirrors ``tools/split_jsonl_set.py`` and stratifies by
``|R| = len(required_agents)`` so the adaptive dataset preserves the same
set-size balance across train/validation/test.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
DEFAULT_SEED = 42
SET_SIZE_MIN = 2
SET_SIZE_MAX = 9


def _read_jsonl(path: str) -> list[dict]:
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                items.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno}: {exc}") from exc
    return items


def _write_jsonl(items: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def _split_group(
    group: list[dict],
    rng: random.Random,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Match the per-group split logic used in ``split_jsonl_set.py``."""
    rng.shuffle(group)
    n = len(group)

    if n == 1:
        return group[:], [], []
    if n == 2:
        return group[:1], [], group[1:]

    n_val = max(1, round(n * val_ratio))
    n_test = max(1, round(n * (1.0 - train_ratio - val_ratio)))
    n_train = n - n_val - n_test

    if n_train < 1:
        n_train = 1
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
    groups: dict[int, list[dict]] = defaultdict(list)
    for rec in items:
        groups[len(rec.get("required_agents", []))].append(rec)

    train_items: list[dict] = []
    val_items: list[dict] = []
    test_items: list[dict] = []

    for set_size in range(SET_SIZE_MIN, SET_SIZE_MAX + 1):
        group = groups.get(set_size, [])
        if not group:
            continue
        if len(group) == 1:
            print(
                f"Warning: |R|={set_size} has only 1 record; assigning it to train",
                file=sys.stderr,
            )
        g_train, g_val, g_test = _split_group(group, rng, train_ratio, val_ratio)
        train_items.extend(g_train)
        val_items.extend(g_val)
        test_items.extend(g_test)

    rng.shuffle(train_items)
    rng.shuffle(val_items)
    rng.shuffle(test_items)
    return train_items, val_items, test_items


def _count_by_required_size(items: list[dict]) -> dict[int, int]:
    counter = Counter(len(item.get("required_agents", [])) for item in items)
    return {k: counter[k] for k in sorted(counter)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split data/tasks_set_adaptive_full.jsonl into train/val/test",
    )
    parser.add_argument(
        "--input",
        default="data/tasks_set_adaptive_full.jsonl",
        help="Path to adaptive JSONL dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="data/splits_adaptive",
        help="Directory for adaptive train/val/test splits",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic splitting",
    )
    args = parser.parse_args()

    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    items = _read_jsonl(args.input)
    if len(items) < 3:
        raise ValueError("Adaptive dataset must contain at least 3 records to split")

    rng = random.Random(args.seed)
    train_items, val_items, test_items = _stratified_split(
        items=items,
        rng=rng,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )

    if len(train_items) + len(val_items) + len(test_items) != len(items):
        raise RuntimeError("Split sizes do not sum to the input dataset size")

    os.makedirs(args.output_dir, exist_ok=True)
    _write_jsonl(train_items, os.path.join(args.output_dir, "train.jsonl"))
    _write_jsonl(val_items, os.path.join(args.output_dir, "val.jsonl"))
    _write_jsonl(test_items, os.path.join(args.output_dir, "test.jsonl"))

    train_dist = _count_by_required_size(train_items)
    val_dist = _count_by_required_size(val_items)
    test_dist = _count_by_required_size(test_items)

    print("Adaptive dataset split complete:")
    print(f"  train: {len(train_items)} records")
    print(f"  val:   {len(val_items)} records")
    print(f"  test:  {len(test_items)} records")
    print(
        "  |R| distribution per split: "
        f"train={train_dist}, val={val_dist}, test={test_dist}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
