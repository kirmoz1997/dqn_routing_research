#!/usr/bin/env python3
"""Convert data/tasks_draft.tsv to data/tasks.jsonl.

Run: python tools/tsv_to_jsonl.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


EXPECTED_HEADER = ["id", "label", "difficulty", "eval_hint", "text", "notes"]


def fail(line_num: int, message: str) -> None:
    print(f"Error on line {line_num}: {message}", file=sys.stderr)
    sys.exit(1)


def parse_tsv(tsv_path: Path) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    seen_ids: set[str] = set()

    header: list[str] | None = None
    with tsv_path.open("r", encoding="utf-8", newline="") as tsv_file:
        for line_num, raw_line in enumerate(tsv_file, start=1):
            if not raw_line.strip():
                continue

            if header is None:
                header = next(csv.reader([raw_line], delimiter="\t"))
                if header != EXPECTED_HEADER:
                    fail(
                        line_num,
                        f"unexpected header {header!r}, expected {EXPECTED_HEADER!r}",
                    )
                continue

            row = next(csv.reader([raw_line], delimiter="\t"))
            if len(row) != len(header):
                fail(
                    line_num,
                    f"expected {len(header)} columns, got {len(row)}",
                )

            record = dict(zip(header, row))

            raw_id = record["id"].strip()
            if not raw_id:
                fail(line_num, "id is empty")
            if raw_id in seen_ids:
                fail(line_num, f"duplicate id {raw_id!r}")
            seen_ids.add(raw_id)
            record["id"] = raw_id

            raw_label = record["label"].strip()
            try:
                label_value = int(raw_label)
            except ValueError:
                fail(line_num, f"label is not an int: {raw_label!r}")
            if not 0 <= label_value <= 8:
                fail(line_num, f"label {label_value} out of range 0..8")
            record["label"] = label_value

            rows.append(record)

    if header is None:
        fail(1, "missing header")

    return rows


def write_jsonl(records: list[dict[str, str | int]], jsonl_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for record in records:
            jsonl_file.write(json.dumps(record, ensure_ascii=False))
            jsonl_file.write("\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tsv_path = repo_root / "data" / "tasks_draft.tsv"
    jsonl_path = repo_root / "data" / "tasks.jsonl"

    if not tsv_path.exists():
        fail(1, f"input file not found: {tsv_path}")

    records = parse_tsv(tsv_path)
    write_jsonl(records, jsonl_path)


if __name__ == "__main__":
    main()
