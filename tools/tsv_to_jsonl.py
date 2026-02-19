#!/usr/bin/env python3
"""Convert TSV drafts to JSONL datasets.

Supported conversions:
  1. data/tasks_set_draft.tsv   → data/tasks_set.jsonl

Run: python tools/tsv_to_jsonl.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


# ── helpers ──────────────────────────────────────────────────────────────────

def fail(line_num: int, message: str) -> None:
    print(f"Error on line {line_num}: {message}", file=sys.stderr)
    sys.exit(1)


def _read_rows(
    tsv_path: Path,
    expected_header: list[str],
) -> list[tuple[int, dict[str, str]]]:
    """Read TSV into a list of (line_num, row_dict) tuples."""
    rows: list[tuple[int, dict[str, str]]] = []
    header: list[str] | None = None

    with tsv_path.open("r", encoding="utf-8", newline="") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            if not raw_line.strip():
                continue

            if header is None:
                header = next(csv.reader([raw_line], delimiter="\t"))
                if header != expected_header:
                    fail(
                        line_num,
                        f"unexpected header {header!r}, expected {expected_header!r}",
                    )
                continue

            row = next(csv.reader([raw_line], delimiter="\t"))
            if len(row) != len(header):
                fail(line_num, f"expected {len(header)} columns, got {len(row)}")

            rows.append((line_num, dict(zip(header, row))))

    if header is None:
        fail(1, "missing header")

    return rows


def _validate_unique_id(
    line_num: int,
    raw_id: str,
    seen_ids: set[str],
) -> str:
    raw_id = raw_id.strip()
    if not raw_id:
        fail(line_num, "id is empty")
    if raw_id in seen_ids:
        fail(line_num, f"duplicate id {raw_id!r}")
    seen_ids.add(raw_id)
    return raw_id


def write_jsonl(records: list[dict[str, Any]], jsonl_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")


# ── tasks_set_draft.tsv ─────────────────────────────────────────────────────

TASKS_SET_HEADER = ["id", "required_agents", "difficulty", "eval_hint", "text", "notes"]


def parse_tasks_set(tsv_path: Path) -> list[dict[str, Any]]:
    rows = _read_rows(tsv_path, TASKS_SET_HEADER)
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for line_num, record in rows:
        record["id"] = _validate_unique_id(line_num, record["id"], seen_ids)

        # --- required_agents: "0,2,4" → [0, 2, 4] ---
        raw_agents = record["required_agents"].strip()
        if not raw_agents:
            fail(line_num, "required_agents is empty")
        parts = [p.strip() for p in raw_agents.split(",")]
        agents: list[int] = []
        for part in parts:
            try:
                val = int(part)
            except ValueError:
                fail(line_num, f"required_agents: {part!r} is not an int")
            if not 0 <= val <= 8:
                fail(line_num, f"required_agents: {val} out of range 0..8")
            agents.append(val)

        if len(agents) != len(set(agents)):
            fail(line_num, f"required_agents has duplicates: {agents}")
        if not 2 <= len(agents) <= 9:
            fail(
                line_num,
                f"required_agents must have 2..9 elements, got {len(agents)}",
            )
        record["required_agents"] = agents

        # --- difficulty: int, default 2 ---
        raw_diff = record["difficulty"].strip()
        if raw_diff == "":
            record["difficulty"] = 2
        else:
            try:
                record["difficulty"] = int(raw_diff)
            except ValueError:
                fail(line_num, f"difficulty is not an int: {raw_diff!r}")

        records.append(record)

    return records


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    # 1. tasks_set_draft.tsv → tasks_set.jsonl
    tasks_set_tsv = repo_root / "data" / "tasks_set_draft.tsv"
    if tasks_set_tsv.exists():
        records = parse_tasks_set(tasks_set_tsv)
        write_jsonl(records, repo_root / "data" / "tasks_set.jsonl")
        print(f"✓ {tasks_set_tsv.name} → tasks_set.jsonl  ({len(records)} records)")
    else:
        print(f"⚠ {tasks_set_tsv} not found, skipping", file=sys.stderr)


if __name__ == "__main__":
    main()
