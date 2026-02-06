from __future__ import annotations

import json
from typing import List, Dict


def load_jsonl(path: str) -> List[Dict]:
    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e

            if "text" not in obj or "label" not in obj:
                raise ValueError(f"Missing required fields on line {line_no}: need 'text' and 'label'")

            items.append(obj)

    return items
