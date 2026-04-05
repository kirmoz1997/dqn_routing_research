#!/usr/bin/env python3
"""
Generate adaptive dataset annotations for routing experiments.

The script reads records from JSONL, requests adaptive agent trajectories from
an OpenAI-compatible local API, and writes enriched records to output JSONL.

Example:
  source .env
  python tools/generate_adaptive_dataset.py --limit 5
  python tools/generate_adaptive_dataset.py --dry_run
"""

# pip install openai  (if not already installed)

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

INPUT_PATH = "data/tasks_set.jsonl"
OUTPUT_PATH = "data/tasks_set_adaptive_full.jsonl"
MAX_RETRIES = 3
STEP_BY_STEP_THRESHOLD = 6  # use per-agent generation for |R| >= this
REQUEST_TIMEOUT = 60  # seconds
DEFAULT_MODEL_NAME = "qwen3-32b"
DEFAULT_API_KEY_ENV = "ADAPTIVE_LLM_API_KEY"

AGENTS = {
    0: "Code Agent (Python) — пишет и исправляет Python-код",
    1: "SQL Agent — SQL-запросы (SELECT/JOIN/GROUP BY)",
    2: "Data Analysis Agent (Pandas) — анализ данных с Pandas",
    3: "Math Formula Solver — математические формулы и вычисления",
    4: "Structured Extraction Agent (JSON) — извлечение данных в JSON",
    5: "Summarization & Formatting Agent — суммаризация и форматирование",
    6: "Requirements / ТЗ Agent — ТЗ и требования (FR/NFR)",
    7: "Rewrite / Style Constraints Agent — рерайт и стилистика",
    8: "Finance / Numeric Computation Agent — финансовые расчёты",
}

SYSTEM_PROMPT = """
Ты — эксперт по мультиагентным системам. Твоя задача — сгенерировать
реалистичную симуляцию пошагового выполнения запроса набором агентов.

Отвечай ТОЛЬКО валидным JSON. Никакого текста до или после JSON.
Никаких markdown-блоков (```json). Только чистый JSON-объект.
Все текстовые поля (output, remaining_gap) должны быть
на русском языке.
""".strip()


def agent_name_from_dict(agent_id: int) -> str:
    raw = AGENTS.get(agent_id, f"Unknown Agent {agent_id}")
    return raw.split(" — ", 1)[0].strip()


def build_user_prompt(text: str, agents_list: str, allowed_agent_names: str) -> str:
    return f"""
Запрос пользователя: "{text}"

Набор агентов, которые должны обработать этот запрос (в порядке
логического выполнения): {agents_list}

Сгенерируй для каждого агента:
1. "output" — краткий результат работы агента (1-2 предложения),
   описывающий что именно он сделал для данного запроса
2. "remaining_gap" — что ещё не решено после этого агента
   (пустая строка "" если после данного агента задача решена полностью)
3. "is_last" — true если этот агент последний в цепочке, иначе false

Верни JSON строго в следующем формате:
{{
  "trajectory": [
    {{
      "agent_id": <int>,
      "agent_name": "<строго одно из: {allowed_agent_names}>",
      "output": "<string>",
      "remaining_gap": "<string>",
      "is_last": <bool>
    }},
    ...
  ]
}}
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate adaptive annotations for tasks_set JSONL."
    )
    parser.add_argument("--input", default=INPUT_PATH, help="Input JSONL path")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Output JSONL path")
    parser.add_argument(
        "--base_url",
        default=os.getenv("ADAPTIVE_LLM_BASE_URL"),
        help=(
            "OpenAI-compatible base URL. "
            "Defaults to env ADAPTIVE_LLM_BASE_URL; if omitted, SDK default is used."
        ),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("ADAPTIVE_LLM_MODEL", DEFAULT_MODEL_NAME),
        help=(
            "Model name to use. "
            f"Defaults to env ADAPTIVE_LLM_MODEL or '{DEFAULT_MODEL_NAME}'."
        ),
    )
    parser.add_argument(
        "--api_key_env",
        default=DEFAULT_API_KEY_ENV,
        help=(
            "Name of the environment variable holding the API key. "
            f"Default: {DEFAULT_API_KEY_ENV}"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N records from input",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print first 2 prompts without API calls, then exit",
    )
    return parser.parse_args()


def _resolve_runtime_config(args: argparse.Namespace) -> tuple[str | None, str, str]:
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise ValueError(
            "API key is not configured. "
            f"Set environment variable {args.api_key_env} before running."
        )
    if not args.model or not str(args.model).strip():
        raise ValueError("Model name must not be empty. Use --model or ADAPTIVE_LLM_MODEL.")
    base_url = str(args.base_url).strip() if args.base_url else None
    return base_url, str(args.model).strip(), api_key


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{line_no}, got {type(obj)}")
            records.append(obj)
    return records


def load_processed_ids(path: Path) -> set[str]:
    processed: set[str] = set()
    if not path.exists():
        return processed

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec_id = obj.get("id")
            adaptive = obj.get("adaptive")
            trajectory = adaptive.get("trajectory") if isinstance(adaptive, dict) else None
            has_valid_adaptive = isinstance(trajectory, list) and len(trajectory) > 0
            if isinstance(rec_id, str) and rec_id and has_valid_adaptive:
                processed.add(rec_id)
    return processed


def format_agents_list(required_agents: list[int]) -> str:
    parts: list[str] = []
    for agent_id in required_agents:
        agent_name = agent_name_from_dict(agent_id)
        parts.append(f"[{agent_id}] {agent_name}")
    return ", ".join(parts)


def _extract_json_text(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def parse_model_json(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = _extract_json_text(raw)
        return json.loads(cleaned)


def validate_adaptive_payload(
    payload: dict[str, Any], required_agents: list[int]
) -> tuple[bool, str]:
    trajectory = payload.get("trajectory")
    if not isinstance(trajectory, list):
        return False, "missing or non-list 'trajectory'"
    if len(trajectory) != len(required_agents):
        return (
            False,
            f"trajectory length {len(trajectory)} != expected {len(required_agents)}",
        )

    required_keys = {"agent_id", "agent_name", "output", "remaining_gap", "is_last"}
    for idx, step in enumerate(trajectory):
        if not isinstance(step, dict):
            return False, f"trajectory[{idx}] is not object"
        missing = required_keys - set(step.keys())
        if missing:
            return False, f"trajectory[{idx}] missing keys: {sorted(missing)}"
        if not isinstance(step["is_last"], bool):
            return False, f"trajectory[{idx}].is_last is not bool"
    return True, ""


def call_api_for_adaptive(
    client: OpenAI, model_name: str, text: str, required_agents: list[int]
) -> dict[str, Any]:
    agents_list = format_agents_list(required_agents)
    allowed_agent_names = ", ".join(agent_name_from_dict(aid) for aid in required_agents)
    user_prompt = build_user_prompt(
        text=text, agents_list=agents_list, allowed_agent_names=allowed_agent_names
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=3000,
        timeout=REQUEST_TIMEOUT,
    )
    raw = (response.choices[0].message.content or "").strip()
    payload = parse_model_json(raw)
    if not isinstance(payload, dict):
        raise ValueError("Model response root is not a JSON object")
    return payload


def generate_trajectory_step_by_step(
    client: OpenAI, model_name: str, item: dict
) -> list[dict] | None:
    """Generate trajectory one agent at a time for large |R| records.

    Each API call generates exactly one agent step (~100-150 tokens),
    avoiding token limit failures that occur with 6-9 agents at once.
    """
    text = item["text"]
    required = item["required_agents"]
    trajectory = []

    for step_idx, agent_id in enumerate(required):
        is_last = step_idx == len(required) - 1
        agent_name = AGENTS[agent_id].split(" — ")[0]
        remaining = required[step_idx + 1 :]

        if not trajectory:
            prev_text = "Нет (это первый агент)"
        else:
            prev_lines = [
                f"  - [{t['agent_id']}] {t['agent_name']}: {t['output']}"
                for t in trajectory
            ]
            prev_text = "\n".join(prev_lines)

        if is_last:
            remaining_text = "нет (это последний агент)"
        else:
            remaining_text = ", ".join(
                f"[{rid}] {AGENTS[rid].split(' — ')[0]}" for rid in remaining
            )

        system_prompt = (
            "Ты — эксперт по мультиагентным системам.\n"
            "Генерируй вывод одного агента в пошаговом выполнении запроса.\n"
            "Отвечай ТОЛЬКО валидным JSON. Никакого текста до или после.\n"
            "Никаких markdown-блоков. Только чистый JSON-объект.\n"
            "Все текстовые поля должны быть на русском языке."
        )

        last_note = "Это последний агент в цепочке." if is_last else ""
        user_prompt = (
            f'Запрос пользователя: "{text}"\n\n'
            f"Уже выполненные шаги:\n{prev_text}\n\n"
            f"Текущий агент: [{agent_id}] {agent_name}\n"
            f"Агенты после этого: {remaining_text}\n"
            f"{last_note}\n\n"
            f"Сгенерируй результат работы текущего агента.\n\n"
            f"Верни JSON строго в формате:\n"
            f'{{\n'
            f'  "output": "<что сделал этот агент, 1-2 предложения>",\n'
            f'  "remaining_gap": "<что ещё не решено после этого агента, '
            f'пустая строка если последний>"\n'
            f"}}"
        )

        step_result = None
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                    max_tokens=600,
                    timeout=REQUEST_TIMEOUT,
                )
                raw = response.choices[0].message.content.strip()
                parsed = json.loads(raw)

                if "output" not in parsed or not parsed["output"].strip():
                    raise ValueError("missing or empty output")
                if "remaining_gap" not in parsed:
                    raise ValueError("missing remaining_gap")
                if is_last:
                    parsed["remaining_gap"] = ""

                step_result = parsed
                break

            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(
                        f"  DEBUG {item['id']} agent={agent_id} "
                        f"attempt={attempt} error={type(e).__name__}: {str(e)[:120]}"
                    )
                    print(
                        f"  DEBUG raw response: "
                        f"{raw[:200] if 'raw' in dir() else 'no response'}"
                    )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)
                continue

        if step_result is None:
            return None

        trajectory.append(
            {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "output": step_result["output"],
                "remaining_gap": step_result["remaining_gap"],
                "is_last": is_last,
            }
        )

    return trajectory


def generate_adaptive_annotation(client: OpenAI, model_name: str, item: dict) -> dict | None:
    text = str(item.get("text", ""))
    required_agents = item.get("required_agents", [])
    if not isinstance(required_agents, list):
        return None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            payload = call_api_for_adaptive(client, model_name, text, required_agents)
            valid, reason = validate_adaptive_payload(payload, required_agents)
            if not valid:
                raise ValueError(f"validation failed: {reason}")
            result = dict(item)
            result["adaptive"] = {"trajectory": payload["trajectory"]}
            return result
        except Exception:
            if attempt < MAX_RETRIES:
                time.sleep(2)
            continue
    return None


def print_dry_run(records: list[dict[str, Any]]) -> None:
    print("DRY RUN: printing first 2 prompts, no API calls.")
    for idx, record in enumerate(records[:2], start=1):
        text = str(record.get("text", ""))
        required_agents = record.get("required_agents", [])
        if not isinstance(required_agents, list):
            required_agents = []
        agents_list = format_agents_list(required_agents)
        allowed_agent_names = ", ".join(agent_name_from_dict(aid) for aid in required_agents)
        user_prompt = build_user_prompt(
            text=text, agents_list=agents_list, allowed_agent_names=allowed_agent_names
        )
        print("-" * 80)
        print(f"Record #{idx} id={record.get('id', '<no_id>')}")
        print("\nSYSTEM PROMPT:\n")
        print(SYSTEM_PROMPT)
        print("\nUSER PROMPT:\n")
        print(user_prompt)
    print("-" * 80)
    print("Dry run complete.")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = read_jsonl(input_path)

    if args.dry_run:
        if args.limit is not None:
            records = records[: args.limit]
        print_dry_run(records)
        return

    base_url, model_name, api_key = _resolve_runtime_config(args)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_ids = load_processed_ids(output_path)
    print(f"Resuming: found {len(processed_ids)} already processed records, skipping.")
    pending_records = [
        record
        for record in records
        if not (str(record.get("id", "")) and str(record.get("id", "")) in processed_ids)
    ]
    if args.limit is not None:
        pending_records = pending_records[: args.limit]
    records = pending_records

    client_kwargs: dict[str, str] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    print(
        "Using LLM config: "
        f"model={model_name} "
        f"base_url={base_url or '<sdk default>'} "
        f"api_key_env={args.api_key_env}"
    )

    total = len(records)
    success = 0
    skipped = 0
    processed_counter = 0

    with output_path.open("a", encoding="utf-8") as out_fh:
        for record in records:
            processed_counter += 1
            rec_id = str(record.get("id", ""))
            if rec_id and rec_id in processed_ids:
                continue

            text = str(record.get("text", ""))
            required_agents = record.get("required_agents", [])
            if not isinstance(required_agents, list):
                print(f"SKIP id={rec_id or '<no_id>'} due to invalid required_agents")
                skipped += 1
                if processed_counter % 10 == 0:
                    pct = (processed_counter / total * 100) if total else 100.0
                    print(
                        f"Progress: {processed_counter}/{total} ({pct:.1f}%) "
                        f"| success={success} skipped={skipped}"
                    )
                continue

            item = record
            if len(item["required_agents"]) >= STEP_BY_STEP_THRESHOLD:
                trajectory = generate_trajectory_step_by_step(client, model_name, item)
                if trajectory is None:
                    print(f"SKIP id={item['id']} after {MAX_RETRIES} failed attempts")
                    skipped += 1
                    continue
                result = dict(item)
                result["adaptive"] = {"trajectory": trajectory}
            else:
                result = generate_adaptive_annotation(client, model_name, item)
                if result is None:
                    print(f"SKIP id={item['id']} after {MAX_RETRIES} failed attempts")
                    skipped += 1
                    continue

            out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_fh.flush()

            if rec_id:
                processed_ids.add(rec_id)
            success += 1

            if processed_counter % 10 == 0:
                pct = (processed_counter / total * 100) if total else 100.0
                print(
                    f"Progress: {processed_counter}/{total} ({pct:.1f}%) "
                    f"| success={success} skipped={skipped}"
                )

    print(
        f"Done. Total: {total} | Success: {success} | "
        f"Skipped: {skipped} | Output: {output_path}"
    )


if __name__ == "__main__":
    main()
