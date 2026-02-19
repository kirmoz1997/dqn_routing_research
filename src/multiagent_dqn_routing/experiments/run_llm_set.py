"""LLM-based baseline for multi-agent set routing.

Calls an OpenAI-compatible Chat Completions API to predict the required
agent set for each item.  Results are cached to avoid redundant API calls.

Run:
    python -m multiagent_dqn_routing.experiments.run_llm_set
    python -m multiagent_dqn_routing.experiments.run_llm_set --split_path data/splits/test.jsonl
    python -m multiagent_dqn_routing.experiments.run_llm_set --model gpt-4o --temperature 0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from multiagent_dqn_routing.agents import AGENTS, N_AGENTS
from multiagent_dqn_routing.data.dataset import load_jsonl_set
from multiagent_dqn_routing.eval.evaluator_set import evaluate_set_router, print_bucket_metrics
from multiagent_dqn_routing.experiments.snapshot_utils import (
    build_meta,
    normalize_metrics,
    parse_reward_args,
)
from multiagent_dqn_routing.sim.reward_set import RewardSetConfig, RewardSetModel

PROMPT_VERSION = "v2-recall-biased"

# ── system prompt ────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    """Build the system prompt describing the routing task and agent list."""
    agent_lines = "\n".join(
        f"  {a['id']}. {a['name']} — {a['description']}" for a in AGENTS
    )
    return (
        "Ты — роутер запросов. Твоя задача: по тексту пользовательского запроса "
        "определить, какие агенты нужны для его решения.\n\n"
        "Доступные агенты (id 0..8):\n"
        f"{agent_lines}\n\n"
        "Главная цель: НЕ ПРОПУСТИТЬ нужных агентов (максимизировать полноту набора). "
        "Если сомневаешься между включить/не включить агента — лучше включи. "
        "Но не выбирай все подряд: старайся держать набор минимальным при условии, "
        "что не пропустишь нужных.\n\n"
        "Перед ответом мысленно проверь чеклист (НЕ включай его в ответ):\n"
        "- Есть ли в запросе подзадачи про написание/исправление кода?\n"
        "- Нужен ли SQL-запрос или работа с базой данных?\n"
        "- Требуется ли анализ данных, таблиц, метрик?\n"
        "- Есть ли математические формулы или вычисления?\n"
        "- Нужно ли извлечь данные в JSON или структурированный формат?\n"
        "- Требуется ли краткое резюме, отчёт или саммари?\n"
        "- Нужно ли составить ТЗ, требования или спецификацию?\n"
        "- Нужно ли переписать/адаптировать текст под определённый стиль?\n"
        "- Есть ли финансовые расчёты (проценты, ROI, NPV, бюджет)?\n\n"
        "Правила формата ответа:\n"
        "1. Верни ТОЛЬКО JSON-массив уникальных целых чисел (id агентов), "
        "отсортированных по возрастанию.\n"
        "2. Длина массива — от 2 до 9 включительно.\n"
        "3. Каждый id — целое число от 0 до 8.\n"
        "4. Никакого дополнительного текста, пояснений или markdown — "
        "только JSON-массив.\n\n"
        "Пример ответа: [0,2,6]"
    )


SYSTEM_PROMPT = _build_system_prompt()


# ── keyword fallback ─────────────────────────────────────────────────────────

_FALLBACK_TRIGGERS: list[tuple[int, list[str]]] = [
    (0, ["python", "скрипт", "код", "функци", "программ", "реализуй",
         "разработай", "создай", "парсер", "пайплайн", "модуль"]),
    (1, ["sql", "таблиц", "запрос", "выгрузи", "базы данных", "базе"]),
    (2, ["анализ", "проанализируй", "метрик", "данных", "статистик",
         "pandas", "csv", "датасет"]),
    (3, ["вычисли", "формул", "рассчитай", "посчитай", "уравнен",
         "математическ", "оптимизац"]),
    (4, ["json", "структурирован", "извлеч", "формат", "поля"]),
    (5, ["резюме", "саммари", "краткий", "краткое", "отчёт", "сформируй",
         "подготовь"]),
    (6, ["техническое задание", "тз", "требовани", "спецификац", "составь",
         "гайдлайн", "методолог"]),
    (7, ["переформулируй", "адаптируй", "перепиши", "стил", "тон",
         "презентац"]),
    (8, ["финанс", "процент", "доходност", "инвестиц", "бюджет", "roi",
         "npv", "рентабельност", "маржа"]),
]


def _keyword_fallback(text: str) -> List[int]:
    """Simple keyword-based fallback when LLM response is unparseable."""
    t = text.lower()
    chosen: list[int] = []
    for agent_id, markers in _FALLBACK_TRIGGERS:
        if any(m in t for m in markers):
            chosen.append(agent_id)
    # guarantee at least 2
    if len(chosen) < 2:
        # fill with generic pair: Summarization + Requirements
        for default_id in [5, 6, 0, 2, 3, 4, 7, 8, 1]:
            if default_id not in chosen:
                chosen.append(default_id)
            if len(chosen) >= 2:
                break
    return sorted(set(chosen))


# ── LLM API ─────────────────────────────────────────────────────────────────

def call_llm(
    messages: List[Dict[str, str]],
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call an OpenAI-compatible Chat Completions endpoint.

    Returns the assistant message content as a string.
    Raises ``RuntimeError`` on HTTP or protocol errors.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(
            f"LLM API HTTP {exc.code}: {error_body[:500]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM API connection error: {exc.reason}") from exc

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected API response structure: {data}") from exc


# ── response parsing ─────────────────────────────────────────────────────────

def _parse_llm_response(raw: str) -> Optional[List[int]]:
    """Try to extract a valid agent-id list from raw LLM output.

    Returns ``None`` if parsing or validation fails.
    """
    # strip markdown code fences if present
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # try to find first JSON array in the text
        match = re.search(r"\[[\d,\s]+\]", cleaned)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

    if not isinstance(parsed, list):
        return None

    # validate elements
    try:
        ids = [int(x) for x in parsed]
    except (ValueError, TypeError):
        return None

    ids = sorted(set(ids))

    # range check
    if any(x < 0 or x >= N_AGENTS for x in ids):
        return None

    # length check
    if not (2 <= len(ids) <= 9):
        return None

    return ids


# ── cache ────────────────────────────────────────────────────────────────────

def _load_cache(
    cache_path: str,
    *,
    model: str,
    prompt_version: str,
) -> tuple[Dict[str, Dict[str, Any]], int]:
    """Load JSONL cache filtered by *model* and *prompt_version*.

    Returns ``(cache_dict, total_lines)`` where *cache_dict* contains
    only entries matching the current model+prompt_version, and
    *total_lines* is the total number of valid records in the file
    (for diagnostics).
    """
    cache: Dict[str, Dict[str, Any]] = {}
    total_lines = 0
    if not os.path.exists(cache_path):
        return cache, total_lines
    with open(cache_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "id" not in rec or "pred" not in rec:
                    continue
                total_lines += 1
                if (rec.get("model") == model
                        and rec.get("prompt_version") == prompt_version):
                    cache[rec["id"]] = rec
            except json.JSONDecodeError:
                continue
    return cache, total_lines


def _append_cache(cache_path: str, record: Dict[str, Any]) -> None:
    """Append a single record to the JSONL cache."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── router factory ───────────────────────────────────────────────────────────

def make_llm_set_router(
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    cache: Dict[str, Dict[str, Any]],
    cache_path: str,
    prompt_version: str,
    max_steps: int,
):
    """Create a router function that calls the LLM API with caching."""
    stats = {"api_calls": 0, "cache_hits": 0, "fallbacks": 0}

    def router(text: str, *, _item_id: str = "") -> List[int]:
        # check cache
        if _item_id and _item_id in cache:
            stats["cache_hits"] += 1
            pred = list(cache[_item_id]["pred"])
            if len(pred) > max_steps:
                pred = pred[:max_steps]
            return pred

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

        raw_response = ""
        parsed: Optional[List[int]] = None

        for attempt in range(1 + max_retries):
            try:
                raw_response = call_llm(
                    messages,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                stats["api_calls"] += 1
                parsed = _parse_llm_response(raw_response)
                if parsed is not None:
                    break
            except RuntimeError as exc:
                print(f"    LLM error (attempt {attempt + 1}): {exc}", file=sys.stderr)
                if attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))  # simple backoff

        if parsed is None:
            # fallback
            stats["fallbacks"] += 1
            parsed = _keyword_fallback(text)
            print(
                f"    ⚠ fallback for id={_item_id}: "
                f"raw={raw_response[:120]!r} → {parsed}",
                file=sys.stderr,
            )

        # write to cache
        if len(parsed) > max_steps:
            parsed = parsed[:max_steps]
        if _item_id and _item_id not in cache:
            rec = {
                "id": _item_id,
                "pred": parsed,
                "raw": raw_response[:500],
                "model": model,
                "prompt_version": prompt_version,
            }
            cache[_item_id] = rec
            _append_cache(cache_path, rec)

        return parsed

    return router, stats


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-based baseline for multi-agent set routing",
    )
    parser.add_argument(
        "--split_path", default="data/splits/test.jsonl",
        help="Путь к JSONL-сплиту для оценки (по умолчанию data/splits/test.jsonl)",
    )
    parser.add_argument(
        "--provider", default="openai_compatible",
        help="Провайдер API (по умолчанию openai_compatible)",
    )
    parser.add_argument(
        "--base_url", default="https://openai-hub.neuraldeep.tech/v1",
        help="Base URL для API (по умолчанию https://openai-hub.neuraldeep.tech/v1)",
    )
    parser.add_argument(
        "--api_key_env", default="LLM_API_KEY",
        help="Имя переменной окружения с API-ключом (по умолчанию LLM_API_KEY)",
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="Название модели (по умолчанию gpt-4o-mini)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Temperature (по умолчанию 0)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=120,
        help="Max tokens в ответе (по умолчанию 120)",
    )
    parser.add_argument(
        "--cache_path", default="cache/llm_router_cache.jsonl",
        help="Путь к JSONL-кэшу (по умолчанию cache/llm_router_cache.jsonl)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed для reward model (по умолчанию 42)",
    )
    parser.add_argument(
        "--max_retries", type=int, default=2,
        help="Макс. повторных попыток при ошибке LLM (по умолчанию 2)",
    )
    parser.add_argument("--max_steps", type=int, default=9, help="Max selected agents")
    parser.add_argument("--dataset_path", default="data/tasks_set.jsonl")
    parser.add_argument("--json_out", default=None, help="Write snapshot JSON to path")
    parser.add_argument("--reward_config_json", default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--p_good", type=float, default=0.85)
    parser.add_argument("--p_bad", type=float, default=0.30)
    parser.add_argument(
        "--prompt_version",
        default=PROMPT_VERSION,
        help="Prompt version marker used for cache filtering and metadata",
    )
    args = parser.parse_args()
    if args.max_steps < 2:
        raise ValueError("--max_steps must be >= 2")
    reward = parse_reward_args(args)

    # ── resolve base_url ──────────────────────────────────────────────
    base_url = args.base_url.strip() if args.base_url.strip() else "https://api.openai.com/v1"

    # ── resolve API key ───────────────────────────────────────────────
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        print(
            f"Ошибка: переменная окружения {args.api_key_env!r} не задана или пуста.\n"
            f"Установите: export {args.api_key_env}=sk-...",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── load data ─────────────────────────────────────────────────────
    items = load_jsonl_set(args.split_path)

    print(f"Источник       : {args.split_path}")
    print(f"n_items        : {len(items)}")
    print(f"Модель         : {args.model}")
    print(f"Prompt version : {args.prompt_version}")
    print(f"Base URL       : {base_url}")
    print(f"Cache          : {args.cache_path}")
    print(f"Temperature    : {args.temperature}")
    print(f"Max retries    : {args.max_retries}")
    print(f"Seed           : {args.seed}\n")

    # ── load cache ────────────────────────────────────────────────────
    cache, total_cache_lines = _load_cache(
        args.cache_path, model=args.model, prompt_version=args.prompt_version,
    )
    print(f"Записей в кэше (всего в файле)  : {total_cache_lines}")
    print(f"Подходят (model+prompt_version)  : {len(cache)}")

    cached_for_split = sum(1 for item in items if item.get("id") in cache)
    cache_misses = len(items) - cached_for_split
    print(f"Cache hits для текущего сплита   : {cached_for_split}/{len(items)}")
    print(f"Cache misses (потребуют API)     : {cache_misses}\n")

    # ── build router ──────────────────────────────────────────────────
    router_fn, stats = make_llm_set_router(
        base_url=base_url,
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        cache=cache,
        cache_path=args.cache_path,
        prompt_version=args.prompt_version,
        max_steps=min(args.max_steps, 9),
    )

    # ── wrap router to pass item_id ───────────────────────────────────
    # evaluate_set_router expects router_fn(text) → list[int],
    # but we need to pass item_id for caching.
    # We'll use a closure over a mutable index.
    _item_index = {"i": 0}

    def router_with_id(text: str) -> List[int]:
        idx = _item_index["i"]
        item_id = items[idx].get("id", f"__idx_{idx}")
        _item_index["i"] += 1
        return router_fn(text, _item_id=item_id)

    # ── reward model ──────────────────────────────────────────────────
    reward_model = RewardSetModel(
        RewardSetConfig(
            alpha=reward["alpha"],
            beta=reward["beta"],
            gamma=reward["gamma"],
            p_good=reward["p_good"],
            p_bad=reward["p_bad"],
            seed=args.seed,
        )
    )

    # ── evaluate ──────────────────────────────────────────────────────
    raw_metrics = evaluate_set_router(items, router_with_id, reward_model)
    metrics, buckets = normalize_metrics(raw_metrics)

    # ── print results ─────────────────────────────────────────────────
    print(f"\n{'═' * 50}")
    print(f"  LLM set-router metrics ({args.model})")
    print(f"{'═' * 50}")
    print(f"  n_items              = {raw_metrics['n_items']}")
    print(f"  mean_episode_reward  = {raw_metrics['mean_episode_reward']:.4f}")
    print(f"  success_rate         = {raw_metrics['success_rate']:.4f}")
    print(f"  exact_match_rate     = {raw_metrics['exact_match_rate']:.4f}")
    print(f"  mean_jaccard         = {raw_metrics['mean_jaccard']:.4f}")
    print(f"  mean_precision       = {raw_metrics['mean_precision']:.4f}")
    print(f"  mean_recall          = {raw_metrics['mean_recall']:.4f}")
    print(f"  mean_f1              = {raw_metrics['mean_f1']:.4f}")
    print(f"  avg_steps            = {raw_metrics['avg_steps']:.4f}")
    print(f"  avg_coverage         = {raw_metrics['avg_coverage']:.4f}")
    print(f"  avg_overselection    = {raw_metrics['avg_overselection']:.4f}")
    print(f"  avg_underselection   = {raw_metrics['avg_underselection']:.4f}")

    print_bucket_metrics(raw_metrics)

    # ── API stats ─────────────────────────────────────────────────────
    print(f"\n── API stats ──")
    print(f"  API calls  : {stats['api_calls']}")
    print(f"  Cache hits : {stats['cache_hits']}")
    print(f"  Fallbacks  : {stats['fallbacks']}")

    if args.json_out:
        out_dir = os.path.dirname(args.json_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            "baseline": "llm",
            "split_path": args.split_path,
            "seed": args.seed,
            "reward": reward,
            "max_steps": args.max_steps,
            "metrics": metrics,
            "buckets": buckets,
            "meta": build_meta(
                dataset_path=args.dataset_path,
                split_path=args.split_path,
                extra={
                    "prompt_version": args.prompt_version,
                    "model": args.model,
                    "base_url": base_url,
                    "temperature": args.temperature,
                },
            ),
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\nJSON snapshot saved: {args.json_out}")


if __name__ == "__main__":
    main()
