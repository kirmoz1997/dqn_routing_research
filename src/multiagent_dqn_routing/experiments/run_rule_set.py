"""Rule-based baseline for multi-agent set routing.

Run:
    python -m multiagent_dqn_routing.experiments.run_rule_set
    python -m multiagent_dqn_routing.experiments.run_rule_set --split_path data/splits/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable, List

import numpy as np

from multiagent_dqn_routing.agents import N_AGENTS
from multiagent_dqn_routing.data.dataset import load_jsonl_set
from multiagent_dqn_routing.eval.evaluator_set import evaluate_set_router, print_bucket_metrics
from multiagent_dqn_routing.experiments.snapshot_utils import (
    build_meta,
    normalize_metrics,
    parse_reward_args,
)
from multiagent_dqn_routing.sim.reward_set import RewardSetConfig, RewardSetModel

# ── trigger tables ───────────────────────────────────────────────────────────
# Each entry: (agent_id, list_of_markers)

TRIGGERS: list[tuple[int, list[str]]] = [
    (1, [  # SQL Agent
        "select", "join", "group by", "where", "order by",
        "insert", "update", "таблиц", "запрос",
    ]),
    (2, [  # Data Analysis Agent (Pandas)
        "dataframe", "датасет", "groupby", "pivot", "csv",
        "quantile", "percentile", "ретеншн", "когорта",
    ]),
    (0, [  # Code Agent (Python)
        "python", "def ", "traceback", "pytest",
        "исправь функцию", "ошибка в коде",
    ]),
    (3, [  # Math Formula Solver
        "вычисли", "формула", "производная", "интеграл",
        "вероятность", "корни", "sqrt", "sin", "cos",
    ]),
    (4, [  # Structured Extraction Agent (JSON)
        "json", "извлеки", "извлечь", "поля", "ключи", "структур",
    ]),
    (5, [  # Summarization & Formatting Agent
        "tl;dr", "выжимка", "кратко", "тезисы", "bullets",
        "executive summary", "конспект",
    ]),
    (6, [  # Requirements / ТЗ Agent
        "тз", "требования", "критерии приемки",
        "acceptance criteria", "user story", "нефункциональные",
    ]),
    (7, [  # Rewrite / Style Constraints Agent
        "перепиши", "сократи", "тон", "стиль", "деловой",
        "исправь грамматику",
    ]),
    (8, [  # Finance / Numeric Computation Agent
        "npv", "roi", "cac", "ltv", "маржа", "выручка",
        "прибыль", "процент", "аннуитет", "дисконт",
    ]),
]


# ── router factory ───────────────────────────────────────────────────────────

def make_rule_based_set_router(
    seed: int = 123,
    min_len: int = 2,
    max_len: int = 9,
) -> Callable[[str], List[int]]:
    rng = np.random.default_rng(seed)

    def router(text: str) -> List[int]:
        t = text.lower()

        chosen: list[int] = []
        chosen_set: set[int] = set()

        for agent_id, markers in TRIGGERS:
            if agent_id in chosen_set:
                continue
            if any(m in t for m in markers):
                chosen.append(agent_id)
                chosen_set.add(agent_id)

        # pad with random unique agents if too few
        if len(chosen) < min_len:
            remaining = [a for a in range(N_AGENTS) if a not in chosen_set]
            rng.shuffle(remaining)
            for a in remaining:
                chosen.append(a)
                chosen_set.add(a)
                if len(chosen) >= min_len:
                    break

        # trim if too many
        if len(chosen) > max_len:
            chosen = chosen[:max_len]

        return chosen

    return router


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rule-based baseline for multi-agent set routing",
    )
    parser.add_argument(
        "--split_path",
        default="data/splits/test.jsonl",
        help="Путь к JSONL-сплиту для оценки (по умолчанию data/splits/test.jsonl)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default 42)",
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
    args = parser.parse_args()

    if args.max_steps < 2:
        raise ValueError("--max_steps must be >= 2")

    items = load_jsonl_set(args.split_path)
    reward = parse_reward_args(args)

    print(f"Источник : {args.split_path}")
    print(f"n_items  : {len(items)}")
    print(f"Seed     : {args.seed}\n")

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

    router = make_rule_based_set_router(
        seed=args.seed + 81,
        min_len=2,
        max_len=min(args.max_steps, 9),
    )

    raw_metrics = evaluate_set_router(items, router, reward_model)
    metrics, buckets = normalize_metrics(raw_metrics)

    print("Rule-based set-router metrics:")
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

    if args.json_out:
        out_dir = os.path.dirname(args.json_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            "baseline": "rule",
            "split_path": args.split_path,
            "seed": args.seed,
            "reward": reward,
            "max_steps": args.max_steps,
            "metrics": metrics,
            "buckets": buckets,
            "meta": build_meta(
                dataset_path=args.dataset_path,
                split_path=args.split_path,
            ),
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\nJSON snapshot saved: {args.json_out}")


if __name__ == "__main__":
    main()
