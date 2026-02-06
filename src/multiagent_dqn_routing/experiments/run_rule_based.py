from __future__ import annotations

import numpy as np

from multiagent_dqn_routing.data.dataset import load_jsonl
from multiagent_dqn_routing.sim.reward_model import RewardModel, RewardConfig
from multiagent_dqn_routing.eval.evaluator import evaluate_router
from multiagent_dqn_routing.agents import N_AGENTS


def make_rule_based_router(seed: int = 123):
    rng = np.random.default_rng(seed)

    def router(text: str) -> int:
        t = text.lower()

        # 1) SQL
        sql_markers = ["sql", "select", "join", "group by", "where", "order by", "insert", "update"]
        if any(m in t for m in sql_markers):
            return 1

        # 2) Pandas / Data analysis
        pandas_markers = ["pandas", "dataframe", "df", "csv", "groupby", "agg", "pivot"]
        if any(m in t for m in pandas_markers):
            return 2

        # 3) Structured extraction JSON
        extract_markers = ["json", "извлеки", "извлечь", "поля", "ключи", "структур"]
        if any(m in t for m in extract_markers):
            return 4

        # 4) Requirements / ТЗ
        req_markers = ["тз", "техническое задание", "требован", "критерии приемки", "acceptance criteria", "scope"]
        if any(m in t for m in req_markers):
            return 6

        # 5) Summarization
        sum_markers = ["tl;dr", "тл;др", "саммари", "суммар", "краткое резюме", "5 буллет", "выжимка"]
        if any(m in t for m in sum_markers):
            return 5

        # 6) Rewrite / Style
        rewrite_markers = ["перепиши", "переписать", "деловом стиле", "сократи", "тон", "стиль", "формально"]
        if any(m in t for m in rewrite_markers):
            return 7

        # 7) Math
        math_markers = ["вычисли", "формул", "предел", "производн", "вероятност", "округли", "^", "sqrt", "sin", "cos"]
        if any(m in t for m in math_markers):
            return 3

        # 8) Finance
        fin_markers = ["процент", "сложный процент", "npv", "roi", "маржа", "юнит", "выручк", "прибыл"]
        if any(m in t for m in fin_markers):
            return 8

        # 9) Code (Python) — оставляем ближе к концу, потому что слово "ошибка" бывает везде
        code_markers = ["python", "def ", "class ", "pytest", "traceback", "stack trace", "исправь функцию", "ошибка в коде"]
        if any(m in t for m in code_markers):
            return 0

        # fallback: если не распознали — выбираем случайно
        return int(rng.integers(0, N_AGENTS))

    return router


def main():
    items = load_jsonl("data/tasks.jsonl")

    reward_model = RewardModel(RewardConfig(p_good=0.85, p_bad=0.30, seed=42))
    router = make_rule_based_router(seed=123)

    metrics = evaluate_router(items, router, reward_model)
    print("Rule-based router metrics:")
    print(metrics["n_items"], "items")
    print("mean_reward =", round(metrics["mean_reward"], 4))
    print("routing_accuracy =", round(metrics["routing_accuracy"], 4))
    print("confusion_matrix:\n", metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
