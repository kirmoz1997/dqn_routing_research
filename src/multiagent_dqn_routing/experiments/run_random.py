from __future__ import annotations

import numpy as np

from multiagent_dqn_routing.data.dataset import load_jsonl
from multiagent_dqn_routing.sim.reward_model import RewardModel, RewardConfig
from multiagent_dqn_routing.eval.evaluator import evaluate_router
from multiagent_dqn_routing.agents import N_AGENTS


def make_random_router(seed: int = 123):
    rng = np.random.default_rng(seed)

    def router(_text: str) -> int:
        return int(rng.integers(0, N_AGENTS))

    return router


def main():
    items = load_jsonl("data/tasks.jsonl")

    reward_model = RewardModel(RewardConfig(p_good=0.85, p_bad=0.30, seed=42))
    router = make_random_router(seed=123)

    metrics = evaluate_router(items, router, reward_model)
    print("Random router metrics:")
    print(metrics["n_items"], "items")
    print("mean_reward =", round(metrics["mean_reward"], 4))
    print("routing_accuracy =", round(metrics["routing_accuracy"], 4))
    print("confusion_matrix:\n", metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
