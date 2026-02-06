from __future__ import annotations

from typing import Callable, Dict, Any, List
import numpy as np

from multiagent_dqn_routing.agents import N_AGENTS


RouterFn = Callable[[str], int]


def evaluate_router(
    items: List[Dict[str, Any]],
    router_fn: RouterFn,
    reward_model,
) -> Dict[str, Any]:
    rewards = []
    correct = 0
    confusion = np.zeros((N_AGENTS, N_AGENTS), dtype=np.int32)

    for ex in items:
        text = ex["text"]
        true_label = int(ex["label"])
        chosen = int(router_fn(text))

        if chosen < 0 or chosen >= N_AGENTS:
            raise ValueError(f"Router returned invalid agent id: {chosen}")

        confusion[true_label, chosen] += 1
        if chosen == true_label:
            correct += 1

        r = int(reward_model.sample_reward(true_label=true_label, chosen_agent=chosen))
        rewards.append(r)

    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    routing_accuracy = float(correct / len(items)) if items else 0.0

    per_class_acc: Dict[int, float] = {}
    for cls in range(N_AGENTS):
        total = int(confusion[cls].sum())
        per_class_acc[cls] = float(confusion[cls, cls] / total) if total > 0 else 0.0

    return {
        "mean_reward": mean_reward,
        "routing_accuracy": routing_accuracy,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": confusion,
        "n_items": len(items),
    }