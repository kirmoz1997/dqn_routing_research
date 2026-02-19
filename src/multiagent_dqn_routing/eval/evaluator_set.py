"""Evaluator for multi-agent *set* routing.

Given a list of dataset items, a router function, and a RewardSetModel,
runs each item through the router and collects episode-level metrics
defined in the Research Plan (§10).

Supports per-bucket breakdown by |R|:
  bucket A: |R| in {2, 3}
  bucket B: |R| in {4, 5, 6}
  bucket C: |R| in {7, 8, 9}

Per-item set-quality metrics:
  exact_match, jaccard, precision, recall, f1.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np

from multiagent_dqn_routing.agents import N_AGENTS
from multiagent_dqn_routing.sim.reward_set import RewardSetModel


# router_fn(text) → list[int]  (ordered sequence of chosen agent ids)
SetRouterFn = Callable[[str], List[int]]

# Bucket definitions: name → set of |R| values
BUCKETS: Dict[str, set[int]] = {
    "A (|R|∈{2,3})":   {2, 3},
    "B (|R|∈{4,5,6})": {4, 5, 6},
    "C (|R|∈{7,8,9})": {7, 8, 9},
}

# Keys used in per-item accumulator dicts.
_ACCUM_KEYS = (
    "episode_rewards", "successes", "steps_list",
    "coverages", "overselections", "underselections",
    "exact_matches", "jaccards", "precisions", "recalls", "f1s",
)


def _new_accumulator() -> Dict[str, list]:
    """Return a fresh per-item accumulator dict."""
    return {k: [] for k in _ACCUM_KEYS}


def _aggregate_metrics(accum: Dict[str, list]) -> Dict[str, Any]:
    """Compute aggregate metrics from a per-item accumulator."""
    n = len(accum["episode_rewards"])
    if n == 0:
        return {
            "mean_episode_reward": 0.0,
            "success_rate": 0.0,
            "avg_steps": 0.0,
            "avg_coverage": 0.0,
            "avg_overselection": 0.0,
            "avg_underselection": 0.0,
            "exact_match_rate": 0.0,
            "mean_jaccard": 0.0,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "mean_f1": 0.0,
            "n_items": 0,
        }
    return {
        "mean_episode_reward": float(np.mean(accum["episode_rewards"])),
        "success_rate": float(np.mean(accum["successes"])),
        "avg_steps": float(np.mean(accum["steps_list"])),
        "avg_coverage": float(np.mean(accum["coverages"])),
        "avg_overselection": float(np.mean(accum["overselections"])),
        "avg_underselection": float(np.mean(accum["underselections"])),
        "exact_match_rate": float(np.mean(accum["exact_matches"])),
        "mean_jaccard": float(np.mean(accum["jaccards"])),
        "mean_precision": float(np.mean(accum["precisions"])),
        "mean_recall": float(np.mean(accum["recalls"])),
        "mean_f1": float(np.mean(accum["f1s"])),
        "n_items": n,
    }


def _append_item(
    accum: Dict[str, list],
    *,
    total_reward: float,
    underselection: int,
    n_steps: int,
    coverage: int,
    overselection: int,
    exact_match: int,
    jaccard: float,
    precision: float,
    recall: float,
    f1: float,
) -> None:
    """Push one item's metrics into *accum*."""
    accum["episode_rewards"].append(total_reward)
    accum["successes"].append(underselection == 0)
    accum["steps_list"].append(n_steps)
    accum["coverages"].append(coverage)
    accum["overselections"].append(overselection)
    accum["underselections"].append(underselection)
    accum["exact_matches"].append(exact_match)
    accum["jaccards"].append(jaccard)
    accum["precisions"].append(precision)
    accum["recalls"].append(recall)
    accum["f1s"].append(f1)


def evaluate_set_router(
    items: List[Dict[str, Any]],
    router_fn: SetRouterFn,
    reward_model: RewardSetModel,
) -> Dict[str, Any]:
    """Evaluate a set-routing strategy on *items*.

    Parameters
    ----------
    items:
        Dataset records.  Each must have ``"text"`` (str) and
        ``"required_agents"`` (list[int]).
    router_fn:
        ``router_fn(text)`` returns a **list** of agent ids (the order
        is the order the router chose them).  Duplicates are allowed
        but penalised.
    reward_model:
        Instance of :class:`RewardSetModel`.

    Returns
    -------
    dict with aggregate metrics:
        mean_episode_reward, success_rate, avg_steps,
        avg_coverage, avg_overselection, avg_underselection,
        exact_match_rate, mean_jaccard, mean_precision, mean_recall,
        mean_f1, n_items,
        and ``"buckets"`` — a dict mapping bucket name to the same
        metric dict computed on that subset.
    """

    overall = _new_accumulator()
    bucket_accum: Dict[str, Dict[str, list]] = {
        name: _new_accumulator() for name in BUCKETS
    }

    for ex in items:
        text: str = ex["text"]
        required_set: set[int] = set(ex["required_agents"])
        set_size = len(required_set)

        chosen_sequence: list[int] = router_fn(text)

        # validate agent ids
        for aid in chosen_sequence:
            if not (0 <= aid < N_AGENTS):
                raise ValueError(
                    f"Router returned invalid agent id {aid} "
                    f"for item {ex.get('id', '?')}"
                )

        # ── step-level rewards ───────────────────────────────────────
        total_reward = 0.0
        selected: set[int] = set()

        for agent_id in chosen_sequence:
            r, _covered_new = reward_model.step_reward(
                required_set=required_set,
                chosen_agent=agent_id,
                already_selected=selected,
            )
            total_reward += r
            selected.add(agent_id)

        # ── terminal penalty ─────────────────────────────────────────
        total_reward += reward_model.terminal_penalty(required_set, selected)

        # ── set-level metrics ────────────────────────────────────────
        intersection = len(selected & required_set)        # |S ∩ R|
        union = len(selected | required_set)               # |S ∪ R|
        coverage = intersection
        overselection = len(selected - required_set)       # |S \ R|
        underselection = len(required_set - selected)      # |R \ S|

        exact_match = int(selected == required_set)

        jaccard = intersection / union if union > 0 else 0.0

        s_size = len(selected)
        precision = intersection / s_size if s_size > 0 else 0.0
        recall = intersection / len(required_set) if len(required_set) > 0 else 0.0

        denom = precision + recall
        f1 = 2.0 * precision * recall / denom if denom > 0 else 0.0

        # ── accumulate ───────────────────────────────────────────────
        item_kwargs = dict(
            total_reward=total_reward,
            underselection=underselection,
            n_steps=len(chosen_sequence),
            coverage=coverage,
            overselection=overselection,
            exact_match=exact_match,
            jaccard=jaccard,
            precision=precision,
            recall=recall,
            f1=f1,
        )

        _append_item(overall, **item_kwargs)

        for bucket_name, sizes in BUCKETS.items():
            if set_size in sizes:
                _append_item(bucket_accum[bucket_name], **item_kwargs)

    # ── aggregate ─────────────────────────────────────────────────────
    result = _aggregate_metrics(overall)
    result["buckets"] = {
        name: _aggregate_metrics(ba)
        for name, ba in bucket_accum.items()
    }

    return result


def print_bucket_metrics(metrics: Dict[str, Any]) -> None:
    """Print per-bucket breakdown from *metrics* returned by
    :func:`evaluate_set_router`.

    Should be called **after** the main metrics block is printed.
    """
    buckets = metrics.get("buckets")
    if not buckets:
        return

    print(f"\n{'─' * 50}")
    print("  Метрики по bucket-ам (|R| группы)")
    print(f"{'─' * 50}")

    for bucket_name, bm in buckets.items():
        n = bm["n_items"]
        if n == 0:
            print(f"\n  {bucket_name}: нет данных")
            continue
        print(f"\n  {bucket_name}:")
        print(f"    n_items              = {n}")
        print(f"    success_rate         = {bm['success_rate']:.4f}")
        print(f"    exact_match_rate     = {bm['exact_match_rate']:.4f}")
        print(f"    mean_jaccard         = {bm['mean_jaccard']:.4f}")
        print(f"    mean_precision       = {bm['mean_precision']:.4f}")
        print(f"    mean_recall          = {bm['mean_recall']:.4f}")
        print(f"    mean_f1              = {bm['mean_f1']:.4f}")
        print(f"    avg_overselection    = {bm['avg_overselection']:.4f}")
        print(f"    avg_underselection   = {bm['avg_underselection']:.4f}")
        print(f"    avg_steps            = {bm['avg_steps']:.4f}")
        print(f"    mean_episode_reward  = {bm['mean_episode_reward']:.4f}")
