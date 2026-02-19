from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from multiagent_dqn_routing.agents import N_AGENTS
from multiagent_dqn_routing.data.dataset import load_jsonl_set
from multiagent_dqn_routing.envs.set_routing_env import STOP_ACTION, SetRoutingEnv
from multiagent_dqn_routing.eval.evaluator_set import evaluate_set_router
from multiagent_dqn_routing.rl.ddqn_agent import DoubleDQNAgent
from multiagent_dqn_routing.rl.replay_buffer import ReplayBuffer
from multiagent_dqn_routing.rl.state_encoder import TfidfStateEncoder
from multiagent_dqn_routing.sim.reward_set import RewardSetConfig, RewardSetModel

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "max_steps": 9,
    "use_action_mask": False,
    "data": {
        "train_path": "data/splits/train.jsonl",
        "val_path": "data/splits/val.jsonl",
        "test_path": "data/splits/test.jsonl",
    },
    "reward": {
        "alpha": 1.0,
        "beta": 0.5,
        "gamma": 1.0,
        "p_good": 0.85,
        "p_bad": 0.30,
    },
    "train": {
        "total_steps": 30000,
        "learning_starts": 1000,
        "batch_size": 128,
        "buffer_size": 100000,
        "learning_rate": 1e-3,
        "discount": 0.99,
        "hidden_sizes": [256, 256],
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "target_update_every": 500,
        "eval_every": 2000,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError("config JSON must be an object")
    return _deep_merge(DEFAULT_CONFIG, raw)


def _linear_epsilon(step: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 1:
        return float(end)
    frac = min(max(step / float(total_steps - 1), 0.0), 1.0)
    return float(start + frac * (end - start))


def _precompute_text_vecs(items: list[dict[str, Any]], encoder: TfidfStateEncoder) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        ex = dict(item)
        ex["text_vec"] = encoder.transform_text(ex["text"])
        out.append(ex)
    return out


def _make_greedy_router(
    agent: DoubleDQNAgent,
    encoder: TfidfStateEncoder,
    max_steps: int,
    use_action_mask: bool,
):
    def router(text: str) -> list[int]:
        text_vec = encoder.transform_text(text)
        selected: set[int] = set()
        chosen: list[int] = []

        for step_idx in range(max_steps):
            selected_mask = np.zeros(N_AGENTS, dtype=np.float32)
            if selected:
                selected_mask[list(selected)] = 1.0
            state = encoder.encode(text_vec, selected_mask, step_idx, max_steps)

            mask = None
            if use_action_mask:
                mask = np.ones(N_AGENTS + 1, dtype=np.float32)
                for aid in selected:
                    mask[aid] = 0.0

            action = agent.select_action(state, epsilon=0.0, mask=mask)
            if action == STOP_ACTION:
                break

            chosen.append(int(action))
            selected.add(int(action))

        return chosen

    return router


def _print_key_metrics(metrics: dict[str, Any], title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"n_items              = {metrics['n_items']}")
    print(f"mean_episode_reward  = {metrics['mean_episode_reward']:.4f}")
    print(f"success_rate         = {metrics['success_rate']:.4f}")
    print(f"exact_match_rate     = {metrics['exact_match_rate']:.4f}")
    print(f"mean_jaccard         = {metrics['mean_jaccard']:.4f}")
    print(f"mean_precision       = {metrics['mean_precision']:.4f}")
    print(f"mean_recall          = {metrics['mean_recall']:.4f}")
    print(f"mean_f1              = {metrics['mean_f1']:.4f}")
    print(f"avg_steps            = {metrics['avg_steps']:.4f}")
    print(f"avg_overselection    = {metrics['avg_overselection']:.4f}")
    print(f"avg_underselection   = {metrics['avg_underselection']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Double DQN for set-routing")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cpu/cuda")
    parser.add_argument("--smoke_test", action="store_true", help="Run with reduced steps")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    seed = int(cfg["seed"])
    _set_seed(seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    max_steps = int(cfg["max_steps"])
    use_action_mask = bool(cfg.get("use_action_mask", False))

    train_path = cfg["data"]["train_path"]
    val_path = cfg["data"]["val_path"]
    test_path = cfg["data"]["test_path"]

    train_items = load_jsonl_set(train_path)
    val_items = load_jsonl_set(val_path)
    test_items = load_jsonl_set(test_path)
    print(f"Loaded splits: train={len(train_items)} val={len(val_items)} test={len(test_items)}")

    encoder = TfidfStateEncoder()
    encoder.fit([x["text"] for x in train_items])
    train_items = _precompute_text_vecs(train_items, encoder)

    step_cost = float(cfg["reward"].get("step_cost", 0.0))

    reward_cfg = RewardSetConfig(
        alpha=float(cfg["reward"]["alpha"]),
        beta=float(cfg["reward"]["beta"]),
        gamma=float(cfg["reward"]["gamma"]),
        p_good=float(cfg["reward"]["p_good"]),
        p_bad=float(cfg["reward"]["p_bad"]),
        step_cost=step_cost,
        seed=seed,
    )
    reward_model = RewardSetModel(reward_cfg)

    train_cfg = cfg["train"]
    total_steps = int(train_cfg["total_steps"])
    learning_starts = int(train_cfg["learning_starts"])
    batch_size = int(train_cfg["batch_size"])
    buffer_size = int(train_cfg["buffer_size"])
    target_update_every = int(train_cfg["target_update_every"])
    eval_every = int(train_cfg["eval_every"])
    epsilon_start = float(train_cfg["epsilon_start"])
    epsilon_end = float(train_cfg["epsilon_end"])
    hidden_sizes = tuple(int(x) for x in train_cfg["hidden_sizes"])

    if args.smoke_test:
        total_steps = min(total_steps, 2000)
        learning_starts = min(learning_starts, 200)
        eval_every = min(eval_every, 500)
        print("Smoke test mode enabled")

    env = SetRoutingEnv(
        items=train_items,
        encoder=encoder,
        reward_model=reward_model,
        max_steps=max_steps,
        seed=seed,
        use_action_mask=use_action_mask,
        step_cost=step_cost,
    )
    replay = ReplayBuffer(capacity=buffer_size, seed=seed)
    agent = DoubleDQNAgent(
        input_dim=encoder.state_dim,
        n_actions=N_AGENTS + 1,
        hidden_sizes=hidden_sizes,
        learning_rate=float(train_cfg["learning_rate"]),
        discount=float(train_cfg["discount"]),
        device=device,
        seed=seed,
    )

    artifact_dir = Path("artifacts/ddqn")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.pt"
    encoder_path = artifact_dir / "encoder.joblib"
    metrics_val_path = artifact_dir / "metrics_val_best.json"
    metrics_test_path = artifact_dir / "metrics_test.json"
    config_used_path = artifact_dir / "config_used.json"

    state = env.reset().astype(np.float32, copy=False)
    best_f1 = float("-inf")
    best_metrics: dict[str, Any] | None = None

    for step in range(total_steps):
        epsilon = _linear_epsilon(step, total_steps, epsilon_start, epsilon_end)
        action_mask = env.get_action_mask() if use_action_mask else None
        action = agent.select_action(state, epsilon=epsilon, mask=action_mask)
        next_state, reward, done, _info = env.step(action)
        next_state = next_state.astype(np.float32, copy=False)

        replay.add(state, action, reward, next_state, done)
        state = next_state

        if done:
            state = env.reset().astype(np.float32, copy=False)

        if step >= learning_starts and len(replay) >= batch_size:
            batch = replay.sample(batch_size)
            agent.train_step(batch)

        if (step + 1) % target_update_every == 0:
            agent.update_target()

        if (step + 1) % eval_every == 0:
            eval_reward_model = RewardSetModel(
                RewardSetConfig(
                    alpha=reward_cfg.alpha,
                    beta=reward_cfg.beta,
                    gamma=reward_cfg.gamma,
                    p_good=reward_cfg.p_good,
                    p_bad=reward_cfg.p_bad,
                    seed=seed,
                )
            )
            val_router = _make_greedy_router(agent, encoder, max_steps, use_action_mask)
            val_metrics = evaluate_set_router(val_items, val_router, eval_reward_model)
            val_f1 = float(val_metrics["mean_f1"])
            print(
                f"step={step + 1:6d}  epsilon={epsilon:.4f}  "
                f"val_mean_f1={val_f1:.4f}  val_jaccard={val_metrics['mean_jaccard']:.4f}"
            )
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_metrics = val_metrics
                torch.save(agent.q.state_dict(), model_path)
                with open(metrics_val_path, "w", encoding="utf-8") as fh:
                    json.dump(best_metrics, fh, ensure_ascii=False, indent=2)

    if best_metrics is None:
        eval_reward_model = RewardSetModel(reward_cfg)
        val_router = _make_greedy_router(agent, encoder, max_steps, use_action_mask)
        best_metrics = evaluate_set_router(val_items, val_router, eval_reward_model)
        best_f1 = float(best_metrics["mean_f1"])
        torch.save(agent.q.state_dict(), model_path)
        with open(metrics_val_path, "w", encoding="utf-8") as fh:
            json.dump(best_metrics, fh, ensure_ascii=False, indent=2)

    print(f"\nBest val mean_f1 = {best_f1:.4f}")
    _print_key_metrics(best_metrics, "Validation (best checkpoint)")

    saved_state = torch.load(model_path, map_location=agent.device)
    agent.q.load_state_dict(saved_state)
    agent.update_target()

    test_reward_model = RewardSetModel(
        RewardSetConfig(
            alpha=reward_cfg.alpha,
            beta=reward_cfg.beta,
            gamma=reward_cfg.gamma,
            p_good=reward_cfg.p_good,
            p_bad=reward_cfg.p_bad,
            seed=seed,
        )
    )
    test_router = _make_greedy_router(agent, encoder, max_steps, use_action_mask)
    test_metrics = evaluate_set_router(test_items, test_router, test_reward_model)

    _print_key_metrics(test_metrics, "Test (greedy)")

    joblib.dump(encoder, encoder_path)
    with open(metrics_test_path, "w", encoding="utf-8") as fh:
        json.dump(test_metrics, fh, ensure_ascii=False, indent=2)

    config_used = copy.deepcopy(cfg)
    config_used["device"] = device
    config_used["smoke_test"] = bool(args.smoke_test)
    with open(config_used_path, "w", encoding="utf-8") as fh:
        json.dump(config_used, fh, ensure_ascii=False, indent=2)

    print("\nSaved artifacts:")
    print(f"  - {model_path}")
    print(f"  - {encoder_path}")
    print(f"  - {metrics_val_path}")
    print(f"  - {metrics_test_path}")
    print(f"  - {config_used_path}")


if __name__ == "__main__":
    main()
