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
from multiagent_dqn_routing.envs.adaptive_routing_env import AdaptiveRoutingEnv
from multiagent_dqn_routing.envs.set_routing_env import STOP_ACTION, SetRoutingEnv
from multiagent_dqn_routing.eval.evaluator_set import evaluate_set_router
from multiagent_dqn_routing.rl.ddqn_agent import DoubleDQNAgent
from multiagent_dqn_routing.rl.replay_buffer import ReplayBuffer
from multiagent_dqn_routing.rl.state_encoder import TfidfStateEncoder
from multiagent_dqn_routing.sim.reward_set import RewardSetConfig, RewardSetJaccard, RewardSetModel

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "max_steps": 9,
    "use_action_mask": False,
    "env_mode": "static",
    "curriculum": {
        "enabled": False,
        "phase_fractions": [0.33, 0.33, 0.34],
    },
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


def _get_adaptive_output(item: dict[str, Any], agent_id: int) -> str:
    adaptive = item.get("adaptive")
    trajectory = adaptive.get("trajectory") if isinstance(adaptive, dict) else None
    if not isinstance(trajectory, list):
        raise ValueError(f"Adaptive item {item.get('id', '?')} is missing adaptive.trajectory")

    for step in trajectory:
        if int(step.get("agent_id", -1)) == int(agent_id):
            return str(step.get("output", ""))
    return ""


def _build_adaptive_state(
    *,
    item: dict[str, Any],
    encoder: TfidfStateEncoder,
    selected: set[int],
    context_outputs: list[str],
) -> np.ndarray:
    if "text_vec" in item:
        text_vec = np.asarray(item["text_vec"], dtype=np.float32)
    else:
        text_vec = encoder.transform_text(item["text"])

    selected_mask = np.zeros(N_AGENTS, dtype=np.float32)
    if selected:
        selected_mask[list(selected)] = 1.0

    if context_outputs:
        context_vec = encoder.transform_text(" ".join(context_outputs))
    else:
        context_vec = np.zeros(encoder.tfidf_dim, dtype=np.float32)

    return np.concatenate([text_vec, selected_mask, context_vec]).astype(
        np.float32,
        copy=False,
    )


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


def _make_adaptive_greedy_router(
    agent: DoubleDQNAgent,
    encoder: TfidfStateEncoder,
    max_steps: int,
    use_action_mask: bool,
):
    def router(item: dict[str, Any]) -> list[int]:
        selected: set[int] = set()
        chosen: list[int] = []
        context_outputs: list[str] = []

        for _step_idx in range(max_steps):
            state = _build_adaptive_state(
                item=item,
                encoder=encoder,
                selected=selected,
                context_outputs=context_outputs,
            )

            mask = None
            if use_action_mask:
                mask = np.ones(N_AGENTS + 1, dtype=np.float32)
                for aid in selected:
                    mask[aid] = 0.0

            action = agent.select_action(state, epsilon=0.0, mask=mask)
            if action == STOP_ACTION:
                break

            agent_output = _get_adaptive_output(item, int(action))
            if agent_output:
                context_outputs.append(agent_output)

            chosen.append(int(action))
            selected.add(int(action))

        return chosen

    return router


def _evaluate_adaptive_router(
    items: list[dict[str, Any]],
    router_fn,
    reward_fn: RewardSetJaccard,
) -> dict[str, Any]:
    episode_rewards: list[float] = []
    successes: list[float] = []
    steps_list: list[int] = []
    coverages: list[int] = []
    overselections: list[int] = []
    underselections: list[int] = []
    exact_matches: list[float] = []
    jaccards: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for item in items:
        required_set = set(item["required_agents"])
        chosen_sequence = router_fn(item)

        for aid in chosen_sequence:
            if not (0 <= aid < N_AGENTS):
                raise ValueError(
                    f"Router returned invalid agent id {aid} for item {item.get('id', '?')}"
                )

        total_reward = 0.0
        selected: set[int] = set()
        for agent_id in chosen_sequence:
            total_reward += reward_fn.step_reward(
                action=agent_id,
                selected=selected,
                required=required_set,
            )
            selected.add(agent_id)
        total_reward += reward_fn.terminal_reward(
            selected=selected,
            required=required_set,
        )

        intersection = len(selected & required_set)
        union = len(selected | required_set)
        coverage = intersection
        overselection = len(selected - required_set)
        underselection = len(required_set - selected)
        exact_match = float(selected == required_set)
        jaccard = intersection / union if union > 0 else 0.0

        selected_size = len(selected)
        precision = intersection / selected_size if selected_size > 0 else 0.0
        recall = intersection / len(required_set) if required_set else 0.0
        denom = precision + recall
        f1 = 2.0 * precision * recall / denom if denom > 0 else 0.0

        episode_rewards.append(float(total_reward))
        successes.append(float(underselection == 0))
        steps_list.append(len(chosen_sequence))
        coverages.append(coverage)
        overselections.append(overselection)
        underselections.append(underselection)
        exact_matches.append(exact_match)
        jaccards.append(jaccard)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "mean_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "exact_match_rate": float(np.mean(exact_matches)) if exact_matches else 0.0,
        "mean_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
        "mean_precision": float(np.mean(precisions)) if precisions else 0.0,
        "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
        "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
        "avg_steps": float(np.mean(steps_list)) if steps_list else 0.0,
        "avg_overselection": float(np.mean(overselections)) if overselections else 0.0,
        "avg_underselection": float(np.mean(underselections)) if underselections else 0.0,
        "n_items": len(items),
    }


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


def _build_curriculum_phases(
    items: list[dict[str, Any]],
    phase_fractions: tuple[float, float, float] = (0.33, 0.33, 0.34),
) -> list[list[dict[str, Any]]]:
    """Split training items into 3 curriculum phases by |R| size.

    Phase 1 — bucket A: len(required_agents) in {2, 3}
    Phase 2 — bucket A+B: len(required_agents) in {2, 3, 4, 5, 6}
    Phase 3 — all items (full dataset)

    Parameters
    ----------
    items:
        Full training dataset (pre-computed text_vecs expected).
    phase_fractions:
        Fraction of total_steps for each phase. Must sum to 1.0.

    Returns
    -------
    List of 3 item lists, one per phase.
    """
    bucket_a = [x for x in items if len(x["required_agents"]) <= 3]
    bucket_ab = [x for x in items if len(x["required_agents"]) <= 6]
    bucket_all = items

    if len(bucket_a) == 0:
        raise ValueError("No training items with |R| <= 3 found. "
                         "Check your dataset split.")
    if len(bucket_ab) == 0:
        raise ValueError("No training items with |R| <= 6 found.")

    return [bucket_a, bucket_ab, bucket_all]


def _build_adaptive_corpus(items: list[dict]) -> list[str]:
    """Build TF-IDF training corpus from query texts + agent outputs.

    In adaptive routing, context_vec is built from agent output strings.
    If the encoder is trained only on query texts, agent output tokens
    are largely out-of-vocabulary, making context_vec uninformative.

    This function extends the corpus with all agent output strings
    from adaptive.trajectory, giving the encoder shared vocabulary
    that covers both the query space and the agent output space.

    Parameters
    ----------
    items:
        Adaptive dataset records containing "text" and
        "adaptive.trajectory[*].output" fields.

    Returns
    -------
    List of strings: query texts + all agent output strings.
    Duplicates are allowed (standard TF-IDF behavior).
    """
    corpus = []
    for item in items:
        # Always include query text
        corpus.append(item["text"])
        # Include all agent outputs from trajectory
        trajectory = item.get("adaptive", {}).get("trajectory", [])
        for step in trajectory:
            output = step.get("output", "").strip()
            if output:
                corpus.append(output)
    return corpus


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
    env_mode = str(cfg.get("env_mode", "static")).lower()
    if env_mode not in {"static", "adaptive"}:
        raise ValueError(f"env_mode must be 'static' or 'adaptive', got '{env_mode}'")

    if env_mode == "adaptive":
        train_path = cfg["data"].get("adaptive_train_path", "data/splits_adaptive/train.jsonl")
        val_path = cfg["data"].get("adaptive_val_path", "data/splits_adaptive/val.jsonl")
        test_path = cfg["data"].get("adaptive_test_path", "data/splits_adaptive/test.jsonl")
    else:
        train_path = cfg["data"]["train_path"]
        val_path = cfg["data"]["val_path"]
        test_path = cfg["data"]["test_path"]

    train_items = load_jsonl_set(train_path)
    val_items = load_jsonl_set(val_path)
    test_items = load_jsonl_set(test_path)
    print(
        f"Loaded {env_mode} splits: "
        f"train={len(train_items)} val={len(val_items)} test={len(test_items)}"
    )

    encoder = TfidfStateEncoder()
    if env_mode == "adaptive":
        fit_corpus = _build_adaptive_corpus(train_items)
        print(
            f"Adaptive encoder corpus: {len(fit_corpus)} documents "
            f"({len(train_items)} texts + "
            f"{len(fit_corpus) - len(train_items)} agent outputs)"
        )
    else:
        fit_corpus = [x["text"] for x in train_items]
    encoder.fit(fit_corpus)
    train_items = _precompute_text_vecs(train_items, encoder)

    reward_mode = "jaccard" if env_mode == "adaptive" else str(cfg.get("reward_mode", "stochastic"))
    step_cost = float(cfg.get("step_cost", cfg["reward"].get("step_cost", 0.0)))

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
    reward_fn = RewardSetJaccard(step_cost=step_cost)

    train_cfg = cfg["train"]
    total_steps = int(cfg.get("total_steps", train_cfg["total_steps"]))
    curriculum_cfg = cfg.get("curriculum", {})
    use_curriculum = bool(curriculum_cfg.get("enabled", False))
    phase_fractions = tuple(
        float(x) for x in curriculum_cfg.get("phase_fractions", [0.33, 0.33, 0.34])
    )
    learning_starts = int(cfg.get("learning_starts", train_cfg["learning_starts"]))
    batch_size = int(cfg.get("batch_size", train_cfg["batch_size"]))
    buffer_size = int(cfg.get("buffer_size", train_cfg["buffer_size"]))
    target_update_every = int(cfg.get("target_update_every", train_cfg["target_update_every"]))
    eval_every = int(cfg.get("eval_every", train_cfg["eval_every"]))
    epsilon_start = float(cfg.get("epsilon_start", train_cfg["epsilon_start"]))
    epsilon_end = float(cfg.get("epsilon_end", train_cfg["epsilon_end"]))
    hidden_sizes = tuple(int(x) for x in cfg.get("hidden_sizes", train_cfg["hidden_sizes"]))
    learning_rate = float(cfg.get("lr", train_cfg["learning_rate"]))
    discount = float(cfg.get("discount", train_cfg["discount"]))

    if args.smoke_test:
        total_steps = min(total_steps, 2000)
        learning_starts = min(learning_starts, 200)
        eval_every = min(eval_every, 500)
        print("Smoke test mode enabled")

    if env_mode == "adaptive":
        env = AdaptiveRoutingEnv(
            items=train_items,
            encoder=encoder,
            reward_fn=reward_fn,
            max_steps=max_steps,
            seed=seed,
            use_action_mask=use_action_mask,
        )
        input_dim = env.state_dim
    else:
        env = SetRoutingEnv(
            items=train_items,
            encoder=encoder,
            reward_model=reward_model,
            max_steps=max_steps,
            seed=seed,
            use_action_mask=use_action_mask,
            step_cost=step_cost,
            reward_mode=reward_mode,
        )
        input_dim = encoder.state_dim
    replay = ReplayBuffer(capacity=buffer_size, seed=seed)
    agent = DoubleDQNAgent(
        input_dim=input_dim,
        n_actions=N_AGENTS + 1,
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        discount=discount,
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

    best_f1 = float("-inf")
    best_metrics: dict[str, Any] | None = None

    # ── Curriculum setup ────────────────────────────────────────────
    if use_curriculum:
        curriculum_phases = _build_curriculum_phases(train_items, phase_fractions)
        phase_boundaries = [
            int(total_steps * phase_fractions[0]),
            int(total_steps * (phase_fractions[0] + phase_fractions[1])),
        ]
        current_phase = 0
        env.set_items(curriculum_phases[0])
        print(
            f"Curriculum enabled: phase 1 starts "
            f"(|R|≤3, n={len(curriculum_phases[0])} items)"
        )

    state = env.reset().astype(np.float32, copy=False)

    for step in range(total_steps):
        # ── Curriculum phase switching ───────────────────────────────────
        if use_curriculum:
            new_phase = current_phase
            if step >= phase_boundaries[1] and current_phase < 2:
                new_phase = 2
            elif step >= phase_boundaries[0] and current_phase < 1:
                new_phase = 1
            if new_phase != current_phase:
                current_phase = new_phase
                env.set_items(curriculum_phases[current_phase])
                state = env.reset().astype(np.float32, copy=False)
                phase_sizes = [len(p) for p in curriculum_phases]
                r_labels = ["|R|≤3", "|R|≤6", "all |R|"]
                print(
                    f"Curriculum: phase {current_phase + 1} starts at step {step} "
                    f"({r_labels[current_phase]}, "
                    f"n={phase_sizes[current_phase]} items)"
                )

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
            if env_mode == "adaptive":
                val_router = _make_adaptive_greedy_router(agent, encoder, max_steps, use_action_mask)
                val_metrics = _evaluate_adaptive_router(val_items, val_router, reward_fn)
            else:
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
        if env_mode == "adaptive":
            val_router = _make_adaptive_greedy_router(agent, encoder, max_steps, use_action_mask)
            best_metrics = _evaluate_adaptive_router(val_items, val_router, reward_fn)
        else:
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

    if env_mode == "adaptive":
        test_router = _make_adaptive_greedy_router(agent, encoder, max_steps, use_action_mask)
        test_metrics = _evaluate_adaptive_router(test_items, test_router, reward_fn)
    else:
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
    config_used["env_mode"] = env_mode
    config_used["reward_mode"] = reward_mode
    config_used["step_cost"] = step_cost
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
