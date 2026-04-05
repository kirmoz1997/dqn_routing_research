"""Adaptive routing environment for sequential multi-agent selection.

This environment exists to model the *adaptive* formulation of the routing
task, where reinforcement learning is justified because each action changes
what the policy knows before the next decision.  Unlike ``SetRoutingEnv``,
which exposes only the original request text and the selected-agent mask,
``AdaptiveRoutingEnv`` also feeds back the outputs produced by previously
chosen agents.

State layout:
    ``state = [text_vec | selected_mask | context_vec]``

Here ``context_vec`` is a TF-IDF encoding of the concatenated outputs from
``current_item["adaptive"]["trajectory"]`` for all selected agents so far.
The same :class:`~multiagent_dqn_routing.rl.state_encoder.TfidfStateEncoder`
used for ``text_vec`` is also used for ``context_vec``, so both text channels
share one vocabulary and the observation size stays consistent.  When no
agents have been selected yet, ``context_vec`` is all zeros.

The reward is always the deterministic Jaccard formulation from Research
Plan §6.3: a fixed step cost for each non-STOP action plus terminal Jaccard
similarity between the selected set and the required set.
"""

from __future__ import annotations

from typing import Any
import sys

import numpy as np

from multiagent_dqn_routing.agents import N_AGENTS
from multiagent_dqn_routing.rl.state_encoder import TfidfStateEncoder
from multiagent_dqn_routing.sim.reward_set import RewardSetJaccard

STOP_ACTION = N_AGENTS


def _validate_adaptive_items(items: list[dict[str, Any]]) -> None:
    if not items:
        raise ValueError("items must not be empty")

    for idx, item in enumerate(items):
        adaptive = item.get("adaptive")
        trajectory = adaptive.get("trajectory") if isinstance(adaptive, dict) else None
        if not isinstance(trajectory, list):
            item_id = item.get("id", f"index {idx}")
            raise ValueError(
                f"Item {item_id} is missing adaptive.trajectory annotation"
            )


class AdaptiveRoutingEnv:
    """Sequential routing environment with agent-output feedback."""

    def __init__(
        self,
        items: list[dict[str, Any]],
        encoder: TfidfStateEncoder,
        reward_fn: RewardSetJaccard,
        max_steps: int = 9,
        seed: int = 42,
        use_action_mask: bool = True,
    ) -> None:
        _validate_adaptive_items(items)
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")

        if len(items) < 100:
            print(
                "Warning: AdaptiveRoutingEnv received fewer than 100 items; "
                "training may be unstable.",
                file=sys.stderr,
            )

        self.items = items
        self.encoder = encoder
        self.reward_fn = reward_fn
        self.max_steps = int(max_steps)
        self.use_action_mask = bool(use_action_mask)
        self.rng = np.random.default_rng(seed)

        self.current_item: dict[str, Any] | None = None
        self.required_set: set[int] = set()
        self.selected_set: set[int] = set()
        self.context_outputs: list[str] = []
        self.step_idx = 0
        self.done = False

    @property
    def state_dim(self) -> int:
        return self.encoder.tfidf_dim * 2 + N_AGENTS

    def set_items(self, items: list[dict[str, Any]]) -> None:
        """Replace the active training item pool (used for curriculum)."""
        _validate_adaptive_items(items)
        self.items = items

    def reset(self) -> np.ndarray:
        idx = int(self.rng.integers(0, len(self.items)))
        self.current_item = self.items[idx]
        self.required_set = set(self.current_item["required_agents"])
        self.selected_set = set()
        self.context_outputs = []
        self.step_idx = 0
        self.done = False
        return self._get_obs()

    def _text_vec(self) -> np.ndarray:
        if self.current_item is None:
            raise RuntimeError("reset() must be called before stepping the environment")

        if "text_vec" in self.current_item:
            return np.asarray(self.current_item["text_vec"], dtype=np.float32)
        return self.encoder.transform_text(self.current_item["text"])

    def _selected_mask(self) -> np.ndarray:
        mask = np.zeros(N_AGENTS, dtype=np.float32)
        if self.selected_set:
            mask[list(self.selected_set)] = 1.0
        return mask

    def _get_agent_output(self, agent_id: int) -> str:
        if self.current_item is None:
            raise RuntimeError("reset() must be called before stepping the environment")

        trajectory = self.current_item["adaptive"]["trajectory"]
        for step in trajectory:
            if int(step.get("agent_id", -1)) == int(agent_id):
                return str(step.get("output", ""))
        return ""

    def _build_context_vec(self) -> np.ndarray:
        if not self.context_outputs:
            return np.zeros(self.encoder.tfidf_dim, dtype=np.float32)

        combined = " ".join(self.context_outputs)
        return self.encoder.transform_text(combined)

    def _get_obs(self) -> np.ndarray:
        if self.current_item is None:
            raise RuntimeError("reset() must be called before stepping the environment")

        text_vec = self._text_vec()
        selected_mask = self._selected_mask()
        context_vec = self._build_context_vec()
        return np.concatenate([text_vec, selected_mask, context_vec]).astype(
            np.float32,
            copy=False,
        )

    def get_action_mask(self) -> np.ndarray:
        mask = np.ones(N_AGENTS + 1, dtype=np.float32)
        if self.use_action_mask:
            for aid in self.selected_set:
                mask[aid] = 0.0
        return mask

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode is done, call reset() to start a new one")
        if not (0 <= int(action) <= STOP_ACTION):
            raise ValueError(f"action must be in [0, {STOP_ACTION}]")

        action = int(action)
        reward = 0.0

        if action != STOP_ACTION:
            agent_output = self._get_agent_output(action)
            if agent_output:
                self.context_outputs.append(agent_output)

            reward += self.reward_fn.step_reward(
                action=action,
                selected=self.selected_set,
                required=self.required_set,
            )
            self.selected_set.add(action)

        self.step_idx += 1

        if action == STOP_ACTION or self.step_idx >= self.max_steps:
            reward += self.reward_fn.terminal_reward(
                selected=self.selected_set,
                required=self.required_set,
            )
            self.done = True

        obs = self._get_obs()
        info: dict[str, Any] = {
            "selected_set": sorted(self.selected_set),
            "required_set": sorted(self.required_set),
            "context_length": len(self.context_outputs),
        }
        if self.use_action_mask:
            info["action_mask"] = self.get_action_mask()
        return obs, float(reward), self.done, info
