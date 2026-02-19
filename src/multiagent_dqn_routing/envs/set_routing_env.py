from __future__ import annotations

from typing import Any

import numpy as np

from multiagent_dqn_routing.agents import N_AGENTS
from multiagent_dqn_routing.rl.state_encoder import TfidfStateEncoder
from multiagent_dqn_routing.sim.reward_set import RewardSetModel

STOP_ACTION = N_AGENTS


class SetRoutingEnv:
    """Environment for set-routing with 9 agents + STOP action."""

    def __init__(
        self,
        items: list[dict[str, Any]],
        encoder: TfidfStateEncoder,
        reward_model: RewardSetModel,
        max_steps: int = 9,
        seed: int = 42,
        use_action_mask: bool = False,
        step_cost: float = 0.0,
    ) -> None:
        if not items:
            raise ValueError("items must not be empty")
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")

        self.items = items
        self.encoder = encoder
        self.reward_model = reward_model
        self.max_steps = int(max_steps)
        self.use_action_mask = bool(use_action_mask)
        self.step_cost = float(step_cost)
        self.rng = np.random.default_rng(seed)

        self.current_item: dict[str, Any] | None = None
        self.required_set: set[int] = set()
        self.selected_set: set[int] = set()
        self.step_idx = 0
        self.done = False

    def reset(self) -> np.ndarray:
        idx = int(self.rng.integers(0, len(self.items)))
        self.current_item = self.items[idx]
        self.required_set = set(self.current_item["required_agents"])
        self.selected_set = set()
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

    def _get_obs(self) -> np.ndarray:
        return self.encoder.encode(
            text_vec=self._text_vec(),
            selected_mask=self._selected_mask(),
            step_idx=self.step_idx,
            max_steps=self.max_steps,
        )

    def get_action_mask(self) -> np.ndarray:
        """Action mask for future masked policies (1 = valid, 0 = invalid)."""
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
            step_reward, _ = self.reward_model.step_reward(
                required_set=self.required_set,
                chosen_agent=action,
                already_selected=self.selected_set,
            )
            reward += step_reward
            reward -= self.step_cost
            self.selected_set.add(action)

        self.step_idx += 1
        if action == STOP_ACTION or self.step_idx >= self.max_steps:
            reward += self.reward_model.terminal_penalty(
                required_set=self.required_set,
                selected_set=self.selected_set,
            )
            self.done = True

        obs2 = self._get_obs()
        info = {
            "selected_set": sorted(self.selected_set),
            "required_set": sorted(self.required_set),
        }
        if self.use_action_mask:
            info["action_mask"] = self.get_action_mask()
        return obs2, float(reward), self.done, info
