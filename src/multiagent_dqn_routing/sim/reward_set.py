"""Reward model for multi-agent *set* routing (v1.0.0).

Rewards follow the Research Plan:
  • +alpha  for covering a new required agent,
  • −beta   for selecting an unnecessary agent,
  • −step_cost  for every non-STOP action (encourages early STOP),
  • −gamma × missing_count  terminal penalty on STOP / max-steps.

The stochastic component mirrors RewardModel: a required agent
"fires" with probability p_good; an unnecessary agent "fires"
with probability p_bad (the reward sign is still negative).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet

import numpy as np


@dataclass
class RewardSetConfig:
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 1.0
    p_good: float = 0.85
    p_bad: float = 0.30
    step_cost: float = 0.0
    seed: int = 42


class RewardSetModel:
    """Step-level and terminal reward for set-routing episodes."""

    def __init__(self, cfg: RewardSetConfig | None = None) -> None:
        self.cfg = cfg or RewardSetConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    # ── per-step reward ──────────────────────────────────────────────────

    def step_reward(
        self,
        required_set: AbstractSet[int],
        chosen_agent: int,
        already_selected: AbstractSet[int],
    ) -> tuple[float, bool]:
        """Return ``(reward, covered_new)``.

        Parameters
        ----------
        required_set:
            Ground-truth set *R* of required agent ids.
        chosen_agent:
            The agent id selected at this step.
        already_selected:
            Set of agent ids that have **already** been selected
            in previous steps (does **not** include *chosen_agent*).

        Returns
        -------
        reward : float
            Positive if a new required agent was covered and "fired",
            negative if the agent is unnecessary.
        covered_new : bool
            ``True`` when *chosen_agent* ∈ *required_set* and was not
            in *already_selected*.
        """
        is_required = chosen_agent in required_set
        is_new = chosen_agent not in already_selected

        if is_required and is_new:
            fires = float(self.rng.random()) < self.cfg.p_good
            reward = self.cfg.alpha if fires else 0.0
            return reward, True

        if is_required and not is_new:
            # duplicate pick of a required agent — no new coverage
            return 0.0, False

        # unnecessary agent
        fires = float(self.rng.random()) < self.cfg.p_bad
        penalty = -self.cfg.beta if fires else 0.0
        return penalty, False

    # ── terminal penalty ─────────────────────────────────────────────────

    def terminal_penalty(
        self,
        required_set: AbstractSet[int],
        selected_set: AbstractSet[int],
    ) -> float:
        """Return ``-gamma * missing_count`` (≤ 0).

        Parameters
        ----------
        required_set:
            Ground-truth set *R*.
        selected_set:
            Final set *S* chosen by the router.
        """
        missing = len(required_set - selected_set)
        return -self.cfg.gamma * missing
