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
import math
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


class RewardSetJaccard:
    """Deterministic terminal Jaccard reward for set-routing episodes.

    Motivation
    ----------
    The original ``RewardSetModel`` uses stochastic per-step rewards
    (p_good / p_bad firing probabilities) plus a terminal gamma penalty
    for missing agents.  In practice this creates high-variance Q-targets
    and conflicting signals: the agent becomes afraid of missing agents
    (large gamma) and never learns to press STOP, selecting ~9/9 agents
    every episode.

    This class replaces the stochastic scheme with a clean two-part
    signal:

    * **Per-step:** a small fixed cost ``-step_cost`` that penalises
      every agent selection, encouraging the router to stop early.
    * **Terminal:** the Jaccard similarity ``|S ∩ R| / |S ∪ R|``
      between the selected set *S* and the ground-truth set *R*.
      This directly matches the evaluation metric and provides an
      unambiguous, deterministic reward at the end of the episode.

    Formulas
    --------
    ``r_step  = -step_cost``
    ``r_term  = |S ∩ R| / |S ∪ R|``  (Jaccard index)
    ``F1      = 2·|S ∩ R| / (|S| + |R|)``  (set-level F1)
    """

    def __init__(self, step_cost: float = 0.05) -> None:
        self.step_cost = float(step_cost)

    def step_reward(
        self,
        action: int,
        selected: AbstractSet[int],
        required: AbstractSet[int],
    ) -> float:
        """Fixed penalty per agent-selection step (no stochasticity)."""
        return -self.step_cost

    def terminal_reward(
        self,
        selected: set[int],
        required: set[int],
    ) -> float:
        """Jaccard index ``|S ∩ R| / |S ∪ R|`` at episode end."""
        if not selected and not required:
            return 1.0
        union = len(selected | required)
        if union == 0:
            return 0.0
        return len(selected & required) / union

    def terminal_reward_f1(
        self,
        selected: set[int],
        required: set[int],
    ) -> float:
        """Set-level F1: ``2·|S ∩ R| / (|S| + |R|)``."""
        if not selected or not required:
            return 0.0
        return 2 * len(selected & required) / (len(selected) + len(required))


class RewardSetLogJaccard:
    """Logarithmic step penalty + terminal Jaccard reward.

    Motivation
    ----------
    Both RewardSetModel (stochastic) and RewardSetJaccard (flat step_cost)
    suffer from "select-all collapse": the agent selects all 9 agents
    every episode because marginal Jaccard gain (~0.06) always exceeds
    flat step_cost (0.05). This is a structural local optimum that cannot
    be escaped by tuning step_cost alone.

    This class implements the logarithmic penalty from Puppeteer
    (Dang et al., NeurIPS 2025), adapted for set routing:

        r_step(t) = -lambda * log(1 + t / max_steps)
        r_terminal = Jaccard(S, R)

    The TOTAL episode cost is:
        R_episode = Jaccard(S, R) - lambda * sum(log(1 + t / max_steps))

    Key property: step cost grows monotonically with t.
    - Step 1 (lambda=0.10, max_steps=9): cost = 0.10 * log(1+1/9) ~= 0.011
    - Step 5:                           cost = 0.10 * log(1+5/9) ~= 0.044
    - Step 9:                           cost = 0.10 * log(1+9/9) ~= 0.069

    This creates a "cheap early steps, expensive late steps" structure:
    the agent is not penalized heavily for initial exploration but faces
    growing costs for over-selection. The select-all strategy becomes
    increasingly irrational as T grows.

    Comparison with flat step_cost=0.05 (9 steps):
        Flat total penalty:    0.05 * 9 = 0.450
        Log total penalty:     lambda * sum(log(1 + t/9)) ~= 0.382
    Absolute values are similar, but the SHAPE is different.

    Parameters
    ----------
    lambda_eff : float
        Efficiency weight lambda. Controls trade-off between Jaccard quality
        and episode length. Recommended starting value: 0.10 (Puppeteer).
    max_steps : int
        Maximum episode length. Used to normalize step index in log formula.

    Reference
    ---------
    Dang et al. "Multi-Agent Collaboration via Evolving Orchestration"
    NeurIPS 2025 (Puppeteer). Adapted logarithmic cost from Section 3.2.
    """

    def __init__(self, lambda_eff: float = 0.10, max_steps: int = 9) -> None:
        self.lambda_eff = float(lambda_eff)
        self.max_steps = int(max_steps)

    def step_reward(
        self,
        step_idx: int,
        action: int,
        selected: AbstractSet[int],
        required: AbstractSet[int],
    ) -> float:
        """Logarithmic penalty for step t (1-indexed).

        Parameters
        ----------
        step_idx : int
            Current step index, 0-indexed (first step = 0).
            Internally converted to 1-indexed for formula.
        action : int
            The agent selected (unused in reward, kept for API consistency).
        selected : AbstractSet[int]
            Agents selected so far (before this step).
        required : AbstractSet[int]
            Ground-truth required agents.
        """
        del action, selected, required
        t = step_idx + 1
        cost = self.lambda_eff * math.log(1.0 + t / self.max_steps)
        return -cost

    def terminal_reward(
        self,
        selected: set[int],
        required: set[int],
    ) -> float:
        """Jaccard similarity at episode end (no additional penalty here).

        The log penalties are accumulated per-step in step_reward().
        Terminal reward is pure Jaccard: clean, metric-aligned signal.
        """
        if not selected and not required:
            return 1.0
        union = len(selected | required)
        if union == 0:
            return 0.0
        return len(selected & required) / union

    def cumulative_log_cost(self, n_steps: int) -> float:
        """Compute total log penalty for n_steps (for analysis/logging).

        Returns lambda * sum(log(1 + t / max_steps)) for t in [1, n_steps].
        """
        return self.lambda_eff * sum(
            math.log(1.0 + t / self.max_steps)
            for t in range(1, n_steps + 1)
        )
