from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RewardConfig:
    p_good: float = 0.85
    p_bad: float = 0.30
    seed: int = 42


class RewardModel:
    def __init__(self, cfg: RewardConfig | None = None):
        self.cfg = cfg or RewardConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    def sample_reward(self, true_label: int, chosen_agent: int) -> int:
        p = self.cfg.p_good if chosen_agent == true_label else self.cfg.p_bad
        return 1 if float(self.rng.random()) < p else 0
