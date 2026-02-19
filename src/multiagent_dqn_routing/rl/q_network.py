from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class QNetwork(nn.Module):
    """MLP that maps state vectors to Q-values."""

    def __init__(
        self,
        input_dim: int,
        n_actions: int = 10,
        hidden_sizes: Iterable[int] = (256, 256),
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if n_actions <= 0:
            raise ValueError("n_actions must be > 0")

        layers: list[nn.Module] = []
        prev_dim = int(input_dim)
        for h in hidden_sizes:
            h = int(h)
            if h <= 0:
                raise ValueError("hidden layer size must be > 0")
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, int(n_actions)))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
