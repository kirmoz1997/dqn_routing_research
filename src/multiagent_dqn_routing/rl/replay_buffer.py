from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque

import numpy as np
import torch


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: float


class ReplayBuffer:
    """Simple uniform replay buffer for off-policy RL."""

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = int(capacity)
        self._buffer: Deque[Transition] = deque(maxlen=self.capacity)
        self._rng = random.Random(seed)

    def add(
        self,
        s: np.ndarray,
        a: int,
        r: float,
        s2: np.ndarray,
        done: bool | float,
    ) -> None:
        transition = Transition(
            state=np.asarray(s, dtype=np.float32),
            action=int(a),
            reward=float(r),
            next_state=np.asarray(s2, dtype=np.float32),
            done=float(done),
        )
        self._buffer.append(transition)

    def sample(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if batch_size > len(self._buffer):
            raise ValueError("batch_size cannot exceed replay buffer length")

        batch = self._rng.sample(list(self._buffer), batch_size)

        states = np.stack([t.state for t in batch], axis=0).astype(np.float32, copy=False)
        actions = np.asarray([t.action for t in batch], dtype=np.int64)
        rewards = np.asarray([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch], axis=0).astype(np.float32, copy=False)
        dones = np.asarray([t.done for t in batch], dtype=np.float32)

        return (
            torch.from_numpy(states),
            torch.from_numpy(actions),
            torch.from_numpy(rewards),
            torch.from_numpy(next_states),
            torch.from_numpy(dones),
        )

    def __len__(self) -> int:
        return len(self._buffer)
