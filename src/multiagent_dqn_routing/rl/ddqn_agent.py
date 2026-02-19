from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F

from multiagent_dqn_routing.rl.q_network import QNetwork


class DoubleDQNAgent:
    """Double DQN agent with online and target Q-networks."""

    def __init__(
        self,
        input_dim: int,
        n_actions: int = 10,
        hidden_sizes: tuple[int, ...] = (256, 256),
        learning_rate: float = 1e-3,
        discount: float = 0.99,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.device = torch.device(device)
        self.n_actions = int(n_actions)
        self.discount = float(discount)
        self._rng = random.Random(seed)

        self.q = QNetwork(
            input_dim=input_dim,
            n_actions=self.n_actions,
            hidden_sizes=hidden_sizes,
        ).to(self.device)
        self.q_target = QNetwork(
            input_dim=input_dim,
            n_actions=self.n_actions,
            hidden_sizes=hidden_sizes,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=learning_rate)
        self.update_target()

    def select_action(
        self,
        state: np.ndarray | torch.Tensor,
        epsilon: float,
        mask: np.ndarray | torch.Tensor | None = None,
    ) -> int:
        if self._rng.random() < float(epsilon):
            if mask is None:
                return self._rng.randrange(self.n_actions)
            mask_np = np.asarray(mask, dtype=np.float32).reshape(-1)
            valid = np.where(mask_np > 0.0)[0]
            if valid.size == 0:
                return self._rng.randrange(self.n_actions)
            return int(self._rng.choice(valid.tolist()))

        if not torch.is_tensor(state):
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        else:
            state_t = state.to(self.device, dtype=torch.float32)
        if state_t.ndim == 1:
            state_t = state_t.unsqueeze(0)

        with torch.no_grad():
            q_values = self.q(state_t).squeeze(0)
            if mask is not None:
                mask_t = torch.as_tensor(mask, dtype=torch.float32, device=self.device).reshape(-1)
                if mask_t.numel() != self.n_actions:
                    raise ValueError("mask size must match n_actions")
                q_values = q_values.masked_fill(mask_t <= 0.0, float("-inf"))
            action = torch.argmax(q_values, dim=-1).item()
        return int(action)

    def train_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> float:
        states, actions, rewards, next_states, dones = batch
        states = states.to(self.device, dtype=torch.float32)
        actions = actions.to(self.device, dtype=torch.int64)
        rewards = rewards.to(self.device, dtype=torch.float32)
        next_states = next_states.to(self.device, dtype=torch.float32)
        dones = dones.to(self.device, dtype=torch.float32)

        q_values = self.q(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.q(next_states).argmax(dim=1, keepdim=True)
            next_q_target = self.q_target(next_states).gather(1, next_actions).squeeze(1)
            target = rewards + self.discount * next_q_target * (1.0 - dones)

        loss = F.smooth_l1_loss(q_sa, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def update_target(self) -> None:
        self.q_target.load_state_dict(self.q.state_dict())
