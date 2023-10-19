from __future__ import annotations

import gym
import numpy as np

__all__ = ["register_gym_env", "MultiDiscretizeEnvWrapper", "ActionWrapper"]

import torch


def register_gym_env(env_id: str, **kwargs):
    """Decorator for gym env registration."""

    def _register(cls):
        gym.register(
            id=env_id, entry_point=f"{cls.__module__}:{cls.__name__}", kwargs=kwargs
        )
        return cls

    return _register


class MultiDiscretizeEnvWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        num_bins: int = 7,
    ):
        super().__init__(env=env)
        assert isinstance(self.action_space, gym.spaces.Box), self.action_space
        self.action_wrapper = ActionWrapper(
            action_space=self.action_space, num_bins=num_bins
        )
        self.action_space = self.action_wrapper.action_space

    def step(self, action: np.ndarray | int) -> tuple:
        continuous_action = self.action_wrapper.undiscretize_action(action)
        return super().step(continuous_action)


class ActionWrapper:
    def __init__(self, action_space: gym.spaces.Box, num_bins: int):
        self.original_action_space = action_space
        low, high = action_space.low, action_space.high
        action_dim = len(low)
        self.low, self.high = low[0], high[0]
        self.action_ranges = np.array(
            [np.linspace(low[i], high[i], num_bins) for i in range(action_dim)]
        )
        self.action_dim = action_dim
        self._action_space = gym.spaces.MultiDiscrete(
            [num_bins for _ in range(action_dim)]
        )
        self.num_bins = num_bins

    @property
    def action_space(self) -> gym.spaces.MultiDiscrete:
        return gym.spaces.MultiDiscrete([self.num_bins for _ in range(self.action_dim)])

    def discretize_action(
        self, action: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Continuous to discrete."""
        if torch.is_tensor(action):
            assert action.ndim == 2, action.shape  # batch-wise
            return ((action - self.low) / (self.high - self.low) * self.num_bins).long()
        discretized_action = (
            (action - self.low) / (self.high - self.low) * self.num_bins
        ).astype(int)
        return np.clip(discretized_action, 0, self.num_bins - 1)

    def undiscretize_action(
        self, action: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Discrete to continuous."""
        if torch.is_tensor(action):
            raise NotImplementedError
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        return self.action_ranges[np.arange(self.action_dim), action]
