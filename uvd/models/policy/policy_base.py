from __future__ import annotations

import abc
from typing import Optional

import gym
import numpy as np
import torch
from omegaconf import DictConfig

from uvd.models.distributions.distributions import (
    DistributionBase,
    DistributionHeadBase,
)
from uvd.models.nn.net_base import NetBase
from uvd.models.preprocessors import Preprocessor

__all__ = ["PolicyBase"]


class PolicyBase(NetBase):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Box | gym.spaces.MultiDiscrete,
        *,
        preprocessor: DictConfig | Preprocessor | None = None,
        visual: DictConfig | None = None,
        obs_encoder: DictConfig,
        act_head: DictConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.is_multi_discrete = isinstance(action_space, gym.spaces.MultiDiscrete)
        if not self.is_multi_discrete:
            assert isinstance(action_space, gym.spaces.Box), action_space
            self.action_dim: int = action_space.shape[0]
        else:
            self.action_dim = np.sum(action_space.nvec)
        self.action_space = action_space
        # placeholders
        self.preprocessor: Optional[Preprocessor] = None
        self.visual: Optional[NetBase] = None
        self.policy: Optional[NetBase] = None
        self.act_head: Optional[DistributionHeadBase] = None

    @property
    def output_dim(self) -> tuple | int | np.ndarray:
        return self.action_dim

    @abc.abstractmethod
    def forward(
        self,
        obs: dict[str, torch.Tensor] | torch.Tensor | np.ndarray,
        goal: torch.Tensor | np.ndarray | None,
        deterministic: bool = False,
        return_embeddings: bool = False,
        **kwargs,
    ) -> torch.Tensor | DistributionBase | tuple:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return (
            f"{self.preprocessor.__repr__()}" if self.preprocessor is not None else ""
        )
