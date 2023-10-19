from __future__ import annotations

import gym
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

import uvd.utils as U
from uvd.models.preprocessors import get_preprocessor
from .policy_base import PolicyBase
from .. import MLP
from ..distributions import DistributionBase

__all__ = ["LanguageConditionedMLPPolicy"]


class LanguageConditionedMLPPolicy(PolicyBase):
    def __init__(
        self,
        *,
        observation_space: gym.spaces.Dict,
        action_space: gym.Space,
        preprocessor: DictConfig | None = None,
        visual: DictConfig | None = None,
        obs_encoder: DictConfig,
        act_head: DictConfig | None = None,
        use_distribution: bool = False,
        bn_to_gn_all: bool = False,
        visual_as_attention_mask: bool = False,
        condition_embed_diff: bool = False,
        **kwargs,
    ):
        super().__init__(**U.prepare_locals_for_super(locals()))

        if visual is not None:
            raise NotImplementedError
            # assert preprocessor is not None
            # preprocessor = {**preprocessor, "remove_pool": True}
        else:
            # frozen embedding during training and/or preprocessor only used during rollout
            preprocessor = {**preprocessor, "remove_pool": False}
        self.preprocessor = get_preprocessor(
            device=torch.cuda.current_device(),
            **preprocessor,
        )

        obs_keys = observation_space.spaces.keys()
        rgb_obs_dims = observation_space["rgb"].shape
        self.rgb_out_dim = 0
        if "rgb" in obs_keys:
            # (embed_dim, )
            self.rgb_out_dim = rgb_obs_dims[0] * 2

        self.proprio_dim = (
            observation_space["proprio"].shape[0] if "proprio" in obs_keys else 0
        )

        mlp_input_dim = self.rgb_out_dim
        self.mlp: MLP = U.hydra_instantiate(
            obs_encoder,
            input_dim=mlp_input_dim,
            proprio_dim=self.proprio_dim,
            output_dim=self.action_dim,
            actor_critic=False,
        )
        self.act_head = U.hydra_instantiate(
            act_head,
            action_dim=action_space if self.is_multi_discrete else self.action_dim,
        )

        if bn_to_gn_all:
            U.bn_to_gn(self)

    def forward(
        self,
        obs: dict[str, torch.Tensor] | torch.Tensor | np.ndarray,
        goal: torch.Tensor | np.ndarray | None,
        deterministic: bool = False,
        return_embeddings: bool = False,
    ) -> torch.Tensor | DistributionBase | tuple:
        if isinstance(obs, dict):
            rgb_embed = obs["rgb"]
            proprio = obs.get("proprio", None)
        else:
            rgb_embed = obs
            proprio = None

        if self.preprocessor is not None:
            preprocessor_output_dim = self.preprocessor.output_dim
            preprocessor_output_dim = (
                (preprocessor_output_dim,)
                if isinstance(preprocessor_output_dim, int)
                else preprocessor_output_dim
            )
            if rgb_embed.shape[1:] != preprocessor_output_dim:
                # B, H, W, 3 or B, 3, H, W after transformed
                assert (
                    rgb_embed.ndim == 4
                ), f"{rgb_embed.shape}, {preprocessor_output_dim}"
                rgb_embed = self.preprocessor.process(rgb_embed, return_numpy=False)

            # language goal, could be different from each substask or for entire task
            if goal is not None and goal.shape[1:] != preprocessor_output_dim:
                # goal = self.preprocessor.encode_text(goal)
                raise NotImplementedError(goal)
            if not torch.is_tensor(goal):
                goal = torch.as_tensor(
                    goal, dtype=rgb_embed.dtype, device=rgb_embed.device
                )

        x = torch.cat([rgb_embed, goal], dim=-1) if goal is not None else rgb_embed
        assert x.shape[0] == rgb_embed.shape[0] and x.ndim == 2, x.shape
        # L, action_dim
        x = self.mlp(x=x, proprio=proprio)
        x = self.act_head(x)
        if deterministic:
            x = x.mode()
        if return_embeddings:
            # return "frozen" embeddings
            return x, rgb_embed, goal
        return x
