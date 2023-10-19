from __future__ import annotations

from typing import Literal

import einops
import gym
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

import uvd.utils as U
from uvd.models.preprocessors import Preprocessor, get_preprocessor
from .policy_base import PolicyBase
from ..distributions import DistributionBase

__all__ = ["MLPPolicy"]


class MLPPolicy(PolicyBase):
    def __init__(
        self,
        *,
        observation_space: gym.spaces.Dict,
        action_space: gym.Space,
        preprocessor: DictConfig | Preprocessor | None = None,
        visual: DictConfig | None = None,
        obs_encoder: DictConfig,
        mlp_output_dim: int | None = None,
        act_head: DictConfig | None = None,
        use_distribution: bool = False,
        bn_to_gn_all: bool = False,
        milestones_compressor: DictConfig | Literal["linear", "flatten"] | None = None,
        obs_add: bool = False,
        **kwargs,
    ):
        super().__init__(**U.prepare_locals_for_super(locals()))

        if visual is not None:
            assert preprocessor is not None
            if isinstance(preprocessor, DictConfig):
                preprocessor = {**preprocessor, "remove_pool": True}
        elif isinstance(preprocessor, DictConfig):
            # frozen embedding during training and/or preprocessor only used during rollout
            preprocessor = {**preprocessor, "remove_pool": False}
        if isinstance(preprocessor, DictConfig):
            self.preprocessor = get_preprocessor(
                device=torch.cuda.current_device(),
                **preprocessor,
            )
        else:
            self.preprocessor = preprocessor

        if visual is not None:
            self.visual = U.hydra_instantiate(
                visual,
                input_shape=self.preprocessor.output_dim,
                preprocessor_fc=self.preprocessor.preprocessor_fc,
            )

        obs_keys = observation_space.spaces.keys()
        rgb_obs_dims = observation_space["rgb"].shape
        self.obs_add = obs_add
        self.rgb_out_dim = 0
        if "rgb" in obs_keys:
            if len(rgb_obs_dims) > 1:
                # output from visual encoder
                assert isinstance(self.visual, nn.Module), rgb_obs_dims
                self.rgb_out_dim = np.prod(self.visual.output_dim)
            else:
                # preprocessed embedding without vis enc
                self.rgb_out_dim = (
                    rgb_obs_dims[0] * 2 if not self.obs_add else rgb_obs_dims[0]
                )

        if milestones_compressor is not None:
            assert not self.obs_add
            assert "milestones" in observation_space.spaces
            if isinstance(milestones_compressor, DictConfig):
                self.milestones_compressor = U.hydra_instantiate(
                    milestones_compressor,
                    milestones_dim=observation_space["milestones"].shape,
                )
                self.rgb_out_dim = (
                    rgb_obs_dims[0] + self.milestones_compressor.output_dim
                )
            # elif milestones_compressor == "linear":
            #     # e.g. 4 * 1024 -> 1024
            #     self.milestones_compressor = nn.Linear(
            #         np.prod(observation_space["milestones"].shape), rgb_obs_dims[0]
            #     )
            elif milestones_compressor == "flatten":
                self.milestones_compressor = nn.Flatten(start_dim=1)
                self.rgb_out_dim = rgb_obs_dims[0] + np.prod(
                    observation_space["milestones"].shape
                )
            else:
                raise NotImplementedError(milestones_compressor)
        else:
            self.milestones_compressor = None

        mlp_input_dim = self.rgb_out_dim

        self.proprio_dim = (
            observation_space["proprio"].shape[0] if "proprio" in obs_keys else 0
        )

        mlp_output_dim = mlp_output_dim or self.action_dim
        self.policy = U.hydra_instantiate(
            obs_encoder,
            input_dim=mlp_input_dim,
            proprio_dim=self.proprio_dim,
            output_dim=mlp_output_dim,
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
        milestone_mask: torch.Tensor | None = None,
        **kwargs,
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
                # B, 2048/1024 or B, 2048, 7, 7
                rgb_embed = self.preprocessor.process(rgb_embed, return_numpy=False)
                if goal is not None and self.milestones_compressor is not None:
                    # multiple milestones, B, N, D or B, N, H, W, 3
                    assert goal.ndim == 3 or goal.ndim == 5, goal.shape
                    if goal.ndim == 5:
                        goal = einops.rearrange(goal, "b n ... -> (b n) ...")
                        goal = self.preprocessor.process(goal, return_numpy=False)
                elif goal is not None and goal.shape[1:] != preprocessor_output_dim:
                    goal = self.preprocessor.process(goal, return_numpy=False)
                if not torch.is_tensor(goal):
                    goal = torch.as_tensor(
                        goal, dtype=rgb_embed.dtype, device=rgb_embed.device
                    )

        if self.milestones_compressor is not None:
            # if goal.ndim != 2:
            #     goal = einops.rearrange(goal, "b n d -> b (n d)")
            goal = self.milestones_compressor(goal, masks=milestone_mask)

        if self.visual is not None:
            # L, D
            x = self.visual(rgb_embed, goal)
        else:
            # fused rgbs frozen embed, L, 2048/1024 directly
            x = torch.cat([rgb_embed, goal], dim=-1) if goal is not None else rgb_embed
        assert x.shape[0] == rgb_embed.shape[0] and x.ndim == 2, x.shape
        # L, action_dim/policy hidden dim with trainable action head
        x = self.policy(x=x, proprio=proprio)
        x = self.act_head(x)
        if deterministic:
            x = x.mode()
        if return_embeddings:
            # return "frozen" embeddings
            return x, rgb_embed, goal
        return x
