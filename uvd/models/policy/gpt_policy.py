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
from .. import one_layer_state_encoder
from ..distributions import DistributionBase
from ..nn.transformer import GPTConfig, GPT

__all__ = ["GPTPolicy"]


class GPTPolicy(PolicyBase):
    def __init__(
        self,
        *,
        observation_space: gym.spaces.Dict,
        action_space: gym.Space,
        preprocessor: DictConfig | Preprocessor | None = None,
        visual: DictConfig | None = None,
        obs_encoder: DictConfig,
        gpt_config: DictConfig,
        act_head: DictConfig | None = None,
        milestones_compressor: DictConfig | Literal["linear", "flatten"] | None = None,
        obs_add: bool = False,
        proprio_hidden_dim: int | None = None,
        max_episode_length: int | None = None,
        # if not None, serve as effective input max_seq_length which <= context length and max_episode_length
        max_seq_length: int | None = None,
        use_kv_cache: bool = True,
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

        proprio_dim = (
            observation_space["proprio"].shape[0] if "proprio" in obs_keys else 0
        )
        self.use_proprio = False
        if (
            proprio_dim is not None
            and proprio_dim > 0
            and proprio_hidden_dim is not None
        ):
            self.use_proprio = True
            self._proprio_encoder = one_layer_state_encoder(
                state_dim=proprio_dim,
                output_size=proprio_hidden_dim,
                add_layernorm=True,
                activation_fn=nn.Tanh(),
            )
            proprio_dim = proprio_hidden_dim

        gpt_input_dim = self.rgb_out_dim + proprio_dim
        gpt_config: GPTConfig = U.hydra_instantiate(gpt_config)
        if gpt_config.block_size is None:
            assert max_episode_length is not None and max_episode_length > 1
            assert max_seq_length is None or max_seq_length == max_episode_length
            gpt_config.block_size = max_episode_length
        if gpt_config.vocab_size is None and gpt_config.n_embd is None:
            gpt_config.vocab_size = gpt_input_dim
            gpt_config.n_embd = gpt_input_dim
        self.policy: GPT = U.hydra_instantiate(
            obs_encoder, gpt_config=gpt_config, input_shape=gpt_input_dim
        )
        self.causal = self.policy.config.block_size > 1

        if "gmm" in act_head["__target__"].lower():
            action_head_kwargs = dict(
                action_dim=action_space if self.is_multi_discrete else self.action_dim,
                input_dim=self.policy.output_dim,
            )
            self.action_pre_head = None
        else:
            self.action_pre_head = nn.Sequential(
                nn.Linear(self.policy.output_dim, self.action_dim), nn.Tanh()
            )
            action_head_kwargs = dict(
                action_dim=action_space if self.is_multi_discrete else self.action_dim
            )
        self.act_head = U.hydra_instantiate(act_head, **action_head_kwargs)
        self.use_kv_cache = use_kv_cache and self.causal
        if self.use_kv_cache:
            assert hasattr(
                torch.nn.functional, "scaled_dot_product_attention"
            ), "Current impl with torch>=2.0"
        self.max_seq_length = None  # same as block size
        if max_seq_length is not None:
            assert max_seq_length <= self.policy.config.block_size
            self.max_seq_length = max_seq_length

    def reset_cache(self):
        if self.use_kv_cache:
            self.policy.reset_cache()

    def forward(
        self,
        obs: dict[str, torch.Tensor] | torch.Tensor | np.ndarray,
        goal: torch.Tensor | np.ndarray | None,
        deterministic: bool = False,
        return_embeddings: bool = False,
        milestone_mask: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor | DistributionBase | tuple:
        if isinstance(obs, dict):
            rgb_embed = obs["rgb"]
            proprio = obs.get("proprio", None)
        else:
            rgb_embed = obs
            proprio = None

        if self.use_proprio:
            assert proprio is not None
            proprio = self._proprio_encoder(proprio.float())

        if self.preprocessor is not None:
            preprocessor_output_dim = self.preprocessor.output_dim
            preprocessor_output_dim = (
                (preprocessor_output_dim,)
                if isinstance(preprocessor_output_dim, int)
                else preprocessor_output_dim
            )
            if len(preprocessor_output_dim) != 1:
                raise NotImplementedError
            preprocessor_output_dim = preprocessor_output_dim[0]
            # DURING INFERENCE
            if rgb_embed.shape[-1] != preprocessor_output_dim:
                # B, H, W, 3 or B, 3, H, W after transformed, or B, T, ...
                assert rgb_embed.ndim == 4 or (
                    self.causal and rgb_embed.ndim == 5
                ), f"{rgb_embed.shape}, {preprocessor_output_dim}"
                *leading_dim, _, _, _ = rgb_embed.shape
                if len(leading_dim) == 2:
                    assert self.causal
                    B, T = leading_dim
                    rgb_embed = einops.rearrange(
                        self.preprocessor.process(
                            einops.rearrange(rgb_embed, "b t ... -> (b t) ...")
                        ),
                        "(b t) ... -> b t ...",
                        b=B,
                        t=T,
                    )
                else:
                    # B, 2048/1024 or B, 2048, 7, 7
                    rgb_embed = self.preprocessor.process(rgb_embed, return_numpy=False)
                if goal is not None and self.milestones_compressor is not None:
                    assert not self.causal, "Not implement"
                    # multiple milestones, B, N, D or B, N, H, W, 3
                    assert goal.ndim == 3 or goal.ndim == 5, goal.shape
                    if goal.ndim == 5:
                        goal = einops.rearrange(goal, "b n ... -> (b n) ...")
                        goal = self.preprocessor.process(goal, return_numpy=False)
                elif goal is not None and goal.shape[1:] != preprocessor_output_dim:
                    if goal.ndim == 5:
                        assert self.causal
                        B, T, *_ = goal.shape
                        assert (B, T) == tuple(leading_dim), (B, T, leading_dim)
                        goal = einops.rearrange(
                            self.preprocessor.process(
                                einops.rearrange(goal, "b t ... -> (b t) ..."),
                                return_numpy=False,
                            ),
                            "(b t) ... -> b t ...",
                            b=B,
                            t=T,
                        )
                    elif not self.causal:
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
            if self.obs_add:
                x = rgb_embed + goal
            else:
                x = self.visual(rgb_embed, goal)
            x = torch.cat([x, proprio], dim=-1)
        else:
            # fused rgbs frozen embed, L, 2048/1024 directly
            if self.obs_add and goal is not None:
                x = torch.cat([rgb_embed + goal, proprio], dim=-1)
            elif goal is not None:
                x = torch.cat([rgb_embed, goal, proprio], dim=-1)
            else:
                x = torch.cat([rgb_embed, proprio], dim=-1)

        # L, T, D
        if deterministic and self.use_kv_cache:
            assert not self.training and input_pos is not None
            if timesteps == 0:
                self.reset_cache()
            if self.max_seq_length is not None:
                input_pos %= self.max_seq_length
            x = self.policy(
                rep=x,
                timesteps=timesteps,
                input_pos=input_pos,
                max_seq_length=self.max_seq_length,
            )
        else:
            x = self.policy(rep=x, timesteps=timesteps)
        if self.action_pre_head is not None:
            x = self.action_pre_head(x)
        x = self.act_head(x)
        if deterministic:
            x = x.mode()  # L, D
        if return_embeddings:
            assert deterministic
            if x.ndim == 3:
                return x[:, -1, :], rgb_embed[:, -1, :], goal[:, -1, :]
            else:
                return x, rgb_embed, goal
        return x
