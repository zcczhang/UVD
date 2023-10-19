from __future__ import annotations

from typing import Callable

import torch
from einops import rearrange
from torch import nn

from uvd.models.nn.net_base import NetBase
import uvd.utils as U

__all__ = ["make_mlp_hidden", "convert_activation", "MLP", "one_layer_state_encoder"]


def make_mlp_hidden(
    nl: Callable,
    hidden_dims: tuple,
    *,
    normalization: bool = False,
    dropouts: tuple | None = None,
) -> list:
    """Create mlp hidden layers list."""
    if dropouts is not None:
        assert len(dropouts) == len(hidden_dims)
    res = []
    for it, dim in enumerate(hidden_dims[:-1]):
        res.append(nn.Linear(dim, hidden_dims[it + 1]))
        if normalization:
            res.append(nn.LayerNorm(hidden_dims[it + 1]))
        res.append(nl())
        if dropouts is not None and dropouts[it] > 0.0:
            res.append(nn.Dropout(dropouts[it]))
    return res


def convert_activation(activation: str | Callable) -> Callable:
    if isinstance(activation, Callable):
        return activation
    if activation == "ReLU":
        return nn.ReLU
    elif activation == "Tanh":
        return nn.Tanh
    elif activation == "LeakyReLU":
        return nn.LeakyReLU
    else:
        raise NotImplementedError(activation)


class MLP(NetBase):
    def __init__(
        self,
        input_dim: int,  # assume rgb output dim
        output_dim: int,
        hidden_dims: tuple = (),
        activation: str | Callable = nn.ReLU,
        dropouts: tuple | None = None,
        normalization: bool = False,
        input_normalization: str | None = None,
        input_normalization_full_obs: bool = False,
        proprio_dim: int | None = None,
        proprio_output_dim: int | None = None,
        proprio_add_layernorm: bool = True,
        proprio_activation: str | Callable | None = None,
        proprio_add_noise: float | None = None,
        proprio_add_noise_eval: bool = False,
        actor_act: bool | str | Callable | None = None,
        actor_critic: bool = False,
        ft_frozen_bn: bool = False,
        ft_actor_last_layer_only: bool = False,
    ):
        super().__init__()
        self._batch_norm = None
        self.input_normalization_full_obs = input_normalization_full_obs
        if input_normalization is not None:
            if isinstance(input_normalization, str):
                input_normalization = getattr(nn, input_normalization)
            bn_dim = (
                input_dim + (proprio_output_dim or proprio_dim or 0)
                if input_normalization_full_obs
                else input_dim
            )
            self._batch_norm = input_normalization(bn_dim)

        self.use_proprio = False
        if proprio_dim is not None and proprio_dim > 0:
            self.use_proprio = True
            self.proprio_add_noise = proprio_add_noise
            self.proprio_add_noise_eval = proprio_add_noise_eval
            self._proprio_encoder = one_layer_state_encoder(
                state_dim=proprio_dim,
                output_size=proprio_output_dim,
                add_layernorm=proprio_add_layernorm,
                activation_fn=convert_activation(proprio_activation)()
                if proprio_activation is not None
                else None,
            )
            input_dim += proprio_output_dim or proprio_dim

        hidden_dims = (input_dim,) + tuple(hidden_dims or ())
        self._actor = nn.Sequential(
            *make_mlp_hidden(
                convert_activation(activation),
                hidden_dims=hidden_dims,
                normalization=normalization,
                dropouts=dropouts,
            ),
            nn.Linear(hidden_dims[-1], output_dim),
        )

        if actor_act is True:
            self.actor_act = convert_activation(activation)()
        elif actor_act is None:
            self.actor_act = nn.Identity()
        else:
            self.actor_act = convert_activation(actor_act)()

        self._output_dim = output_dim

        self.actor_critic = actor_critic
        self.ft_frozen_bn = (
            actor_critic and ft_frozen_bn and self._batch_norm is not None
        )
        self.ft_actor_last_layer_only = actor_critic and ft_actor_last_layer_only
        if actor_critic:
            self._critic = nn.Sequential(
                *make_mlp_hidden(
                    convert_activation(activation),
                    hidden_dims=hidden_dims,
                    normalization=normalization,
                    dropouts=dropouts,
                ),
                nn.Linear(hidden_dims[-1], 1),
            )

    def train(self, mode: bool = True) -> "MLP":
        super().train(mode)
        if mode and self.ft_frozen_bn:
            self._batch_norm.eval()
        if mode and self.ft_actor_last_layer_only:
            if self._proprio_encoder is not None:
                U.freeze_module(self._proprio_encoder)
            U.freeze_module(self._actor)
            U.unfreeze_module(self._actor[-1])
        return self

    @property
    def output_dim(self):
        return self._output_dim

    def forward(
        self, x: torch.Tensor, proprio: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        *leading_dim, D = x.shape
        if self._batch_norm is not None and not self.input_normalization_full_obs:
            if self.actor_critic:
                assert len(leading_dim) == 2, leading_dim
                x = rearrange(x, "L B D -> (L B) D")
            x = self._batch_norm(x)
            if self.actor_critic:
                x = rearrange(x, "(L B) D -> L B D", L=leading_dim[0], B=leading_dim[1])
        if self.use_proprio:
            assert proprio is not None
            if self.proprio_add_noise is not None and (
                self.training or self.proprio_add_noise_eval
            ):
                proprio = (
                    proprio.float() + torch.randn_like(proprio) * self.proprio_add_noise
                )
            proprio_rep = self._proprio_encoder(proprio.float())
            x = torch.cat([x, proprio_rep], dim=-1)
            if self._batch_norm is not None and self.input_normalization_full_obs:
                if self.actor_critic:
                    x = rearrange(x, "L B D -> (L B) D")
                x = self._batch_norm(x)
                if self.actor_critic:
                    x = rearrange(x, "(L B) D -> L B D", L=leading_dim[0])

        actor = self._actor(x)
        actor = torch.clip(actor, -0.999, 0.999)
        actor = self.actor_act(actor)
        if self.actor_critic:
            return actor, self._critic(x)
        return self.actor_act(self._actor(x))


def one_layer_state_encoder(
    state_dim: int,
    output_size: int | None = None,
    num_stacked_frames: int | None = None,
    add_layernorm: bool = True,
    activation_fn: nn.Module = nn.Tanh(),
) -> torch.nn.Sequential | nn.Identity | None:
    if output_size is None:
        return nn.Identity()
    if state_dim is None or state_dim == 0:
        return None
    enc = nn.Sequential(nn.Linear(state_dim * (num_stacked_frames or 1), output_size))
    if add_layernorm:
        enc.append(nn.LayerNorm(output_size))
    if activation_fn is not None:
        enc.append(activation_fn)
    return enc
