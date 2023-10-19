from __future__ import annotations

import einops
import numpy as np
import torch
from typing import Type

from omegaconf import DictConfig

from uvd.models.nn.net_base import NetBase
import torch.nn as nn
import uvd.utils as U

__all__ = ["LinearResampler", "SepResampler"]


class LinearResampler(NetBase):
    def __init__(
        self,
        milestones_dim: tuple | None = None,
        in_dim: int | None = None,
        *,
        out_dim: int,
        **kwargs,
    ):
        super().__init__()
        if milestones_dim is not None and in_dim is not None:
            raise ValueError(
                f"Ambiguous input shape since {milestones_dim=} and {in_dim=} both specified"
            )
        if milestones_dim is not None:
            self.linear = nn.Linear(np.prod(milestones_dim), out_dim)
        else:
            assert in_dim is not None
            self.linear = nn.Linear(in_dim, out_dim, **kwargs)
        self._out_dim = out_dim

    @property
    def output_dim(self) -> tuple | int:
        return self._out_dim

    def forward(self, milestones: torch.Tensor, **kwargs):
        if milestones.ndim == 3:
            milestones = einops.rearrange(milestones, "b n d -> b (n d)")
        x = self.linear(milestones)
        return x


class SepResampler(NetBase):
    def __init__(
        self, milestones_dim: tuple, subsample_module: Type[NetBase], **subsample_kwargs
    ):
        super().__init__()
        self.n_milestone, self.milestone_embed_dim = milestones_dim
        if isinstance(subsample_module, DictConfig):
            self.subsample_module = U.hydra_instantiate(
                subsample_module,
                milestones_dim=(1, self.milestone_embed_dim),
                **subsample_kwargs,
            )
        else:
            self.subsample_module = subsample_module(
                milestones_dim=(1, self.milestone_embed_dim), **subsample_kwargs
            )

    @property
    def output_dim(self) -> tuple | int:
        return self.n_milestone * self.subsample_module.output_dim

    def forward(self, milestones: torch.Tensor, masks: torch.Tensor | None = None):
        assert milestones.ndim == 3, "Not Implement other that B, N, D"
        if masks is not None:
            #  B, N
            assert masks.ndim == 2, masks.shape
            assert masks.shape == milestones.shape[:2], (masks.shape, milestones.shape)
        B, N, *_ = milestones.shape
        outputs = []
        for m in range(N):
            output_l = self.subsample_module(
                milestones[:, m, :],
                masks=None if masks is None else masks[:, m][:, None],  # N, 1
            )  # B, d
            outputs.append(output_l)
        output = torch.concat(outputs, dim=-1)  # B, N*d
        return output
